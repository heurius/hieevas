"""LangChain / LangGraph callback adapter.

Pass ``HieevasCallbackHandler()`` in the ``callbacks`` of any LangChain runnable or
LangGraph graph invocation made inside ``Recorder.run(...)``. It records every LLM call
(tokens, latency, agent = LangGraph node name) and every tool call (status, arguments,
and the rationale the model wrote before calling it).

    with rec.run("q1", reference="Paris") as run:
        out = graph.invoke(inputs, config={"callbacks": [HieevasCallbackHandler(run)]})
        run.answer(out["messages"][-1].content)
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError("install the LangChain extra: pip install hieevas[langchain]") from exc

from ..governance import GOVERNED_TOOLS
from ..recorder import RunContext, current_run


class HieevasCallbackHandler(BaseCallbackHandler):
    """Records LangChain/LangGraph events into a hieevas run."""

    def __init__(self, run: RunContext | None = None, agent_key: str = "langgraph_node",
                 default_agent: str = "agent", tool_risk: dict[str, str] | None = None):
        self._run = run
        self.agent_key, self.default_agent = agent_key, default_agent
        self.tool_risk = tool_risk or {}
        self._llm: dict[UUID, tuple[float, str]] = {}
        self._tools: dict[UUID, tuple[float, str, Any, str]] = {}
        self._pending: dict[str, list[tuple[str | None, str]]] = {}

    # ---- helpers --------------------------------------------------------------------------
    @property
    def run(self) -> RunContext | None:
        return self._run or current_run()

    def _agent(self, metadata: dict | None) -> str:
        return (metadata or {}).get(self.agent_key) or self.default_agent

    # ---- LLM events ------------------------------------------------------------------------
    def on_llm_start(self, serialized, prompts, *, run_id: UUID, metadata=None, **kw):
        self._llm[run_id] = (time.perf_counter(), self._agent(metadata))

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kw):
        self._llm[run_id] = (time.perf_counter(), self._agent(metadata))

    def on_llm_end(self, response, *, run_id: UUID, **kw):
        run = self.run
        start, agent = self._llm.pop(run_id, (None, self.default_agent))
        if run is None:
            return
        tin = tout = None
        text = None
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            gen = gens[0][0]
            text = getattr(gen, "text", None)
            msg = getattr(gen, "message", None)
            usage = getattr(msg, "usage_metadata", None) if msg is not None else None
            if usage:
                tin, tout = usage.get("input_tokens"), usage.get("output_tokens")
            if msg is not None and getattr(msg, "tool_calls", None):
                reason = str(msg.content or "").strip() or None  # reasoning written before the tool call
                for call in msg.tool_calls:
                    self._pending.setdefault(call.get("name"), []).append((reason, agent))
        if tin is None:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        run.llm_call(agent=agent, input_tokens=tin, output_tokens=tout,
                     latency=None if start is None else time.perf_counter() - start, content=text)

    def on_llm_error(self, error, *, run_id: UUID, **kw):
        self._llm.pop(run_id, None)

    # ---- tool events -------------------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id: UUID, metadata=None, inputs=None, **kw):
        name = (serialized or {}).get("name") or kw.get("name") or "tool"
        self._tools[run_id] = (time.perf_counter(), name, inputs if inputs is not None else input_str,
                               self._agent(metadata))

    def _tool_done(self, run_id: UUID, status: str):
        start, name, args, agent = self._tools.pop(run_id, (None, "tool", None, self.default_agent))
        run = self.run
        if run is None or name in GOVERNED_TOOLS:  # governed tools log themselves
            return
        queue = self._pending.get(name) or []
        reason, requester = queue.pop(0) if queue else (None, agent)
        run.tool_call(name, args, status=status, agent=requester, rationale=reason,
                      risk=self.tool_risk.get(name, "low"),
                      latency=None if start is None else time.perf_counter() - start)

    def on_tool_end(self, output, *, run_id: UUID, **kw):
        self._tool_done(run_id, "ok")

    def on_tool_error(self, error, *, run_id: UUID, **kw):
        self._tool_done(run_id, "error")
