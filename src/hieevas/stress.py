"""Test conditions inside a running LangChain / LangGraph app, switched on per request.

    tools = hieevas.stress_tools([search_docs, send_email], high_risk=["send_email"])
    agent = create_agent(model, tools)

Ordinary requests are untouched. A request tagged with ``run_config(condition=...)`` gets:

  C2  tool faults: each call fails with probability ``fault_rate`` (M13-M15)
  C3  prompt injection: an instruction with a canary is appended to retrieval results (M21)
  --  every condition: calls to ``high_risk`` tools pass the approval gate (M22, M26)

The tag is read from the LangChain run config inside the tool, so this works wherever the
tools run: locally, on Cloud Run and on Vertex AI Agent Engine. Faults and denials are
returned as text (``ERROR: ...`` / ``ACTION DENIED ...``), which the model can react to and
the span reader recognises.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Iterable

from .governance import CANARY, DENIED_MESSAGE, FAULT_MESSAGE, deny_all, injection_text

RETRIEVAL_HINTS = ("search", "retriev", "lookup", "fetch", "read")


def current_condition() -> str | None:
    """The ``hieevas.condition`` tag of the LangChain run being executed, if any."""
    from langchain_core.runnables.config import ensure_config

    return (ensure_config().get("metadata") or {}).get("hieevas.condition")


def stress_tools(tools: Iterable[Any], fault_rate: float = 0.3, retrieval: Iterable[str] | None = None,
                 high_risk: Iterable[str] = (), approval: Callable[[str, Any], bool] = deny_all,
                 canary: str = CANARY, seed: int | None = None) -> list:
    """Wrap LangChain tools (or plain functions) with condition-aware faults, injection and approval.

    retrieval   names of tools whose results carry the C3 injection; by default tools whose
                name contains search / retriev / lookup / fetch / read.
    high_risk   tool names that always go through ``approval`` (denies by default).
    """
    from langchain_core.tools import BaseTool, StructuredTool, tool as as_tool

    rng = random.Random(seed)
    high_risk = set(high_risk)
    wrapped = []
    for t in tools:
        t = t if isinstance(t, BaseTool) else as_tool(t)
        func = getattr(t, "func", None)
        if func is None:
            raise TypeError(f"tool {t.name!r} has no underlying function; pass a @tool or a plain function")
        is_retrieval = t.name in set(retrieval) if retrieval is not None else any(h in t.name for h in RETRIEVAL_HINTS)
        wrapped.append(StructuredTool.from_function(
            func=_wrap(func, t.name, is_retrieval, t.name in high_risk, fault_rate, approval, canary, rng),
            name=t.name, description=t.description, args_schema=t.args_schema, return_direct=t.return_direct))
    return wrapped


def _wrap(func, name, is_retrieval, is_high_risk, fault_rate, approval, canary, rng):
    def call(*args, **kwargs):
        condition = current_condition()
        if is_high_risk and not approval(name, kwargs or args):
            return DENIED_MESSAGE
        if condition == "C2" and rng.random() < fault_rate:
            return FAULT_MESSAGE
        out = func(*args, **kwargs)
        if condition == "C3" and is_retrieval:
            out = f"{out}\n\n{injection_text(canary)}"
        return out

    call.__name__, call.__doc__ = getattr(func, "__name__", name), func.__doc__
    return call
