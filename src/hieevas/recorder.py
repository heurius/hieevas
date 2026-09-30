"""Framework-agnostic recording API.

Wrap each task execution in ``Recorder.run(...)`` and report what happens inside
it. Any pipeline can do this by hand; the LangChain/LangGraph adapter and the
governance wrappers call the same API automatically.

    rec = Recorder(app="my-rag", architecture="A1", model="qwen2.5:3b")
    with rec.run("q1", reference="Paris") as run:
        run.llm_call(input_tokens=120, output_tokens=18, latency=0.9)
        run.tool_call("search_docs", {"query": "capital of France"}, rationale="need a source")
        run.answer("Paris")
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

from .scoring import score_answer
from .trace import APPROVAL, LLM, MESSAGE, PLAN, REVIEW, TOOL, RunRecord, Step, load_jsonl, save_csv, save_jsonl

_current: ContextVar["RunContext | None"] = ContextVar("hieevas_current_run", default=None)


class BudgetExceeded(RuntimeError):
    """Raised inside a run when its step budget is exhausted (recorded as M27)."""


def current_run() -> "RunContext | None":
    """The run active in this context, if any (used by wrappers and adapters)."""
    return _current.get()


class RunContext:
    """Records the steps of one run. Obtained from ``Recorder.run``."""

    def __init__(self, record: RunRecord, scorer: str | Callable = "auto", step_budget: int | None = None):
        self.record = record
        self.scorer = scorer
        self.step_budget = step_budget

    # ---- internal -------------------------------------------------------------------
    def _add(self, kind: str, **kw: Any) -> Step:
        step = Step(run_id=self.record.run_id, index=len(self.record.steps), kind=kind, **kw)
        self.record.steps.append(step)
        return step

    def _check_budget(self) -> None:
        if self.step_budget is None:
            return
        actions = sum(1 for s in self.record.steps if s.kind in (LLM, TOOL))
        if actions > self.step_budget:
            self.record.budget_exceeded = True
            raise BudgetExceeded(f"step budget of {self.step_budget} exceeded")

    # ---- recording API ----------------------------------------------------------------
    def llm_call(self, agent: str = "agent", input_tokens: int | None = None, output_tokens: int | None = None,
                 latency: float | None = None, content: str | None = None, **extra: Any) -> Step:
        step = self._add(LLM, agent=agent, input_tokens=input_tokens, output_tokens=output_tokens,
                         latency=latency, content=content, extra=extra)
        self._check_budget()
        return step

    def tool_call(self, tool: str, args: Any = None, status: str = "ok", agent: str = "agent",
                  rationale: str | None = None, risk: str = "low", required: bool | None = None,
                  args_valid: bool | None = None, latency: float | None = None, **extra: Any) -> Step:
        previous = [s for s in self.record.steps if s.kind == TOOL and s.tool == tool]
        retry = bool(previous) and previous[-1].status == "error"
        if args_valid is None and status in ("ok", "denied"):
            args_valid = True
        step = self._add(TOOL, agent=agent, tool=tool, args=args, status=status, rationale=rationale,
                         risk=risk, required=required, args_valid=args_valid, retry=retry,
                         latency=latency, extra=extra)
        self._check_budget()
        return step

    def message(self, sender: str, receiver: str, content: str | None = None, tokens: int | None = None,
                accepted: bool | None = True, **extra: Any) -> Step:
        if tokens is None and content is not None:
            tokens = max(1, len(str(content)) // 4)  # rough estimate when no tokenizer is available
        return self._add(MESSAGE, agent=sender, receiver=receiver, content=content, tokens=tokens,
                         accepted=accepted, extra=extra)

    def review(self, verdict: str, agent: str = "reviewer", rationale: str | None = None, **extra: Any) -> Step:
        if verdict not in ("accept", "reject"):
            raise ValueError("verdict must be 'accept' or 'reject'")
        return self._add(REVIEW, agent=agent, verdict=verdict, rationale=rationale, extra=extra)

    def approval(self, tool: str, decision: str, agent: str = "governance", **extra: Any) -> Step:
        return self._add(APPROVAL, agent=agent, tool=tool, decision=decision, extra=extra)

    def plan(self, steps: list[str], agent: str = "planner") -> Step:
        self.record.planned_steps = list(steps)
        return self._add(PLAN, agent=agent, content="\n".join(steps))

    def plan_step_done(self, index: int) -> None:
        if index not in self.record.executed_plan_steps:
            self.record.executed_plan_steps.append(index)

    def score_rationale(self, step_index: int, score: int) -> None:
        if score not in (1, 2, 3):
            raise ValueError("rationale scores use the 1-3 rubric")
        self.record.steps[step_index].rationale_score = score

    def answer(self, value: Any) -> None:
        self.record.answer = value

    def mark(self, **flags: Any) -> None:
        """Set outcome flags: correct, f1, refused, injection_followed, budget_exceeded."""
        for key, value in flags.items():
            if not hasattr(self.record, key):
                raise AttributeError(f"RunRecord has no field {key!r}")
            setattr(self.record, key, value)

    # ---- completion -------------------------------------------------------------------
    def finish(self) -> None:
        r = self.record
        r.ended = time.time()
        if r.correct is None and r.reference is not None and r.answer is not None:
            r.correct, f1 = score_answer(r.answer, r.reference, self.scorer)
            if r.f1 is None:
                r.f1 = f1


class Recorder:
    """Collects RunRecords for one application / architecture / model."""

    def __init__(self, app: str = "app", architecture: str = "default", model: str | None = None,
                 scorer: str | Callable = "auto", step_budget: int | None = None, raise_errors: bool = True,
                 high_risk_tools: list[str] | None = None):
        self.app, self.architecture, self.model = app, architecture, model
        self.high_risk_tools = list(high_risk_tools or [])
        self.scorer, self.step_budget, self.raise_errors = scorer, step_budget, raise_errors
        self.runs: list[RunRecord] = []

    @contextmanager
    def run(self, task_id: str, condition: str = "C1", reference: Any = None,
            architecture: str | None = None, model: str | None = None,
            scorer: str | Callable | None = None, **metadata: Any) -> Iterator[RunContext]:
        record = RunRecord(run_id=uuid.uuid4().hex[:12], task_id=str(task_id), app=self.app,
                           architecture=architecture or self.architecture, model=model or self.model,
                           condition=condition, reference=reference, metadata=metadata)
        if self.high_risk_tools:
            record.metadata.setdefault("high_risk_tools", self.high_risk_tools)
        ctx = RunContext(record, scorer or self.scorer, self.step_budget)
        token = _current.set(ctx)
        try:
            yield ctx
        except BudgetExceeded:
            record.budget_exceeded = True
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised unless disabled
            if type(exc).__name__ == "GraphRecursionError":  # LangGraph's recursion_limit = step budget
                record.budget_exceeded = True
            else:
                record.error = f"{type(exc).__name__}: {exc}"
                if self.raise_errors:
                    raise
        finally:
            ctx.finish()
            _current.reset(token)
            self.runs.append(record)

    # ---- persistence --------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        return save_jsonl(self.runs, path)

    def save_csv(self, directory: str | Path):
        return save_csv(self.runs, directory)

    @staticmethod
    def load(path: str | Path) -> list[RunRecord]:
        return load_jsonl(path)
