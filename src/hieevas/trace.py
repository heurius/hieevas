"""Trace data model: one RunRecord per task execution, one Step per agent action.

The fields follow the logging schema of the paper (Table 13). Every metric is
computed from these records, so an agent never has to be re-run to be scored.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

# Kinds of step a run can contain.
LLM, TOOL, MESSAGE, REVIEW, APPROVAL, PLAN = "llm", "tool", "message", "review", "approval", "plan"
STEP_KINDS = (LLM, TOOL, MESSAGE, REVIEW, APPROVAL, PLAN)
ACTION_KINDS = (LLM, TOOL)  # steps counted as "actions" (M9)


@dataclass
class Step:
    run_id: str
    index: int
    kind: str
    agent: str = "agent"
    timestamp: float = field(default_factory=time.time)
    # LLM calls
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency: float | None = None
    # tool calls
    tool: str | None = None
    args: Any = None
    args_valid: bool | None = None
    status: str | None = None  # "ok" | "error" | "denied"
    risk: str = "low"  # "low" | "high"
    required: bool | None = None  # was a high-risk call actually needed by the task?
    retry: bool = False
    rationale: str | None = None
    rationale_score: int | None = None  # 1-3 rubric (M24)
    # messages, reviews, approvals
    receiver: str | None = None
    content: str | None = None
    tokens: int | None = None
    accepted: bool | None = None
    verdict: str | None = None  # review: "accept" | "reject"
    decision: str | None = None  # approval: "approved" | "denied"
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    app: str = "app"
    architecture: str = "default"
    model: str | None = None
    condition: str = "C1"
    started: float = field(default_factory=time.time)
    ended: float | None = None
    answer: Any = None
    reference: Any = None
    correct: bool | None = None
    f1: float | None = None
    refused: bool | None = None
    injection_followed: bool | None = None
    budget_exceeded: bool = False
    error: str | None = None
    planned_steps: list[str] | None = None
    executed_plan_steps: list[int] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        return None if self.ended is None else self.ended - self.started

    def steps_of(self, *kinds: str) -> list[Step]:
        return [s for s in self.steps if s.kind in kinds]

    # ---- serialisation --------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        data = dict(data)
        steps = [Step(**s) for s in data.pop("steps", [])]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known}, steps=steps)


def save_jsonl(runs: Iterable[RunRecord], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r.to_dict(), default=str) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[RunRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        return [RunRecord.from_dict(json.loads(line)) for line in fh if line.strip()]


def save_csv(runs: Iterable[RunRecord], directory: str | Path) -> tuple[Path, Path]:
    """Write the two logs of the paper's schema: runs.csv and steps.csv."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    runs = list(runs)
    run_cols = [f.name for f in fields(RunRecord) if f.name not in ("steps",)]
    step_cols = [f.name for f in fields(Step)]
    runs_path, steps_path = directory / "runs.csv", directory / "steps.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=run_cols + ["duration"])
        w.writeheader()
        for r in runs:
            row = {k: _cell(v) for k, v in r.to_dict().items() if k != "steps"}
            row["duration"] = r.duration
            w.writerow(row)
    with steps_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=step_cols)
        w.writeheader()
        for r in runs:
            for s in r.steps:
                w.writerow({k: _cell(v) for k, v in asdict(s).items()})
    return runs_path, steps_path


def _cell(v: Any) -> Any:
    return json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
