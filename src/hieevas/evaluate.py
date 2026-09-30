"""Compute metrics over groups of runs."""
from __future__ import annotations

import time
from typing import Iterable, Sequence

from . import metrics as M
from .metrics.base import Metric, MetricResult
from .report import Report
from .trace import RunRecord


def evaluate(runs: Iterable[RunRecord], metrics: Sequence[Metric] | None = None,
             group_by: Sequence[str] = ("architecture", "condition"),
             baseline_condition: str = "C1", fault_condition: str = "C2", use_scopes: bool = True) -> Report:
    """Evaluate runs, one row per group.

    ``group_by`` names RunRecord fields (e.g. ``architecture``, ``condition``, ``model``, ``app``)
    or the time buckets ``day`` and ``hour`` (UTC, from the run start) for trends in live traffic.
    Cross-condition metrics (M13) compare ``baseline_condition`` with ``fault_condition``
    across the other grouping keys and are reported on the fault-condition row.

    When runs are pooled across conditions (``condition`` not in ``group_by``) and
    ``use_scopes`` is true, each metric is computed only on the conditions it belongs to
    (e.g. task success on C1, recovery on C2, injection on C3), giving one complete
    column per architecture.
    """
    runs = list(runs)
    metrics = list(metrics or M.ALL)
    group_by = tuple(group_by)
    groups: dict[tuple, list[RunRecord]] = {}
    for r in runs:
        groups.setdefault(tuple(group_value(r, k) for k in group_by), []).append(r)

    rows: list[dict] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        members = groups[key]
        row: dict = {"group": dict(zip(group_by, key)), "n_runs": len(members), "results": {}}
        for m in metrics:
            if m.cross_condition:
                row["results"][m.id] = _cross_condition(m, runs, group_by, key, baseline_condition, fault_condition)
            elif m.scope and "condition" in group_by:
                cond = key[group_by.index("condition")]
                row["results"][m.id] = (m.compute(members) if cond in m.scope else MetricResult(
                    None, 0, f"not applicable: measured under {', '.join(m.scope)}", False))
            elif m.scope and use_scopes:
                scoped = [r for r in members if r.condition in m.scope]
                row["results"][m.id] = (m.compute(scoped) if scoped else MetricResult(
                    None, 0, f"no runs under {', '.join(m.scope)}"))
            else:
                row["results"][m.id] = m.compute(members)
        rows.append(row)
    return Report(rows=rows, metrics=metrics, group_by=group_by, runs=runs)


def _cross_condition(m: Metric, runs: list[RunRecord], group_by: tuple, key: tuple,
                     baseline: str, fault: str) -> MetricResult:
    if "condition" in group_by:
        pos = group_by.index("condition")
        if key[pos] != fault:
            return MetricResult(None, 0, f"not applicable: reported on the {fault} row", False)
        others = [(k, v) for k, v in zip(group_by, key) if k != "condition"]
    else:
        others = list(zip(group_by, key))
    subset = [r for r in runs if all(group_value(r, k) == v for k, v in others)]
    return m.compute(subset, baseline, fault)


TIME_BUCKETS = {"day": "%Y-%m-%d", "hour": "%Y-%m-%d %H:00"}


def group_value(run: RunRecord, key: str):
    """Value of a grouping key: a RunRecord field, or a UTC time bucket of the run start."""
    if key in TIME_BUCKETS and not hasattr(run, key):
        return time.strftime(TIME_BUCKETS[key], time.gmtime(run.started))
    return getattr(run, key)
