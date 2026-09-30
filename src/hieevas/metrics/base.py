"""Metric base class and result type."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from ..trace import RunRecord

EFFECTIVENESS = "Effectiveness"
EFFICIENCY = "Efficiency"
PLANNING = "Planning & reasoning"
ROBUSTNESS = "Robustness & recovery"
COORDINATION = "Coordination"
SAFETY = "Safety & security"
GOVERNANCE = "Transparency & governance"
DIMENSIONS = (EFFECTIVENESS, EFFICIENCY, PLANNING, ROBUSTNESS, COORDINATION, SAFETY, GOVERNANCE)


@dataclass(frozen=True)
class MetricResult:
    value: float | None
    n: int = 0  # number of observations the value is based on
    note: str | None = None  # why a value is missing, or how it was computed
    applicable: bool = True  # False when the metric does not apply to this group by design
    detail: str | None = None  # the calculation with its counts, e.g. "8 correct ÷ 8 scored runs"

    @property
    def available(self) -> bool:
        return self.value is not None


class Metric:
    """A metric from the codebook (paper, Table 8).

    Subclasses set the class attributes and implement ``compute`` over a list of runs.
    """

    id: str = ""
    name: str = ""
    dimension: str = ""
    level: str = ""  # "agent" | "interaction" | "system"
    data_type: str = "quantitative"  # "quantitative" | "rubric"
    focus: str = ""  # "outcome" | "process"
    definition: str = ""
    higher_is_better: bool | None = None
    cross_condition: bool = False  # needs runs from two conditions (M13)
    scope: tuple[str, ...] | None = None  # test conditions the metric is computed on (None = all)

    def compute(self, runs: list[RunRecord]) -> MetricResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.id} {self.name}>"


def count(x: float) -> str:
    """Format a count or total for a calculation string: 5339 -> '5,339', 2.5 -> '2.5'."""
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"


def ratio(num: float, den: float, missing: str, num_label: str = "", den_label: str = "") -> MetricResult:
    """``num / den``; the labels name both counts, e.g. ("correct", "scored runs")."""
    if not den:
        return MetricResult(None, 0, missing)
    detail = f"{count(num)} {num_label} ÷ {count(den)} {den_label}".replace("  ", " ").strip()
    return MetricResult(num / den, int(den), detail=detail)


def average(values: Iterable[float | None], missing: str, total_label: str | None = "",
            per_label: str = "runs") -> MetricResult:
    """Mean of ``values``, shown as "total ÷ count", e.g. "5,339 tokens ÷ 8 runs".

    With ``total_label=None`` the total is not meaningful and the detail reads "mean over 8 runs".
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return MetricResult(None, 0, missing)
    if total_label is None:
        detail = f"mean over {len(vals)} {per_label}"
    else:
        detail = f"{count(sum(vals))} {total_label} ÷ {len(vals)} {per_label}".replace("  ", " ")
    return MetricResult(mean(vals), len(vals), detail=detail)
