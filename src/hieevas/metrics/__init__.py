"""Metric registry: ``metrics.ALL``, ``metrics.by_id("M14")``, ``metrics.dimension(SAFETY)``."""
from .base import (COORDINATION, DIMENSIONS, EFFECTIVENESS, EFFICIENCY, GOVERNANCE, PLANNING, ROBUSTNESS,
                   SAFETY, Metric, MetricResult)
from .definitions import ALL

BY_ID = {m.id: m for m in ALL}


def by_id(*ids: str) -> list[Metric]:
    return [BY_ID[i] for i in ids]


def dimension(*names: str) -> list[Metric]:
    return [m for m in ALL if m.dimension in names]


# Convenience groups
PERFORMANCE = dimension(EFFECTIVENESS, EFFICIENCY, PLANNING)
DEPENDABILITY = dimension(ROBUSTNESS, COORDINATION, SAFETY, GOVERNANCE)

__all__ = ["ALL", "BY_ID", "by_id", "dimension", "Metric", "MetricResult", "DIMENSIONS", "PERFORMANCE",
           "DEPENDABILITY", "EFFECTIVENESS", "EFFICIENCY", "PLANNING", "ROBUSTNESS", "COORDINATION",
           "SAFETY", "GOVERNANCE"]
