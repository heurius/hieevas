"""Evaluation report: tabular exports and the HTML dashboard."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics.base import DIMENSIONS, Metric, MetricResult
from .trace import RunRecord


@dataclass
class Report:
    rows: list[dict]
    metrics: list[Metric]
    group_by: tuple
    runs: list[RunRecord] = field(default_factory=list)

    # ---- access ----------------------------------------------------------------------
    def value(self, metric_id: str, **group: Any) -> MetricResult | None:
        for row in self.rows:
            if all(row["group"].get(k) == v for k, v in group.items()):
                return row["results"].get(metric_id)
        return None

    def to_records(self) -> list[dict]:
        """One flat dict per group: group keys, n_runs, and one column per metric value."""
        out = []
        for row in self.rows:
            rec = dict(row["group"], n_runs=row["n_runs"])
            for m in self.metrics:
                rec[m.id] = row["results"][m.id].value
            out.append(rec)
        return out

    def unavailable(self) -> dict[str, str]:
        """Metrics with no value in any group where they apply, mapped to the reason."""
        out = {}
        for m in self.metrics:
            results = [row["results"][m.id] for row in self.rows]
            applicable = [r for r in results if r.applicable]
            if applicable and not any(r.available for r in applicable):
                out[m.id] = next((r.note for r in applicable if r.note), "no data")
        return out

    def not_applicable(self) -> dict[str, list[str]]:
        """Groups in which each metric does not apply by design (e.g. coordination for a single agent)."""
        out: dict[str, list[str]] = {}
        for m in self.metrics:
            for row in self.rows:
                res = row["results"][m.id]
                if not res.applicable:
                    label = " · ".join(str(v) for v in row["group"].values()) or "all"
                    out.setdefault(m.id, []).append(label)
        return out

    # ---- exports -----------------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        recs = self.to_records()
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]) if recs else ["group"])
            w.writeheader()
            w.writerows(recs)
        return path

    def to_dict(self) -> dict:
        return {
            "group_by": list(self.group_by),
            "metrics": [{"id": m.id, "name": m.name, "dimension": m.dimension, "level": m.level,
                         "data_type": m.data_type, "focus": m.focus, "definition": m.definition,
                         "higher_is_better": m.higher_is_better} for m in self.metrics],
            "rows": [{"group": r["group"], "n_runs": r["n_runs"],
                      "results": {k: {"value": v.value, "n": v.n, "note": v.note, "applicable": v.applicable,
                                      "detail": v.detail}
                                  for k, v in r["results"].items()}}
                     for r in self.rows],
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def to_dataframe(self):
        import pandas as pd  # optional dependency
        return pd.DataFrame(self.to_records())

    def summary(self) -> str:
        """Plain-text table: metrics as rows, groups as columns."""
        labels = [" / ".join(str(v) for v in r["group"].values()) or "all" for r in self.rows]
        width = max([28] + [len(l) for l in labels])
        lines = [f"{'metric':<34}" + "".join(f"{l:>{width}}" for l in labels)]
        for dim in DIMENSIONS:
            dm = [m for m in self.metrics if m.dimension == dim]
            if not dm:
                continue
            lines.append(f"-- {dim}")
            for m in dm:
                cells = [_fmt(row["results"][m.id].value) for row in self.rows]
                lines.append(f"{(m.id + ' ' + m.name)[:33]:<34}" + "".join(f"{c:>{width}}" for c in cells))
        return "\n".join(lines)

    def html(self, title: str = "HIEEVAS – Hierarchical Evaluation of Agentic Systems", subtitle: str = "") -> str:
        """The self-contained dashboard as a string, e.g. to serve from a web route."""
        from .dashboard import render_dashboard
        return render_dashboard(self, title=title, subtitle=subtitle)

    def to_html(self, path: str | Path, title: str = "HIEEVAS – Hierarchical Evaluation of Agentic Systems", subtitle: str = "") -> Path:
        path = Path(path)
        path.write_text(self.html(title=title, subtitle=subtitle), encoding="utf-8")
        return path


def _fmt(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"
