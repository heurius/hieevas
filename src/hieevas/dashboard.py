"""Self-contained HTML dashboard (no external scripts, works offline)."""
from __future__ import annotations

import datetime as _dt
from html import escape

from . import metrics as M
from .metrics.base import DIMENSIONS, Metric
from .metrics.guide import GUIDE
from .trace import LLM, TOOL

# Validated categorical palette (colour-blind separation checked), light / dark.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Metrics expressed as proportions are shown as percentages on a fixed 0-100 % axis.
PERCENT = {"M1", "M3", "M10", "M11", "M12", "M13", "M14", "M16", "M17", "M18", "M19", "M20", "M21", "M23",
           "M25", "M27"}

CSS = """
:root{--bg:#f7f7f5;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;
--track:#efeeea;--accent:#2a78d6;--good:#1a7f4b;--bad:#b3261e;
""" + "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES)) + """}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--ink:#ffffff;
--ink2:#c3c2b7;--muted:#8f8e86;--line:#2e2d2a;--track:#262623;--accent:#3987e5;--good:#4cc38a;--bad:#ff8a80;
""" + "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES_DARK)) + """}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--ink2);margin:0 0 4px}.meta{color:var(--muted);font-size:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(170px,100%),1fr));gap:12px;margin-top:18px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile .k{color:var(--ink2);font-size:12px}.tile .v{font-size:26px;font-weight:600;margin-top:4px;font-variant-numeric:tabular-nums}
.tile .d{color:var(--muted);font-size:11px;margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0 0;color:var(--ink2);font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(360px,100%),1fr));gap:12px}
.tiles,.grid,.card{min-width:0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h3{font-size:13.5px;margin:0}.card .def{color:var(--ink2);font-size:12px;margin:2px 0 8px}
.badge{display:inline-block;font-size:10.5px;color:var(--ink2);border:1px solid var(--line);border-radius:999px;padding:0 7px;margin-right:4px}
.na{color:var(--muted);font-size:12px;font-style:italic;padding:8px 0}
svg text{fill:var(--ink2);font-size:11px}svg .val{fill:var(--ink);font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
th{background:var(--track);color:var(--ink2);font-weight:600}th:first-child,td:first-child{text-align:left}
tr.dim td{background:var(--track);color:var(--ink2);font-weight:600;text-align:left}
.scroll{overflow-x:auto}.ok{color:var(--good)}.no{color:var(--bad)}
.notes li{margin:3px 0;color:var(--ink2)}.muted{color:var(--muted)}.h3{font-size:14px;margin:14px 0 6px}.foot{margin-top:28px;color:var(--muted);font-size:12px}
.q{font-size:12.5px;margin:2px 0 6px}
.calc{font-size:12px;color:var(--ink2);margin-top:6px;font-variant-numeric:tabular-nums}.calc div{margin:2px 0}
details.about{margin-top:8px;border-top:1px solid var(--line);padding-top:6px}
details.about summary{cursor:pointer;color:var(--accent);font-size:12px;list-style:none}
details.about summary::-webkit-details-marker{display:none}details.about summary::before{content:"▸ "}
details.about[open] summary::before{content:"▾ "}
dl.g{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;margin:8px 0 0;font-size:12px}
dl.g dt{color:var(--muted);white-space:nowrap}dl.g dd{margin:0;color:var(--ink2)}
details.flow{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:18px}
details.flow>summary{cursor:pointer;font-weight:600}
.pipe{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:14px 0 6px}
.stage{flex:1 1 180px;min-width:0;border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--bg)}
.stage b{display:block;font-size:13px}.stage .t{display:block;color:var(--ink2);font-size:12px;margin:3px 0 8px}
.arrow{align-self:center;color:var(--muted);font-size:18px}
.chip{display:inline-block;font-size:11px;border-radius:999px;padding:1px 7px;margin:2px 3px 0 0;
background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--ink)}
.flow p{color:var(--ink2);font-size:12.5px;margin:6px 0}
"""

# Where in a LangGraph run each metric's data comes from (the "How a run becomes metrics" panel).
FLOW = [
    ("Input", "the user message, plus the tags given with run_config(): task, reference answer, test condition",
     ()),
    ("Model node · LLM call", "each model invocation: tokens in and out, latency, and the text written "
     "alongside a tool call (its reason)", ("M5", "M6", "M8", "M23")),
    ("Tools node · tool call", "each tool execution: name, arguments, output, error or 'ACTION DENIED' from "
     "the approval gate", ("M10", "M12", "M14", "M15", "M21", "M22", "M26")),
    ("Final answer", "the last message of the graph's final state, compared with the reference",
     ("M1", "M2", "M3", "M7", "M20", "M21")),
]
WHOLE_RUN = ("M4", "M9", "M13", "M25", "M27")
MULTI_AGENT = ("M11", "M16", "M17", "M18", "M19")
LEVEL_TIP = {"agent": "Agent level: one agent's behaviour", "interaction": "Interaction level: exchanges "
             "between agents", "system": "System level: the end-to-end result"}
FOCUS_TIP = {"outcome": "Outcome: describes the result", "process": "Process: describes how the result was reached"}
TYPE_TIP = {"quantitative": "Computed directly from the trace", "rubric": "Should be confirmed by human raters"}


def _flow_panel() -> str:
    def chips(ids):
        return "".join(f"<span class='chip' title='{escape(M.BY_ID[i].name)}'>{i}</span>" for i in ids)

    stages = []
    for i, (title, text, ids) in enumerate(FLOW):
        if i:
            stages.append("<div class='arrow'>" + ("⇄" if i == 2 else "→") + "</div>")
        stages.append(f"<div class='stage'><b>{escape(title)}</b><span class='t'>{escape(text)}</span>"
                      f"{chips(ids)}</div>")
    return ("<details class='flow' open><summary>How a run becomes metrics</summary>"
            "<p>One request to the app (one <code>graph.invoke</code>) is one <b>run</b>. Its trace records the "
            "input, every model call and tool call made by the graph's nodes (the model and tools nodes loop "
            "until the model answers), and the final state. Each metric reads one part of that record:</p>"
            f"<div class='pipe'>{''.join(stages)}</div>"
            f"<p><b>Whole run</b> (time, step count, audit trail, fault comparison): {chips(WHOLE_RUN)}</p>"
            f"<p><b>Between agents</b>, multi-agent graphs only (plans, hand-overs, reviews): {chips(MULTI_AGENT)}</p>"
            "<p><b>Test conditions</b> are set per request with the condition tag: C1 normal operation, "
            "C2 tool faults, C3 prompt injection in retrieved text, C4 out-of-policy requests. Task success "
            "and efficiency are read from C1 runs, recovery from C2, injection from C3, refusal from C4. "
            "Open <i>What is this?</i> on any card for the details.</p></details>")


def _about(m: Metric) -> str:
    g = GUIDE.get(m.id)
    if not g:
        return ""
    rows = [("Measures", g["measures"]), ("How", g["how"]), ("From", g["source"]), ("Reading it", g["read"])]
    if g.get("needs"):
        rows.append(("Needs", g["needs"]))
    return ("<details class='about'><summary>What is this?</summary><dl class='g'>" +
            "".join(f"<dt>{k}</dt><dd>{escape(v)}</dd>" for k, v in rows) + "</dl></details>")


def _fmt(metric_id: str, v: float | None) -> str:
    if v is None:
        return "–"
    if metric_id in PERCENT:
        return f"{v * 100:.0f}%"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.2f}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _label(group: dict) -> str:
    return " · ".join(str(v) for v in group.values()) or "all"


def _bar_chart(metric: Metric, rows: list[dict], colour_of) -> str:
    items = [(r, r["results"][metric.id]) for r in rows]
    items = [(r, res) for r, res in items if res.available]
    if not items:
        return ""
    vmax = 1.0 if metric.id in PERCENT else max(max(res.value for _, res in items), 1e-9)
    if metric.id in PERCENT:
        vmax = max(1.0, max(res.value for _, res in items))
    left, width, bar_h, gap = 132, 330, 14, 6
    height = len(items) * (bar_h + gap) + 4
    parts = [f'<svg viewBox="0 0 {left + width + 56} {height}" width="100%" role="img" '
             f'aria-label="{escape(metric.name)} by group">']
    for i, (row, res) in enumerate(items):
        y = i * (bar_h + gap) + 2
        w = max(2.0, width * max(res.value, 0) / vmax)
        colour = colour_of(row["group"])
        tip = f"{_label(row['group'])}: {_fmt(metric.id, res.value)} = {res.detail or f'n={res.n}'}"
        parts.append(f'<g><title>{escape(tip)}</title>'
                     f'<text x="{left - 8}" y="{y + bar_h - 3}" text-anchor="end">{escape(_label(row["group"]))[:22]}</text>'
                     f'<rect x="{left}" y="{y}" width="{width}" height="{bar_h}" rx="3" fill="var(--track)"/>'
                     f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{colour}"/>'
                     f'<text class="val" x="{left + w + 6:.1f}" y="{y + bar_h - 3}">{_fmt(metric.id, res.value)}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def _heat_cell(metric: Metric, value: float | None, values: list[float], tip: str | None = None) -> str:
    text = _fmt(metric.id, value)
    if tip:
        text = f"<span title='{escape(tip)}'>{text}</span>"
    if value is None or metric.higher_is_better is None or len(values) < 2 or max(values) == min(values):
        return f"<td>{text}</td>"
    t = (value - min(values)) / (max(values) - min(values))
    if not metric.higher_is_better:
        t = 1 - t
    alpha = 0.08 + 0.42 * t
    return f'<td style="background:color-mix(in srgb, var(--accent) {alpha * 100:.0f}%, transparent)">{text}</td>'


def render_dashboard(report, title: str = "HIEEVAS – Hierarchical Evaluation of Agentic Systems", subtitle: str = "") -> str:
    rows, metrics, runs = report.rows, report.metrics, report.runs
    first_key = report.group_by[0] if report.group_by else None
    entities = []
    for r in rows:
        e = r["group"].get(first_key) if first_key else "all"
        if e not in entities:
            entities.append(e)

    def colour_of(group: dict) -> str:
        e = group.get(first_key) if first_key else "all"
        return f"var(--s{entities.index(e) % len(SERIES)})"

    available = sum(1 for m in metrics if any(r["results"][m.id].available for r in rows))
    conditions = sorted({r.condition for r in runs})

    def headline(metric_id: str, label: str, desc: str, conds: tuple | None = None):
        subset = [r for r in runs if conds is None or r.condition in conds]
        if not subset:
            return (label, "–", f"{metric_id} · no {'/'.join(conds)} runs yet")
        res = M.BY_ID[metric_id].compute(subset)
        return (label, _fmt(metric_id, res.value), f"{metric_id} · {res.detail or desc}" if res.available
                else f"{metric_id} · {res.note}")

    per_condition = " · ".join(f"{c}: {sum(r.condition == c for r in runs)}" for c in conditions)
    tiles = [
        ("Runs executed", str(len(runs)), f"{per_condition} · {len(entities)} {first_key or 'group'}(s)"),
        headline("M1", "Task success (C1)", "correct ÷ scored", ("C1",)),
        headline("M7", "Tokens per success (C1)", "lower is better", ("C1",)),
        headline("M14", "Recovery rate (C2)", "solved despite tool errors", ("C2",)),
        headline("M21", "Injection success (C3)", "lower is better", ("C3",)),
        headline("M20", "Refusal rate (C4)", "out-of-policy requests declined", ("C4",)),
        headline("M25", "Audit completeness", "complete logs"),
        ("Metrics with data", f"{available}/{len(metrics)}", "see coverage notes below"),
    ]

    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{escape(title)}</title><style>{CSS}</style></head><body><div class='wrap'>",
           f"<h1>{escape(title)}</h1>"]
    if subtitle:
        out.append(f"<p class='sub'>{escape(subtitle)}</p>")
    out.append(f"<div class='meta'>Generated {_dt.datetime.now():%Y-%m-%d %H:%M} · hieevas · "
               f"grouped by {', '.join(report.group_by) or 'nothing'}</div>")
    out.append("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='k'>{escape(k)}</div><div class='v'>{escape(v)}</div>"
        f"<div class='d'>{escape(d)}</div></div>" for k, v, d in tiles) + "</div>")
    out.append(_flow_panel())
    if first_key:
        out.append("<div class='legend'>" + "".join(
            f"<span><i style='background:var(--s{i % len(SERIES)})'></i>{escape(str(e))}</span>"
            for i, e in enumerate(entities)) + "</div>")

    # ---- one section per dimension ----
    for dim in DIMENSIONS:
        dm = [m for m in metrics if m.dimension == dim]
        if not dm:
            continue
        out.append(f"<h2>{escape(dim)}</h2><div class='grid'>")
        for m in dm:
            chart = _bar_chart(m, rows, colour_of)
            missing = [r["results"][m.id] for r in rows if not r["results"][m.id].available]
            notes = sorted({x.note for x in missing if x.note and x.applicable})
            na = sorted({x.note for x in missing if x.note and not x.applicable})
            body = chart or f"<div class='na'>{escape('No data: ' + '; '.join(notes) if notes else _cap('; '.join(na)) or 'No data')}</div>"
            calcs = [(row["group"], row["results"][m.id]) for row in rows if row["results"][m.id].available]
            if calcs:
                body += "<div class='calc'>" + "".join(
                    f"<div><span class='muted'>{escape(_label(g))}:</span> {_fmt(m.id, res.value)} = "
                    f"{escape(res.detail or f'n = {res.n}')}</div>" for g, res in calcs) + "</div>"
            if chart and na:
                scope = f"runs under {', '.join(m.scope)}" if m.scope else "some groups"
                body += (f"<div class='na'>Measured on {escape(scope)} only; "
                         f"groups under other conditions are left out.</div>")
            better = {True: "higher is better", False: "lower is better", None: "context-dependent"}[m.higher_is_better]
            question = GUIDE.get(m.id, {}).get("measures", "")
            out.append(f"<div class='card'><h3>{m.id} · {escape(m.name)}</h3>"
                       f"<div class='q'>{escape(question)}</div>"
                       f"<div class='def'>{escape(m.definition)}</div>"
                       f"<span class='badge' title='{LEVEL_TIP[m.level]}'>{m.level}</span>"
                       f"<span class='badge' title='{FOCUS_TIP[m.focus]}'>{m.focus}</span>"
                       f"<span class='badge' title='{TYPE_TIP[m.data_type]}'>{m.data_type}</span>"
                       f"<span class='badge'>{better}</span>"
                       f"<div style='margin-top:8px'>{body}</div>{_about(m)}</div>")
        out.append("</div>")

    # ---- comparison matrix ----
    out.append("<h2>Comparison matrix</h2><p class='sub'>Shading marks the better value within each row "
               "(direction-aware); unshaded rows have no preferred direction or a single value.</p>"
               "<div class='scroll'><table><tr><th>Metric</th>" +
               "".join(f"<th>{escape(_label(r['group']))}</th>" for r in rows) + "</tr>")
    for dim in DIMENSIONS:
        dm = [m for m in metrics if m.dimension == dim]
        if not dm:
            continue
        out.append(f"<tr class='dim'><td colspan='{len(rows) + 1}'>{escape(dim)}</td></tr>")
        for m in dm:
            vals = [r["results"][m.id].value for r in rows]
            present = [v for v in vals if v is not None]
            cells = []
            for r in rows:
                res = r["results"][m.id]
                cells.append("<td class='muted' title='" + escape(res.note or "") + "'>n/a</td>" if not res.applicable
                             else _heat_cell(m, res.value, present, res.detail or res.note))
            out.append(f"<tr><td>{m.id} {escape(m.name)}</td>" + "".join(cells) + "</tr>")
    out.append("</table></div>")

    # ---- coverage notes ----
    missing = report.unavailable()
    out.append("<h2>Coverage notes</h2><h3 class='h3'>No data yet</h3>")
    if missing:
        out.append("<ul class='notes'>" + "".join(
            f"<li><b>{mid} {escape(M.BY_ID[mid].name)}</b>: {escape(reason)}</li>" for mid, reason in missing.items())
                   + "</ul>")
    else:
        out.append("<p class='sub'>Every metric has a value wherever it applies.</p>")
    na = report.not_applicable()
    if na:
        out.append("<h3 class='h3'>Not applicable by design</h3><ul class='notes'>" + "".join(
            f"<li><b>{mid} {escape(M.BY_ID[mid].name)}</b>: {escape(', '.join(groups))}</li>"
            for mid, groups in na.items()) + "</ul>")

    # ---- run explorer ----
    out.append("<h2>Runs</h2><div class='scroll'><table><tr><th>Task</th><th>Architecture</th><th>Condition</th>"
               "<th>Correct</th><th>F1</th><th>Tokens</th><th>Time (s)</th><th>LLM calls</th><th>Tool calls</th>"
               "<th>Tool errors</th><th>Refused</th><th>Injection followed</th><th>Budget hit</th></tr>")
    for r in sorted(runs, key=lambda x: (x.architecture, x.condition, x.task_id)):
        tools = r.steps_of(TOOL)
        tok = sum(s.total_tokens for s in r.steps_of(LLM))

        def flag(v):
            return "–" if v is None else ("<span class='ok'>yes</span>" if v else "<span class='no'>no</span>")

        out.append(f"<tr><td>{escape(r.task_id)}</td><td>{escape(str(r.architecture))}</td><td>{escape(r.condition)}</td>"
                   f"<td>{flag(r.correct)}</td><td>{'–' if r.f1 is None else f'{r.f1:.2f}'}</td><td>{tok:,}</td>"
                   f"<td>{'–' if r.duration is None else f'{r.duration:.1f}'}</td><td>{len(r.steps_of(LLM))}</td>"
                   f"<td>{len(tools)}</td><td>{sum(s.status == 'error' for s in tools)}</td>"
                   f"<td>{flag(r.refused)}</td><td>{'–' if r.injection_followed is None else ('<span class=no>yes</span>' if r.injection_followed else '<span class=ok>no</span>')}</td>"
                   f"<td>{'yes' if r.budget_exceeded else 'no'}</td></tr>")
    out.append("</table></div>")
    out.append("<p class='foot'>Metric definitions follow the multi-level evaluation framework (27 metrics, 7 dimensions, "
               "agent / interaction / system levels). Rubric-scored metrics (M20, M24) should be confirmed by human "
               "coders with agreement reported.</p></div></body></html>")
    return "".join(out)
