# Changelog

## 0.1.0 — 2026-10-02

First release.

- Framework-agnostic `Recorder` for LLM calls, tool calls, plans, inter-agent messages,
  reviews and approvals; JSONL and CSV (runs/steps) logs.
- 27 metrics in 7 dimensions at agent, interaction and system level, with condition scopes
  and explicit "not applicable" / "no data" reasons.
- `evaluate()` with grouping by architecture, condition, model or app; CSV/JSON exports and a
  self-contained offline HTML dashboard.
- Governance layer: `governed_tool` approval gate, fault injection (C2), prompt injection (C3).
- LangChain / LangGraph callback adapter recording tokens, latency, tool calls and rationales.
- OpenTelemetry adapter (GenAI, OpenInference and OpenLLMetry conventions): live span processor,
  OTLP JSON files, and experimental loaders for AgentCore (CloudWatch) and Vertex AI (Cloud Trace).
- Live mode for LangGraph apps: `hieevas.init("local" | "cloud" | "vertex")`, `run_config()` tags
  carried in LangGraph config metadata, file / Cloud Trace / OTLP export, `load_runs` with time
  windows, `day` / `hour` grouping, `send_probes`, and a dependency-free dashboard server
  (`python -m hieevas.serve`, also `hieevas-dashboard`). Refusal is inferred for `C4` runs.
- `stress_tools()`: C2 tool faults, C3 injection and the approval gate inside a deployed LangChain /
  LangGraph app, switched on per request by its `run_config` tags; `send_probes(repeats=)` for M3.
- Span reader: the answer is the final message, not the whole graph state (retrieved text no longer
  counts as a correct answer); tool faults and gate denials are read from ToolMessage output;
  denials are recorded as approval requests (M26); injection is inferred for `C3` runs.
- Scoring: `auto` now scores text references by whole-phrase containment (`contains`); the paper's
  strict exact match remains available as `text`. Consistency (M3) compares scored outcomes.
- Metric guide (`hieevas.metrics.guide`, `METRICS.md`): what each metric measures, how, from which
  part of the run, how to read it and what it needs; shown on the dashboard with a
  "How a run becomes metrics" panel.
- Production safety: `init()` joins an application's existing OpenTelemetry provider (with a
  warning) instead of replacing it; README "Using hieevas in production" and `SECURITY.md` cover the
  approval-gate default, probe-only stress conditions and user data in traces.
- Every metric value carries its calculation with counts (`MetricResult.detail`, e.g.
  "8 correct ÷ 8 scored runs"), shown in tiles, under each chart, in tooltips and in `report.json`;
  the run tile breaks runs down by condition.
