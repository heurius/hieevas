# hieevas

**HIE**rarchical **EV**aluation of **A**gentic **S**ystems: evaluate **how** an agentic AI or RAG
pipeline reaches its answers, not only **whether** it does.

Answer-quality tools such as Ragas judge the final answer and its grounding; hieevas measures the
path to it (cost, robustness under tool faults, prompt-injection resistance, governance and audit)
at three levels of the hierarchy: the single agent, the interaction between agents, and the whole
system. The two complement each other.

`hieevas` records what your pipeline does (LLM calls, tool calls, messages between agents,
reviews, approvals) and scores it on **27 metrics in 7 dimensions** at **agent, interaction and
system level**, following the multi-level evaluation framework in *Measuring the Process, Not Just
the Outcome* (Table 8). It works offline, has no required dependencies, and produces a
self-contained HTML dashboard.

| Dimension | Metrics |
|---|---|
| Effectiveness | M1 task success · M2 answer F1 · M3 consistency |
| Efficiency | M4 completion time · M5 tokens/task · M6 LLM calls · M7 tokens per success · M8 peak context |
| Planning & reasoning | M9 steps · M10 redundant calls · M11 plan adherence · M12 tool-call validity |
| Robustness & recovery | M13 fault-induced drop · M14 recovery rate · M15 retries per error |
| Coordination | M16 hand-over success · M17 reviewer rejections · M18 correction success · M19 communication overhead |
| Safety & security | M20 refusal rate · M21 injection success · M22 unauthorised high-risk attempts |
| Transparency & governance | M23 rationale coverage · M24 rationale quality · M25 audit completeness · M26 approval triggers · M27 budget violations |

When the data for a metric was not recorded, the metric returns *not available* with the reason,
never a misleading zero. [METRICS.md](https://github.com/heurius/hieevas/blob/main/METRICS.md) explains every metric in plain language: what it
measures, how it is computed, and which part of a LangGraph run it reads (input, LLM call, tool call,
final state). The same text appears under *What is this?* on each dashboard card.

## Install

```bash
pip install hieevas              # core, no dependencies
pip install "hieevas[langgraph]" # + LangChain / LangGraph callback adapter
pip install "hieevas[live]"      # + live tracing of LangGraph apps (local mode, any OTLP collector)
pip install "hieevas[gcp]"       # + Cloud Run / VM -> Cloud Trace, and reading Cloud Trace (incl. Agent Engine)
```

Until the first PyPI release, install from GitHub:
`pip install "hieevas[live] @ git+https://github.com/heurius/hieevas"`.
For development: `git clone https://github.com/heurius/hieevas && pip install -e "./hieevas[dev]"`.

## 0. Live dashboards for any LangGraph app

Add one line at start-up; tag requests only if you want known-answer or stress probes.

```python
import hieevas

session = hieevas.init("local")        # or "cloud" (Cloud Run / VM) or "vertex" (Agent Engine)
graph.invoke(inputs)                                                    # live traffic: no changes
graph.invoke(inputs, config=hieevas.run_config(task_id="q1", condition="C1", reference="Paris"))  # probe
session.save_dashboard("dashboard.html")
```

```bash
python -m hieevas.serve --source file:traces/spans.jsonl                        # local mode
python -m hieevas.serve --source cloud-trace:MY_PROJECT --tool-risk send_email  # cloud and vertex modes
```

| Mode | App and LLM | Traces written by | Dashboard reads |
|---|---|---|---|
| `local` | both on this machine (e.g. Ollama) | `hieevas.init("local")` → JSON-lines file | `file:traces/spans.jsonl` |
| `cloud` | app on Cloud Run / a VM, LLM via an API | `hieevas.init("cloud")` → Cloud Trace (or `endpoint=` any OTLP collector) | `cloud-trace:PROJECT` |
| `vertex` | agent on Vertex AI Agent Engine | `LanggraphAgent(..., enable_tracing=True)` → Cloud Trace | `cloud-trace:PROJECT` |

All three produce the same OpenInference spans, so one reader and one dashboard serve them.
Tags passed with `run_config` travel in the LangGraph config metadata and are read back
from the spans, which also works on Agent Engine where the app does not own its spans.
The dashboard server (`/`, `/report.json`, `/healthz`; query `?hours=6&group_by=architecture,hour`)
needs only the standard library; on Cloud Run deploy it with `--no-allow-unauthenticated`.

**What live traffic can and cannot show.** Real user requests have no reference answers, so
they yield the efficiency, planning, coordination, high-risk-attempt and governance metrics.
Task success, recovery, refusal and injection metrics come from probes: requests tagged with a
condition (`C1` known answer, `C2` tool faults, `C3` injection, `C4` out-of-policy) sent with
`hieevas.send_probes(...)`, for example on a schedule. For `C4` runs without an explicit
flag, refusal is scored as *declined in words and no high-risk tool attempted*.

Examples for each mode are in [`examples/`](https://github.com/heurius/hieevas/tree/main/examples): `local_ollama.py`, `cloud_run/`,
`vertex_agent_engine.py`, and a stand-alone dashboard container in `dashboard/`.

## Using hieevas in production

hieevas is alpha software, provided "as is" under the MIT licence without warranty. Before
running it next to real users, check these four points:

1. **The approval gate blocks real actions by default.** `stress_tools(..., high_risk=[...])`
   routes every call to a high-risk tool through `approval`, for probes *and* real users, and the
   default `deny_all` refuses them all. In production pass your own approval function (for example
   one that asks a person), or `approval=approve_all` from `hieevas.governance` to record without
   blocking.
2. **Stress conditions are for probes only.** Tool faults (`C2`) and injected text (`C3`) are
   applied only to requests tagged with that condition. Never let end users set tags: the Cloud
   Run example rejects tags unless `HIEEVAS_ACCEPT_TAGS=1`, and probes should come from a
   trusted caller.
3. **Traces contain user data.** Spans hold prompts, retrieved passages, tool arguments and
   answers, which may include personal or confidential information. Treat trace files and Cloud
   Trace like application logs: restrict access, set retention, sample (`sample_rate=`) and
   follow your privacy obligations. The dashboard server has no login of its own; locally it
   listens on 127.0.0.1 only, and on Cloud Run it must be deployed with
   `--no-allow-unauthenticated` or behind IAP.
4. **Existing OpenTelemetry setups are joined, not replaced.** If the application already
   configured a tracer provider, `hieevas.init()` adds its exporter to that provider and warns;
   the application's service name and sampling stay in force.

Metric values are measurements from recorded behaviour, not certifications: keyword-based
refusal detection and phrase-matching of answers are heuristics (see [METRICS.md](https://github.com/heurius/hieevas/blob/main/METRICS.md)).
See [SECURITY.md](https://github.com/heurius/hieevas/blob/main/SECURITY.md) to report a vulnerability.

## 1. Any pipeline (framework-agnostic)

```python
from hieevas import Recorder, evaluate

rec = Recorder(app="my-rag", architecture="A1", model="qwen2.5:1.5b")
with rec.run("q1", reference="Paris") as run:
    run.llm_call(agent="retriever", input_tokens=120, output_tokens=18, latency=0.9)
    run.tool_call("search_docs", {"query": "capital of France"}, rationale="need a source")
    run.answer("Paris")

report = evaluate(rec.runs, group_by=("architecture", "condition"))
print(report.summary())
report.to_html("dashboard.html")      # offline dashboard
report.to_csv("metrics.csv")
rec.save("traces.jsonl"); rec.save_csv("logs/")   # runs.csv + steps.csv
```

Multi-agent pipelines can also record `run.plan([...])`, `run.plan_step_done(i)`,
`run.message(sender, receiver, text)`, `run.review("accept"|"reject")` and
`run.mark(refused=..., injection_followed=...)`.

## 2. LangChain / LangGraph

```python
from hieevas.adapters.langchain import HieevasCallbackHandler

with rec.run("q1", reference="Paris") as run:
    out = graph.invoke(inputs, config={"callbacks": [HieevasCallbackHandler(run)],
                                       "recursion_limit": 40})      # recursion limit = step budget (M27)
    run.answer(out["messages"][-1].content)
```

The handler records every LLM call (tokens, latency, agent = LangGraph node) and every tool call
(status, arguments, and the text the model wrote before calling it, used as the rationale).

## 3. Governance layer and test conditions

```python
from hieevas import governed_tool, inject, injection_followed

search = governed_tool(search_fn, name="search_docs", fault_rate=0.3)          # C2 tool faults
send_email = governed_tool(send_fn, name="send_email", risk="high")           # approval gate (denies by default)
docs = inject(retrieved_docs)                                                  # C3 prompt injection
run.mark(injection_followed=injection_followed(run.record))
```

## 4. OpenTelemetry: AgentCore, Vertex AI and any OTel backend

One trace becomes one run. Spans using the OpenTelemetry GenAI conventions (`gen_ai.*`),
OpenInference or OpenLLMetry are understood.

```python
# a) In-process, wherever the agent runs (locally, AgentCore Runtime, Vertex Agent Engine)
from hieevas.adapters.otel import HieevasSpanProcessor
proc = HieevasSpanProcessor()
tracer_provider.add_span_processor(proc)          # alongside your normal exporter
...                                               # run the agent
report = evaluate(proc.runs(architecture="A1"))

# b) From files written by an OpenTelemetry Collector (file exporter)
from hieevas.adapters.otel import load_otlp_json, spans_to_runs
runs = spans_to_runs(load_otlp_json("traces.json"))

# c) Amazon Bedrock AgentCore: spans in CloudWatch (log group aws/spans)   pip install "hieevas[agentcore]"
from hieevas.adapters.otel import fetch_cloudwatch_spans
runs = spans_to_runs(fetch_cloudwatch_spans(start_ms, end_ms, region="us-east-1"))

# d) Google Vertex AI Agent Engine: spans in Cloud Trace                   pip install "hieevas[vertex]"
from hieevas.adapters.otel import fetch_cloud_trace_spans
runs = spans_to_runs(fetch_cloud_trace_spans("my-gcp-project"))
```

Set `hieevas.task_id`, `hieevas.condition`, `hieevas.reference` and `hieevas.architecture`
as span attributes to control grouping and scoring; `hieevas.rationale` and `hieevas.risk`
on tool spans enable M23 and M22. The CloudWatch and Cloud Trace loaders are experimental:
check the span record format for your platform version.

## Example

`../rag_eval/` runs a RAG pipeline over four PDF papers with a local Ollama model in three
architectures (single agent, planner–executor, planner–executor–reviewer, built with LangGraph)
under four conditions, and writes the dashboard.

## Notes

- M20 and M24 are rubric-scored. `detect_refusal` is a keyword heuristic for screening only;
  confirm codes with human raters and report agreement with `cohen_kappa`.
- Token counts come from the model provider; message tokens are estimated (4 characters ≈ 1 token)
  when not supplied.
