"""Live evaluation of deployed agents: export traces while the app runs, read them back to score.

Three deployment modes share the same application code and the same dashboard:

  ========  =============================================  ==========================================
  mode      where the app and the LLM run                  where traces go / are read from
  ========  =============================================  ==========================================
  local     app and LLM on one machine (e.g. Ollama)       ``setup_tracing("file")`` -> ``file:``
  cloud     app on Cloud Run / a VM, LLM through an API    ``setup_tracing("cloud_trace")`` -> ``cloud-trace:``
                                                           (or ``"otlp"`` to any OpenTelemetry collector)
  vertex    LangGraph agent on Vertex AI Agent Engine      ``LanggraphAgent(..., enable_tracing=True)``
                                                           -> ``cloud-trace:``
  ========  =============================================  ==========================================

In the local and cloud modes call :func:`setup_tracing` once at start-up; on Agent Engine
the Vertex wrapper sets up tracing itself. In every mode, tag requests with
:func:`~hieevas.adapters.otel.run_config` (task, condition, reference) and view the result
with ``python -m hieevas.serve --source <source>`` or :func:`load_runs` + ``evaluate``.

A trace store only holds what the agent did. Correctness needs a reference answer, so
live user traffic yields the process, coordination, safety and governance metrics, while
task success, recovery, refusal and injection metrics come from tagged probe requests
(see :func:`send_probes`).
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .adapters.otel import fetch_cloud_trace_spans, load_otlp_json, run_config, spans_to_runs
from .evaluate import evaluate
from .report import Report
from .trace import RunRecord, load_jsonl

EXPORTERS = ("file", "cloud_trace", "otlp")


# ---------------------------------------------------------------- export (local and cloud modes)
def setup_tracing(exporter: str = "file", service_name: str = "agent", path: str | Path = "traces/spans.jsonl",
                  project_id: str | None = None, endpoint: str | None = None, sample_rate: float = 1.0,
                  instrument_langchain: bool = True, batch: bool = True):
    """Configure OpenTelemetry for a LangChain / LangGraph app and return the tracer provider.

    exporter      "file" (local mode: spans appended to ``path`` as JSON lines),
                  "cloud_trace" (Google Cloud Trace; needs ``opentelemetry-exporter-gcp-trace``),
                  "otlp" (any OpenTelemetry collector at ``endpoint``, OTLP over HTTP).
    sample_rate   share of requests traced (parent-based), e.g. 0.2 for busy services.
    batch         export in a background thread. On Cloud Run use ``--no-cpu-throttling`` or
                  ``batch=False``, otherwise spans may wait until the next request.

    LangChain / LangGraph calls are instrumented with OpenInference
    (``openinference-instrumentation-langchain``), the same span format that Vertex AI
    Agent Engine produces, so all three modes are read by the same code.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    if exporter == "file":
        span_exporter = FileSpanExporter(path, service_name)
    elif exporter == "cloud_trace":
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
        span_exporter = CloudTraceSpanExporter(project_id=project_id or _default_project())
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        span_exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    else:
        raise ValueError(f"exporter must be one of {EXPORTERS}, not {exporter!r}")

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}),
                              sampler=ParentBased(TraceIdRatioBased(sample_rate)))
    provider.add_span_processor(BatchSpanProcessor(span_exporter) if batch else SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)
    if instrument_langchain:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
        except ImportError:
            warnings.warn("openinference-instrumentation-langchain is not installed: LangChain / LangGraph "
                          "calls will not be traced (pip install 'hieevas[live]')", stacklevel=2)
        else:
            LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider


def _default_project() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if project:
        return project
    import google.auth  # installed with the Cloud Trace exporter
    return google.auth.default()[1]


try:
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter, SpanExportResult as _Result
except ImportError:  # pragma: no cover
    _SpanExporter, _Result = object, None


class FileSpanExporter(_SpanExporter):
    """Append finished spans to a JSON-lines file (one flat span record per line).

    The format is read by :func:`~hieevas.adapters.otel.load_otlp_json` and by ``file:`` sources.
    """

    def __init__(self, path: str | Path, service_name: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.service_name = service_name
        self._lock = threading.Lock()

    def export(self, spans):
        lines = []
        for span in spans:
            ctx, parent = span.context, span.parent
            attrs = {k: list(v) if isinstance(v, tuple) else v for k, v in (span.attributes or {}).items()}
            service = self.service_name or (span.resource.attributes.get("service.name") if span.resource else None)
            if service:
                attrs.setdefault("service.name", service)
            lines.append(json.dumps({
                "traceId": format(ctx.trace_id, "032x"), "spanId": format(ctx.span_id, "016x"),
                "parentSpanId": format(parent.span_id, "016x") if parent else None, "name": span.name,
                "startTimeUnixNano": span.start_time, "endTimeUnixNano": span.end_time, "attributes": attrs,
                "status": {"code": span.status.status_code.name},
            }, default=str))
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))
        return _Result.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ---------------------------------------------------------------- sources (all modes)
@dataclass
class FileSource:
    """Spans in a JSON-lines or OTLP JSON file (local mode, or an OpenTelemetry Collector file export)."""
    path: str | Path

    def runs(self, start: float | None = None, end: float | None = None, **convert: Any) -> list[RunRecord]:
        path = Path(self.path)
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return []
        return _window(spans_to_runs(load_otlp_json(path), **convert), start, end)


@dataclass
class RunsFileSource:
    """RunRecords saved with ``Recorder.save`` / ``save_jsonl`` (direct recording, no OpenTelemetry)."""
    path: str | Path

    def runs(self, start: float | None = None, end: float | None = None, **convert: Any) -> list[RunRecord]:
        path = Path(self.path)
        return _window(load_jsonl(path), start, end) if path.exists() else []


@dataclass
class CloudTraceSource:
    """Google Cloud Trace (cloud mode on Cloud Run / a VM, and Vertex AI Agent Engine)."""
    project_id: str
    filter: str = ""
    max_traces: int = 500

    def runs(self, start: float | None = None, end: float | None = None, **convert: Any) -> list[RunRecord]:
        spans = fetch_cloud_trace_spans(self.project_id, filter_=self.filter, page_size=100, start=start, end=end,
                                        max_traces=self.max_traces)
        return spans_to_runs(spans, **convert)


def source_from_uri(uri: str):
    """``file:traces/spans.jsonl``, ``runs:runs.jsonl`` or ``cloud-trace:PROJECT_ID``.

    A bare path is read as a span file.
    """
    scheme, _, rest = uri.partition(":")
    if scheme == "cloud-trace" and rest:
        return CloudTraceSource(rest)
    if scheme == "runs" and rest:
        return RunsFileSource(rest)
    if scheme == "file" and rest:
        return FileSource(rest)
    return FileSource(uri)


def _window(runs: list[RunRecord], start: float | None, end: float | None) -> list[RunRecord]:
    return [r for r in runs if (start is None or r.started >= start) and (end is None or r.started <= end)]


def load_runs(source, hours: float | None = 24, app: str | None = None, **convert: Any) -> list[RunRecord]:
    """Runs from a source (object or URI) started in the last ``hours`` (None = all).

    ``app`` keeps only runs of one service (the OpenTelemetry ``service.name``). Remaining
    keyword arguments go to :func:`~hieevas.adapters.otel.spans_to_runs`, e.g.
    ``tool_risk={"send_email": "high"}``, ``architecture=``, ``condition=`` (defaults for
    untagged runs) or ``scorer=``.
    """
    src = source_from_uri(source) if isinstance(source, str) else source
    end = time.time()
    runs = src.runs(start=None if hours is None else end - hours * 3600, end=end, **convert)
    return [r for r in runs if app is None or r.app == app]


def live_report(source, hours: float | None = 24, group_by: Sequence[str] = ("architecture", "condition"),
                app: str | None = None, **convert: Any) -> Report:
    """Load recent runs and evaluate them in one call."""
    return evaluate(load_runs(source, hours=hours, app=app, **convert), group_by=group_by)


# ---------------------------------------------------------------- one-call entry point
MODES = ("local", "cloud", "vertex")


@dataclass
class Session:
    """Returned by :func:`init`: where traces are read from, plus dashboard shortcuts."""
    mode: str
    source: Any
    provider: Any = None
    convert: dict | None = None

    def runs(self, hours: float | None = 24, app: str | None = None) -> list[RunRecord]:
        return load_runs(self.source, hours=hours, app=app, **(self.convert or {}))

    def report(self, hours: float | None = 24, group_by: Sequence[str] = ("architecture", "condition"),
               app: str | None = None) -> Report:
        return evaluate(self.runs(hours, app), group_by=group_by)

    def save_dashboard(self, path: str | Path = "hieevas_dashboard.html", hours: float | None = 24,
                       group_by: Sequence[str] = ("architecture", "condition"), app: str | None = None,
                       title: str = "HIEEVAS – Hierarchical Evaluation of Agentic Systems") -> Path:
        if self.provider is not None:
            self.provider.force_flush()
        report = self.report(hours, group_by, app)
        return report.to_html(path, title=title, subtitle=f"{len(report.runs)} runs · {self.mode} mode")

    def serve(self, port: int | None = None, host: str | None = None, **kwargs: Any) -> None:
        from .serve import serve
        serve(self.source, host=host, port=port, **{**(self.convert or {}), **kwargs})


def init(mode: str = "local", service_name: str = "agent", path: str | Path = "traces/spans.jsonl",
         project_id: str | None = None, endpoint: str | None = None, sample_rate: float = 1.0,
         tool_risk: dict[str, str] | None = None, instrument_langchain: bool = True,
         batch: bool = True) -> Session:
    """Start evaluating a LangGraph app in one of three deployment modes.

    ``"local"``   app and LLM on this machine: spans go to ``path``.
    ``"cloud"``   app on Cloud Run / a VM: spans go to Cloud Trace in ``project_id``
                  (or to an OpenTelemetry collector when ``endpoint`` is given).
    ``"vertex"``  agent on Vertex AI Agent Engine: tracing is done by the Vertex wrapper
                  (``LanggraphAgent(..., enable_tracing=True)``), so this call only
                  prepares reading from Cloud Trace. Call it where you view the dashboard
                  or send probes, not inside the deployed agent.

    ``tool_risk`` marks high-risk tools by name, e.g. ``{"send_email": "high"}``.
    """
    convert = {"tool_risk": tool_risk} if tool_risk else {}
    if mode == "local":
        provider = setup_tracing("file", service_name, path=path, sample_rate=sample_rate,
                                 instrument_langchain=instrument_langchain, batch=batch)
        return Session(mode, FileSource(path), provider, convert)
    if mode == "cloud":
        exporter = "otlp" if endpoint else "cloud_trace"
        project = None if endpoint else (project_id or _default_project())
        provider = setup_tracing(exporter, service_name, project_id=project, endpoint=endpoint,
                                 sample_rate=sample_rate, instrument_langchain=instrument_langchain, batch=batch)
        return Session(mode, CloudTraceSource(project) if project else None, provider, convert)
    if mode == "vertex":
        return Session(mode, CloudTraceSource(project_id or _default_project()), None, convert)
    raise ValueError(f"mode must be one of {MODES}, not {mode!r}")


# ---------------------------------------------------------------- probes (known-answer and stress requests)
def send_probes(send: Callable[[str, dict], Any], probes: Iterable[dict], architecture: str | None = None,
                pause: float = 0.0, repeats: int = 1) -> list[dict]:
    """Send tagged probe requests to a running agent, in any mode.

    ``send(question, config)`` calls the agent with a LangGraph run config, e.g.
    ``lambda q, cfg: graph.invoke({"messages": [("user", q)]}, config=cfg)`` locally,
    ``lambda q, cfg: remote.query(input={"messages": [("user", q)]}, config=cfg)`` on Agent
    Engine, or an HTTP call that forwards ``cfg["metadata"]`` to the app.
    Each probe is a dict with ``question`` and optionally ``task_id``, ``condition``
    (``C1``-``C4``) and ``reference``. ``repeats`` sends every probe that many times under the
    same task id, which gives consistency (M3). Returns one result dict per request.

    C2 and C3 probes only change behaviour when the app's tools are wrapped with
    :func:`~hieevas.stress.stress_tools`.
    """
    results = []
    probes = [{"task_id": f"probe-{i + 1}", **p} for i, p in enumerate(probes)]  # same id across repeats
    for p in [p for p in probes for _ in range(repeats)]:
        cfg = run_config(task_id=p["task_id"], condition=p.get("condition", "C1"),
                         architecture=architecture, reference=p.get("reference"))
        t0 = time.time()
        try:
            output, error = send(p["question"], cfg), None
        except Exception as exc:  # recorded; the remaining probes still run
            output, error = None, f"{type(exc).__name__}: {exc}"
        results.append({**p, "config": cfg, "output": output, "error": error, "seconds": time.time() - t0})
        if pause:
            time.sleep(pause)
    return results
