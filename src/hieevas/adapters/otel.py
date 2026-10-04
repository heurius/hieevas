"""OpenTelemetry adapter: turn agent traces into hieevas runs.

Any platform that emits OpenTelemetry spans can be evaluated this way, including
Amazon Bedrock AgentCore (spans in CloudWatch, log group ``aws/spans``), Google
Vertex AI Agent Engine (Cloud Trace), LangSmith, Langfuse, Arize Phoenix, or a local
OpenTelemetry Collector.

Three attribute conventions are understood:
  * OpenTelemetry GenAI semantic conventions (``gen_ai.*``)
  * OpenInference (``openinference.span.kind``, ``llm.token_count.*``, ``tool.name``)
  * OpenLLMetry / Traceloop (``traceloop.span.kind``, ``llm.usage.*``)

Three input shapes are accepted:
  * live spans from the OpenTelemetry SDK (``HieevasSpanProcessor``)
  * OTLP JSON (``resourceSpans`` ... ``spans``), e.g. an OTel Collector file export
  * flat span records (``attributes`` as a dict), e.g. CloudWatch ``aws/spans`` or Cloud Trace labels

One trace becomes one run. Add these span attributes (on any span of the trace) to
control grouping and scoring: ``hieevas.task_id``, ``hieevas.condition``,
``hieevas.architecture``, ``hieevas.reference``, ``hieevas.rationale`` (on tool spans),
and the outcome flags ``hieevas.correct``, ``hieevas.refused``, ``hieevas.injection_followed``.
Failed spans should also carry ``error.type``: some backends (Cloud Trace v1 reads) drop span status.

With LangChain / LangGraph the same keys can travel in the run config instead
(``graph.invoke(x, config=run_config(task_id="T1", condition="C3"))``): OpenInference
copies config metadata onto the ``metadata`` attribute of every span, which is read here.
This also works where the application does not own the spans, e.g. Vertex AI Agent Engine.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

TAG_KEYS = ("task_id", "condition", "architecture", "reference", "correct", "refused", "injection_followed")

from ..governance import DENIED_MESSAGE, FAULT_MESSAGE, injection_followed
from ..scoring import detect_refusal, score_answer
from ..trace import LLM, TOOL, RunRecord, Step

# Tool outputs that mark a call as denied by the approval gate, or as failed. Recognising the
# text matters because tool code cannot reach its OpenInference span to set an error status.
DENIED_PREFIX = DENIED_MESSAGE.split(":")[0]  # "ACTION DENIED"
ERROR_PREFIXES = (FAULT_MESSAGE.split(":")[0] + ":", "Error:")  # "ERROR:" and LangGraph's ToolNode errors

LLM_OPS = {"chat", "text_completion", "generate_content", "completion"}
TOOL_OPS = {"execute_tool"}
AGENT_OPS = {"invoke_agent", "create_agent"}


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start: float  # seconds
    end: float
    attrs: dict = field(default_factory=dict)
    error: bool = False


# ---------------------------------------------------------------- normalisation
def _otlp_value(v: dict) -> Any:
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_otlp_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _attrs(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):  # OTLP: [{"key": ..., "value": {...}}]
        return {a["key"]: _otlp_value(a.get("value", {})) for a in raw}
    return {}


def _seconds(v: Any) -> float:
    if v is None:
        return 0.0
    v = float(v)
    # Unix epoch in ns (~1.7e18), us (~1.7e15), ms (~1.7e12) or s (~1.7e9)
    if v > 1e17:
        return v / 1e9
    if v > 1e14:
        return v / 1e6
    if v > 1e11:
        return v / 1e3
    return v


def _is_error(status: Any) -> bool:
    if isinstance(status, dict):
        status = status.get("code", status.get("status_code"))
    return str(status).upper() in ("2", "ERROR", "STATUS_CODE_ERROR", "STATUSCODE.ERROR")


def normalise(record: dict) -> Span:
    """Normalise one span given as an OTLP JSON span or a flat record."""
    return Span(
        trace_id=str(record.get("traceId") or record.get("trace_id") or ""),
        span_id=str(record.get("spanId") or record.get("span_id") or ""),
        parent_id=(record.get("parentSpanId") or record.get("parent_span_id") or record.get("parentId")) or None,
        name=str(record.get("name", "")),
        start=_seconds(record.get("startTimeUnixNano") or record.get("start_time") or record.get("startTime")),
        end=_seconds(record.get("endTimeUnixNano") or record.get("end_time") or record.get("endTime")),
        attrs=_attrs(record.get("attributes") or record.get("labels") or {}),
        error=_is_error(record.get("status")),
    )


def from_sdk_span(span: Any) -> Span:
    """Normalise an OpenTelemetry SDK ``ReadableSpan``."""
    ctx, parent = span.context, span.parent
    return Span(
        trace_id=format(ctx.trace_id, "032x"), span_id=format(ctx.span_id, "016x"),
        parent_id=format(parent.span_id, "016x") if parent else None, name=span.name,
        start=(span.start_time or 0) / 1e9, end=(span.end_time or 0) / 1e9,
        attrs=dict(span.attributes or {}), error=span.status.status_code.name == "ERROR",
    )


def load_otlp_json(path: str | Path) -> list[Span]:
    """Read spans from an OTLP JSON file (one request per line, or one JSON document)."""
    text = Path(path).read_text(encoding="utf-8")
    docs = [json.loads(l) for l in text.splitlines() if l.strip()] if "\n" in text.strip() else [json.loads(text)]
    spans = []
    for doc in docs:
        for rs in doc.get("resourceSpans", []) if isinstance(doc, dict) else []:
            for ss in rs.get("scopeSpans", rs.get("instrumentationLibrarySpans", [])):
                spans += [normalise(s) for s in ss.get("spans", [])]
        if isinstance(doc, dict) and "traceId" in doc:  # flat record per line
            spans.append(normalise(doc))
    return spans


# ---------------------------------------------------------------- classification
def _num(attrs: dict, *keys: str) -> int | None:
    for k in keys:
        if attrs.get(k) is not None:
            try:
                return int(attrs[k])
            except (TypeError, ValueError):
                continue
    return None


def kind_of(span: Span) -> str | None:
    a = span.attrs
    oi = str(a.get("openinference.span.kind", "")).upper()
    op = str(a.get("gen_ai.operation.name", "")).lower()
    tl = str(a.get("traceloop.span.kind", "")).lower()
    if oi == "LLM" or op in LLM_OPS or a.get("llm.request.type") or (
            a.get("gen_ai.system") and _num(a, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens") is not None):
        return LLM
    if oi == "TOOL" or op in TOOL_OPS or tl == "tool" or a.get("gen_ai.tool.name"):
        return TOOL
    if oi == "AGENT" or op in AGENT_OPS or tl == "agent":
        return "agent"
    return None


def _agent_name(span: Span, by_id: dict[str, Span]) -> str:
    meta = span.attrs.get("metadata")
    if isinstance(meta, str) and "langgraph_node" in meta:
        try:
            return json.loads(meta).get("langgraph_node") or "agent"
        except ValueError:
            pass
    node, seen = span, set()
    while node is not None and node.span_id not in seen:
        seen.add(node.span_id)
        name = node.attrs.get("gen_ai.agent.name") or node.attrs.get("agent.name")
        if name:
            return str(name)
        if kind_of(node) == "agent" and node is not span:
            return node.name
        node = by_id.get(node.parent_id) if node.parent_id else None
    return "agent"


def _failed(span: Span) -> bool:
    """Span status, or the attributes that survive backends which drop status (e.g. Cloud Trace v1 reads)."""
    a = span.attrs
    return (span.error or bool(a.get("error.type")) or str(a.get("otel.status_code", "")).upper() == "ERROR"
            or bool(a.get("/error/message")))


def _flag(spans: list[Span], key: str) -> bool | None:
    v = _find(spans, key)
    if v is None:
        return None
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def _find(spans: list[Span], key: str) -> Any:
    for s in spans:
        if s.attrs.get(key) not in (None, ""):
            return s.attrs[key]
    if key.startswith("hieevas."):
        for s in spans:
            value = _from_metadata(s.attrs.get("metadata"), key)
            if value not in (None, ""):
                return value
    return None


def _from_metadata(meta: Any, key: str) -> Any:
    """Read a ``hieevas.*`` key from an OpenInference ``metadata`` attribute (a JSON string).

    Backends truncate long attribute values (Cloud Trace keeps 256 bytes), which breaks
    the JSON, so a truncated document is searched with a regular expression instead.
    """
    if isinstance(meta, dict):
        return meta.get(key)
    if not isinstance(meta, str) or key not in meta:
        return None
    try:
        return json.loads(meta).get(key)
    except (ValueError, AttributeError):
        m = re.search(rf'"{re.escape(key)}"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|-?[\d.]+)', meta)
        return json.loads(m.group(1)) if m else None


# ---------------------------------------------------------------- tagging runs from the application
def run_config(task_id: str | None = None, condition: str | None = None, architecture: str | None = None,
               reference: Any = None, config: dict | None = None, **flags: Any) -> dict:
    """LangChain / LangGraph run config that tags the resulting trace for hieevas.

        graph.invoke(inputs, config=run_config(task_id="probe-7", condition="C3"))
        remote_agent.query(input=inputs, config=run_config(condition="live"))   # Agent Engine

    ``flags`` may carry outcome flags such as ``refused=True``. An existing ``config`` is
    extended, not replaced.
    """
    tags = {"task_id": task_id, "condition": condition, "architecture": architecture, "reference": reference,
            **flags}
    unknown = set(tags) - set(TAG_KEYS)
    if unknown:
        raise ValueError(f"unknown hieevas tag(s): {', '.join(sorted(unknown))}")
    config = dict(config or {})
    meta = dict(config.get("metadata") or {})
    meta.update({f"hieevas.{k}": v for k, v in tags.items() if v is not None})
    config["metadata"] = meta
    return config


def annotate(**tags: Any) -> bool:
    """Set ``hieevas.*`` attributes on the current OpenTelemetry span.

    Use it where the application opens its own spans (e.g. a Cloud Run request handler).
    Returns False when no span is recording, so it is safe to call anywhere.
    """
    from opentelemetry import trace  # optional dependency

    span = trace.get_current_span()
    if not span.is_recording():
        return False
    for k, v in tags.items():
        if v is not None:
            span.set_attribute(f"hieevas.{k}", v if isinstance(v, (bool, int, float, str)) else str(v))
    return True


def _tool_output_text(value: Any) -> str:
    """Text a tool returned. OpenInference stores a serialised ToolMessage (JSON), which a
    backend may have truncated; the leading part of its content is enough for the markers."""
    text = str(value or "")
    if not text.lstrip().startswith("{"):
        return text
    try:
        doc = json.loads(text)
        data = doc.get("data") or doc.get("kwargs") or doc
        return str(data.get("content", "")) if isinstance(data, dict) else text
    except (ValueError, AttributeError):
        m = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        return m.group(1) if m else text


ANSWER_KEYS = ("answer", "final_answer", "response", "output")


def _final_answer(output: Any, last_llm_text: str | None) -> Any:
    """The agent's answer, not the whole graph state.

    LangGraph root spans carry the final state (every message, including tool results), so
    scoring it would credit answers found only in retrieved text. Use the last message of
    that state, else a text field named in ``ANSWER_KEYS`` (e.g. ``state["answer"]``), else the
    text of the last LLM call; a plain-text output is kept as is.
    """
    if not isinstance(output, str):
        return output if output is not None else last_llm_text
    text = output.strip()
    if not text.startswith(("{", "[")):
        return output
    try:
        state = json.loads(text)
    except ValueError:  # truncated by the backend: fall back to the last model output
        return last_llm_text
    messages = state.get("messages") if isinstance(state, dict) else None
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            data = last.get("data") or last.get("kwargs") or last
            content = data.get("content") if isinstance(data, dict) else None
            if isinstance(content, list):  # content blocks
                content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            if content:
                return content
    if isinstance(state, dict):  # graphs that keep the answer in a state field instead of messages
        for key in ANSWER_KEYS:
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return last_llm_text if last_llm_text is not None else output


# ---------------------------------------------------------------- conversion
def spans_to_runs(spans: Iterable[Span | dict], app: str = "otel", architecture: str = "default",
                  condition: str = "C1", scorer: str | Callable = "auto",
                  answer_attr: str | None = None, tool_risk: dict[str, str] | None = None,
                  pending_seconds: float | None = 600) -> list[RunRecord]:
    """Group spans by trace and convert each trace into a RunRecord.

    ``tool_risk`` marks tools as high-risk by name (e.g. ``{"send_email": "high"}``) when the
    spans themselves carry no ``hieevas.risk`` attribute.

    A request that is still running has exported its finished child spans but not its root
    span. Such a trace (no span without a parent, last span ended under ``pending_seconds``
    ago) is skipped until it completes; ``None`` keeps every trace.
    """
    tool_risk = tool_risk or {}
    norm = [s if isinstance(s, Span) else normalise(s) for s in spans]
    traces: dict[str, list[Span]] = {}
    for s in norm:
        traces.setdefault(s.trace_id, []).append(s)

    runs = []
    now = time.time()
    for trace_id, members in traces.items():
        if (pending_seconds is not None and all(s.parent_id for s in members)
                and now - max(s.end for s in members) < pending_seconds):
            continue  # still running: its root span has not been exported yet
        pending: dict[str, list[str | None]] = {}  # tool name -> reasons written before the call
        last_llm_text = None  # final model output = the answer when the root span holds the whole state
        members.sort(key=lambda s: s.start)
        by_id = {s.span_id: s for s in members}
        root = next((s for s in members if not s.parent_id or s.parent_id not in by_id), members[0])
        rec = RunRecord(run_id=trace_id[:16] or root.span_id, task_id=str(_find(members, "hieevas.task_id") or trace_id),
                        app=str(_find(members, "service.name") or app),
                        architecture=str(_find(members, "hieevas.architecture") or architecture),
                        model=_find(members, "gen_ai.request.model") or _find(members, "llm.model_name"),
                        condition=str(_find(members, "hieevas.condition") or condition),
                        started=min(s.start for s in members), ended=max(s.end for s in members),
                        reference=_find(members, "hieevas.reference"), metadata={"source": "otel"})
        high_risk = sorted(t for t, level in tool_risk.items() if level == "high")
        if high_risk:  # declared, so zero attempts is a measurement rather than missing data (M22)
            rec.metadata["high_risk_tools"] = high_risk
        for s in members:
            kind = kind_of(s)
            if kind not in (LLM, TOOL):
                continue
            a, agent = s.attrs, _agent_name(s, by_id)
            step = Step(run_id=rec.run_id, index=len(rec.steps), kind=kind, agent=agent, timestamp=s.start,
                        latency=max(0.0, s.end - s.start))
            if kind == LLM:
                step.input_tokens = _num(a, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens",
                                         "llm.token_count.prompt", "llm.usage.prompt_tokens")
                step.output_tokens = _num(a, "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens",
                                          "llm.token_count.completion", "llm.usage.completion_tokens")
                # OpenInference: text the model wrote alongside its tool calls = rationale for those calls
                reason = str(a.get("llm.output_messages.0.message.content") or "").strip() or None
                if reason is not None:
                    last_llm_text = reason
                i = 0
                while a.get(f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name"):
                    name = str(a[f"llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name"])
                    pending.setdefault(name, []).append(reason)
                    i += 1
            else:
                name = a.get("gen_ai.tool.name") or a.get("tool.name") or s.name.removeprefix("execute_tool ").strip()
                previous = [x for x in rec.steps if x.kind == TOOL and x.tool == name]
                step.tool = str(name)
                step.args = a.get("gen_ai.tool.call.arguments") or a.get("tool.parameters") or a.get("input.value")
                output = _tool_output_text(a.get("output.value"))
                denied = output.startswith(DENIED_PREFIX)
                failed = not denied and (_failed(s) or output.startswith(ERROR_PREFIXES))
                step.status = "denied" if denied else "error" if failed else "ok"
                step.args_valid = None if failed else True
                step.retry = bool(previous) and previous[-1].status == "error"
                queue = pending.get(step.tool) or []
                step.rationale = a.get("hieevas.rationale") or (queue.pop(0) if queue else None)
                step.risk = str(a.get("hieevas.risk") or tool_risk.get(step.tool, "low"))
                if denied:  # the approval gate answered this call (M26)
                    rec.steps.append(Step(run_id=rec.run_id, index=len(rec.steps), kind="approval",
                                          agent="governance", timestamp=s.start, tool=step.tool, decision="denied"))
                    step.index = len(rec.steps)
            rec.steps.append(step)
        answer = root.attrs.get(answer_attr) if answer_attr else None
        rec.answer = answer if answer is not None else _final_answer(root.attrs.get("output.value"), last_llm_text)
        if rec.reference is not None and rec.answer is not None:
            rec.correct, rec.f1 = score_answer(rec.answer, rec.reference, scorer)
        # outcomes judged by the application and attached to the trace as attributes
        for key in ("correct", "refused", "injection_followed"):
            flag = _flag(members, f"hieevas.{key}")
            if flag is not None:
                setattr(rec, key, flag)
        if rec.condition == "C4" and rec.refused is None and rec.answer is not None:
            # full refusal (paper, Section 4.7): declined in words and never tried a high-risk tool
            attempted = any(s.kind == TOOL and s.risk == "high" for s in rec.steps)
            rec.refused = not attempted and detect_refusal(rec.answer)
        if rec.condition == "C3" and rec.injection_followed is None:
            # canary echoed in the answer, or a high-risk action attempted after the injected text
            rec.injection_followed = injection_followed(rec)
        runs.append(rec)
    return runs


# ---------------------------------------------------------------- live collection
try:
    from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor
except ImportError:  # pragma: no cover
    _SpanProcessor = object


class HieevasSpanProcessor(_SpanProcessor):
    """Collect finished spans in-process; call ``runs()`` to convert them.

        provider.add_span_processor(proc := HieevasSpanProcessor())
        ... run your agent (AgentCore Runtime, Vertex Agent Engine or locally) ...
        evaluate(proc.runs(architecture="A1"))
    """

    def __init__(self):
        self.spans: list[Span] = []

    def on_start(self, span, parent_context=None):  # noqa: D401 - SDK interface
        pass

    def on_end(self, span):
        self.spans.append(from_sdk_span(span))

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def runs(self, **kwargs) -> list[RunRecord]:
        return spans_to_runs(self.spans, **kwargs)


# ---------------------------------------------------------------- platform loaders (optional)
def fetch_cloudwatch_spans(start_ms: int, end_ms: int, log_group: str = "aws/spans",
                           filter_pattern: str = "", region: str | None = None) -> list[Span]:
    """Read AgentCore spans from CloudWatch Logs (requires ``boto3`` and AWS credentials).

    AgentCore Observability writes OpenTelemetry spans as JSON log events to the
    ``aws/spans`` log group once Transaction Search is enabled. Experimental: verify
    the record format for your AgentCore version.
    """
    import boto3  # optional dependency

    logs = boto3.client("logs", region_name=region)
    spans, token = [], None
    while True:
        kw = dict(logGroupName=log_group, startTime=start_ms, endTime=end_ms, filterPattern=filter_pattern)
        if token:
            kw["nextToken"] = token
        page = logs.filter_log_events(**kw)
        for event in page.get("events", []):
            try:
                spans.append(normalise(json.loads(event["message"])))
            except (ValueError, KeyError):
                continue
        token = page.get("nextToken")
        if not token:
            return spans


def fetch_cloud_trace_spans(project_id: str, filter_: str = "", page_size: int = 100,
                            start: float | None = None, end: float | None = None,
                            max_traces: int | None = None) -> list[Span]:
    """Read spans from Google Cloud Trace (requires ``google-cloud-trace``).

    Works for Vertex AI Agent Engine (``enable_tracing=True``) and for any service, e.g. on
    Cloud Run, that exports OpenTelemetry spans to Cloud Trace. Uses the Cloud Trace v1 API,
    whose span labels carry the OpenTelemetry attributes. ``start``/``end`` are unix seconds;
    traces are requested newest first, and ``max_traces`` stops paging after that many.
    """
    from google.cloud import trace_v1  # optional dependency

    client = trace_v1.TraceServiceClient()
    from google.protobuf.timestamp_pb2 import Timestamp

    window = {}
    for key, value in (("start_time", start), ("end_time", end)):  # unix seconds
        if value is not None:
            ts = Timestamp()
            ts.FromNanoseconds(int(value * 1e9))
            window[key] = ts
    request = trace_v1.ListTracesRequest(project_id=project_id, filter=filter_, page_size=page_size,
                                         view=trace_v1.ListTracesRequest.ViewType.COMPLETE,
                                         order_by="start desc", **window)
    spans = []
    for n, tr in enumerate(client.list_traces(request=request)):
        if max_traces is not None and n >= max_traces:
            break
        for s in tr.spans:
            spans.append(Span(trace_id=tr.trace_id, span_id=str(s.span_id),
                              parent_id=str(s.parent_span_id) if s.parent_span_id else None, name=s.name,
                              start=s.start_time.timestamp(), end=s.end_time.timestamp(),
                              attrs=dict(s.labels), error=str(s.labels.get("/error/message", "")) != ""))
    return spans
