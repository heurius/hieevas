"""OpenTelemetry adapter: live SDK spans, OTLP JSON files and flat (CloudWatch-style) records."""
import json

import pytest

from hieevas import evaluate
from hieevas.adapters.otel import load_otlp_json, spans_to_runs

otel = pytest.importorskip("opentelemetry.sdk.trace")


def test_live_spans_with_genai_conventions():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Status, StatusCode

    from hieevas.adapters.otel import HieevasSpanProcessor

    provider = TracerProvider()
    proc = HieevasSpanProcessor()
    provider.add_span_processor(proc)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("invoke_agent researcher") as root:
        root.set_attributes({"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "researcher",
                             "hieevas.task_id": "T01", "hieevas.condition": "C2",
                             "hieevas.reference": "143", "output.value": "There were 143 studies."})
        with tracer.start_as_current_span("chat qwen2.5") as llm:
            llm.set_attributes({"gen_ai.operation.name": "chat", "gen_ai.request.model": "qwen2.5:1.5b",
                                "gen_ai.usage.input_tokens": 700, "gen_ai.usage.output_tokens": 40})
        with tracer.start_as_current_span("execute_tool search_docs") as t1:
            t1.set_attributes({"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_docs",
                               "hieevas.rationale": "need the study count"})
            t1.set_status(Status(StatusCode.ERROR))
        with tracer.start_as_current_span("execute_tool search_docs") as t2:
            t2.set_attributes({"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_docs"})

    runs = proc.runs(architecture="A1")
    assert len(runs) == 1
    r = runs[0]
    assert (r.task_id, r.condition, r.model, r.correct) == ("T01", "C2", "qwen2.5:1.5b", True)
    llm_steps, tools = r.steps_of("llm"), r.steps_of("tool")
    assert llm_steps[0].input_tokens == 700 and llm_steps[0].agent == "researcher"
    assert [t.status for t in tools] == ["error", "ok"] and tools[1].retry
    report = evaluate(runs, group_by=("architecture", "condition"))
    assert report.value("M14", condition="C2").value == 1.0  # solved despite a tool error
    assert report.value("M15", condition="C2").value == 1.0


def test_otlp_json_with_openinference(tmp_path):
    def attr(k, v):
        key = "intValue" if isinstance(v, int) else "stringValue"
        return {"key": k, "value": {key: str(v) if key == "intValue" else v}}

    doc = {"resourceSpans": [{"scopeSpans": [{"spans": [
        {"traceId": "a" * 32, "spanId": "1" * 16, "name": "agent", "startTimeUnixNano": "1790000000000000000",
         "endTimeUnixNano": "1790000002000000000", "attributes": [attr("openinference.span.kind", "AGENT")]},
        {"traceId": "a" * 32, "spanId": "2" * 16, "parentSpanId": "1" * 16, "name": "ChatOllama",
         "startTimeUnixNano": "1790000000100000000", "endTimeUnixNano": "1790000000900000000",
         "attributes": [attr("openinference.span.kind", "LLM"), attr("llm.token_count.prompt", 500),
                        attr("llm.token_count.completion", 20),
                        attr("metadata", json.dumps({"langgraph_node": "planner"}))]},
        {"traceId": "a" * 32, "spanId": "3" * 16, "parentSpanId": "1" * 16, "name": "lookup",
         "startTimeUnixNano": "1790000001000000000", "endTimeUnixNano": "1790000001100000000",
         "attributes": [attr("openinference.span.kind", "TOOL"), attr("tool.name", "lookup")]},
    ]}]}]}
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(doc))
    runs = spans_to_runs(load_otlp_json(path))
    r = runs[0]
    assert r.duration == pytest.approx(2.0)
    assert r.steps_of("llm")[0].agent == "planner" and r.steps_of("llm")[0].output_tokens == 20
    assert r.steps_of("tool")[0].tool == "lookup"


def test_flat_cloudwatch_style_records():
    records = [
        {"traceId": "t1", "spanId": "s1", "name": "invoke_agent", "startTimeUnixNano": 1e18,
         "endTimeUnixNano": 1e18 + 5e9, "attributes": {"gen_ai.operation.name": "invoke_agent"},
         "status": {"code": "UNSET"}},
        {"traceId": "t1", "spanId": "s2", "parentSpanId": "s1", "name": "chat", "startTimeUnixNano": 1e18,
         "endTimeUnixNano": 1e18 + 2e9, "attributes": {"gen_ai.operation.name": "chat",
                                                       "gen_ai.usage.input_tokens": 300,
                                                       "gen_ai.usage.output_tokens": 10}},
        {"traceId": "t1", "spanId": "s3", "parentSpanId": "s1", "name": "execute_tool send_email",
         "startTimeUnixNano": 1e18 + 3e9, "endTimeUnixNano": 1e18 + 4e9,
         "attributes": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "send_email",
                        "hieevas.risk": "high"}, "status": {"code": "ERROR"}},
    ]
    r = spans_to_runs(records, architecture="agentcore")[0]
    assert r.duration == pytest.approx(5.0)
    tool = r.steps_of("tool")[0]
    assert (tool.tool, tool.status, tool.risk) == ("send_email", "error", "high")
    report = evaluate([r], group_by=(), use_scopes=False)
    assert report.value("M5").value == 310 and report.value("M22").value == 1


def test_final_answer_reads_state_answer_field():
    import json
    from hieevas.adapters.otel import _final_answer
    state = {"query_text": "Java developer", "retrieved": [{"text": "resume"}], "answer": "John Doe matches."}
    assert _final_answer(json.dumps(state), last_llm_text="raw model text") == "John Doe matches."
    assert _final_answer(json.dumps({"retrieved": []}), last_llm_text="raw model text") == "raw model text"


def test_running_request_is_skipped_until_its_root_span_arrives():
    import time
    from hieevas.adapters.otel import Span, spans_to_runs
    now = time.time()
    child = Span(trace_id="t1", span_id="c1", parent_id="root1", name="retrieval_agent",
                 start=now - 2, end=now - 1, attrs={})
    assert spans_to_runs([child]) == []                        # root still running
    assert len(spans_to_runs([child], pending_seconds=None)) == 1
    root = Span(trace_id="t1", span_id="root1", parent_id=None, name="LangGraph", start=now - 3, end=now, attrs={})
    assert len(spans_to_runs([child, root])) == 1               # complete
