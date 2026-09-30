"""Live mode: run tagging, span files, sources, time buckets and the dashboard server."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from hieevas import evaluate, run_config
from hieevas.adapters.otel import _from_metadata, spans_to_runs

otel = pytest.importorskip("opentelemetry.sdk.trace")


def test_run_config_merges_and_validates():
    cfg = run_config(task_id="T1", condition="C3", config={"metadata": {"user": "u1"}, "recursion_limit": 9})
    assert cfg["recursion_limit"] == 9
    assert cfg["metadata"] == {"user": "u1", "hieevas.task_id": "T1", "hieevas.condition": "C3"}
    with pytest.raises(ValueError):
        run_config(colour="red")


def test_metadata_tags_survive_truncation():
    full = json.dumps({"langgraph_node": "agent", "hieevas.condition": "C2", "hieevas.refused": True})
    assert _from_metadata(full, "hieevas.condition") == "C2"
    cut = full[:-5]  # backend truncated the attribute: no longer valid JSON
    assert _from_metadata(cut, "hieevas.condition") == "C2"
    assert _from_metadata(cut, "hieevas.task_id") is None


def test_tags_from_metadata_attribute():
    meta = json.dumps({"hieevas.task_id": "probe-1", "hieevas.condition": "C4", "hieevas.refused": True,
                       "hieevas.architecture": "A3"})
    records = [
        {"traceId": "t1", "spanId": "s1", "name": "LangGraph", "startTimeUnixNano": 1e18, "endTimeUnixNano": 1e18 + 1e9,
         "attributes": {"openinference.span.kind": "CHAIN", "metadata": meta}},
        {"traceId": "t1", "spanId": "s2", "parentSpanId": "s1", "name": "send_email", "startTimeUnixNano": 1e18,
         "endTimeUnixNano": 1e18 + 5e8, "attributes": {"openinference.span.kind": "TOOL", "tool.name": "send_email"}},
    ]
    r = spans_to_runs(records, tool_risk={"send_email": "high"})[0]
    assert (r.task_id, r.condition, r.architecture, r.refused) == ("probe-1", "C4", "A3", True)
    assert r.steps_of("tool")[0].risk == "high"


def test_out_of_policy_refusal_and_declared_risk():
    def trace(tid, answer, tools):
        spans = [{"traceId": tid, "spanId": "r", "name": "LangGraph", "startTimeUnixNano": 1e18,
                  "endTimeUnixNano": 1e18 + 1e9,
                  "attributes": {"openinference.span.kind": "CHAIN", "output.value": answer,
                                 "metadata": json.dumps({"hieevas.condition": "C4"})}}]
        spans += [{"traceId": tid, "spanId": f"t{i}", "parentSpanId": "r", "name": t, "startTimeUnixNano": 1e18,
                   "endTimeUnixNano": 1e18 + 1e8, "attributes": {"openinference.span.kind": "TOOL", "tool.name": t}}
                  for i, t in enumerate(tools)]
        return spans

    records = trace("a", "I can't send emails.", []) + trace("b", "I cannot do that, it was denied.", ["send_email"])
    runs = {r.run_id: r for r in spans_to_runs(records, tool_risk={"send_email": "high"})}
    assert runs["a"].refused is True
    assert runs["b"].refused is False  # tried the high-risk tool first: not a full refusal
    report = evaluate([runs["a"]], group_by=())
    assert report.value("M22").value == 0 and report.value("M20").value == 1


def _traced_runs(tracer, n, condition="C1"):
    for i in range(n):
        with tracer.start_as_current_span("invoke_agent") as root:
            root.set_attributes({"gen_ai.operation.name": "invoke_agent", "hieevas.task_id": f"T{i}",
                                 "hieevas.condition": condition, "hieevas.architecture": "A1"})
            with tracer.start_as_current_span("chat") as llm:
                llm.set_attributes({"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 100,
                                    "gen_ai.usage.output_tokens": 10})


def test_file_exporter_and_source(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from hieevas.live import FileSource, FileSpanExporter, load_runs

    path = tmp_path / "spans.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(FileSpanExporter(path, service_name="my-app")))
    _traced_runs(provider.get_tracer("t"), 3)
    runs = load_runs(f"file:{path}", hours=1)
    assert len(runs) == 3 and {r.app for r in runs} == {"my-app"}
    assert load_runs(FileSource(path), hours=1, app="other") == []
    assert FileSource(tmp_path / "missing.jsonl").runs() == []
    report = evaluate(runs, group_by=("architecture", "day"))
    assert report.value("M5", architecture="A1").value == 110
    assert report.rows[0]["group"]["day"] == time.strftime("%Y-%m-%d", time.gmtime())


def test_dashboard_server(tmp_path):
    from http.server import ThreadingHTTPServer

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from hieevas.live import FileSpanExporter
    from hieevas.serve import DashboardService, make_handler

    path = tmp_path / "spans.jsonl"
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(FileSpanExporter(path)))
    _traced_runs(provider.get_tracer("t"), 2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DashboardService(f"file:{path}")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        html = urllib.request.urlopen(base + "/?hours=2&group_by=architecture,hour").read().decode()
        assert "<html" in html and "2 runs" in html
        data = json.loads(urllib.request.urlopen(base + "/report.json").read())
        assert data["rows"][0]["n_runs"] == 2
        assert urllib.request.urlopen(base + "/healthz").read() == b"ok"
        for bad in ("/?hours=-1", "/?group_by=password", "/nope"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + bad)
    finally:
        server.shutdown()


def test_langgraph_metadata_reaches_spans():
    """End to end: a LangGraph run tagged with run_config() is read back with its tags."""
    pytest.importorskip("openinference.instrumentation.langchain")
    langgraph = pytest.importorskip("langgraph.graph")
    from typing import TypedDict

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    from hieevas.adapters.otel import HieevasSpanProcessor

    class State(TypedDict):
        question: str
        answer: str

    g = langgraph.StateGraph(State)
    g.add_node("answer", lambda s: {"answer": "Paris"})
    g.add_edge(langgraph.START, "answer")
    g.add_edge("answer", langgraph.END)
    graph = g.compile()

    provider = TracerProvider()
    proc = HieevasSpanProcessor()
    provider.add_span_processor(proc)
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        graph.invoke({"question": "capital of France?"},
                     config=run_config(task_id="T9", condition="C1", architecture="A1", reference="Paris"))
    finally:
        instrumentor.uninstrument()
    runs = proc.runs()
    assert len(runs) == 1
    assert (runs[0].task_id, runs[0].architecture, runs[0].reference) == ("T9", "A1", "Paris")
