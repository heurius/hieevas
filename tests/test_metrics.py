import pytest

from hieevas import Recorder, evaluate, governed_tool, metrics
from hieevas.governance import approve_all
from hieevas.recorder import BudgetExceeded


def make_runs():
    """Two architectures under C1 and C2 with hand-checkable outcomes (test fixtures)."""
    rec = Recorder(app="test", architecture="A1")
    # A1, C1: two tasks, one correct
    with rec.run("t1", reference="4") as run:
        run.llm_call(input_tokens=100, output_tokens=10)
        run.tool_call("calc", {"x": "2+2"}, rationale="compute")
        run.tool_call("calc", {"x": "2+2"})  # redundant, no rationale
        run.answer("4")
    with rec.run("t2", reference="5") as run:
        run.llm_call(input_tokens=200, output_tokens=20)
        run.answer("6")
    # A1, C2: one task with a tool error then a retry, solved
    with rec.run("t1", condition="C2", reference="4") as run:
        run.llm_call(input_tokens=100, output_tokens=10)
        run.tool_call("calc", {"x": "2+2"}, status="error")
        run.tool_call("calc", {"x": "2+2"}, status="ok")
        run.answer("4")
    # A3, C1: planner/executor/reviewer with one rejection that is corrected
    with rec.run("t1", architecture="A3", reference="4") as run:
        run.plan(["search", "compute"])
        run.message("planner", "executor", "do search then compute", tokens=10, accepted=True)
        run.llm_call(agent="executor", input_tokens=300, output_tokens=40)
        run.plan_step_done(0)
        run.review("reject")
        run.review("accept")
        run.answer("4")
    return rec.runs


def test_group_level_metrics():
    report = evaluate(make_runs())
    r = lambda mid, **g: report.value(mid, **g).value
    assert r("M1", architecture="A1", condition="C1") == 0.5
    assert r("M5", architecture="A1", condition="C1") == pytest.approx((110 + 220) / 2)
    assert r("M7", architecture="A1", condition="C1") == pytest.approx(330 / 1)
    assert r("M10", architecture="A1", condition="C1") == pytest.approx(1 / 2)
    assert r("M23", architecture="A1", condition="C1") == pytest.approx(1 / 2)
    assert r("M14", architecture="A1", condition="C2") == 1.0
    assert r("M15", architecture="A1", condition="C2") == 1.0  # one retry per error
    assert r("M11", architecture="A3", condition="C1") == 0.5
    assert r("M16", architecture="A3", condition="C1") == 1.0
    assert r("M17", architecture="A3", condition="C1") == 0.5
    assert r("M18", architecture="A3", condition="C1") == 1.0
    assert r("M19", architecture="A3", condition="C1") == pytest.approx(10 / 340)
    assert r("M25", architecture="A1", condition="C1") == 1.0


def test_fault_induced_drop_is_reported_on_fault_row():
    report = evaluate(make_runs())
    assert report.value("M13", architecture="A1", condition="C1").value is None
    # C1 success 0.5, C2 success 1.0 -> drop = (0.5 - 1.0) / 0.5 = -1.0 (improved under faults)
    assert report.value("M13", architecture="A1", condition="C2").value == pytest.approx(-1.0)


def test_missing_data_is_explained_not_zero():
    report = evaluate(make_runs())
    res = report.value("M16", architecture="A1", condition="C1")
    assert res.value is None and "single-agent" in res.note
    assert "M24" in report.unavailable()


def test_consistency_over_repeats():
    rec = Recorder()
    for ans in ("4", "4", "5"):
        with rec.run("t1", reference="4") as run:
            run.answer(ans)
    assert metrics.BY_ID["M3"].compute(rec.runs).value == pytest.approx(2 / 3)


def test_budget_and_governed_tools():
    rec = Recorder(step_budget=2)
    send = governed_tool(lambda to: "sent", name="send_email", risk="high")
    flaky = governed_tool(lambda q: "ok", name="search", fault_rate=1.0, seed=1)
    with rec.run("t1") as run:
        assert "DENIED" in send("boss@example.com")
        assert "ERROR" in flaky("x")
        run.llm_call()
        run.llm_call()  # third action exceeds the budget of 2
        raise AssertionError("budget should have stopped the run")
    r = rec.runs[0]
    assert r.budget_exceeded
    kinds = [(s.kind, s.tool, s.status) for s in r.steps]
    assert ("approval", "send_email", None) in kinds
    assert ("tool", "send_email", "denied") in kinds and ("tool", "search", "error") in kinds
    report = evaluate(rec.runs, group_by=(), use_scopes=False)
    assert report.value("M26").value == 1 and report.value("M22").value == 1 and report.value("M27").value == 1


def test_approved_high_risk_call_runs():
    send = governed_tool(lambda to: "sent", name="send_ok", risk="high", approval=approve_all)
    with Recorder().run("t") as run:
        assert send("x") == "sent"
    assert run.record.steps[-1].status == "ok"


def test_errors_are_recorded():
    rec = Recorder(raise_errors=False)
    with rec.run("t1"):
        raise KeyError("boom")
    assert "KeyError" in rec.runs[0].error
    with pytest.raises(KeyError):
        with Recorder().run("t2"):
            raise KeyError("boom")


def test_scopes_give_one_complete_column_per_architecture():
    rec = Recorder(high_risk_tools=["send_email"])
    with rec.run("t1", condition="C1", reference="4") as run:
        run.answer("4")
    with rec.run("t1", condition="C3", reference="4") as run:
        run.answer("4")
        run.mark(injection_followed=False)
    report = evaluate(rec.runs, group_by=("architecture",))
    assert report.value("M1").value == 1.0 and report.value("M1").n == 1   # C1 only
    assert report.value("M21").value == 0.0                              # C3 only
    assert report.value("M22").value == 0                                # declared tools, never attempted
    by_cond = evaluate(rec.runs)
    res = by_cond.value("M21", condition="C1")
    assert res.value is None and not res.applicable
