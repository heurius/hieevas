from hieevas import evaluate, load_jsonl, save_csv, save_jsonl
from hieevas.metrics import ALL

from test_metrics import make_runs


def test_roundtrip_jsonl(tmp_path):
    runs = make_runs()
    path = save_jsonl(runs, tmp_path / "runs.jsonl")
    back = load_jsonl(path)
    assert [r.to_dict() for r in back] == [r.to_dict() for r in runs]


def test_csv_logs(tmp_path):
    runs_csv, steps_csv = save_csv(make_runs(), tmp_path)
    assert runs_csv.read_text().startswith("run_id,") and "kind" in steps_csv.read_text().splitlines()[0]


def test_exports_and_dashboard(tmp_path):
    report = evaluate(make_runs())
    assert len(ALL) == 27
    report.to_csv(tmp_path / "m.csv")
    report.to_json(tmp_path / "m.json")
    html = report.to_html(tmp_path / "d.html", title="Test").read_text(encoding="utf-8")
    assert "<svg" in html and "M14" in html and "Coverage notes" in html
    assert "http" not in html.split("<body>")[0]  # no external resources in <head>
    assert "M1 Task success rate" in report.summary()


def test_every_metric_is_explained_on_the_dashboard():
    from hieevas.metrics.guide import GUIDE, guide_markdown

    assert set(GUIDE) == {m.id for m in ALL}
    assert all(g["measures"] and g["how"] and g["source"] and g["read"] for g in GUIDE.values())
    html = evaluate(make_runs()).html()
    assert html.count("<summary>What is this?</summary>") == 27 and "How a run becomes metrics" in html
    assert guide_markdown().count("### M") == 27


def test_values_carry_their_calculation():
    from hieevas import Recorder

    rec = Recorder(architecture="A1")
    for i, answer in enumerate(["4", "4", "5"]):
        with rec.run(f"t{i}", reference="4") as run:
            run.llm_call(input_tokens=100, output_tokens=10)
            run.answer(answer)
    report = evaluate(rec.runs, group_by=("architecture",))
    assert report.value("M1", architecture="A1").detail == "2 correct ÷ 3 scored runs"
    assert report.value("M5", architecture="A1").detail == "330 tokens ÷ 3 runs"
    assert report.value("M7", architecture="A1").detail == "330 tokens in 3 scored runs ÷ 2 correct runs"
    html = report.html()
    assert "2 correct ÷ 3 scored runs" in html and "Runs executed" in html
