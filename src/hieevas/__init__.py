"""hieevas: multi-level evaluation of agentic AI and RAG pipelines.

Record what an agent does, then score it on 27 metrics across seven dimensions
(effectiveness, efficiency, planning, robustness, coordination, safety, governance)
at agent, interaction and system level.

    from hieevas import Recorder, evaluate
    rec = Recorder(app="my-rag", architecture="A1")
    with rec.run("q1", reference="Paris") as run:
        ...
    evaluate(rec.runs).to_html("dashboard.html")

Or trace a LangGraph app while it runs and view the dashboard (local, cloud or vertex mode):

    session = hieevas.init("local")                 # or "cloud" / "vertex"
    graph.invoke(inputs, config=hieevas.run_config(task_id="q1", reference="Paris"))
    session.save_dashboard("dashboard.html")          # or: python -m hieevas.serve --source ...
"""
from . import metrics
from .adapters.otel import annotate, run_config
from .evaluate import evaluate
from .governance import CANARY, governed_tool, inject, injection_followed
from .live import Session, init, live_report, load_runs, send_probes, setup_tracing
from .recorder import BudgetExceeded, Recorder, RunContext, current_run
from .report import Report
from .scoring import cohen_kappa, detect_refusal, score_answer, token_f1
from .stress import stress_tools
from .trace import RunRecord, Step, load_jsonl, save_csv, save_jsonl

__version__ = "0.1.1"
__all__ = ["Recorder", "RunContext", "current_run", "BudgetExceeded", "evaluate", "Report", "metrics",
           "governed_tool", "inject", "injection_followed", "CANARY", "RunRecord", "Step", "save_jsonl",
           "load_jsonl", "save_csv", "score_answer", "token_f1", "detect_refusal", "cohen_kappa",
           "init", "Session", "setup_tracing", "load_runs", "live_report", "send_probes", "run_config", "annotate", "stress_tools"]
