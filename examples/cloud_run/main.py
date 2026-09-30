"""Mode 2 (cloud): the LangGraph app on Cloud Run (or any VM) calling a hosted LLM, traced to Cloud Trace.

Local test (uses your gcloud credentials; spans go to Cloud Trace in GOOGLE_CLOUD_PROJECT):
    pip install "hieevas[gcp]" langchain langchain-google-vertexai fastapi uvicorn
    set MODEL=google_vertexai:gemini-2.5-flash-lite
    uvicorn main:app --port 8080

Deploy: see README.md in this folder. The dashboard is at /eval when HIEEVAS_DASHBOARD=1,
or run it as a separate service with ``python -m hieevas.serve --source cloud-trace:PROJECT``.
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import hieevas
from hieevas.serve import DashboardService

from agent import HIGH_RISK, ask, build_agent

session = hieevas.init("cloud", service_name=os.environ.get("K_SERVICE", "docs-agent"), tool_risk=HIGH_RISK,
                         sample_rate=float(os.environ.get("TRACE_SAMPLE_RATE", "1.0")))
agent = build_agent()  # MODEL env var, e.g. "google_vertexai:gemini-2.5-flash-lite"
app = FastAPI()

# Probe tags (condition, reference, ...) are accepted only when explicitly enabled, so that
# ordinary callers cannot label their own traffic.
ACCEPT_TAGS = os.environ.get("HIEEVAS_ACCEPT_TAGS") == "1"


class Ask(BaseModel):
    question: str
    tags: dict | None = None  # e.g. {"task_id": "gaia", "condition": "C1", "reference": "466"}


@app.post("/ask")
def ask_route(body: Ask):
    config = None
    if body.tags:
        if not ACCEPT_TAGS:
            raise HTTPException(403, "probe tags are disabled (set HIEEVAS_ACCEPT_TAGS=1)")
        try:
            config = hieevas.run_config(**{"architecture": "react-cloud", **body.tags})
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"answer": ask(agent, body.question, config)}


if os.environ.get("HIEEVAS_DASHBOARD") == "1":
    dashboard = DashboardService(session.source, tool_risk=HIGH_RISK,
                                 architecture="react-cloud")

    @app.get("/eval", response_class=HTMLResponse)
    def eval_route(hours: float = 24, group_by: str = "architecture,condition"):
        return dashboard.html(hours=hours, group_by=tuple(group_by.split(",")))
