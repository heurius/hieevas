"""Mode 3 (vertex): a LangGraph agent on Vertex AI Agent Engine, traced by Google's wrapper.

Nothing from hieevas is deployed with the agent: ``enable_tracing=True`` makes Agent Engine
send OpenInference spans to Cloud Trace, and hieevas reads them from there. Tags travel in
the ``config`` of each query.

    pip install "google-cloud-aiplatform[agent_engines,langchain]" "hieevas[gcp]"
    python vertex_agent_engine.py deploy     # create the Agent Engine (takes a few minutes)
    python vertex_agent_engine.py probe      # send tagged probes, write the dashboard
    python vertex_agent_engine.py delete     # remove the Agent Engine
    python -m hieevas.serve --source cloud-trace:PROJECT_ID --tool-risk send_email,delete_record

Agent Engine pickles the agent: use the same Python minor version locally as in the
``requirements`` below.
"""
import json
import os
import sys
import time
from pathlib import Path

import vertexai
from vertexai import agent_engines

import hieevas
from agent import HIGH_RISK, PROBES, PROMPT, TOOLS

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STATE = Path("agent_engine.json")


def make_agent():
    # TOOLS are wrapped with hieevas.stress_tools, so hieevas must be installed in the agent
    return agent_engines.LanggraphAgent(model="gemini-2.5-flash-lite", tools=TOOLS,
                                        model_kwargs={"temperature": 0.2}, enable_tracing=True)


def ask(remote, question: str, config: dict) -> str:
    out = remote.query(input={"messages": [{"role": "user", "content": f"{PROMPT}\n\n{question}"}]}, config=config)
    last = (out.get("messages") or [{}])[-1] if isinstance(out, dict) else {}
    content = last.get("kwargs", {}).get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    return content if isinstance(content, str) else " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def main(cmd: str):
    vertexai.init(project=PROJECT, location=LOCATION, staging_bucket=f"gs://{PROJECT}-staging")
    if cmd == "deploy":
        remote = agent_engines.create(make_agent(), display_name="docs-agent", extra_packages=["agent.py"],
                                      requirements=["google-cloud-aiplatform[agent_engines,langchain]",
                                                    "langchain", "langgraph", "cloudpickle",
                                                    "openinference-instrumentation-langchain",
                                                    "hieevas"])  # until on PyPI: add the wheel to extra_packages
        STATE.write_text(json.dumps({"resource_name": remote.resource_name}))
        print("deployed", remote.resource_name)
    elif cmd == "probe":
        remote = agent_engines.get(json.loads(STATE.read_text())["resource_name"])
        session = hieevas.init("vertex", project_id=PROJECT, tool_risk=HIGH_RISK)
        for r in hieevas.send_probes(lambda q, cfg: ask(remote, q, cfg), PROBES, architecture="agent-engine",
                                       repeats=2):
            print(r["task_id"], r["condition"], repr(str(r["output"])[:80]), r["error"] or "")
        time.sleep(60)  # Cloud Trace needs a little time before new traces can be read
        print("dashboard:", session.save_dashboard("dashboard_agent_engine.html", hours=1))
    elif cmd == "delete":
        name = json.loads(STATE.read_text())["resource_name"]
        agent_engines.delete(name, force=True)
        print("deleted", name)


if __name__ == "__main__":
    main(sys.argv[1])
