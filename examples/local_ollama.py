"""Mode 1 (local): app and LLM on this machine, traced to a file, scored on a dashboard.

    pip install "hieevas[live]" langchain langchain-ollama
    ollama pull qwen2.5:3b-instruct
    python examples/local_ollama.py
    python -m hieevas.serve --source file:traces/spans.jsonl --tool-risk send_email,delete_record

The only hieevas lines an existing LangGraph app needs are ``hieevas.init(...)`` at
start-up and, optionally, ``config=hieevas.run_config(...)`` to tag a request.
"""
import hieevas
from agent import HIGH_RISK, PROBES, ask, build_agent

session = hieevas.init("local", service_name="docs-agent", path="traces/spans.jsonl", tool_risk=HIGH_RISK)
agent = build_agent()  # MODEL env var, default "ollama:qwen2.5:3b-instruct"

if __name__ == "__main__":
    results = hieevas.send_probes(lambda q, cfg: ask(agent, q, cfg), PROBES, architecture="react-local",
                                    repeats=2)  # repeats give consistency (M3)
    for r in results:
        print(r["task_id"], r["condition"], repr(str(r["output"])[:80]), r["error"] or "")
    print("dashboard:", session.save_dashboard("traces/dashboard.html"))
    print(session.report(group_by=("architecture",)).summary())
