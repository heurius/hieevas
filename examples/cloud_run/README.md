# Mode 2: LangGraph app on Cloud Run, traced to Cloud Trace

```
 user ──▶ Cloud Run: main.py (LangGraph + hieevas.init("cloud")) ──▶ Vertex AI Gemini
                          │ OpenInference spans
                          ▼
                    Cloud Trace ◀── Cloud Run: dashboard (python -m hieevas.serve)
```

## 1. Deploy the app

Copy `../agent.py` next to `main.py`, then:

```bash
gcloud run deploy docs-agent --source . --region us-central1 \
  --set-env-vars MODEL=google_vertexai:gemini-2.5-flash-lite,HIEEVAS_ACCEPT_TAGS=1 \
  --no-cpu-throttling --no-allow-unauthenticated
```

- `--no-cpu-throttling` lets the background exporter send spans after the response is
  returned. Without it, use `hieevas.init(..., batch=False)` (adds export time to requests).
- The service account needs **Cloud Trace Agent** (`roles/cloudtrace.agent`) and
  **Vertex AI User** (`roles/aiplatform.user`).
- For busy services set `TRACE_SAMPLE_RATE=0.2` to trace one request in five.

## 2. View the dashboard

Either inside the app (set `HIEEVAS_DASHBOARD=1`, open `/eval`), or as its own service,
which also works for Vertex AI Agent Engine traces:

```bash
gcloud run deploy hieevas-dashboard --source ../dashboard --region us-central1 \
  --set-env-vars HIEEVAS_SOURCE=cloud-trace:PROJECT_ID,HIEEVAS_TOOL_RISK=send_email,delete_record \
  --no-allow-unauthenticated
```

The dashboard's service account needs **Cloud Trace User** (`roles/cloudtrace.user`).
Neither service has its own login: keep `--no-allow-unauthenticated` (or put them behind IAP).

## 3. Send probes (known answers and stress conditions)

```python
import requests, hieevas
from agent import PROBES

url, token = "https://docs-agent-....run.app", "<gcloud auth print-identity-token>"
hieevas.send_probes(lambda q, cfg: requests.post(
    f"{url}/ask", json={"question": q, "tags": {k.removeprefix("hieevas."): v for k, v in cfg["metadata"].items()}},
    headers={"Authorization": f"Bearer {token}"}).json(), PROBES)
```

Schedule this (Cloud Scheduler → Cloud Run job) to keep task-success, refusal and
injection metrics current alongside the process metrics from real traffic.
