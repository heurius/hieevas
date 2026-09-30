# Stand-alone hieevas dashboard for Cloud Run. Reads Cloud Trace, so it serves apps in
# cloud mode (Cloud Run / VM) and on Vertex AI Agent Engine alike.
#   HIEEVAS_SOURCE=cloud-trace:PROJECT_ID   HIEEVAS_TOOL_RISK=send_email,delete_record
FROM python:3.12-slim
# until hieevas is on PyPI: COPY hieevas-0.1.0-py3-none-any.whl . and install that file instead
RUN pip install --no-cache-dir "hieevas[gcp]"
CMD exec python -m hieevas.serve
