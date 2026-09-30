"""Serve the live evaluation dashboard over HTTP (standard library only).

    python -m hieevas.serve --source file:traces/spans.jsonl                 # local mode
    python -m hieevas.serve --source cloud-trace:MY_PROJECT --tool-risk send_email=high

Routes: ``/`` dashboard (query parameters ``hours``, ``group_by``, ``app``), ``/report.json``,
``/healthz``. Results are cached for ``--cache-seconds`` so that reloading the page does not
re-read the trace store. The server has no login of its own: on Cloud Run deploy it with
``--no-allow-unauthenticated`` (or behind IAP); locally it listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .evaluate import TIME_BUCKETS, evaluate
from .live import load_runs, source_from_uri
from .trace import RunRecord

GROUP_KEYS = {"architecture", "condition", "model", "app", "task_id", *TIME_BUCKETS}
MAX_HOURS = 24 * 30


class DashboardService:
    """Load runs from a source, evaluate them and render the dashboard, with a short cache."""

    def __init__(self, source, hours: float = 24, group_by: tuple = ("architecture", "condition"),
                 title: str = "HIEEVAS – Hierarchical Evaluation of Agentic Systems", cache_seconds: float = 60, **convert: Any):
        self.source = source_from_uri(source) if isinstance(source, str) else source
        self.hours, self.group_by, self.title = hours, tuple(group_by), title
        self.cache_seconds, self.convert = cache_seconds, convert
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def report(self, hours: float | None = None, group_by: tuple | None = None, app: str | None = None):
        hours = self.hours if hours is None else hours
        group_by = self.group_by if group_by is None else group_by
        key = (hours, group_by, app)
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < self.cache_seconds:
                return hit[1]
        runs: list[RunRecord] = load_runs(self.source, hours=hours, app=app, **self.convert)
        report = evaluate(runs, group_by=group_by)
        with self._lock:
            self._cache[key] = (time.time(), report)
        return report

    def html(self, hours: float | None = None, group_by: tuple | None = None, app: str | None = None) -> str:
        report = self.report(hours, group_by, app)
        window = f"last {hours or self.hours:g} h"
        subtitle = (f"{len(report.runs)} runs · {window} · source {type(self.source).__name__}"
                    + (f" · app {app}" if app else ""))
        return report.html(title=self.title, subtitle=subtitle)


def _params(query: str) -> dict:
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    out: dict = {}
    if "hours" in q:
        hours = float(q["hours"])
        if not 0 < hours <= MAX_HOURS:
            raise ValueError(f"hours must be between 0 and {MAX_HOURS}")
        out["hours"] = hours
    if "group_by" in q:
        keys = tuple(k for k in q["group_by"].split(",") if k)
        bad = set(keys) - GROUP_KEYS
        if bad:
            raise ValueError(f"cannot group by {', '.join(sorted(bad))}; use {', '.join(sorted(GROUP_KEYS))}")
        out["group_by"] = keys
    if q.get("app"):
        out["app"] = q["app"]
    return out


def make_handler(service: DashboardService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server interface
            url = urlparse(self.path)
            if url.path == "/healthz":
                return self._send(200, "text/plain", "ok")
            if url.path not in ("/", "/report.json"):
                return self._send(404, "text/plain", "not found")
            try:
                params = _params(url.query)
            except ValueError as exc:
                return self._send(400, "text/plain", str(exc))
            try:
                if url.path == "/report.json":
                    body = json.dumps(service.report(**params).to_dict(), default=str)
                    return self._send(200, "application/json", body)
                return self._send(200, "text/html; charset=utf-8", service.html(**params))
            except Exception as exc:  # show the reason instead of a bare 500
                self.log_error("dashboard failed: %r", exc)
                return self._send(500, "text/html; charset=utf-8",
                                  f"<h1>Dashboard unavailable</h1><pre>{escape(type(exc).__name__)}: "
                                  f"{escape(str(exc))}</pre>")

        def _send(self, code: int, ctype: str, body: str):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(source, host: str | None = None, port: int | None = None, **kwargs: Any) -> None:
    """Run the dashboard server until interrupted. ``kwargs`` go to :class:`DashboardService`."""
    port = port or int(os.environ.get("PORT", 8050))
    host = host or ("0.0.0.0" if "PORT" in os.environ else "127.0.0.1")  # Cloud Run sets PORT
    server = ThreadingHTTPServer((host, port), make_handler(DashboardService(source, **kwargs)))
    print(f"hieevas dashboard on http://{'localhost' if host == '127.0.0.1' else host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _tool_risk(text: str) -> dict[str, str]:
    pairs = (p.split("=", 1) for p in text.split(",") if p)
    return {k.strip(): (v.strip() if v else "high") for k, v in ((p[0], p[1] if len(p) > 1 else "") for p in pairs)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m hieevas.serve", description="Serve the hieevas dashboard.")
    p.add_argument("--source", default=os.environ.get("HIEEVAS_SOURCE"),
                   help="file:PATH, runs:PATH or cloud-trace:PROJECT_ID (env HIEEVAS_SOURCE)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--group-by", default="architecture,condition")
    p.add_argument("--tool-risk", default=os.environ.get("HIEEVAS_TOOL_RISK", ""),
                   help="high-risk tools, e.g. send_email=high,delete_record=high")
    p.add_argument("--architecture", default="default", help="label for runs without a hieevas.architecture tag")
    p.add_argument("--title", default="HIEEVAS – Hierarchical Evaluation of Agentic Systems")
    p.add_argument("--cache-seconds", type=float, default=60)
    a = p.parse_args(argv)
    if not a.source:
        p.error("--source is required (or set HIEEVAS_SOURCE)")
    serve(a.source, host=a.host, port=a.port, hours=a.hours, group_by=tuple(a.group_by.split(",")),
          title=a.title, cache_seconds=a.cache_seconds, tool_risk=_tool_risk(a.tool_risk),
          architecture=a.architecture)


if __name__ == "__main__":
    main()
