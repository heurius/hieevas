# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Report them privately through
GitHub: **Security → Report a vulnerability** on https://github.com/heurius/hieevas, or by
e-mail to heurius.tech@gmail.com with "hieevas security" in the subject.

Include the version, a description, and steps to reproduce. You can expect an acknowledgement
within a week. Fixes are released as a new patch version and noted in the changelog.

## Supported versions

hieevas is in alpha (0.x). Only the latest release receives fixes.

## Data handled by hieevas

hieevas records and reads execution traces of AI applications. Traces can contain user prompts,
retrieved documents, tool arguments and model answers, and therefore personal or confidential
data. hieevas does not send data anywhere by itself: traces go only to the destination you
configure (a local file, Google Cloud Trace, or an OpenTelemetry collector), and the dashboard
reads from there.

- Restrict who can read trace files, Cloud Trace and the dashboard.
- The dashboard server (`python -m hieevas.serve`) has no authentication. It listens on
  127.0.0.1 unless the `PORT` environment variable is set (as on Cloud Run); deploy it only
  behind authentication (`--no-allow-unauthenticated`, IAP or a private network).
- Use sampling and retention limits appropriate to your data.

## Test conditions

`stress_tools` can inject tool failures and a harmless prompt-injection text (with a canary code)
into requests tagged `C2` / `C3`. Only trusted probe callers should be able to set these tags.
