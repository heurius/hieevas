# Releasing hieevas

Releases are published to PyPI automatically by `.github/workflows/publish.yml` when a
GitHub release is published. No API token is stored anywhere (PyPI Trusted Publishing).

## One-time setup

1. The code lives at https://github.com/heurius/hieevas (done).
2. On PyPI, go to **Your account → Publishing → Add a new pending publisher** and enter:
   - PyPI project name: `hieevas`
   - Owner: `heurius` (organisation)
   - Repository name: `hieevas`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. On GitHub, go to **Settings → Environments → New environment** and create `pypi`
   (optionally add yourself as a required reviewer, so each release waits for your approval).

## Each release

1. Update `version` in `pyproject.toml` (e.g. `0.1.1`) and add an entry to `CHANGELOG.md`.
2. Commit and push to `main`; wait for CI to pass.
3. Create a GitHub release with the tag `v0.1.1` (it must match the version) and publish it.
4. The workflow runs the tests, builds, checks and uploads the package. Within a few minutes
   `pip install hieevas==0.1.1` works.

PyPI never accepts the same version twice, so every release needs a new version number.
