# Contributing to opik-hermes

Thanks for contributing to `opik-hermes`.

## Before opening an issue

1. Search existing issues first: <https://github.com/comet-ml/opik-hermes/issues>
2. Use the matching template:
   - Bug report: [.github/ISSUE_TEMPLATE/bug_report.yml](.github/ISSUE_TEMPLATE/bug_report.yml)
   - Feature request: [.github/ISSUE_TEMPLATE/feature_request.yml](.github/ISSUE_TEMPLATE/feature_request.yml)
3. Include reproducible steps, Hermes version, and plugin version.

## Local setup

Prerequisites:

- Python `>=3.11,<3.14`

Clone and bootstrap into a virtual environment:

```bash
git clone https://github.com/comet-ml/opik-hermes.git
cd opik-hermes
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional runtime config:

```bash
cp .env.example .env
```

## Development workflow

1. Create a focused branch for your change.
2. Keep changes scoped and avoid unrelated formatting-only edits.
3. Run local checks before opening/updating a PR.

Recommended local checks (these mirror CI):

```bash
python -m pytest tests/ -v
ruff check observability tests
ruff format --check observability tests
```

The `observability/opik/` directory is byte-compatible with Hermes' bundled
plugin layout, so it can be mounted as a user plugin, vendored into Hermes, or
shipped as the pip package — all from one source. Keep that contract intact:
the wheel must expose the `hermes_agent.plugins` → `opik_hermes:register` entry
point (CI's `build` job asserts this).

## Pull requests

Open PRs here: <https://github.com/comet-ml/opik-hermes/pulls>

Please:

1. Prefer opening a draft PR early for feedback.
2. Follow the PR template: [.github/pull_request_template.md](.github/pull_request_template.md)
3. Link related issues using `Fixes #<issue-number>` (or `Resolves #<issue-number>`) in the PR body.
4. Update tests/docs for behavior changes.
5. Call out compatibility changes clearly.

If you use GitHub CLI, common commands are:

```bash
gh pr create --draft
gh pr view --web
```

## Releases

- Publishing is driven by a GitHub Release + `.github/workflows/publish.yml`.
- Bump `version` in `pyproject.toml` for the release, then publish a GitHub
  Release tagged `v<version>` — the tag must match `pyproject.toml` exactly (CI
  enforces this before uploading to PyPI).
- Release notes are assembled automatically by release-drafter as PRs merge.

## Commit and review expectations

- Keep commits scoped and reviewable.
- Use clear commit messages that describe behavior changes.
- If changing user-facing configuration, update `README.md` and
  `observability/opik/README.md`.

## Security and secrets

- Do not commit API keys, tokens, or `.env` files.
- Use `.env.example` as the template for new configuration fields.

## References

- Opik docs: <https://www.comet.com/docs/opik>
- Hermes agent: <https://github.com/NousResearch/hermes-agent>
