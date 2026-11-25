# Engineering and release guide

## Local checks

From the repository root, use Python 3.11 and a fresh virtual environment:

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
python -m ruff check --select E9,F63,F7,F82 src data/generate_dataset.py run_server.py tests
python -m mypy --ignore-missing-imports --follow-imports=silent --check-untyped-defs src data/generate_dataset.py run_server.py
python -m pytest tests -q --cov=. --cov-report=term-missing --cov-report=xml
python -m bandit -r src data/generate_dataset.py run_server.py -lll -iii
python -m pip_audit -r requirements.txt
docker build -t local-project .
```

Ruff initially gates correctness rather than formatting. Mypy checks function bodies, with missing third-party stubs tolerated. Coverage is a baseline, not a percentage guarantee; copied scripts or subprocesses may not be represented. Bandit gates only high-severity, high-confidence findings; dependency auditing covers known advisories in the resolved runtime requirements. Neither is a comprehensive security audit. Dependabot proposes updates weekly; review and test each before merging.

## Candidate release

Use `0.1.0` as a proposed initial version only if no existing release or tag takes precedence. Check GitHub releases and tags first. Work from a clean main branch, review the existing [release checklist](RELEASE_CHECKLIST.md), confirm dataset and model artifact provenance and redistribution rights, and require Python, Docker and engineering checks to pass on the exact candidate commit.

The local runtime command is `python run_server.py`. Follow [deployment instructions](DEPLOYMENT.md) for container use. Do not include credentials, private data or generated artifacts without approval.

Prepare a changelog entry with actual changes, known limitations, tested Python version, and links to successful workflows. After explicit maintainer approval, create an annotated version tag and publish release notes through GitHub. Verify downloads and installation from the release source archive. Do not upload container images or model/data artifacts until their contents and rights are reviewed.

This guide prepares a release; it does not publish a tag, package, container or deployment.
