# Contributing

Open an issue describing the problem or proposed change before substantial work. Fork the repository, create a focused branch and submit a pull request with the rationale and verification steps.

## Local verification

Use Python 3.11 in an isolated virtual environment from the repository root.

```sh
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

Run data pipelines in a disposable checkout: generated charts, databases and exports may be overwritten. Use synthetic or authorized data only. Include regression tests for behavior changes where practical. Do not commit credentials, private datasets or local environment files.

## Review

Keep changes focused and preserve existing behavior unless the change is documented. Disclose known limitations and failing checks. Contributions must be material you have rights to submit. Licensing is pending ownership review; do not assume permission to redistribute or relicense the repository.
