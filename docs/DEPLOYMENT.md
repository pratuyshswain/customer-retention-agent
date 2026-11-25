# Local container execution

Prerequisite: Docker with Linux containers enabled. Run from the repository root.

```sh
docker build -t local-project .
docker run --rm -p 127.0.0.1:8000:8000 local-project
```

Local API only. Open http://localhost:8000/docs. Do not expose the API publicly without security review. Optional credentials must be injected at runtime, never baked into the image.

The configuration is proposed and has not been built or runtime-tested. Dependency versions are not fully locked. Review the build context for secrets, private datasets and redistribution rights before building. Do not publish container images until licensing and security review are complete. If Phase 2 introduces a dependency manifest, consolidate the Docker install step with it after merge.
