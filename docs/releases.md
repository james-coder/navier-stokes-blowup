# Builds and releases

## Normal development

Pushes to `main`, pull requests, and manual CI dispatches run:

1. CPU tests and runnable examples on Python 3.12 and 3.13.
2. A Sphinx documentation build with warnings treated as errors.
3. Wheel and source-distribution builds, metadata checks, and an installed-wheel smoke test outside the checkout.

The test results, built documentation, and distributions are uploaded as Actions artifacts. CUDA tests remain a separately recorded local check; GitHub-hosted CPU runners do not establish GPU support. Action versions are pinned to release commit hashes, with monthly Dependabot updates. Normal CI has read-only repository permissions.

## Versioned releases

Update `project.version` in `pyproject.toml`, `__version__` in the package, the Sphinx `release` value, and the changelog together. Commit the change and let main CI pass. Then create and push a matching version tag, for example:

```bash
git tag -a v0.1.0 -m 'ns-blowup 0.1.0 research preview'
git push origin v0.1.0
```

The tag workflow runs the reusable CI pipeline again against that tag. A mismatch between tag and package version fails before publication. After validation succeeds, a separate job publishes a GitHub Release containing the wheel, source archive, documentation ZIP, and SHA256 checksums. That publication job alone receives write permission. Prerelease version tags are marked as prereleases.

The workflow uses the repository's automatic `GITHUB_TOKEN`; it requires no personal token. It publishes to GitHub Releases only. PyPI trusted publishing is not configured and no package-index credential is assumed. A failed release upload does not overwrite an existing release; inspect the failure before retrying or changing a published tag.

**No release tag was created as part of the initial repository setup.** The tag-triggered release path is configured but **Not Yet Tested end to end** until a release actually runs. Documentation CI produces a downloadable site; a hosted Read the Docs project still needs connecting separately.
