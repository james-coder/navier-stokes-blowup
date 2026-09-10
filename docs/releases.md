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

The tag workflow runs the reusable CI pipeline again against that tag. A mismatch between tag and package version fails before publication. After validation succeeds, the `pypi` job downloads the validated distributions and publishes them to PyPI. It does not rebuild or run project code and alone receives `id-token: write` permission. After PyPI publication succeeds, a separate job publishes a GitHub Release containing the same wheel and source archive, the documentation ZIP, and SHA256 checksums. Only the GitHub Release job receives repository write permission. Prerelease version tags are marked as prereleases on GitHub.

PyPI uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/): GitHub proves the workflow identity through OIDC, without a stored API token. The configured publisher is:

| Field | Value |
| --- | --- |
| PyPI project | `ns-blowup` |
| GitHub owner | `james-coder` |
| Repository | `navier-stokes-blowup` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

The GitHub environment permits version tags. No personal GitHub token or PyPI secret is needed by the workflow. The first upload creates the PyPI project through the pending publisher; subsequent releases use the resulting project publisher. See [PyPI's project-creation guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

Published PyPI files are immutable. Do not move a published tag or try to replace its files. If PyPI succeeds but the GitHub Release job fails, rerun only the failed jobs; do not upload the distributions again. For code or artifact corrections, increment the version and publish a new tag.

Check [Release Actions](https://github.com/james-coder/navier-stokes-blowup/actions/workflows/release.yml), [GitHub Releases](https://github.com/james-coder/navier-stokes-blowup/releases), and [PyPI](https://pypi.org/project/ns-blowup/) for publication status. Documentation CI produces a downloadable site; hosted Read the Docs deployment remains **Not Yet Tested / Not Connected**.
