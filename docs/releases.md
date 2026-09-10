# Builds and releases

## Normal development

Pushes to `main`, pull requests, and manual CI dispatches run:

1. CPU tests and runnable examples on Python 3.12 and 3.13.
2. A Sphinx documentation build with warnings treated as errors.
3. Chromium interaction and data-download checks on committed and freshly generated Observatory artifacts.
4. Wheel and source-distribution builds, metadata checks, and an installed-wheel smoke test outside the checkout.

The test results, built documentation, and distributions are available as artifacts on the [CI runs](https://github.com/james-coder/navier-stokes-blowup/actions/workflows/ci.yml). You can also [download the latest released documentation ZIP](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/documentation.zip). CUDA tests remain a [separately recorded local check](validation.md); GitHub-hosted CPU runners do not establish GPU support. Action versions are pinned to release commit hashes, with [monthly Dependabot updates](https://github.com/james-coder/navier-stokes-blowup/blob/main/.github/dependabot.yml). Normal CI has read-only repository permissions.

## Versioned releases

Update `project.version` in [pyproject.toml](https://github.com/james-coder/navier-stokes-blowup/blob/main/pyproject.toml), `__version__` in [the package](https://github.com/james-coder/navier-stokes-blowup/blob/main/src/ns_blowup/__init__.py), `release` in [the Sphinx configuration](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/conf.py), and [the changelog](https://github.com/james-coder/navier-stokes-blowup/blob/main/CHANGELOG.md) together. Commit the change and let [main CI](https://github.com/james-coder/navier-stokes-blowup/actions/workflows/ci.yml) pass. Then create and push a matching version tag, for example:

```bash
git tag -a v0.1.0 -m 'ns-blowup 0.1.0 research preview'
git push origin v0.1.0
```

The tag workflow runs the reusable CI pipeline again against that tag. A mismatch between tag and package version fails before publication. After validation succeeds, the `pypi` job downloads the validated distributions and publishes them to PyPI. It does not rebuild or run project code and alone receives `id-token: write` permission. After PyPI publication succeeds, a separate job publishes a GitHub Release containing the same wheel and source archive, the documentation ZIP, and SHA256 checksums. Only the GitHub Release job receives repository write permission. Prerelease version tags are marked as prereleases on GitHub.

PyPI uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/): GitHub proves the workflow identity through OIDC, without a stored API token. The configured publisher is:

| Field | Value |
| --- | --- |
| PyPI project | [ns-blowup](https://pypi.org/project/ns-blowup/) |
| GitHub owner | [james-coder](https://github.com/james-coder) |
| Repository | [navier-stokes-blowup](https://github.com/james-coder/navier-stokes-blowup) |
| Workflow filename | [release.yml](https://github.com/james-coder/navier-stokes-blowup/blob/main/.github/workflows/release.yml) |
| GitHub environment | `pypi` (defined in [release.yml](https://github.com/james-coder/navier-stokes-blowup/blob/main/.github/workflows/release.yml)) |

The GitHub environment permits version tags. No personal GitHub token or PyPI secret is needed by the workflow. The first upload creates the PyPI project through the pending publisher; subsequent releases use the resulting project publisher. See [PyPI's project-creation guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

Published PyPI files are immutable. Do not move a published tag or try to replace its files. If PyPI succeeds but the GitHub Release job fails, rerun only the failed jobs; do not upload the distributions again. For code or artifact corrections, increment the version and publish a new tag.

Check [Release Actions](https://github.com/james-coder/navier-stokes-blowup/actions/workflows/release.yml), [GitHub Releases](https://github.com/james-coder/navier-stokes-blowup/releases), and [PyPI](https://pypi.org/project/ns-blowup/) for publication status. [Download the released documentation](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/documentation.zip) and [SHA256 checksums](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/SHA256SUMS). Hosted Read the Docs deployment remains **Not Yet Tested / Not Connected**, tracked in [#103](https://github.com/james-coder/navier-stokes-blowup/issues/103).
