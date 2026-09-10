# Contributing

The flagship is the [Singularity Observatory](https://htmlpreview.github.io/?https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/observatory.html). Favor changes that make the mathematical construction executable, testable, and usable, with a reusable Navier–Stokes API beneath it. Find concrete tasks in the [issue backlog](https://github.com/james-coder/navier-stokes-blowup/issues/1) and the [roadmap](docs/roadmap.md).

Install with `python -m pip install -e '.[dev,plot]'`, then run `JAX_PLATFORMS=cpu python -m pytest -q`. NVIDIA contributors can also run the suite with `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`. Never silently treat CPU execution as a successful GPU test.

A numerical contribution should explain the equation, discretization, units, data layout, domain, and failure modes. Cite the primary source near any translated formula. For construction components, name the equation/lemma and pin the source commit. Supply a test that can detect a physically meaningful error: independent derivatives, a different discretization, a conserved quantity, analytic substitution, or a convergence study. Tests duplicating the same formula in two places are insufficient.

Do not label a synthetic concentrating field as the full blowup construction. Do not define forcing as a residual and then use the resulting identity to claim force regularity. Finite tests and upstream proofs establish different facts; keep that distinction in the docs and API.

Keep the simple path short: named examples, clear errors, predictable shapes, and portable results. Add a runnable example for a new public workflow. Avoid global changes to JAX device, precision, or memory settings on import. Changes to saved-data formats require a schema revision and migration/compatibility notes.

Make instructions to open, read, or download an existing resource clickable. Use rendered URLs for app launch, actual artifact downloads for offline bundles, and source links for configuration or code. Check the destination behavior in both GitHub and the built documentation. Keep command arguments and user-generated local paths copyable; do not invent web links to files that exist only after the reader runs an example.

Update the README status matrix with measured evidence. Use **Not Yet Tested** when code exists without relevant validation, and **Not Implemented** when it does not exist. A CI workflow is not a passed CI run. Performance claims need fair workloads, completed device execution, separated compilation, memory measurements, and matched error targets.

Before a public release, follow the [release procedure](docs/releases.md), run the [installed-wheel smoke check](scripts/smoke_wheel.py) outside the source checkout, execute the [documented examples](examples/), inspect the observatory in a browser, and have a human reviewer assess the scientific claims and release notes.
