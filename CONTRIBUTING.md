# Contributing

The flagship is the Singularity Observatory. Favor changes that make the mathematical construction executable, testable, and usable, with a reusable Navier–Stokes API beneath it.

Install with `python -m pip install -e '.[dev,plot]'`, then run `JAX_PLATFORMS=cpu python -m pytest -q`. NVIDIA contributors can also run the suite with `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false`. Never silently treat CPU execution as a successful GPU test.

A numerical contribution should explain the equation, discretization, units, data layout, domain, and failure modes. Cite the primary source near any translated formula. For construction components, name the equation/lemma and pin the source commit. Supply a test that can detect a physically meaningful error: independent derivatives, a different discretization, a conserved quantity, analytic substitution, or a convergence study. Tests duplicating the same formula in two places are insufficient.

Do not label a synthetic concentrating field as the full blowup construction. Do not define forcing as a residual and then use the resulting identity to claim force regularity. Finite tests and upstream proofs establish different facts; keep that distinction in the docs and API.

Keep the simple path short: named examples, clear errors, predictable shapes, and portable results. Add a runnable example for a new public workflow. Avoid global changes to JAX device, precision, or memory settings on import. Changes to saved-data formats require a schema revision and migration/compatibility notes.

Update the README status matrix with measured evidence. Use **Not Yet Tested** when code exists without relevant validation, and **Not Implemented** when it does not exist. A CI workflow is not a passed CI run. Performance claims need fair workloads, completed device execution, separated compilation, memory measurements, and matched error targets.

Before a public release, run the built wheel outside the source checkout, execute the documented examples, inspect the observatory in a browser, and have a human reviewer assess the scientific claims and release notes.
