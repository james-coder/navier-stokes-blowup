# Changelog

## 0.2.0 — independent references and safer stepping

- Add RK4 step doubling, tolerance-controlled acceptance, force-aware candidate checks and bounded retries to `Simulation.run`. The accepted solution now uses two half steps; step counts and roundoff may differ from 0.1.
- Add the true `TaylorGreen3D` initial condition and independent Re=20 energy/enstrophy and spatial-convergence reports.
- Add the optional mpmath reference extra with independent coordinate, heat and pressure evaluations and a 25/50/75-digit decimal dataset.
- Fix HTMLPreview JSON execution and run automated Chromium regression tests in CI.
- Publish a real ScreenCLI install/demo recording of version 0.1.0, a vorticity preview, and explanatory Taylor–Green links.
- Map the complete construction to pinned paper equations and Lean declarations, including quantitative choices still required for the inner profiles.

The complete smooth blowup field and its admissible parameter selector remain unimplemented. This release makes no full-construction or near-singularity tracking claim.

## 0.1.0 — research preview

- Import the design conversation with source provenance.
- Add a JAX periodic Navier–Stokes library, analytical checks and a friendly simulation API.
- Implement similarity coordinates, viscosity scaling and heat-exterior quadrature.
- Add the standalone Singularity Observatory for implemented components and numerical precision experiments.
- Document scientific scope, unimplemented construction steps, validation and prior work.
- Configure CI, documentation builds and tag-triggered GitHub releases.

The complete smooth blowup construction is not implemented. No production-speed,
Blender-integration, formal-verification or near-singularity tracking claim is made.
