# Roadmap: a reproducible singularity benchmark

## Flagship acceptance criterion

The completed application should let a researcher select a concrete realization of the published construction, a numerical solver, and an error budget; reproduce a trajectory or field evaluation; and identify where the solver ceases to track the reference. Every result should disclose reference uncertainty, retained spatial scales, time error, precision, and provenance.

**Current status:** a working component observatory plus a general periodic numerical foundation. Complete construction reproduction and solver tracking remain **Not Implemented / Not Yet Tested**. The stages below are outstanding work, not installed features.

## 1. Make all construction choices computational

Record each profile equation, parameter inequality, support cutoff, normalization, and matching condition against a pinned mathematical source. Implement actual coefficient-selection and profile-solving algorithms, not only dataclasses that accept unknown coefficients. Distinguish existence choices from formulas and provide a reproducible choice strategy.

Acceptance: two independent implementations or numerical representations agree on specified compact domains; parameter constraints and profile identities pass; reference error is estimated. The upstream dependency map is in the [paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf) and [Lean tree](https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538).

## 2. Implement matching and corrections

Assemble the inner/outer profiles with moment constraints and axis regularity. Implement background corrections, oscillatory realization, compact mean corrections, residual improvement, and final localization. Each operation needs an evaluator with a stated truncation domain and tolerances. A finite implementation cannot silently equate an infinite limiting construction with its partial sum.

Acceptance: component identities survive composition; divergence and momentum errors are controlled separately; force behavior and localization are checked with independent numerical paths. Numerical tests must include failure controls and cannot certify infinite-order smoothness by finite sampling alone.

## 3. Publish a reference dataset

Define evaluation domains, time-remaining ranges, dimensional normalization, selected parameters, precision, and truncation strategy. Store diagnostics and field samples with hashes. Provide a higher-precision reference and independent comparisons before presenting the dataset as ground truth within any claimed tolerance.

Acceptance: a clean environment reproduces the stated numerical results; the limiting mathematical claims remain clearly attributed to their proof sources. Python translation accuracy and theorem validity are separate questions.

## 4. Measure solver tracking

Introduce solver adapters with explicit domain, initial data, forcing, and comparison norms. Compare trajectories against the assembled construction at equal error goals. Refine time and space independently and compare floating-point formats. Detect aliasing, insufficient spatial sampling, temporal instability, and reference truncation errors separately.

Acceptance: publish reproducible accuracy-versus-cost curves and explain loss of accuracy. Do not rank methods solely by how long they avoid NaN. No claim of “optimal” follows from a finite benchmark sweep.

## 5. Make the flagship broadly usable

Provide a single setup command, presets with known memory budgets, resumable experiments, comparison reports, tutorials, and artifact export. Keep the common path short while exposing the numerical assumptions. Release a compatibility matrix, versioned examples, and documented stability guarantees for the public API.

Acceptance: independent researchers reproduce the example without developer intervention. A published release and CI matrix must actually run before being marked tested. Maintainer review and external validation are milestones, not simulated badges.

## 6. Optional presentation and performance work

Blender/OpenVDB export can follow a validated volume representation and real round-trip tests. Distributed execution and custom kernels should follow profiling and a demonstrated limitation. Free-surface, obstacle, smoke, and engineering-design solvers are separate substantial extensions; none are prerequisites for the first construction-specific flagship.
