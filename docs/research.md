# Research basis and implementation strategy

## Finding and product direction

The strongest distinctive application is a construction-specific numerical observatory: a place to evaluate a newly available mathematical example, expose its computational dependencies, and compare numerical methods against it. A generic GPU Navier–Stokes engine would not establish the desired connection between a new mathematical result and new computational capability. Existing projects already offer accelerated and differentiable CFD. [^1], [^2].

The present implementation is a working library plus a component-level observatory. It is an initial executable research artifact, not the completed distinguishing application. Its API and documentation deliberately expose that boundary. A mature release must earn its claims through a complete field construction, independent checks, and reproducible solver-tracking experiments.

## Evidence for translating the construction

The paper describes a concentrating leading field, matched profiles, oscillatory stress realization, corrections, and localization. These dependencies rule out treating its leading power laws as a finished solution. The implemented subset covers similarity coordinates, viscosity scaling, and the heat exterior; the remaining steps are listed individually in the roadmap. [^3].

The inspected Lean coordinate source explicitly defines the positive coordinate through an existence theorem and `Classical.choose` in a noncomputable section. That is a precise example of why proof code is not automatically a numerical evaluator. For this particular scalar equation the existence characterization also suggests a practical contraction iteration, which the Python module implements and tests against implicit derivatives. That inference is local to the coordinate map; it does not establish that all later choices are equally easy to compute. [^4].

The upstream metadata identifies formalized results and their main declarations. We subsequently built the pinned repository successfully and ran its separate Navier–Stokes Comparator procedure: the statements and axioms matched, and both nanoda and Lean kernels accepted the exported solution. The [formal audit record](formal-audit.md) preserves exact versions, trust assumptions, every attempt log and scope. These upstream checks do not validate the Python finite-precision translation. Recording an upstream commit prevents an implementation note from silently referring to a different source tree later. [^5].

## What the numerical translation adds

An executable reference needs concrete choices, numerical representations, error accounting, and usable evaluation interfaces. For every implicit profile or infinite correction process, it must identify what is truncated, how coefficients are selected, and what controls the resulting error on a specified compact evaluation region. One global “tolerance” setting would conceal several different errors unless those contributions were separately estimated.

The coordinate solver is a manageable example. It uses a fixed iteration count so it composes with JAX transformations, returns a named tuple, and accepts time remaining directly. The tests verify the defining relation and derivatives against independent identities across several scales. These checks establish finite numerical behavior in the test range. They do not turn the iterative evaluator into a proof certificate.

The heat exterior is another tractable piece. Its quadrature nodes and weights are built on the host, and batches of integral evaluations run as JAX array operations. Pressure uses a separate radial integral. Tests compare the heat factor to adaptive quadrature and verify both a scalar heat equation and the vector momentum residual. The observatory intentionally uses a different spatial discretization for its refinement experiment, rather than calculating a residual through exactly the same derivative path used to define a force.

That distinction is necessary: setting `forcing = momentum_residual(velocity, pressure)` guarantees an identity for almost any sufficiently differentiable supplied field. It does not establish the theorem's force regularity or compact-support requirements. This project never labels an arbitrary manufactured field as the completed blowup construction.

## Numerical foundation and prior art

A periodic spectral solver makes a useful initial laboratory because its geometry is explicit, pressure projection is transparent, and smooth benchmarks can be checked closely. SpectralDNS is an established Python pseudospectral code with distributed-memory implementations. Its paper is relevant methodological context and a future comparison target; the new library does not inherit its validation or scaling results. [^6], [^7].

The first implementation uses strict two-thirds truncation and RK4. This is an engineering choice favoring a small inspectable method. It is not evidence that this method is best near a constructed singularity. A final comparison needs equal accuracy targets, careful retained-mode accounting, memory reporting, and independent time/space refinement. A method that produces finite values longest might simply be excessively dissipative; time-to-failure alone is an inadequate metric.

The current benchmark suite includes analytical decay, nonzero forced nonlinear dynamics, energy consistency, projection behavior, and differentiation through evolution. The tests also compare retained nonlinear modes against an overresolved calculation. Their narrow scope is useful precisely because a failure can be traced to a particular operator. Large turbulent flows require separate validation rather than a larger version of the same toy benchmark.

## JAX, precision, and optimization

JAX offers a practical common array and differentiation layer for CPU/GPU evaluation, and its installation documentation specifies the supported accelerator runtimes. This release records the exact tested versions and provides optional CUDA dependencies. Support for a backend in JAX does not mean this library has tested that backend. [^8].

JAX's default precision configuration is material to a scientific library. A request for float64 must not quietly become float32, including on reload from disk. The high-level interface therefore rejects float64 when it is disabled, and saved float64 results demand an appropriate configuration on reload. Float64 still has finite dynamic range and cancellation error; it is not a substitute for arbitrary precision. [^9].

The observatory's finite-difference experiment demonstrates a concrete limitation: subtracting nearby values can amplify roundoff as a spatial stencil shrinks. This is an observed property of the included dataset, not a GPU performance claim. It is an appropriate teaching example for why higher grid resolution, higher precision, and more accurate quadrature solve different problems.

Runtime reports separate initial compilation from warmed execution and wait for completion before recording elapsed time, following JAX guidance. They are correctness-oriented smoke benchmarks on a shared machine, not a throughput comparison against NumPy or a production CFD package. [^10]. Memory preallocation is disabled in GPU commands because the device is shared; production capacity remains unmeasured. [^11].

Differentiable inverse design is a useful secondary workflow, but its novelty does not come from the blowup result. The JAX-Fluids paper already describes end-to-end differentiable compressible two-phase simulation. PhiFlow supports differentiable simulation across several numerical backends. The implemented scalar forcing-inference example demonstrates a usable API and a verified gradient path, without suggesting that this release solves practical airfoil, mixing, or free-surface design tasks. [^1], [^2].

JAX-CFD is relevant historical prior art, but its current README states it is no longer maintained and directs readers toward alternatives. Depending on it without acknowledging that status would create an avoidable maintenance risk. This project implements its small periodic core directly and cites the broader numerical context. [^12].

## Blender integration assessment

Blender can import OpenVDB volume sequences, and its fluid-cache documentation includes a Mantaflow script export. Those are plausible exchange points; they do not establish compatibility with this library's current NPZ/VTK output. A real integration needs explicitly named volume fields, physical transforms, frame timing, units, and validation in Blender itself. [^13], [^14].

OpenVDB's Python interface documents NumPy array exchange and grid transforms, so an exporter can be investigated without reimplementing the file format. A future exporter must still be tested with actual OpenVDB bindings and the target Blender version. Writing a plausible file extension or mocking that dependency would not qualify as integration validation. [^15].

The flagship's first viewer is therefore standalone HTML with data generated by the Python library. It provides immediate access to the research artifact and isolates the mathematical work from an external application's installation and cache conventions. Blender remains an optional future renderer. No speedup, visual-quality advantage, or replacement of Mantaflow has been demonstrated.

## Release and claim criteria

The release should describe capabilities as implemented, tested in a named environment, or explicitly untested. The full application can claim construction reproduction only after all required fields are assembled and independently checked over stated domains and truncation tolerances. A numerical residual alone is insufficient; divergence, forcing behavior, localization, energy estimates, and asymptotic fidelity must also be addressed.

A convincing application would publish a dataset tying each solver's loss of accuracy to its retained scales, timestep, precision, and reference error. It would show competing explanations rather than reducing every discrepancy to “blowup.” It should also let another researcher reproduce the result from a pinned paper/formalization version and a versioned environment.

That is the most defensible path to a library people trust: one distinctive application supported by explicit mathematics, reproducible numerical evidence, and a friendly reusable interface. The present component explorer is a concrete first implementation along that path. The outstanding work is substantive and is not hidden behind “Not Yet Tested” labels for code that does not exist.

## Sources

[^1]: [JAX-Fluids](https://arxiv.org/abs/2203.13760). Inspected 2026-09-10.

[^2]: [PhiFlow](https://github.com/tum-pbs/PhiFlow). Inspected 2026-09-10.

[^3]: [OpenAI paper, §§3–10 and Appendix A](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf). Inspected 2026-09-10.

[^4]: [Pinned SimilarityCoordinates.lean](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/SimilarityCoordinates.lean). Inspected 2026-09-10.

[^5]: [Pinned formalization.yaml](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/formalization.yaml). Inspected 2026-09-10.

[^6]: [Mortensen and Langtangen, 2016](https://arxiv.org/abs/1602.03638). Inspected 2026-09-10.

[^7]: [spectralDNS repository](https://github.com/spectralDNS/spectralDNS). Inspected 2026-09-10.

[^8]: [JAX installation](https://docs.jax.dev/en/latest/installation.html). Inspected 2026-09-10.

[^9]: [JAX default dtypes](https://docs.jax.dev/en/latest/default_dtypes.html). Inspected 2026-09-10.

[^10]: [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html). Inspected 2026-09-10.

[^11]: [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html). Inspected 2026-09-10.

[^12]: [JAX-CFD README](https://github.com/google/jax-cfd). Inspected 2026-09-10.

[^13]: [Blender volume objects](https://docs.blender.org/manual/en/4.5/modeling/volumes/introduction.html). Inspected 2026-09-10.

[^14]: [Blender fluid cache](https://docs.blender.org/manual/en/4.5/physics/fluid/type/domain/cache.html). Inspected 2026-09-10.

[^15]: [OpenVDB Python documentation](https://www.openvdb.org/documentation/doxygen/python.html). Inspected 2026-09-10.
