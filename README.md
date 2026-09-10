# ns-blowup

**A Navier–Stokes library with one flagship: the Singularity Observatory.**

The aim is to make a new mathematical construction something people can inspect, evaluate, and eventually use to challenge numerical solvers. A friendly Python API provides the simulation and diagnostics underneath it.

**Version 0.1.0 is a research preview.** The library runs real 2D/3D periodic simulations and evaluates specific components of OpenAI’s published construction. **The complete smooth blowup solution is Not Implemented.** This release does not yet support the claim that it reproduces the theorem or enables previously impossible fluid simulation.

[Open the observatory](https://htmlpreview.github.io/?https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/observatory.html) · [Download offline app and docs](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/documentation.zip) · [Getting started](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/tutorial.md) · [API](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/api.md) · [Research and implementation gaps](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/research.md) · [Original conversation](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/conversation.md)

## The flagship application

**“How close to a known singularity can a numerical solver remain accurate—and why does it fail?”**

That is the question the completed Singularity Observatory should answer. OpenAI’s paper supplies the motivating construction; its accompanying repository contains Lean formalizations. These are mathematical sources, not a ready-made GPU fluid solver. [Paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf), [formalization](https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538).

The **working preview** lets you:

- Move toward the singular time and inspect the similarity-coordinate geometry.
- Compare normalized velocity and core-energy scaling.
- Change a hypothetical grid resolution and see its spatial coverage.
- Inspect an actual FP32/FP64 refinement experiment for the outer heat-flow component.
- [Download the app and its numerical datasets](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/documentation.zip) for offline use.

The geometry sampling threshold is a heuristic. The scaling plots are not measured norms of a completed blowup solution. The heat exterior is valid away from the axis and is itself singular on the axis at all times; it must not be used as smooth initial data for a purported blowup reproduction. The implementation boundary follows equations (3.2), (4.1), and Lemma A.6 of the [paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf).

Use [the browser preview](https://htmlpreview.github.io/?https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/observatory.html) to run the checked-in app directly. For offline use, [download the app and documentation ZIP](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/documentation.zip), extract it, and open `observatory.html` from that folder. The downloaded app is self-contained and needs no server, account, CDN, or Python installation. To regenerate it:

```bash
python -m ns_blowup.observatory --output outputs/observatory.html
```

You can inspect the checked-in [numerical dataset](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/observatory.json). Regenerating the app also writes `outputs/observatory.json`. The intended mature application and its acceptance criteria are described in [the roadmap](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/roadmap.md).

## Install

Requires **Python 3.12+**. Install the research preview from [PyPI](https://pypi.org/project/ns-blowup/):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'ns-blowup[plot]==0.1.0'
```

For an NVIDIA GPU on Linux/WSL2:

```bash
python -m pip install 'ns-blowup[cuda12,plot]==0.1.0'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

CUDA 12 was exercised on the local RTX 4090. A `cuda13` extra is provided but **Not Yet Tested**. Driver/platform requirements come from the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). Disabling preallocation helps when sharing a GPU; it does not make an oversized calculation fit in memory. See [JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

For development, clone [this repository](https://github.com/james-coder/navier-stokes-blowup) and run `python -m pip install -e '.[dev,plot]'` from its root. See the [contributor setup and checks](https://github.com/james-coder/navier-stokes-blowup/blob/main/CONTRIBUTING.md).

## Your first simulation

```python
from ns_blowup import Simulation

result = Simulation((32, 32), nu=0.05).run(
    initial="taylor-green",
    t_end=1.0,
    frames=21,
)

result.plot(quantity="vorticity", path="outputs/vorticity.png")
result.save("outputs/flow.npz")
print(result.diagnostics()["energy"])
```

![Signed vorticity of the 32 by 32 Taylor–Green simulation at time 1.0, with viscosity 0.05](https://raw.githubusercontent.com/james-coder/navier-stokes-blowup/main/docs/images/vorticity.png)

This is the actual vorticity plot from the example above: a 32×32 periodic Taylor–Green flow at `t=1.0`, with viscosity `nu=0.05`. [View the full-size PNG](https://raw.githubusercontent.com/james-coder/navier-stokes-blowup/main/docs/images/vorticity.png) or [reproduce it with the rendering example](https://github.com/james-coder/navier-stokes-blowup/blob/main/examples/render_vorticity.py).

`Simulation` chooses a JAX device, projects the initial velocity, adapts the timestep, checks for nonfinite states, and returns uniformly timed snapshots. `result.final` is the last velocity field. No array-layout knowledge is needed for the built-in examples.

For a genuinely three-dimensional exact benchmark:

```python
result = Simulation((24, 24, 24), nu=0.02).run("abc", t_end=0.5)
result.export_vtk("outputs/final.vtk")
```

Or run the CLI:

```bash
python -m ns_blowup --dim 3 --flow abc --resolution 24 --time 0.5 --plot
```

The CLI writes `flow.npz`, `final.vtk`, and `diagnostics.json` to `outputs/demo/`. VTK exports follow the structured-points layout with x varying fastest. [VTK file-format documentation](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html).

## Analytical evaluation and numerical checks

```python
import jax.numpy as jnp
from ns_blowup import ABCFlow, FieldDiagnostics

flow = ABCFlow(nu=0.01)
points = jnp.array([[0.2, 0.4, 0.6], [1.0, 2.0, 3.0]])
checks = FieldDiagnostics(flow.velocity, flow.pressure, flow.nu, flow.forcing)

velocity = flow.velocity(points, 0.3)
vorticity = checks.vorticity(points, 0.3)
residual = checks.residual(points, 0.3)
divergence = checks.divergence(points, 0.3)
```

The momentum residual is

```text
R = ∂t u + (u · ∇)u + ∇p − ν Δu − f
```

Pointwise checks differentiate field callbacks independently of the spectral solver. Tests include wrong-pressure/wrong-force controls so a residual that always returns zero would fail. The implemented exact flows are documented with their formulas in [the numerical-methods note](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/numerics.md).

## Differentiable simulation

The lower-level API composes with `jax.jit`, `jax.grad`, and `jax.vmap`:

```python
import jax
from ns_blowup import PeriodicGrid, SpectralSolver, TaylorGreen, kinetic_energy

grid = PeriodicGrid((16, 16))
solver = SpectralSolver(grid, nu=0.05)
u0 = TaylorGreen(nu=0.05).velocity(grid.points(), 0.0)

def final_energy(amplitude):
    final = solver.integrate(amplitude * u0, dt=0.01, steps=50)
    return kinetic_energy(final, grid)

d_energy_d_amplitude = jax.jit(jax.grad(final_energy))(1.0)
```

[The forcing-optimization example](https://github.com/james-coder/navier-stokes-blowup/blob/main/examples/optimize_forcing.py) infers a scalar forcing amplitude through the numerical evolution. It is a small reproducible inverse problem, not evidence of globally optimal engineering designs. Differentiable CFD predates this project and the cited blowup result: see [JAX-Fluids](https://arxiv.org/abs/2203.13760) and [PhiFlow](https://github.com/tum-pbs/PhiFlow).

## What works, and what does not

Outstanding implementation, validation, and future ideas are tracked in the [GitHub backlog](https://github.com/james-coder/navier-stokes-blowup/issues/1), with an [area-by-area guide](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/backlog.md).

“Tested” means the listed checks passed in this checkout on the recorded environment, not that all physical regimes are validated.

| Capability | Status | Evidence / limit |
|---|---|---|
| 2D/3D periodic, constant-density incompressible flow | Tested | Projection, analytic decay, forced nonlinear flow, energy identity |
| JAX CPU / NVIDIA CUDA 12 execution | Tested | Local CPU and RTX 4090 reports below; small grids |
| FP32 and FP64 | Tested | Precision-specific numerical reports; FP64 needs x64 enabled |
| Fourth-order time convergence | Tested | Timestep refinement on a resolved analytic flow |
| Nonlinear dealiasing | Tested | Comparison against a 4× finer grid and inviscid energy production |
| Gradients through initial amplitude and forcing | Tested | Analytic derivative and centered finite-difference comparison |
| Adaptive convenience API, plots, NPZ round-trip | Tested | Workflow tests and executed examples |
| VTK writer | Tested: format/order | **Not Yet Tested** in the ParaView GUI |
| Similarity coordinates / viscosity rescaling | Tested: component identities | Implicit derivatives and equation residuals |
| Heat exterior | Tested: bounded sample ranges | Independent adaptive quadrature and momentum/heat residuals |
| Observatory browser controls and responsive layout | Tested in Chromium | Desktop/mobile viewport, sliders, animation, no JS errors |
| Full smooth blowup construction | **Not Implemented / Not Yet Tested** | Missing inner profiles, matching, pulses, corrections, localization |
| Formal certification of Python results | **Not Yet Tested** | Lean build and Comparator were not run here |
| Near-singularity solver tracking | **Not Yet Tested** | Needs complete executable field and certified truncation control |
| CUDA 13, AMD, Apple GPU, TPU, multi-GPU | **Not Yet Tested** | No hardware/runtime validation here; multi-GPU API not implemented |
| Large turbulent production cases / arbitrary precision | **Not Yet Tested** | No production qualification; arbitrary-precision backend not implemented |
| Walls, obstacles, free surfaces, FLIP/APIC, smoke/fire | **Not Implemented** | Current domain is periodic and single-phase |
| Blender add-on / OpenVDB exporter | **Not Implemented / Not Yet Tested** | Integration research only |
| Speedup over Blender or other CFD packages | **Not Yet Tested** | No comparative benchmark or speed claim |
| Hosted CI | Configured; see [live runs](https://github.com/james-coder/navier-stokes-blowup/actions/workflows/ci.yml) | CPU tests, docs and distribution builds; CUDA remains locally tested |
| Tag-triggered releases / PyPI | Automated validation and Trusted Publishing | Check the [release run](https://github.com/james-coder/navier-stokes-blowup/actions/workflows/release.yml) for publication status |

## Reproducible validation

```bash
JAX_PLATFORMS=cpu python -m pytest -q
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python -m pytest -q
JAX_PLATFORMS=cpu python scripts/validate.py --output outputs/validation-cpu.json
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  python scripts/validate.py --output outputs/validation-gpu.json
python examples/quickstart.py
python examples/optimize_forcing.py
```

**Final local test runs: 34 passed on CPU and 34 passed on CUDA (2026-09-10).**

Recorded environments and measurements: [CPU](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/validation-cpu.json), [GPU](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/validation-gpu.json), [validation explanation](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/validation.md). Tests enable x64 within the test process. The installed library never enables global x64 or memory preallocation on import.

The reports separate compilation from repeated execution and wait for device completion. That follows [JAX’s benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html). These small checks, on an occupied GPU, are **not** a fair performance ranking.

## Numerical scope

The current solver uses periodic Fourier differentiation, a pressure projection, strict two-thirds truncation, and explicit RK4. It solves in physical units chosen consistently by the caller; `nu` is kinematic viscosity and pressure is per unit density. The default box has side length `2π`.

Periodic spectral methods are an established route for this problem; [Mortensen and Langtangen (2016)](https://arxiv.org/abs/1602.03638) and [spectralDNS](https://github.com/spectralDNS/spectralDNS) provide important prior art. Our choices and independent tests are described in [Numerics](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/numerics.md).

Use `JAX_ENABLE_X64=1` and `dtype="float64"` for tighter numerical checks. FP64 is not arbitrary precision. See [JAX’s default-dtype documentation](https://docs.jax.dev/en/latest/default_dtypes.html). The convenience API checks timestep estimates on the host; use `SpectralSolver.integrate` for compiled fixed-step loops and differentiation. An adaptive timestep cannot rescue inadequate spatial resolution.

## Position among existing libraries

| Need today | Existing project worth examining | This project’s role |
|---|---|---|
| Differentiable simulation with multiple ML backends | [PhiFlow](https://github.com/tum-pbs/PhiFlow) | Construction-focused evaluation and an approachable periodic solver |
| Compressible and two-phase differentiable CFD | [JAX-Fluids](https://arxiv.org/abs/2203.13760) | Different equation/domain scope; those features are not implemented here |
| Distributed pseudospectral DNS | [spectralDNS](https://github.com/spectralDNS/spectralDNS) | Single-device JAX reference and diagnostics |
| Historical JAX finite-volume/spectral research | [JAX-CFD](https://github.com/google/jax-cfd) | Useful prior art; its README now states it is no longer maintained |
| Mathematical certificates for the new result | [NavierStokesAndEuler](https://github.com/openai/NavierStokesAndEuler) | Numerical component evaluation, not a substitute for proof checking |

Becoming a widely trusted library requires reliable examples, stable interfaces, independent validation, documented failure modes, and sustained maintenance. That is the development direction, not a release claim.

## NVIDIA Warp and Read the Docs

**Warp:** a promising optional kernel backend, with no performance claim yet. Keep the JAX reference and evaluate targeted kernels first; Warp’s JAX autodiff integration has documented restrictions. [Warp JAX interoperability](https://nvidia.github.io/warp/stable/user_guide/interoperability/jax.html).

**Read the Docs:** configuration and a Sphinx/MyST site are included. The local build is tested; hosted deployment is **Not Yet Tested / Not Connected**. [Build instructions and assessment](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/platforms.md), [Read the Docs Sphinx guide](https://docs.readthedocs.com/platform/stable/intro/sphinx.html).

## Blender

Blender remains a possible presentation frontend for the flagship, rather than the initial killer application. Blender documents OpenVDB volume sequences and Mantaflow cache/script workflows. This repository currently exports NPZ and VTK; it has **no Blender cache integration**. [Blender volume objects](https://docs.blender.org/manual/en/4.5/modeling/volumes/introduction.html), [fluid cache](https://docs.blender.org/manual/en/4.5/physics/fluid/type/domain/cache.html), [OpenVDB Python API](https://www.openvdb.org/documentation/doxygen/python.html).

## Builds and releases

GitHub Actions tests Python 3.12/3.13, builds the documentation, builds and checks wheel/source distributions, and exercises the installed wheel. Matching `v*` tags run the same validation before publishing to PyPI through Trusted Publishing, then attaching the same distributions, documentation, and checksums to a GitHub Release. No stored PyPI API token is needed. See [the release workflow](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/releases.md) and [PyPI's Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/). Read the Docs hosting is not connected.

## Sources and provenance

The shared conversation was extracted with [chatgpt-import-share](https://github.com/james-coder/chatgpt-import-share). The [Markdown transcript](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/conversation.md), [structured export](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/conversation.json), and [source manifest](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/source-manifest.json) preserve provenance and redacted/unavailable-message markers. The conversation is a design input, not independent evidence for its claims.

The manifest pins the inspected Lean commit and records the paper hash. Read the [research note](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/research.md) and [annotated sources](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/sources.md) for the implementation decisions and unresolved questions. [Contributing](https://github.com/james-coder/navier-stokes-blowup/blob/main/CONTRIBUTING.md) describes the evidence required for new features. Code uses the [MIT license](https://github.com/james-coder/navier-stokes-blowup/blob/main/LICENSE); the imported conversation and referenced third-party works retain their respective rights.
