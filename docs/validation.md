# Validation record

Validation was performed locally on 2026-09-10 with Python 3.12.3, JAX 0.11.1, NumPy 2.5.3, Linux/WSL2, and an NVIDIA GeForce RTX 4090 with driver 610.74. The GPU already had substantial unrelated memory use; these results are not a performance comparison. The complete installed environment is recorded in [environment-tested.txt](environment-tested.txt).

## Automated tests

- **CPU: 34 passed**, no failures, errors, or skips.
- **CUDA: 34 passed**, no failures, errors, or skips.

[Test summary](test-summary.json) records the counts from the final JUnit reports. The test suite covers exact fields, wrong-equation controls, nonlinear forcing, time refinement, projection and mean preservation, nonlinear aliasing, energy production, gradients, data export/reload, user-input errors, mathematical component identities, and the observatory data experiment.

The observatory generator prefers CPU. In a CUDA-only process it falls back to the enabled device, which was exercised by the CUDA suite. A test of this report-generation configuration initially failed and was fixed before the final complete runs. No test was skipped to obtain the reported pass counts.

## Numerical reports

The [CPU report](validation-cpu.json) and [GPU report](validation-gpu.json) use 16 points per axis, 20 steps of size 0.01, and viscosity 0.1. The 2D benchmark is the [Taylor–Green vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex); the 3D benchmark is ABC. The numbers below are maximum absolute velocity errors against their analytical values at the final time.

| Backend | Dimension | FP32 error | FP64 error |
|---|---|---|---|
| CPU | 2D | 2.98e-7 | 5.11e-15 |
| CPU | 3D | 4.77e-7 | 9.44e-16 |
| RTX 4090 / CUDA 12 | 2D | 4.17e-7 | 5.11e-15 |
| RTX 4090 / CUDA 12 | 3D | 4.77e-7 | 8.88e-16 |

The reports also contain divergence and analytical residual measurements. FP32 divergence was below 1.4e-6; FP64 divergence was below 2.6e-15 in these checks. Passing these smooth low-mode problems does not qualify the solver for underresolved turbulence or singularity tracking.

The heat exterior was checked against an independent adaptive integral at `Z = 0, 0.001, 0.1, 1, 4`, using 96 quadrature nodes. Heat-equation and momentum checks used selected positive radii and times. Tests do not establish a uniform bound outside these ranges or for arbitrary derivative orders.

## Application and packaging

The 0.2 development suite passed on CPU (42 tests, followed by all four reference tests after adding the production/reference comparison). GitHub Actions then passed the complete updated suite on Python 3.12 and 3.13, browser checks, documentation, and the installed-wheel check in [run 34476201110](https://github.com/james-coder/navier-stokes-blowup/actions/runs/34476201110). Fifteen affected workflow, 3D benchmark and reference tests also passed locally with the CUDA backend on the RTX 4090, x64 enabled and preallocation disabled. The multiprecision calculations themselves run on CPU; the production-component comparisons exercised CUDA. These are functional numerical checks, not performance measurements.

The [automated Chromium checks](https://github.com/james-coder/navier-stokes-blowup/blob/main/tests/browser/observatory.spec.cjs) pass eight tests across the checked-in demo and a freshly generated app with a different `h`. They check time/resolution extremes against the actual dataset, both view modes, animation pause/end/restart, 390/768/1440-pixel layouts, canvas drawing, exact JSON downloads, and console errors. Screenshots are retained as CI artifacts; the mobile layout was also visually inspected. Other browsers and assistive technology are **Not Yet Tested**.

Embedded numerical data now lives in a hidden text element with HTML escaping. This avoids the JSON execution error caused by [HTMLPreview's script reconstruction](https://github.com/htmlpreview/htmlpreview.github.com/blob/master/htmlpreview.js). A local compatibility test exercises reconstruction of every script as JavaScript without depending on that service's availability.

The quickstart, forcing optimization, construction-scale export, figure reproduction, and 3D float64 CLI were executed. The forcing example reduced its objective from about 0.1005 to 5e-15 and recovered the target amplitude 1.5. This is one small inverse problem, not a general optimization guarantee.

A wheel was built and installed into a separate temporary target. Import from that installation, a simulation, and generation of the observatory using its packaged HTML resource were checked outside the source checkout. This is not a fresh-machine dependency-installation test.

The [0.2.0 release run](https://github.com/james-coder/navier-stokes-blowup/actions/runs/34477591808) passed all 43 tests separately on Python 3.12 and 3.13, plus all eight browser checks, documentation, packaging and publication. These hosted CPU checks are separate from the local CUDA record. The [separate upstream formal audit](formal-audit.md) passed the full Lean build and the Navier–Stokes Comparator checks with both configured kernels; this does not certify Python results. CUDA 13, non-NVIDIA accelerators, multi-GPU behavior, external viewer round-trips, and the complete blowup construction are **Not Yet Tested**. The README distinguishes untested implemented code from features that are not implemented at all.

## Reproduction

Use the [README validation commands](https://github.com/james-coder/navier-stokes-blowup#reproducible-validation) and inspect the [validation script](https://github.com/james-coder/navier-stokes-blowup/blob/main/scripts/validate.py). Compare results with tolerances appropriate to the dtype. Floating-point results and timings can vary with hardware, JAX/compiler versions, and device load. The checked-in reports capture a tested environment; they are not promises of bitwise reproducibility or performance.
