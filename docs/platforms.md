# NVIDIA Warp and documentation hosting

## Warp: a complementary backend candidate

NVIDIA Warp compiles Python kernels for CPU/GPU execution and provides simulation and geometry primitives. Its examples include fluids, finite elements, particles, and optimization. That makes it a credible candidate for specialized numerical kernels or a later geometry/particle layer. [Warp 1.17 documentation](https://nvidia.github.io/warp/stable/).

For this project, the recommended initial architecture is **JAX reference implementation plus optional Warp kernels where profiling demonstrates a benefit**. A rewrite of the whole FFT-based solver has no demonstrated justification. Warp does not remove the need to implement the mathematical construction, control truncation errors, or preserve an independent numerical reference.

Warp supports JAX array exchange and kernels callable from compiled JAX code. Its documentation marks the autodiff integration experimental, requires scalar kernel arguments to be static, and lists restrictions on in-place arguments and output dimensions when differentiation is enabled. Those constraints matter for trainable parameters. Our nested spatial/time derivative diagnostics require explicit compatibility tests; generic first-derivative support is not sufficient evidence. [Warp JAX interoperability](https://nvidia.github.io/warp/stable/user_guide/interoperability/jax.html).

| Candidate workload | Assessment for this project |
|---|---|
| Fused evaluation of known profile formulas at many points | Promising experiment; compare values and all required derivatives with JAX |
| Particle advection, geometry queries, sparse spatial structures | Plausible later role if these become part of the flagship |
| Replacement of the current entire spectral solver | No measured benefit yet; preserve the existing reference |
| Arbitrary-precision or formal proof verification | Does not supply these project capabilities |
| Immediate speedup claim | **Not Yet Tested**; Warp is not installed or integrated in this release |

A first Warp experiment should choose one expensive, mathematically settled kernel; preserve FP32/FP64 behavior; measure compilation, transfers, forward time and derivative time separately; and pass independent accuracy tests. It should remain optional until the accuracy and deployment costs are understood. These are project recommendations, not published Warp benchmark results.

## Read the Docs: recommended for the documentation site

Read the Docs supports Sphinx with MyST Markdown and repository-based build configuration. This matches the existing narrative/API docs and makes it possible to publish documentation separately from running simulations. [Sphinx deployment guide](https://docs.readthedocs.com/platform/stable/intro/sphinx.html), [configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html).

The repository contains the [Read the Docs configuration](https://github.com/james-coder/navier-stokes-blowup/blob/main/.readthedocs.yaml), [Sphinx configuration](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/conf.py), [navigation index](index.md), and [pinned documentation requirements](https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/requirements.txt). The build uses only documentation dependencies; it does not launch CUDA or recompute experiments. The [standalone Observatory](https://htmlpreview.github.io/?https://github.com/james-coder/navier-stokes-blowup/blob/main/docs/observatory.html) and [numerical data](observatory.json) are copied into the generated site.

[Download the prebuilt documentation ZIP](https://github.com/james-coder/navier-stokes-blowup/releases/latest/download/documentation.zip), or build locally:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs outputs/site
```

After building, open `outputs/site/index.html` on your machine; after extracting the ZIP, open `index.html` in the extracted folder. The local build is checked. **Hosted Read the Docs deployment is Not Yet Tested / Not Connected.** [Issue #103](https://github.com/james-coder/navier-stokes-blowup/issues/103) tracks connecting the GitHub repository and verifying the hosted deployment.
