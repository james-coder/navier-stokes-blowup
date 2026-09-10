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

The repository contains `.readthedocs.yaml`, `docs/conf.py`, a navigation index, and pinned documentation requirements. The build uses only documentation dependencies; it does not launch CUDA or recompute experiments. The standalone observatory and numerical data are copied into the generated site.

Build locally:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs outputs/site
```

Open `outputs/site/index.html`. The local build is checked. **Hosted Read the Docs deployment is Not Yet Tested / Not Connected.** No hosted project, webhook, or service account was created. Importing the GitHub repository into Read the Docs is the remaining hosting step; until then, the repo does not claim a live Read the Docs URL.
