# Annotated sources

Sources were inspected on 2026-09-10. Dates below are publication years where established; documentation and repository behavior may change. The [source manifest](source-manifest.json) records the downloaded paper hash and exact Lean/importer commits.

| Source | Document | What it supports |
|---|---|---|
| OpenAI | [Finite time blowup for Navier–Stokes](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf), 166 pages | Construction and mathematical component references |
| OpenAI | [NavierStokesAndEuler, pinned commit](https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538) | Formal source provenance and scope |
| OpenAI | [SimilarityCoordinates.lean](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/SimilarityCoordinates.lean) | Existence-defined coordinate choice, uniqueness and derivative identities |
| OpenAI | [formalization.yaml](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/formalization.yaml) | Upstream statements about formalized declarations; not independently checked here |
| Mortensen and Langtangen, 2016 | [High performance Python for direct numerical simulations of turbulent flows](https://arxiv.org/abs/1602.03638) | Pseudospectral CFD implementation context |
| spectralDNS developers | [spectralDNS](https://github.com/spectralDNS/spectralDNS) | Existing periodic/MPI numerical solver; comparison candidate |
| Bezgin, Buhendwa and Adams, 2022 | [JAX-FLUIDS](https://arxiv.org/abs/2203.13760) | Prior differentiable compressible two-phase CFD |
| PhiFlow developers | [PhiFlow](https://github.com/tum-pbs/PhiFlow) | Prior differentiable simulation with multiple backends and visualization |
| JAX-CFD developers | [JAX-CFD](https://github.com/google/jax-cfd) | Historical methods and current unmaintained status |
| JAX developers | [Installation](https://docs.jax.dev/en/latest/installation.html) | CPU/CUDA installation, platform and driver requirements |
| JAX developers | [Default dtypes and the X64 flag](https://docs.jax.dev/en/latest/default_dtypes.html) | Explicit precision configuration |
| JAX developers | [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html) | Compilation, synchronization and timing methodology |
| JAX developers | [GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) | Preallocation controls and memory behavior |
| JAX developers | [Configuration implementation](https://github.com/jax-ml/jax/blob/main/jax/_src/config.py) | Temporary x64 context behavior; version-specific use tested locally |
| Blender Foundation | [4.5 volume objects](https://docs.blender.org/manual/en/4.5/modeling/volumes/introduction.html) | OpenVDB imports and volume sequences |
| Blender Foundation | [4.5 fluid cache](https://docs.blender.org/manual/en/4.5/physics/fluid/type/domain/cache.html) | Cache formats and Mantaflow script export |
| OpenVDB developers | [Python interface](https://www.openvdb.org/documentation/doxygen/python.html) | NumPy exchange and grid transforms for a future exporter |
| VTK developers | [Legacy VTK file formats](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html) | Structured-points header and data ordering |
| NVIDIA | [Warp](https://nvidia.github.io/warp/stable/) and [JAX interoperability](https://nvidia.github.io/warp/stable/user_guide/interoperability/jax.html) | Optional-backend assessment and autodiff constraints |
| Read the Docs | [Sphinx deployment](https://docs.readthedocs.com/platform/stable/intro/sphinx.html) and [configuration](https://docs.readthedocs.com/platform/stable/config-file/v2.html) | Documentation build configuration |
| chatgpt-import-share | [Importer repository](https://github.com/james-coder/chatgpt-import-share) | Local tool used to preserve the design conversation |

The tutorial's example behavior, local test results, and measured precision curves are project-generated evidence, not claims borrowed from these sources. No independent acceptance of the theorem, real-world speedup, or complete solver-benchmark reproduction is inferred from the existence of an upstream repository.
