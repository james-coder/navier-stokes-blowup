# Three-dimensional Taylor–Green benchmark

The [Taylor–Green vortex introduction](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex) explains the physical setup and distinguishes the exact 2D solution from the original 3D flow. `TaylorGreen3D` provides the nonlinear 3D initial condition:

$$u_0=a(\sin(kx)\cos(ky)\cos(kz),-\cos(kx)\sin(ky)\cos(kz),0).$$

```python
from ns_blowup import Simulation, TaylorGreen3D

sim = Simulation((24, 24, 24), nu=0.05, device="cpu")
result = sim.run("taylor-green-3d", t_end=1, frames=21)
result.plot(quantity="vorticity", path="outputs/taylor-green-3d.png")
result.save("outputs/taylor-green-3d.npz")

# Customize amplitude or wavenumber with a periodic initial callback:
initial = TaylorGreen3D(amplitude=0.5).initial_velocity
custom = sim.run(initial, t_end=1, frames=21)
```

The named initial condition is defined at `t0=0`. To restart later, supply the saved velocity array. The separate `TaylorGreen` class still represents the exact 2D decaying solution, optionally independent of z in a 3D box. `TaylorGreen3D` has only an `initial_velocity` evaluator; it does not pretend to know the exact later solution. The third velocity component becomes nonzero during nonlinear evolution.

The [reproducible comparison](taylor-green-3d.json) uses `[0,2π)^3`, `a=k=1`, viscosity `0.05`, characteristic Reynolds number `a/(nu*k)=20`, and times `0,0.25,0.5,0.75,1`. All runs use float64 on CPU. Mean energy is `mean(|u|²)/2` and mean enstrophy is `mean(|curl(u)|²)/2`; their initial values are `1/8` and `3/8`. These are volume averages; `SimulationResult.diagnostics()['energy']` returns the volume integral.

The independent solver implements NumPy Fourier derivatives and rotational-form advection `P(u × curl(u))`, with [SciPy's eighth-order DOP853 integrator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). It does not use the library's initial-condition evaluator, grid, right-hand side, integrator, or diagnostics. Both methods use strict two-thirds spectral truncation. Independent 32³ and 48³ reference runs use `rtol=1e-11`, `atol=1e-13`, and maximum step `0.02`; their maximum differences over the saved times are `2.55e-13` in mean energy and `3.13e-11` in mean enstrophy.

| JAX resolution | Final velocity RMS difference from resampled 48³ reference | Maximum mean-energy difference | Maximum mean-enstrophy difference |
|---|---:|---:|---:|
| 12³ | 3.78e-3 | 1.52e-6 | 7.74e-5 |
| 18³ | 3.48e-4 | 1.36e-8 | 1.01e-6 |
| 24³ | 3.90e-5 | 1.56e-10 | 1.48e-8 |

JAX runs use a maximum step of `0.005`, `rtol=1e-9`, and `atol=1e-11`. Energy alone is an insensitive accuracy check here: compare the field error and enstrophy as well. The reference's mean energy decreases from `0.125` to `0.09187228420651929`; mean enstrophy decreases from `0.375` to `0.29548013130640044`. The complete time series, versions, solver metadata, and reference refinement results are in the linked JSON.

Reproduce using [the independent comparison script](https://github.com/james-coder/navier-stokes-blowup/blob/main/scripts/validate_taylor_green_3d.py):

```bash
python -m pip install scipy
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  python scripts/validate_taylor_green_3d.py --output outputs/taylor-green-3d.json
```

This validates a resolved short-time, moderate-Reynolds-number case. **Not Yet Tested:** high-Re turbulence, long-time accuracy, or agreement with published Re=1600 DNS. [Xcompact3d's Taylor–Green documentation and datasets](https://github.com/xcompact3d/Incompact3d/blob/master/docs/pages/cases/TGV.rst) describe those substantially more demanding cases. The reference refinement differences here are empirical estimates, not rigorous continuum error bounds.
