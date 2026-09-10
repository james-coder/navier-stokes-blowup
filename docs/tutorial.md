# Getting started

## 1. Explore before installing

Open [the Singularity Observatory](observatory.html) as a downloaded local HTML file. Move the time slider, switch between fixed physical scale and following the core, and change the grid resolution. The labels distinguish coordinate geometry from measured numerical residuals. Download the JSON if you want to analyze the plotted data separately.

## 2. Run a simple flow

Install from the repository using the README instructions. Then:

```python
from ns_blowup import Simulation

simulation = Simulation((32, 32), nu=0.05, device="cpu")
result = simulation.run("taylor-green", t_end=0.5, frames=11)
print(result.times)
print(result.final.shape)  # (32, 32, 2)
result.plot(path="outputs/first-flow.png")
```

The default box is `[0,2π)` in each direction, with periodic boundaries. The initial velocity is a built-in decaying vortex. The viscosity controls diffusion. Frame count controls saved output, not the internal timestep. A saved image shows the final speed by default.

Change the shape to `(24,24,24)` and the initial condition to `"abc"` for a three-dimensional exact flow. Use small shapes while learning; storage and intermediate arrays grow with the product of all grid dimensions.

## 3. Define your own initial condition

```python
import jax.numpy as jnp
from ns_blowup import Simulation

sim = Simulation((32, 32), lengths=(4.0, 3.0), nu=0.02)

def initial(points):
    y = points[..., 1]
    return jnp.stack([jnp.sin(2*jnp.pi*y/3.0), jnp.zeros_like(y)], axis=-1)

result = sim.run(initial, t_end=0.2, frames=5)
```

The callback gets all grid points. Its return shape must be the spatial grid shape followed by the velocity component axis. Initial data are projected and high modes removed; inspect `result.metadata["initial_projection_relative_change"]` to see the relative change.

## 4. Apply a force

```python
import jax.numpy as jnp

def force(points, time):
    y = points[..., 1]
    return jnp.stack([0.1*jnp.cos(time)*jnp.sin(2*jnp.pi*y/3.0),
                      jnp.zeros_like(y)], axis=-1)

result = sim.run(initial, t_end=0.2, forcing=force, frames=5, dt=0.005)
```

Here `dt` is a maximum; the adaptive driver may use smaller steps. Strong forcing can demand additional care even when the initial velocity is small. For a differentiable forcing parameter, follow [the optimization example](https://github.com/james-coder/navier-stokes-blowup/blob/main/examples/optimize_forcing.py), which uses the compiled fixed-step interface.

## 5. Save, reload, and inspect

```python
from ns_blowup import SimulationResult

result.save("outputs/flow.npz")
restored = SimulationResult.load("outputs/flow.npz")
metrics = restored.diagnostics()
print(metrics["energy"])
print(metrics["divergence_max"])
restored.export_vtk("outputs/flow-final.vtk")
```

NPZ carries grid lengths, time samples, viscosity, velocity, schema version, and metadata. Reloading a float64 result requires x64 enabled so it cannot silently lose precision. The export contains velocity samples, not a liquid mesh or a Blender bake cache.

## 6. Choose numerical precision

Start a fresh shell command with x64 enabled:

```bash
JAX_ENABLE_X64=1 python examples/quickstart.py
```

In your own script also request `Simulation(..., dtype="float64")`. For the CLI use `--precision float64`; the CLI enables x64 before constructing arrays. Library import does not change global numerical settings. Exact benchmark evaluation uses the precision of your JAX arrays and configuration.

## 7. Evaluate construction components

```python
import jax.numpy as jnp
from ns_blowup import SimilarityCoordinates, HeatExterior

mapping = SimilarityCoordinates(h=0.005)
coordinates = mapping(jnp.array([0.01, 0.02, 0.03]), tau=1e-4)
print(coordinates.q, coordinates.eta, coordinates.X)

exterior = HeatExterior(h=0.005, order=96)
print(exterior.azimuthal_velocity(jnp.array([1.0, 2.0]), tau=0.1))
```

`tau` is time remaining, supplied directly. Avoid computing extremely small `tau` by subtracting nearly equal float32 times. The exterior must not be evaluated at radius zero. Increasing quadrature order and comparing residuals is useful; interpreting it as the complete theorem is not supported.

## Troubleshooting

| Symptom | Action |
|---|---|
| GPU unavailable | Confirm `jax.devices()` and install the matching JAX CUDA extra; CPU execution remains available |
| GPU already in use / allocation error | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, reduce grid size and saved frames |
| Float64 refused | Set `JAX_ENABLE_X64=1` before starting Python |
| Shape mismatch | Put vector components on the final axis; forcing must return the full grid shape |
| Named flow rejected for box length | Supply a callback with a period matching that box |
| Simulation reaches `max_steps` | Inspect the flow, reduce duration, or raise the explicit step budget |
| NaN from a construction component | Check its mathematical domain and finite inputs |
| Large discretization error | Refine space/time and compare with an independent reference; changing precision alone may not help |
