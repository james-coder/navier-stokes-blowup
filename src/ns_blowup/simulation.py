"""Friendly host-driven simulation with checked, adaptive timesteps."""
from dataclasses import dataclass, field
import math
import operator
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from .analytical import TaylorGreen, ABCFlow
from .grid import PeriodicGrid
from .spectral import SpectralSolver


@dataclass
class SimulationResult:
    grid: PeriodicGrid
    times: jax.Array
    velocity: jax.Array
    nu: float
    metadata: dict = field(default_factory=dict)

    @property
    def final(self):
        return self.velocity[-1]

    def diagnostics(self):
        """Per-frame energy, maximum speed, vorticity and divergence (host arrays)."""
        solver = SpectralSolver(self.grid, self.nu)
        u = self.velocity
        omega = jax.vmap(solver.vorticity)(u)
        omega_size = jnp.abs(omega) if self.grid.ndim == 2 else jnp.linalg.norm(omega, axis=-1)
        spatial_axes = tuple(range(1, self.grid.ndim + 1))
        return {"time": np.asarray(self.times),
                "energy": np.asarray(0.5 * self.grid.cell_volume * jnp.sum(u**2, axis=spatial_axes + (-1,))),
                "speed_max": np.asarray(jnp.max(jnp.linalg.norm(u, axis=-1), axis=spatial_axes)),
                "vorticity_max": np.asarray(jnp.max(omega_size, axis=spatial_axes)),
                "divergence_max": np.asarray(jnp.max(jnp.abs(jax.vmap(solver.divergence)(u)), axis=spatial_axes))}

    def save(self, path):
        """Portable compressed NPZ including grid, times, viscosity and JSON metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, times=np.asarray(self.times), velocity=np.asarray(self.velocity),
                                lengths=self.grid.lengths, nu=self.nu, schema_version=1,
                                metadata=json.dumps(self.metadata))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported result schema")
            velocity, times = data["velocity"], data["times"]
            if velocity.dtype == np.float64 and not jax.config.x64_enabled:
                raise ValueError("Loading float64 results requires JAX_ENABLE_X64=1 to preserve precision")
            grid = PeriodicGrid(velocity.shape[1:-1], tuple(data["lengths"]))
            if velocity.shape[-1] != grid.ndim or times.shape != (velocity.shape[0],) or len(times) == 0:
                raise ValueError("invalid result array shapes")
            if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
                raise ValueError("result contains nonfinite data or unordered times")
            nu = float(data["nu"])
            if not math.isfinite(nu) or nu < 0:
                raise ValueError("saved viscosity must be finite and nonnegative")
            return cls(grid, jnp.asarray(times), jnp.asarray(velocity), nu, json.loads(str(data["metadata"])))

    def plot(self, *, frame=-1, quantity="speed", path=None, ax=None):
        """Plot an xy slice (middle z plane in 3D). Returns Matplotlib Axes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Plotting requires: pip install 'ns-blowup[plot]'") from exc
        u = self.velocity[frame]
        if quantity == "speed":
            values = np.asarray(jnp.linalg.norm(u, axis=-1))
        elif quantity == "vorticity":
            omega = SpectralSolver(self.grid, self.nu).vorticity(u)
            values = np.asarray(omega if self.grid.ndim == 2 else jnp.linalg.norm(omega, axis=-1))
        else:
            raise ValueError("quantity must be 'speed' or 'vorticity'")
        if self.grid.ndim == 3:
            values = values[:, :, self.grid.shape[2] // 2]
        if ax is None:
            _, ax = plt.subplots()
        dx, dy = self.grid.spacing[:2]
        # Samples are at i*dx, not cell centers; extend half a cell around them.
        artist = ax.imshow(values.T, origin="lower", extent=(-dx/2, self.grid.lengths[0]-dx/2,
                            -dy/2, self.grid.lengths[1]-dy/2), aspect="equal", cmap="viridis")
        ax.set(xlabel="x", ylabel="y", title=f"{quantity}, t={float(self.times[frame]):.4g}")
        ax.figure.colorbar(artist, ax=ax, label=quantity)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(path, dpi=160, bbox_inches="tight")
        return ax

    def export_vtk(self, path, *, frame=-1):
        """Legacy ASCII VTK structured points for ParaView; no Blender cache claim."""
        from .io import write_vtk
        return write_vtk(path, self.grid, self.velocity[frame])


class Simulation:
    """Run an incompressible periodic flow with adaptive dt and saved frames.

    This convenience API runs a Python control loop and checks each step on the
    host. For differentiable training or throughput use SpectralSolver.integrate.
    """

    def __init__(self, shape=(32, 32), *, nu=0.01, lengths=None, dtype="float32", device="auto"):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if dtype == "float64" and not jax.config.x64_enabled:
            raise ValueError("float64 requires JAX_ENABLE_X64=1 before Python starts")
        if device not in ("auto", "cpu", "gpu"):
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
        try:
            self.device = jax.devices()[0] if device == "auto" else jax.devices(device)[0]
        except RuntimeError as exc:
            raise RuntimeError(f"JAX {device} device unavailable; install the matching JAX runtime or use device='cpu'") from exc
        self.dtype = jnp.dtype(dtype)
        self.grid = PeriodicGrid(tuple(shape), lengths)
        with jax.default_device(self.device):
            self.solver = SpectralSolver(self.grid, nu)
            self.points = self.grid.points(self.dtype)

    def run(self, initial="taylor-green", *, t_end=1.0, t0=0.0, frames=21, dt=None,
            forcing=None, safety=0.4, max_steps=100000, progress=None):
        """Evolve a named benchmark, point callback initial(x), or velocity array.

        dt is an optional upper bound; the stability estimate is recomputed each
        step. progress(time, completed_steps) runs after each saved frame.
        Results contain exactly `frames` uniformly timed snapshots including t0.
        """
        if not math.isfinite(t0) or not math.isfinite(t_end) or t_end <= t0:
            raise ValueError("t_end must be finite and greater than finite t0")
        frames, max_steps = operator.index(frames), operator.index(max_steps)
        if frames < 2 or max_steps < 1:
            raise ValueError("frames must be >=2 and max_steps must be positive")
        if dt is not None and (not math.isfinite(dt) or dt <= 0):
            raise ValueError("dt must be finite and positive")
        with jax.default_device(self.device):
            if isinstance(initial, str):
                if initial == "taylor-green":
                    model = TaylorGreen(nu=self.solver.nu)
                elif initial == "abc" and self.grid.ndim == 3:
                    model = ABCFlow(nu=self.solver.nu)
                else:
                    raise ValueError("initial must be 'taylor-green', 'abc' (3D), a callback, or an array")
                if any(not math.isclose(length / (2 * math.pi), round(length / (2 * math.pi)), abs_tol=1e-12)
                       for length in self.grid.lengths):
                    raise ValueError("named benchmarks need lengths that are integer multiples of 2*pi; supply a periodic callback for other boxes")
                u = model.velocity(self.points, t0)
            else:
                u = initial(self.points) if callable(initial) else initial
            u = jnp.asarray(u, dtype=self.dtype)
            self.solver._check(u)
            if not bool(jnp.all(jnp.isfinite(u))):
                raise ValueError("initial velocity must be finite")
            projected = self.solver.project(u).astype(self.dtype)
            projection_change = float(jnp.linalg.norm(projected-u) / jnp.maximum(jnp.linalg.norm(u), 1e-30))
            u = projected
            step = jax.jit(lambda u, t, dt: self.solver.step(u, t, dt, forcing).astype(self.dtype))
            bound = jax.jit(lambda u: self.solver.suggest_dt(u, safety))
            times = np.linspace(t0, t_end, frames)
            snapshots, t, steps = [u], float(t0), 0
            for target in times[1:]:
                while t < target:
                    if steps >= max_steps:
                        raise RuntimeError("max_steps reached; shorten the run or raise max_steps")
                    step_dt = min(float(bound(u)), float(target - t), math.inf if dt is None else dt)
                    if not math.isfinite(step_dt) or step_dt <= 0 or t + step_dt == t:
                        raise FloatingPointError("timestep cannot advance; check resolution, forcing and precision")
                    u = step(u, jnp.asarray(t, self.dtype), jnp.asarray(step_dt, self.dtype))
                    if not bool(jnp.all(jnp.isfinite(u))):
                        raise FloatingPointError("flow became nonfinite; reduce dt/safety and inspect forcing")
                    t = min(float(target), t + step_dt)
                    steps += 1
                snapshots.append(u)
                if progress is not None:
                    progress(t, steps)
            return SimulationResult(self.grid, jnp.asarray(times), jnp.stack(snapshots), self.solver.nu,
                                    {"steps": steps, "device": str(self.device), "dtype": str(self.dtype),
                                     "initial_projection_relative_change": projection_change,
                                     "integrator": "RK4", "dealiasing": "strict 2/3 truncation"})
