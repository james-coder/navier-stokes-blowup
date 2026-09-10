"""Independent pointwise autodiff diagnostics and grid norm reductions."""
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .grid import PeriodicGrid


def _pointwise(fn, points, t):
    points = jnp.asarray(points)
    if not jnp.issubdtype(points.dtype, jnp.floating):
        points = points.astype(jnp.result_type(1.0))
    t = jnp.asarray(t, dtype=points.dtype)
    if t.ndim != 0:
        raise ValueError("time must be scalar; use jax.vmap for multiple times")
    if points.ndim < 1 or points.shape[-1] not in (2, 3):
        raise ValueError("points must have shape (..., 2) or (..., 3)")
    result = jax.vmap(fn, in_axes=(0, None))(points.reshape((-1, points.shape[-1])), t)
    return result.reshape(points.shape[:-1] + result.shape[1:])


@dataclass(frozen=True)
class FieldDiagnostics:
    """Functions accept a single point x and scalar t; methods also accept point batches.

    A velocity callback returns (d,), pressure returns (), forcing returns (d,).
    No forcing is assumed if the callback is omitted. Methods compose with jit.
    """
    velocity: Callable
    pressure: Callable
    nu: float
    forcing: Callable | None = None

    def divergence(self, points, t):
        return _pointwise(lambda x, t: jnp.trace(jax.jacfwd(self.velocity)(x, t)), points, t)

    def vorticity(self, points, t):
        def curl(x, t):
            g = jax.jacfwd(self.velocity)(x, t)
            if x.shape[-1] == 2:
                return g[1, 0] - g[0, 1]
            return jnp.stack([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]])
        return _pointwise(curl, points, t)

    def gradient(self, points, t):
        return _pointwise(jax.jacfwd(self.velocity), points, t)

    def residual(self, points, t):
        """u_t + (u·grad)u + grad(p) - nu*laplacian(u) - f."""
        def residual(x, t):
            u = self.velocity(x, t)
            grad = jax.jacfwd(self.velocity)(x, t)
            hessian = jax.jacfwd(jax.jacfwd(self.velocity))(x, t)
            laplacian = jnp.trace(hessian, axis1=-2, axis2=-1)
            force = 0 if self.forcing is None else self.forcing(x, t)
            return (jax.jacfwd(self.velocity, argnums=1)(x, t) + grad @ u
                    + jax.grad(self.pressure)(x, t) - self.nu * laplacian - force)
        return _pointwise(residual, points, t)


def kinetic_energy(velocity, grid: PeriodicGrid):
    """E = 1/2 integral |u|² dx, rectangular quadrature over the periodic box."""
    if velocity.shape != grid.shape + (grid.ndim,):
        raise ValueError("velocity shape does not match grid")
    return 0.5 * grid.cell_volume * jnp.sum(jnp.abs(velocity)**2)


def sampled_norms(velocity, vorticity, gradient):
    """Sample maxima, not certified continuum bounds; gradient uses Frobenius norm."""
    speed = jnp.linalg.norm(velocity, axis=-1)
    omega = jnp.abs(vorticity) if velocity.shape[-1] == 2 else jnp.linalg.norm(vorticity, axis=-1)
    return {"speed_max": jnp.max(speed), "vorticity_max": jnp.max(omega),
            "gradient_max": jnp.max(jnp.linalg.norm(gradient, axis=(-2, -1)))}
