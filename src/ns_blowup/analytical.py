"""Exact smooth benchmarks, evaluated at points with shape (..., dimension)."""
from dataclasses import dataclass
import math

import jax.numpy as jnp


def _viscosity(nu):
    if not math.isfinite(nu) or nu < 0:
        raise ValueError("viscosity must be finite and nonnegative")


@dataclass(frozen=True)
class TaylorGreen:
    """Exact 2D decaying vortex, optionally embedded unchanged along the third axis.

    This is NOT the three-dimensional Taylor–Green initial condition with cos(z).
    Periods are 2*pi/wavenumber. Pressure has zero spatial mean.
    """
    nu: float = 0.01
    amplitude: float = 1.0
    wavenumber: float = 1.0

    def __post_init__(self):
        _viscosity(self.nu)
        if not math.isfinite(self.wavenumber) or self.wavenumber <= 0:
            raise ValueError("wavenumber must be finite and positive")

    def velocity(self, xyz, t):
        xyz = jnp.asarray(xyz)
        if xyz.shape[-1] not in (2, 3):
            raise ValueError("points need 2 or 3 coordinates")
        x, y = self.wavenumber * xyz[..., 0], self.wavenumber * xyz[..., 1]
        a = self.amplitude * jnp.exp(-2 * self.nu * self.wavenumber**2 * t)
        components = [a * jnp.sin(x) * jnp.cos(y), -a * jnp.cos(x) * jnp.sin(y)]
        if xyz.shape[-1] == 3:
            components.append(jnp.zeros_like(components[0]))
        return jnp.stack(components, axis=-1)

    def pressure(self, xyz, t):
        xyz = jnp.asarray(xyz)
        a = self.amplitude * jnp.exp(-2 * self.nu * self.wavenumber**2 * t)
        return a**2 / 4 * (jnp.cos(2 * self.wavenumber * xyz[..., 0])
                            + jnp.cos(2 * self.wavenumber * xyz[..., 1]))

    def forcing(self, xyz, t):
        return jnp.zeros_like(self.velocity(xyz, t))


@dataclass(frozen=True)
class ABCFlow:
    """3D decaying Beltrami flow: curl(u)=k*u, p=-|u|²/2 (up to a constant)."""
    nu: float = 0.01
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    wavenumber: float = 1.0

    def __post_init__(self):
        _viscosity(self.nu)
        if not math.isfinite(self.wavenumber) or self.wavenumber <= 0:
            raise ValueError("wavenumber must be finite and positive")

    def velocity(self, xyz, t):
        xyz = jnp.asarray(xyz)
        if xyz.shape[-1] != 3:
            raise ValueError("ABC flow needs 3 coordinates")
        x, y, z = (xyz[..., i] * self.wavenumber for i in range(3))
        u = jnp.stack([self.a * jnp.sin(z) + self.c * jnp.cos(y),
                       self.b * jnp.sin(x) + self.a * jnp.cos(z),
                       self.c * jnp.sin(y) + self.b * jnp.cos(x)], axis=-1)
        return jnp.exp(-self.nu * self.wavenumber**2 * t) * u

    def pressure(self, xyz, t):
        return -jnp.sum(self.velocity(xyz, t)**2, axis=-1) / 2

    def forcing(self, xyz, t):
        return jnp.zeros_like(self.velocity(xyz, t))
