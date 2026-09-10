"""Executable components of the cited construction, not the full blowup solution.

References: OpenAI, Finite time blowup for Navier–Stokes, (3.2), (4.1),
Lemma 4.1, (10.22)–(10.23). See docs/research.md for the implementation boundary.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Coordinates(NamedTuple):
    q: jax.Array
    eta: jax.Array
    X: jax.Array


@dataclass(frozen=True)
class SimilarityCoordinates:
    """Solve q - z² q^(2h) = tau, eta = z/q^(1/2-h), X = r²/(2q).

    tau is time remaining, supplied directly to avoid loss in subtracting T-t.
    Invalid tau produces NaN, including under jit. h is static. The 32-step
    fixed-point iteration has contraction factor <=2h on the positive branch;
    this numerical approximation has no interval-certified error bounds.
    """
    h: float = 0.005

    def __post_init__(self):
        if not math.isfinite(self.h) or not 0 < self.h < 0.01:
            raise ValueError("h must lie strictly between 0 and 0.01")

    def __call__(self, xyz, tau):
        xyz, tau = jnp.asarray(xyz), jnp.asarray(tau)
        if xyz.shape[-1] != 3:
            raise ValueError("similarity coordinates require 3D points")
        tau = jnp.where((tau > 0) & jnp.isfinite(tau), tau, jnp.nan)
        z = xyz[..., 2]
        q0 = tau + jnp.abs(z)**(2 / (1 - 2 * self.h))
        q = jax.lax.fori_loop(0, 32, lambda i, q: tau + z**2 * q**(2 * self.h), q0)
        return Coordinates(q, z / q**(0.5 - self.h), jnp.sum(xyz[..., :2]**2, axis=-1) / (2 * q))

    def scales(self, tau):
        """Unit-normalized power laws; neither exact norms nor a velocity field."""
        tau = jnp.asarray(tau)
        tau = jnp.where((tau > 0) & jnp.isfinite(tau), tau, jnp.nan)
        return {"radial_length": tau**0.5, "axial_length": tau**(0.5 - self.h),
                "tangential_speed": tau**(-0.5 - self.h),
                "core_energy_scale": tau**(0.5 - 3 * self.h)}


@dataclass(frozen=True)
class ViscosityScaled:
    """Map a supplied viscosity-one solution to viscosity nu by spatial scaling.

    The caller must supply actual velocity/pressure/forcing functions solving the
    viscosity-one equation. This transformation cannot construct missing profiles.
    """
    base: object
    nu: float

    def __post_init__(self):
        if not math.isfinite(self.nu) or self.nu <= 0:
            raise ValueError("nu must be finite and positive")
        if getattr(self.base, "nu", 1.0) != 1.0:
            raise ValueError("base solution must have viscosity one")

    def velocity(self, xyz, t):
        scale = math.sqrt(self.nu)
        return scale * self.base.velocity(jnp.asarray(xyz) / scale, t)

    def pressure(self, xyz, t):
        return self.nu * self.base.pressure(jnp.asarray(xyz) / math.sqrt(self.nu), t)

    def forcing(self, xyz, t):
        scale = math.sqrt(self.nu)
        return scale * self.base.forcing(jnp.asarray(xyz) / scale, t)

    @property
    def energy_factor(self):
        """Factor for the whole-space 3D energy integral (scaled domain)."""
        return self.nu**2.5
