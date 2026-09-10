"""Initial conditions whose later nonlinear evolution must be computed."""
from dataclasses import dataclass
import math

import jax.numpy as jnp


@dataclass(frozen=True)
class TaylorGreen3D:
    """True 3D Taylor–Green initial condition, distinct from exact TaylorGreen.

    u0 = amplitude*(sin(kx)cos(ky)cos(kz), -cos(kx)sin(ky)cos(kz), 0).
    Period 2*pi/k in all directions, characteristic Re=amplitude/(nu*k).
    No exact time-dependent velocity or pressure is claimed by this class.
    See docs/taylor-green-3d.md for independent numerical validation.
    """
    amplitude: float = 1.0
    wavenumber: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.amplitude):
            raise ValueError('amplitude must be finite')
        if not math.isfinite(self.wavenumber) or self.wavenumber <= 0:
            raise ValueError('wavenumber must be finite and positive')

    def initial_velocity(self, xyz):
        xyz = jnp.asarray(xyz)
        if xyz.shape[-1] != 3:
            raise ValueError('TaylorGreen3D needs three coordinates')
        x, y, z = (self.wavenumber*xyz[..., axis] for axis in range(3))
        return self.amplitude*jnp.stack((jnp.sin(x)*jnp.cos(y)*jnp.cos(z),
                                        -jnp.cos(x)*jnp.sin(y)*jnp.cos(z),
                                        jnp.zeros_like(z)), axis=-1)
