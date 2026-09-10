"""Periodic, endpoint-excluding Cartesian grids; vectors use a final component axis."""
from dataclasses import dataclass
import math
import operator

import jax.numpy as jnp


@dataclass(frozen=True)
class PeriodicGrid:
    shape: tuple[int, ...]
    lengths: tuple[float, ...] | None = None

    def __post_init__(self):
        try:
            shape = tuple(operator.index(n) for n in self.shape)
        except TypeError as exc:
            raise ValueError("shape must contain integers") from exc
        if len(shape) not in (2, 3) or any(n < 6 for n in shape):
            raise ValueError("use a 2D or 3D grid with at least 6 points per axis")
        lengths = (2 * math.pi,) * len(shape) if self.lengths is None else tuple(self.lengths)
        if len(lengths) != len(shape) or any(not math.isfinite(x) or x <= 0 for x in lengths):
            raise ValueError("one finite positive length is required per axis")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lengths", lengths)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def spacing(self):
        return tuple(length / n for length, n in zip(self.lengths, self.shape))

    @property
    def cell_volume(self):
        return math.prod(self.spacing)

    @property
    def volume(self):
        return math.prod(self.lengths)

    def points(self, dtype=None):
        axes = [jnp.arange(n, dtype=dtype) * dx for n, dx in zip(self.shape, self.spacing)]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
