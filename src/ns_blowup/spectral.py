"""Differentiable Fourier–Galerkin incompressible solver on periodic 2D/3D boxes."""
import math
import operator

import jax
import jax.numpy as jnp

from .grid import PeriodicGrid


class SpectralSolver:
    """Strict 2/3 truncation, Leray projection and explicit RK4.

    u has shape (*grid.shape, grid.ndim). All physical lengths are supported.
    Force callbacks have signature force(points, time) -> u-shaped array.
    Their closed-over parameters may be JAX tracers for inverse design.
    No global JAX configuration or device selection is changed by the library.
    """

    def __init__(self, grid: PeriodicGrid, nu=0.01):
        if not math.isfinite(nu) or nu < 0:
            raise ValueError("nu must be finite and nonnegative")
        self.grid, self.nu = grid, nu
        self.axes = tuple(range(grid.ndim))
        modes = [jnp.fft.fftfreq(n) * n for n in grid.shape]
        wave = [m * (2 * math.pi / length) for m, length in zip(modes, grid.lengths)]
        self.k = jnp.stack(jnp.meshgrid(*wave, indexing="ij"), axis=-1)
        self.k2 = jnp.sum(self.k**2, axis=-1)
        self.safe_k2 = jnp.where(self.k2 == 0, 1, self.k2)
        mode_mesh = jnp.meshgrid(*modes, indexing="ij")
        # Strict inequality excludes the ambiguous aliased boundary when n is divisible by 3.
        self.mask = jnp.ones(grid.shape, dtype=bool)
        for m, n in zip(mode_mesh, grid.shape):
            self.mask = self.mask & (jnp.abs(m) < n / 3)

    def _check(self, u):
        if u.shape != self.grid.shape + (self.grid.ndim,):
            raise ValueError("velocity shape does not match grid")
        if not jnp.issubdtype(u.dtype, jnp.floating):
            raise ValueError("velocity must be real floating point")

    def _fft(self, a):
        return jnp.fft.fftn(a, axes=self.axes)

    def _ifft(self, a):
        return jnp.fft.ifftn(a, axes=self.axes).real

    def _project_hat(self, uhat):
        k = self.k.astype(uhat.real.dtype)
        dot = jnp.sum(k * uhat, axis=-1)
        return (uhat - k * (dot / self.safe_k2.astype(uhat.real.dtype))[..., None]) * self.mask[..., None]

    def project(self, u):
        """Project and truncate initial data; preserve its spatial mean."""
        self._check(u)
        return self._ifft(self._project_hat(self._fft(u)))

    def divergence(self, u):
        self._check(u)
        return self._ifft(1j * jnp.sum(self.k * self._fft(u), axis=-1))

    def vorticity(self, u):
        self._check(u)
        uhat = self._fft(u)
        if self.grid.ndim == 2:
            return self._ifft(1j * (self.k[..., 0] * uhat[..., 1] - self.k[..., 1] * uhat[..., 0]))
        return self._ifft(1j * jnp.cross(self.k, uhat))

    def _advection_hat(self, uhat):
        u = self._ifft(uhat)
        advection = jnp.zeros_like(u)
        for axis in self.axes:
            derivative = self._ifft(1j * self.k[..., axis, None].astype(uhat.real.dtype) * uhat)
            advection = advection + u[..., axis, None] * derivative
        return self._fft(advection) * self.mask[..., None]

    def _rhs_hat(self, uhat, t, forcing):
        uhat = self._project_hat(uhat)
        acceleration = -self._advection_hat(uhat)
        if forcing is not None:
            force = forcing(self.grid.points(), t)
            self._check(force)
            acceleration = acceleration + self._fft(force.astype(uhat.real.dtype))
        return self._project_hat(acceleration) - self.nu * self.k2[..., None].astype(uhat.real.dtype) * uhat

    def pressure(self, u, t=0.0, forcing=None):
        """Recover zero-mean pressure from div(f - (u·grad)u)."""
        self._check(u)
        acceleration = -self._advection_hat(self._project_hat(self._fft(u)))
        if forcing is not None:
            force = forcing(self.grid.points(), t)
            self._check(force)
            acceleration = acceleration + self._fft(force)
        phat = -1j * jnp.sum(self.k * acceleration, axis=-1) / self.safe_k2
        return self._ifft(jnp.where(self.k2 == 0, 0, phat) * self.mask)

    def rhs(self, u, t=0.0, forcing=None):
        self._check(u)
        return self._ifft(self._rhs_hat(self._fft(u), t, forcing))

    def _step_hat(self, uhat, t, dt, forcing):
        k1 = self._rhs_hat(uhat, t, forcing)
        k2 = self._rhs_hat(uhat + dt * k1 / 2, t + dt / 2, forcing)
        k3 = self._rhs_hat(uhat + dt * k2 / 2, t + dt / 2, forcing)
        k4 = self._rhs_hat(uhat + dt * k3, t + dt, forcing)
        return self._project_hat(uhat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    def step(self, u, t, dt, forcing=None):
        """One RK4 step. dt must be positive and stable (see suggest_dt)."""
        self._check(u)
        return self._ifft(self._step_hat(self._project_hat(self._fft(u)), t, dt, forcing))

    def integrate(self, u, dt, steps, *, t0=0.0, forcing=None, save_every=None):
        """Return final u, or (times, frames) when save_every is supplied.

        steps and save_every are static integers; dt and force parameters may be
        differentiated. Saved frames include projected initial data and the final
        state, even if steps is not divisible by save_every. No history is retained
        by the forward loop when save_every=None (autodiff can need extra memory).
        """
        self._check(u)
        steps = operator.index(steps)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if not isinstance(dt, jax.core.Tracer) and (not math.isfinite(float(dt)) or float(dt) <= 0):
            raise ValueError("dt must be finite and positive")
        uhat = self._project_hat(self._fft(u))

        def advance(uhat, start, count):
            def body(i, current):
                return self._step_hat(current, t0 + (start + i) * dt, dt, forcing)
            return jax.lax.fori_loop(0, count, body, uhat)

        if save_every is None:
            return self._ifft(advance(uhat, 0, steps))
        save_every = operator.index(save_every)
        if save_every <= 0:
            raise ValueError("save_every must be positive")
        blocks, remainder = divmod(steps, save_every)

        def block(current, i):
            current = advance(current, i * save_every, save_every)
            return current, self._ifft(current)

        final, frames = jax.lax.scan(block, uhat, jnp.arange(blocks))
        frames = jnp.concatenate([self._ifft(uhat)[None], frames], axis=0)
        times = t0 + jnp.arange(blocks + 1) * save_every * dt
        if remainder:
            last = self._ifft(advance(final, blocks * save_every, remainder))
            frames = jnp.concatenate([frames, last[None]], axis=0)
            times = jnp.concatenate([times, jnp.asarray([t0 + steps * dt])])
        return times, frames

    def suggest_dt(self, u, safety=0.4):
        """Conservative instantaneous advection/diffusion estimate; not a guarantee.

        Recheck as velocities change, particularly with strong forcing. The bound
        includes both imaginary advection and negative-real diffusion rates.
        """
        self._check(u)
        if not math.isfinite(safety) or not 0 < safety <= 1:
            raise ValueError("safety must lie in (0, 1]")
        kmax = jnp.max(jnp.where(self.mask[..., None], jnp.abs(self.k), 0), axis=self.axes)
        advective = jnp.sum(jnp.max(jnp.abs(u), axis=self.axes) * kmax)
        viscous = self.nu * jnp.max(jnp.where(self.mask, self.k2, 0))
        rate = advective + viscous
        return safety / jnp.maximum(rate, jnp.finfo(u.dtype).tiny)
