"""Numerical quadrature of the paper's heat exterior (Appendix A, Lemma A.6).

This is a component valid away from the axis. It is singular on the axis for
all times and is NOT the smooth initial-data blowup solution of Theorem 1.1.
"""
import math
import operator

import jax.numpy as jnp
import numpy as np


class HeatExterior:
    """Differentiable azimuthal heat exterior with independent pressure quadrature.

    Generalized Gauss–Laguerre quadrature evaluates H; Gauss–Legendre quadrature
    evaluates pressure. order controls both. Increase order and compare residuals
    in your evaluation range; accuracy is not uniform over arbitrary Z or radius.
    c is the user-supplied c_infinity normalization, not a matched inner profile.
    """
    def __init__(self, h=0.005, *, c=1.0, nu=1.0, order=64, blowup_time=1.0):
        if not math.isfinite(h) or not 0 < h < .01:
            raise ValueError("h must lie strictly between 0 and 0.01")
        if not math.isfinite(c) or c <= 0 or not math.isfinite(nu) or nu <= 0:
            raise ValueError("c and nu must be finite and positive")
        order = operator.index(order)
        if order < 8 or order > 256:
            raise ValueError("quadrature order must be between 8 and 256")
        if not math.isfinite(blowup_time):
            raise ValueError("blowup_time must be finite")
        self.h, self.c, self.nu, self.order, self.blowup_time = h, c, nu, order, blowup_time
        # Golub–Welsch Jacobi matrix for normalized weight exp(-v)*v**h / Gamma(1+h).
        i = np.arange(order)
        off = np.sqrt(np.arange(1,order) * (np.arange(1,order)+h))
        matrix = np.diag(2*i+1+h) + np.diag(off,1) + np.diag(off,-1)
        nodes, vectors = np.linalg.eigh(matrix)
        self.nodes = jnp.asarray(nodes)
        self.weights = jnp.asarray(vectors[0]**2)
        legendre, weights = np.polynomial.legendre.leggauss(order)
        self.pressure_nodes = jnp.asarray((legendre+1)/2)
        self.pressure_weights = jnp.asarray(weights/2)

    def factor(self, Z):
        """H(Z) for finite Z>=0; invalid arguments return NaN under jit too."""
        Z = jnp.asarray(Z)
        Z = jnp.where((Z>=0) & jnp.isfinite(Z), Z, jnp.nan)
        return jnp.sum(self.weights * jnp.exp(-self.h*jnp.log1p(Z[...,None]*self.nodes)), axis=-1)

    def azimuthal_velocity(self, radius, tau):
        """K_nu(r,tau), using time remaining directly; r>0 and tau>=0."""
        r, tau = jnp.asarray(radius), jnp.asarray(tau)
        r = jnp.where((r>0) & jnp.isfinite(r),r,jnp.nan)
        tau = jnp.where((tau>=0) & jnp.isfinite(tau),tau,jnp.nan)
        s = r**2 / (2*self.nu)
        return math.sqrt(self.nu)*self.c*s**(-.5-self.h)*self.factor(2*tau/s)

    def velocity(self, xyz, t):
        xyz = jnp.asarray(xyz)
        if xyz.shape[-1] != 3:
            raise ValueError("heat exterior requires 3D points")
        r = jnp.linalg.norm(xyz[...,:2],axis=-1)
        k = self.azimuthal_velocity(r,self.blowup_time-t)
        return jnp.stack([-k*xyz[...,1]/r,k*xyz[...,0]/r,jnp.zeros_like(k)],axis=-1)

    def pressure(self, xyz, t):
        xyz = jnp.asarray(xyz)
        r = jnp.linalg.norm(xyz[...,:2],axis=-1)
        # p(r) = -integral_r^infinity K(s)^2/s ds, transformed by s=r/v.
        k = self.azimuthal_velocity(r[...,None]/self.pressure_nodes,self.blowup_time-t)
        return -jnp.sum(self.pressure_weights*k**2/self.pressure_nodes,axis=-1)

    def forcing(self, xyz, t):
        return jnp.zeros_like(jnp.asarray(xyz))
