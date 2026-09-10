import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ns_blowup import SimilarityCoordinates, ViscosityScaled, TaylorGreen, FieldDiagnostics


def test_similarity_identity_and_implicit_derivatives():
    c=SimilarityCoordinates(.005)
    for tau in [1.,1e-4,1e-12]:
        for z in [0.,1e-5,.3,-2.]:
            point=jnp.array([.1,.2,z])
            q,eta,X=jax.jit(c)(point,tau)
            np.testing.assert_allclose(q-z*z*q**(2*c.h),tau,atol=1e-15,rtol=1e-10)
            np.testing.assert_allclose(X,.025/q,rtol=1e-14)
            L=1-2*c.h*eta**2
            qt=jax.grad(lambda tau:c(point,tau).q)(tau)
            qz=jax.grad(lambda z:c(jnp.array([.1,.2,z]),tau).q)(z)
            np.testing.assert_allclose(qt,1/L,rtol=1e-13)
            np.testing.assert_allclose(qz,2*z*q**(2*c.h)/L,atol=2e-14)
            assert abs(float(eta)) <= 1 # Can round to 1 away from the core as tau -> 0.


def test_scaling_and_invalid_domain():
    c=SimilarityCoordinates()
    scales=c.scales(jnp.array([1.,1e-8]))
    assert scales['tangential_speed'][1]>scales['tangential_speed'][0]
    assert scales['core_energy_scale'][1]<scales['core_energy_scale'][0]
    assert jnp.isnan(jax.jit(c)(jnp.zeros(3),0.).q)
    for h in [0.,.01,-1.,np.nan]:
        with pytest.raises(ValueError): SimilarityCoordinates(h)


def test_viscosity_rescaling_preserves_equations():
    model=ViscosityScaled(TaylorGreen(nu=1.),nu=.017)
    d=FieldDiagnostics(model.velocity,model.pressure,model.nu,model.forcing)
    x=jnp.array([[.1,.2,.3],[-.3,.1,.7]])
    np.testing.assert_allclose(d.residual(x,.1),0,atol=1e-14)
    assert model.energy_factor==model.nu**2.5
    with pytest.raises(ValueError): ViscosityScaled(TaylorGreen(nu=.1),.3)
