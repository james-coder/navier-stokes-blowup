import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from ns_blowup import HeatExterior, FieldDiagnostics


def test_heat_factor_against_independent_adaptive_quadrature():
    exterior=HeatExterior(order=96)
    for Z in [0.,.001,.1,1.,4.]:
        reference=quad(lambda v:np.exp(-v)*v**exterior.h*(1+Z*v)**(-exterior.h),0,np.inf,
                       epsabs=1e-13,epsrel=1e-13)[0]/gamma(1+exterior.h)
        np.testing.assert_allclose(exterior.factor(Z),reference,atol=2e-10)
    derivative=jax.grad(exterior.factor)(0.)
    np.testing.assert_allclose(derivative,-exterior.h*(1+exterior.h),rtol=2e-14)


def test_exterior_heat_equation_and_radial_monotonicity():
    exterior=HeatExterior(nu=.3,order=96)
    for radius,tau in [(1.,.01),(2.,.3),(3.,.8)]:
        k=exterior.azimuthal_velocity
        kr=jax.grad(k,0)(radius,tau)
        krr=jax.grad(jax.grad(k,0),0)(radius,tau)
        kt=-jax.grad(k,1)(radius,tau)
        residual=kt-exterior.nu*(krr+kr/radius-k(radius,tau)/radius**2)
        assert abs(float(residual))<1e-11
        assert float(kr)<0
    assert np.isnan(exterior.azimuthal_velocity(0.,.1))


def test_exterior_momentum_with_pressure_quadrature():
    e=HeatExterior(order=96,nu=.7)
    d=FieldDiagnostics(e.velocity,e.pressure,e.nu,e.forcing)
    points=jnp.array([[2.,.3,.7],[-3.,1.,.2]])
    np.testing.assert_allclose(jax.jit(d.residual)(points,.7),0,atol=3e-10)
    np.testing.assert_allclose(d.divergence(points,.7),0,atol=2e-14)
