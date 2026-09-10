import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ns_blowup import ABCFlow, TaylorGreen, FieldDiagnostics, PeriodicGrid, kinetic_energy


@pytest.mark.parametrize('flow,dim', [(TaylorGreen(nu=.17), 2), (TaylorGreen(nu=.17), 3), (ABCFlow(nu=.17,a=.3,b=.7,c=1.2), 3)])
def test_exact_momentum_and_incompressibility(flow, dim):
    points = jnp.asarray(np.random.default_rng(21).uniform(0, 2*np.pi, (2, 4, dim)))
    diagnostic = FieldDiagnostics(flow.velocity, flow.pressure, flow.nu, flow.forcing)
    assert np.max(np.abs(jax.jit(diagnostic.residual)(points, .3))) < 2e-14
    assert np.max(np.abs(diagnostic.divergence(points, .3))) < 2e-14
    assert diagnostic.gradient(points, .3).shape == (2, 4, dim, dim)


def test_curl_sign_and_beltrami_identity():
    x = jnp.array([[.3,.5,.7],[1.,2.,3.]])
    model = ABCFlow(wavenumber=2.0)
    d = FieldDiagnostics(model.velocity,model.pressure,model.nu)
    np.testing.assert_allclose(d.vorticity(x,.2), 2*model.velocity(x,.2), atol=1e-14)
    tg = TaylorGreen()
    d = FieldDiagnostics(tg.velocity,tg.pressure,tg.nu)
    expected = 2*jnp.sin(x[:,0])*jnp.sin(x[:,1])*jnp.exp(-2*tg.nu*.2)
    np.testing.assert_allclose(d.vorticity(x[:,:2],.2),expected,atol=1e-14)


def test_energy_normalization_against_closed_form():
    g = PeriodicGrid((12,12))
    model = TaylorGreen(nu=.2,amplitude=1.7)
    expected = g.volume * model.amplitude**2 / 4 * np.exp(-4*model.nu*.4)
    np.testing.assert_allclose(kinetic_energy(model.velocity(g.points(),.4),g),expected,rtol=1e-14)


def test_residual_detects_wrong_pressure_and_force():
    model = TaylorGreen()
    points = jnp.array([[.3,.7]])
    d = FieldDiagnostics(model.velocity,lambda x,t: jnp.array(0.),model.nu)
    assert np.max(np.abs(d.residual(points,.2))) > .1
    d = FieldDiagnostics(model.velocity,model.pressure,model.nu,lambda x,t: jnp.ones_like(x))
    np.testing.assert_allclose(d.residual(points,.2),-1,atol=1e-14)


def test_diagnostics_accept_integer_coordinates_and_time():
    model=TaylorGreen()
    d=FieldDiagnostics(model.velocity,model.pressure,model.nu)
    np.testing.assert_allclose(d.residual([[0,1],[1,0]],0),0,atol=1e-14)
    with pytest.raises(ValueError,match='scalar'): d.residual([[0,1]],[0.,1.])
