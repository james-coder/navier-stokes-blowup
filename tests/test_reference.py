import mpmath
import pytest
from ns_blowup.reference import ComponentReference


def test_precision_convergence_and_independent_integral():
    old_dps = mpmath.mp.dps
    refs = [ComponentReference(dps=dps) for dps in [25, 50]]
    for ref in refs:
        m = ref.mp
        for Z in ['.001', '1', '1000']:
            assert abs(ref.heat_factor(Z) - ref.heat_factor(Z, method='quadrature')) < 100*m.eps
        for tau, z in [('1e-12', '.5'), ('.3', '-2'), ('2', '0')]:
            q = ref.coordinates('1.5', z, tau)['q']
            assert abs(q-m.mpf(z)**2*q**(2*ref.h)-m.mpf(tau)) < 100*m.eps*max(1, q)
        assert abs(ref.heat_residual('1.5', '.2')) < 100*m.eps
    m = refs[1].mp
    assert abs(m.mpf(refs[0].heat_factor('1'))-refs[1].heat_factor('1')) < m.mpf('1e-24')
    assert mpmath.mp.dps == old_dps


def test_pressure_normalization_and_radial_balance():
    ref = ComponentReference(dps=25, h='.007', nu='.3')
    m = ref.mp
    r, tau, delta = m.mpf('1.5'), m.mpf('.2'), m.mpf('1e-6')
    p = ref.pressure(r, tau)
    assert p < 0
    assert abs(ref.pressure(100*r, tau)) < abs(p)/1000
    derivative = (ref.pressure(r+delta, tau)-ref.pressure(r-delta, tau))/(2*delta)
    assert abs(derivative-ref.swirl(r, tau)**2/r) < m.mpf('1e-12')
    velocity = ref.velocity(['1.5', '0', '3'], tau)
    assert velocity[0] == velocity[2] == 0
    assert velocity[1] == ref.swirl(r, tau)


def test_reference_domain_errors():
    for kwargs in [dict(dps=10), dict(h='.01'), dict(nu=0), dict(c='nan')]:
        with pytest.raises(ValueError):
            ComponentReference(**kwargs)
    ref = ComponentReference()
    for operation in [lambda: ref.swirl(0, 1), lambda: ref.pressure(1, 0),
                      lambda: ref.coordinates(-1, 0, 1), lambda: ref.heat_factor(-1)]:
        with pytest.raises(ValueError):
            operation()


def test_production_components_against_independent_reference():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ns_blowup import HeatExterior, SimilarityCoordinates
    with jax.enable_x64(True):
        ref = ComponentReference(dps=50, h='.005', nu='.3')
        exterior = HeatExterior(h=.005, nu=.3, order=128)
        expected = np.array([float(x) for x in ref.velocity(['1.5', '.5', '0'], '.2')])
        np.testing.assert_allclose(exterior.velocity(jnp.array([1.5, .5, 0.]), .8), expected, rtol=1e-12)
        np.testing.assert_allclose(exterior.pressure(jnp.array([1.5, 0., 0.]), .8), float(ref.pressure('1.5', '.2')), rtol=1e-9)
        expected = ref.coordinates('1.5', '.5', '1e-12')
        actual = SimilarityCoordinates()(jnp.array([1.5, 0., .5]), 1e-12)
        for key in ['q', 'eta', 'X']:
            np.testing.assert_allclose(getattr(actual, key), float(expected[key]), rtol=1e-12)
