import jax.numpy as jnp
import numpy as np
import pytest

from ns_blowup import Simulation


def test_force_from_rest_retries_and_converges_with_exact_frames():
    # Spatially uniform forcing reduces the PDE to u'=exp(12t). Its rapid
    # growth is invisible to the zero-velocity, zero-viscosity initial bound.
    sim = Simulation((6, 6), nu=0, dtype='float64', device='cpu')
    zero = jnp.zeros((6, 6, 2))
    def forcing(x, t):
        return jnp.ones_like(x) * jnp.exp(12*t)
    errors = []
    for tolerance in [1e-3, 1e-8]:
        result = sim.run(zero, forcing=forcing, t_end=.5, frames=2,
                         rtol=tolerance, atol=tolerance * .01)
        expected = np.expm1(6) / 12
        errors.append(float(jnp.max(jnp.abs(result.final - expected))))
        assert result.metadata['rejected_steps'] > 0
        assert result.metadata['max_accepted_error_ratio'] <= 1
        np.testing.assert_array_equal(result.times, [0., .5])
    assert errors[1] < errors[0] / 20
    assert errors[1] < 1e-6
    calls = []
    result = sim.run(zero, forcing=forcing, t_end=.5, frames=7,
                     progress=lambda t, n: calls.append((t, n)))
    np.testing.assert_array_equal(result.times, np.linspace(0, .5, 7))
    np.testing.assert_allclose(result.velocity[:, 0, 0, 0], np.expm1(12*np.asarray(result.times))/12, rtol=1e-6)
    assert [t for t, _ in calls] == list(np.asarray(result.times)[1:])
    assert all(b[1] > a[1] for a, b in zip(calls, calls[1:]))


def test_retry_exhaustion_and_invalid_tolerances():
    sim = Simulation((6, 6), nu=0, dtype='float64', device='cpu')
    zero = jnp.zeros((6, 6, 2))
    for force in [lambda x, t: jnp.ones_like(x)*jnp.exp(12*t),
                  lambda x, t: jnp.full_like(x, jnp.nan)]:
        with pytest.raises(RuntimeError, match='max_rejections.*t=0'):
            sim.run(zero, forcing=force, t_end=.5, frames=2, max_rejections=0)
    for kwargs in [dict(rtol=-1), dict(atol=np.nan), dict(rtol=0, atol=0), dict(max_rejections=-1)]:
        with pytest.raises(ValueError):
            sim.run(zero, **kwargs)


def test_zero_flow_and_zero_relative_tolerance():
    result = Simulation((6, 6), nu=0, device='cpu').run(
        jnp.zeros((6, 6, 2)), t_end=1, frames=3, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.velocity, 0)
    assert result.metadata['rejected_steps'] == 0
