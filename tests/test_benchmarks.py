import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ns_blowup import PeriodicGrid, Simulation, SpectralSolver, TaylorGreen3D


def test_true_3d_initial_invariants_and_nonlinear_evolution():
    grid = PeriodicGrid((12, 12, 12))
    model = TaylorGreen3D()
    u = model.initial_velocity(grid.points(jnp.float64))
    solver = SpectralSolver(grid, .05)
    np.testing.assert_allclose(jnp.mean(jnp.sum(u*u, axis=-1))/2, 1/8, atol=1e-15)
    omega = solver.vorticity(u)
    np.testing.assert_allclose(jnp.mean(jnp.sum(omega*omega, axis=-1))/2, 3/8, atol=1e-15)
    assert float(jnp.max(jnp.abs(solver.divergence(u)))) < 1e-14
    assert float(jnp.max(jnp.abs(u[:, :, 0]-u[:, :, 3]))) > .9
    result = Simulation(grid.shape, nu=.05, dtype='float64').run('taylor-green-3d', t_end=.1, frames=2)
    assert float(jnp.max(jnp.abs(result.final[..., 2]))) > .005
    assert result.diagnostics()['energy'][1] < result.diagnostics()['energy'][0]
    # Public point evaluator stays differentiable.
    derivative = jax.jacfwd(model.initial_velocity)(jnp.array([.2, .3, .4]))
    assert abs(float(jnp.trace(derivative))) < 1e-14


def test_3d_benchmark_domain_and_parameters():
    with pytest.raises(ValueError, match='t0=0'):
        Simulation((6, 6, 6)).run('taylor-green-3d', t0=.1)
    with pytest.raises(ValueError):
        Simulation((6, 6)).run('taylor-green-3d')
    with pytest.raises(ValueError):
        TaylorGreen3D().initial_velocity(jnp.zeros(2))
    for kwargs in [dict(amplitude=np.inf), dict(wavenumber=0)]:
        with pytest.raises(ValueError):
            TaylorGreen3D(**kwargs)
