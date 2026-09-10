import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ns_blowup import PeriodicGrid, SpectralSolver, TaylorGreen, ABCFlow, kinetic_energy


@pytest.mark.parametrize('shape,lengths', [((12,15),(3.,7.)), ((8,9,10),(2.,3.,4.))])
def test_projection_divergence_idempotence_mean(shape,lengths):
    grid = PeriodicGrid(shape,lengths)
    solver = SpectralSolver(grid)
    u = jnp.asarray(np.random.default_rng(2).normal(size=shape+(grid.ndim,)))
    p = jax.jit(solver.project)(u)
    assert np.max(np.abs(solver.divergence(p))) < 2e-14
    np.testing.assert_allclose(solver.project(p),p,atol=2e-15)
    np.testing.assert_allclose(np.mean(p,axis=solver.axes),np.mean(u,axis=solver.axes),atol=2e-15)
    assert float(kinetic_energy(p,grid)) < float(kinetic_energy(u,grid))


@pytest.mark.parametrize('shape,model', [((12,12),TaylorGreen(nu=.2)),((8,8,8),ABCFlow(nu=.2))])
def test_analytic_solution_pressure_vorticity_and_energy(shape,model):
    grid = PeriodicGrid(shape)
    solver = SpectralSolver(grid,model.nu)
    points = grid.points()
    u = model.velocity(points,0.)
    final = jax.jit(lambda u: solver.integrate(u,.01,20))(u)
    np.testing.assert_allclose(final,model.velocity(points,.2),atol=2e-12)
    expected_p = model.pressure(points,0.)
    expected_p -= expected_p.mean()
    np.testing.assert_allclose(solver.pressure(u),expected_p,atol=3e-15)
    assert float(kinetic_energy(final,grid)) < float(kinetic_energy(u,grid))
    assert np.max(np.abs(solver.divergence(final))) < 2e-14
    if grid.ndim == 3:
        np.testing.assert_allclose(solver.vorticity(u),u,atol=3e-15)


def test_fourth_order_time_convergence():
    grid=PeriodicGrid((8,8))
    flow=TaylorGreen(nu=.5)
    solver=SpectralSolver(grid,flow.nu)
    initial=flow.velocity(grid.points(),0.)
    errors=[]
    for steps in [4,8,16]:
        u=solver.integrate(initial,1./steps,steps)
        errors.append(float(jnp.max(jnp.abs(u-flow.velocity(grid.points(),1.)))))
    assert errors[0]/errors[1] > 14
    assert errors[1]/errors[2] > 14


def test_time_dependent_forcing_and_pressure():
    grid=PeriodicGrid((10,10,10))
    solver=SpectralSolver(grid,nu=.13)
    points=grid.points()

    def velocity(x,t):
        return .3*jnp.exp(t)*jnp.stack([jnp.sin(x[...,1]),jnp.sin(x[...,2]),jnp.sin(x[...,0])],axis=-1)

    def force(x,t):
        amp=.3*jnp.exp(t)
        sx,sy,sz=(jnp.sin(x[...,i]) for i in range(3))
        cx,cy,cz=(jnp.cos(x[...,i]) for i in range(3))
        adv=amp**2*jnp.stack([cy*sz,cz*sx,cx*sy],axis=-1)
        gradp=jnp.stack([cx*sy*sz,sx*cy*sz,sx*sy*cz],axis=-1)
        return (1+solver.nu)*velocity(x,t)+adv+gradp

    final=jax.jit(lambda u: solver.integrate(u,.005,20,forcing=force))(velocity(points,0.))
    np.testing.assert_allclose(final,velocity(points,.1),atol=3e-12)
    expected=jnp.prod(jnp.sin(points),axis=-1)
    np.testing.assert_allclose(solver.pressure(velocity(points,.1),.1,force),expected,atol=3e-15)


def test_nonlinear_transfer_against_independent_overresolved_grid():
    coarse=PeriodicGrid((12,12,12))
    fine=PeriodicGrid((48,48,48))
    a=SpectralSolver(coarse,nu=0.)
    b=SpectralSolver(fine,nu=0.)
    # Populate all retained modes with a generic divergence-free field.
    u=a.project(jnp.asarray(np.random.default_rng(12).normal(size=(12,12,12,3))))
    hat=np.fft.fftn(np.asarray(u),axes=(0,1,2))/12**3
    fine_hat=np.zeros((48,48,48,3),dtype=complex)
    for i in range(12):
        for j in range(12):
            for k in range(12):
                modes=tuple(n if n<6 else n-12 for n in (i,j,k))
                fine_hat[tuple(n%48 for n in modes)]=hat[i,j,k]*48**3
    fine_u=jnp.asarray(np.fft.ifftn(fine_hat,axes=(0,1,2)).real)
    rhs_fine=np.fft.fftn(np.asarray(b.rhs(fine_u)),axes=(0,1,2))/48**3
    rhs_coarse=np.fft.fftn(np.asarray(a.rhs(u)),axes=(0,1,2))/12**3
    for i,j,k in np.argwhere(np.asarray(a.mask)):
        modes=tuple(n if n<6 else n-12 for n in (i,j,k))
        np.testing.assert_allclose(rhs_coarse[i,j,k],rhs_fine[tuple(n%48 for n in modes)],atol=2e-15)
    # Inviscid energy production vanishes for the truncated nonlinear operator.
    assert abs(float(jnp.sum(u*a.rhs(u)))) < 2e-12


def test_autodiff_initial_amplitude_and_forcing():
    grid=PeriodicGrid((8,8))
    solver=SpectralSolver(grid,nu=.1)
    initial=TaylorGreen(nu=.1).velocity(grid.points(),0.)
    def objective(a):
        u=solver.integrate(a*initial,.01,10)
        return kinetic_energy(u,grid)
    derivative=jax.jit(jax.grad(objective))(1.3)
    np.testing.assert_allclose(derivative,2*objective(1.3)/1.3,rtol=1e-12)

    def forced(a):
        def f(x,t):
            return jnp.stack([a*jnp.sin(x[...,1]),jnp.zeros_like(x[...,1])],axis=-1)
        return kinetic_energy(solver.integrate(jnp.zeros_like(initial),.01,10,forcing=f),grid)
    derivative=jax.grad(forced)(.7)
    fd=(forced(.70001)-forced(.69999))/.00002
    np.testing.assert_allclose(derivative,fd,rtol=2e-9)


def test_saved_frames_remainder_and_zero_steps():
    solver=SpectralSolver(PeriodicGrid((8,8)))
    u=TaylorGreen().velocity(solver.grid.points(),0.)
    times,frames=solver.integrate(u,.01,5,save_every=2,t0=.3)
    np.testing.assert_allclose(times,[.3,.32,.34,.35])
    np.testing.assert_allclose(frames[-1],solver.integrate(u,.01,5,t0=.3),atol=1e-14)
    times,frames=solver.integrate(u,.01,0,save_every=2)
    assert frames.shape==(1,8,8,2)
    for dt in [0,-1,np.nan]:
        with pytest.raises(ValueError): solver.integrate(u,dt,1)


@pytest.mark.parametrize('dtype', [jnp.float32,jnp.float64])
def test_precision_and_constant_mode(dtype):
    solver=SpectralSolver(PeriodicGrid((9,9)),nu=.01)
    u=jnp.ones((9,9,2),dtype=dtype)
    result=solver.integrate(u,.01,2)
    assert result.dtype == dtype
    np.testing.assert_allclose(result,u,atol=1e-6 if dtype==jnp.float32 else 1e-14)
