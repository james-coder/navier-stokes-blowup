import jax.numpy as jnp
import numpy as np
import pytest

from ns_blowup import Simulation, SimulationResult, PeriodicGrid, TaylorGreen


def test_friendly_workflow_npz_plot_and_vtk(tmp_path):
    sim=Simulation((8,8),nu=.2,dtype='float64')
    calls=[]
    result=sim.run(t_end=.1,frames=3,dt=.01,progress=lambda t,n:calls.append((t,n)))
    assert len(calls)==2
    assert result.velocity.shape==(3,8,8,2)
    expected=TaylorGreen(nu=.2).velocity(sim.points,.1)
    np.testing.assert_allclose(result.final,expected,atol=1e-12)
    metrics=result.diagnostics()
    assert np.max(metrics['divergence_max'])<1e-14
    assert np.all(np.diff(metrics['energy'])<0)
    path=result.save(tmp_path/'flow.npz')
    loaded=SimulationResult.load(path)
    np.testing.assert_array_equal(loaded.velocity,result.velocity)
    assert loaded.metadata==result.metadata
    assert loaded.grid==result.grid
    vtk=result.export_vtk(tmp_path/'final.vtk').read_text()
    assert 'DIMENSIONS 8 8 1' in vtk and 'POINT_DATA 64' in vtk
    values=np.loadtxt(vtk.splitlines()[9:])
    # First two VTK points differ in x, not y.
    np.testing.assert_allclose(values[1,:2],np.asarray(result.final)[1,0],atol=1e-14)
    import matplotlib
    matplotlib.use('Agg')
    ax=result.plot(path=tmp_path/'plot.png')
    assert (tmp_path/'plot.png').stat().st_size>1000
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_3d_float32_and_custom_periodic_box():
    sim=Simulation((8,8,8),dtype='float32')
    result=sim.run('abc',t_end=.01,frames=2)
    assert result.velocity.dtype==jnp.float32
    sim=Simulation((8,8),lengths=(3.,4.))
    def initial(x):
        return jnp.stack([jnp.sin(2*jnp.pi*x[...,1]/4),jnp.zeros_like(x[...,0])],axis=-1)
    r=sim.run(initial,t_end=.01,frames=2)
    assert r.final.shape==(8,8,2)
    with pytest.raises(ValueError,match='integer multiples'): sim.run(t_end=.01)


@pytest.mark.parametrize('kwargs', [{'t_end':0},{'frames':1},{'dt':0},{'safety':0},{'max_steps':0}])
def test_validation(kwargs):
    with pytest.raises(ValueError): Simulation((8,8)).run(**kwargs)


def test_bad_initial_and_step_limit():
    s=Simulation((8,8))
    with pytest.raises(ValueError): s.run(jnp.zeros((8,8,3)))
    with pytest.raises(ValueError): s.run(jnp.full((8,8,2),jnp.nan))
    with pytest.raises(ValueError): s.run('abc')
    with pytest.raises(RuntimeError,match='max_steps'): s.run(dt=.0001,max_steps=1)
    for shape in [(4,4),(8,),(8.5,8)]:
        with pytest.raises(ValueError): PeriodicGrid(shape)


def test_loading_preserves_precision_and_rejects_corrupt_metadata(tmp_path):
    import jax
    r=Simulation((8,8),dtype='float64').run(t_end=.01,frames=2)
    path=r.save(tmp_path/'result.npz')
    with jax.enable_x64(False):
        with pytest.raises(ValueError,match='preserve precision'): SimulationResult.load(path)
    with np.load(path) as contents:
        data={k:contents[k] for k in contents.files}
    data['nu']=np.nan
    np.savez(path,**data)
    with pytest.raises(ValueError,match='viscosity'): SimulationResult.load(path)
