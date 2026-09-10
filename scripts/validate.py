"""Generate a reproducible, device-specific numerical report. No performance ranking."""
import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ns_blowup import ABCFlow, FieldDiagnostics, PeriodicGrid, SpectralSolver, TaylorGreen, __version__


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    rows=[]
    for dim in (2,3):
        for dtype in ('float32','float64'):
            # Create constants in the selected precision, not just the initial array.
            with jax.enable_x64(dtype=='float64'):
                grid=PeriodicGrid((16,)*dim)
                flow=TaylorGreen(nu=.1) if dim==2 else ABCFlow(nu=.1)
                solver=SpectralSolver(grid,nu=.1)
                points=grid.points(getattr(jnp,dtype))
                u=flow.velocity(points,0.)
                solve=jax.jit(lambda u:solver.integrate(u,.01,20))
                start=time.perf_counter()
                result=solve(u).block_until_ready()
                compile_run=time.perf_counter()-start
                samples=[]
                for _ in range(3):
                    start=time.perf_counter()
                    result=solve(u).block_until_ready()
                    samples.append(time.perf_counter()-start)
                error=float(jnp.max(jnp.abs(result-flow.velocity(points,.2))))
                divergence=float(jnp.max(jnp.abs(solver.divergence(result))))
                diagnostic=FieldDiagnostics(flow.velocity,flow.pressure,flow.nu)
                residual=float(jnp.max(jnp.abs(diagnostic.residual(points.reshape(-1,dim)[:8],.2))))
                tolerance=2e-5 if dtype=='float32' else 1e-11
                if max(error,divergence,residual)>tolerance:
                    raise RuntimeError(f'validation failed for {dim}D {dtype}')
                rows.append(dict(dimension=dim,shape=grid.shape,dtype=dtype,steps=20,dt=.01,
                                 velocity_max_error=error,divergence_max=divergence,
                                 analytical_residual_max=residual,compile_and_run_seconds=compile_run,
                                 warmed_median_seconds=float(np.median(samples))))
            jax.clear_caches()
    payload=dict(package_version=__version__,python=platform.python_version(),jax=jax.__version__,
                 numpy=np.__version__,devices=[str(d) for d in jax.devices()],
                 device_kind=jax.devices()[0].device_kind,platform=platform.platform(),
                 scope='Small exact-flow checks; occupied GPU; not a CPU/GPU performance comparison',results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
