# API guide

All primary names are imported from `ns_blowup`. Python docstrings describe arguments and array conventions. Version 0.2 is a preview; additions should preserve these conventions, and breaking changes must be called out before release.

| API | Purpose | Returns |
|---|---|---|
| `Simulation(shape, nu=..., lengths=..., dtype=..., device=...)` | Friendly simulation setup | Simulation object |
| `sim.run(initial, t_end=..., frames=..., dt=..., forcing=...)` | Adaptively evolve a flow | `SimulationResult` |
| `SimulationResult.load(path)` | Read a saved NPZ without pickle | Result object |
| `result.final` | Last saved velocity | JAX array |
| `result.diagnostics()` | Per-frame energy, speed, curl and divergence maxima | Dictionary of NumPy arrays |
| `result.plot(quantity=..., path=...)` | 2D image or middle xy slice in 3D | Matplotlib Axes |
| `result.save(path)` | NPZ with grid, time and metadata | Path |
| `result.export_vtk(path, frame=-1)` | Legacy structured-point velocity export | Path |
| `PeriodicGrid(shape, lengths=None)` | Endpoint-excluding 2D/3D box | Grid object |
| `grid.points(dtype=None)` | Coordinates in physical units | `(*shape, dimension)` array |
| `SpectralSolver(grid, nu=...)` | Compilable periodic solver | Solver object |
| `solver.project(u)` | Truncate and make velocity divergence free | Velocity array |
| `solver.rhs(u, t=..., forcing=...)` | Semidiscrete time derivative | Velocity array |
| `solver.step(u, t, dt, forcing=None)` | One RK4 step; caller controls positive stable dt | Velocity array |
| `solver.integrate(u, dt, steps, t0=..., forcing=..., save_every=...)` | Fixed-step differentiable evolution | Final velocity, or `(times, frames)` |
| `solver.pressure(u, t=..., forcing=...)` | Recover zero-mean pressure | Scalar grid array |
| `solver.divergence(u)`, `solver.vorticity(u)` | Fourier diagnostics | Scalar/vector grid array |
| `solver.suggest_dt(u, safety=0.4)` | Instantaneous step estimate | Scalar |
| `TaylorGreen(nu=..., amplitude=..., wavenumber=...)` | Exact 2D vortex, with optional 3D embedding | Field object |
| `TaylorGreen3D(amplitude=..., wavenumber=...).initial_velocity(points)` | True nonlinear 3D benchmark initial data | Velocity array |
| `ABCFlow(nu=..., a=..., b=..., c=..., wavenumber=...)` | Exact 3D Beltrami flow | Field object |
| `FieldDiagnostics(velocity, pressure, nu, forcing=None)` | Autodiff diagnostics | Diagnostic object |
| `checks.residual(points,t)`, `.divergence(...)`, `.vorticity(...)`, `.gradient(...)` | Batched analytical checks | JAX array |
| `kinetic_energy(u, grid)` | Integral energy estimate | Scalar |
| `sampled_norms(u, omega, gradient)` | Sampled norm maxima | Dictionary |
| `SimilarityCoordinates(h=0.005)(xyz,tau)` | Positive-branch similarity coordinates | Named tuple `(q, eta, X)` |
| `SimilarityCoordinates(h).scales(tau)` | Unit-normalized power laws | Dictionary |
| `ViscosityScaled(base, nu)` | Rescale a supplied viscosity-one field | Field object |
| `HeatExterior(h=..., c=..., nu=..., order=...)` | Outer-flow quadrature, away from axis | Field object |
| `exterior.factor(Z)` | Heat-factor integral approximation | Scalar/batched array |
| `exterior.azimuthal_velocity(radius,tau)` | Swirl speed with direct time remaining | Scalar/batched array |

`sim.run` also accepts `rtol`, `atol`, and `max_rejections` for [error-controlled acceptance and retries](numerics.md). Use `"taylor-green-3d"` on a 3D grid for the [independently checked 3D benchmark](taylor-green-3d.md).

The optional `ns_blowup.reference.ComponentReference(dps=...)` provides [independent multiprecision component evaluation](multiprecision.md). Its scalar results are mpmath numbers, and its `tau` argument is time remaining, not physical time.

Analytical field objects expose `.velocity(points,t)`, `.pressure(points,t)`, and `.forcing(points,t)`. Diagnostics require callbacks that accept a **single point** and return respectively `(d,)`, `()`, and `(d,)`; the supplied field objects also support batches. Time is scalar per evaluation. Use `jax.vmap` for multiple times.

Simulation initial callbacks accept the **whole point grid** and return a velocity grid. Solver forcing callbacks accept `(points, time)` and must return exactly the velocity-grid shape. Uniform forces should explicitly broadcast to that shape.

Choose `device="gpu"` to require a GPU, `"cpu"` to require CPU, or `"auto"` to use JAX's default. `JAX_PLATFORMS` can restrict which backends exist in a process. A required unavailable device produces an error, rather than a hidden fallback. The observatory generator is the exception: it prefers CPU for its small report and uses the enabled device if CPU is disabled.

Construct grids, solver objects, and quadrature objects outside JIT. Close over them in a compiled function; keep `steps` and `save_every` static. The friendly `Simulation.run` driver, plotting, filesystem operations, and data loading are host APIs and do not belong inside JIT.

Positive `tau` is required for similarity coordinates; the heat exterior permits `tau=0` only at positive radius. Invalid mathematical-domain inputs return NaN even under JIT. Validate evaluation ranges before large batched work. Geometry h is a selected static parameter; this library does not supply the theorem's complete parameter-selection machinery.
