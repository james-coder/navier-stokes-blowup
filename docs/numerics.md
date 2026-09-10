# Numerical methods and conventions

## Equations and domain

For constant density absorbed into pressure, the implemented equation is

$$
\partial_t u = P[-(u\cdot\nabla)u+f]+\nu\Delta u,
\qquad \nabla\cdot u=0.
$$

The box is periodic in every direction. Samples are at `x_i = i L/N`, omitting the repeated endpoint. Vectors use shape `(nx, ny, 2)` or `(nx, ny, nz, 3)`; the last axis stores components. The Fourier transform acts only on spatial axes. Mixed units are not detected automatically.

`PeriodicGrid` supports unequal side lengths, odd grid sizes, and unequal point counts. Named initial conditions have period `2π` and require compatible box lengths. A callback can define another periodic initial field. Walls and obstacle geometries need a different discretization.

## Projection, nonlinear terms, and pressure

For each nonzero Fourier wavevector, the Leray projection is `P_k(v) = v − k(k·v)/|k|²`. The constant mode is preserved, including mean acceleration from forcing. The numerical solution retains only modes satisfying `|m_i| < N_i/3` in **every** direction. The strict inequality eliminates a boundary ambiguity when a size is divisible by three. Initial conditions and forcing are projected/truncated too; initial projection change is recorded in the result metadata.

The quadratic term is formed from inverse transforms of the retained velocity and its spectral derivatives, then transformed back and truncated. Because the input bandwidth is strictly below `N/3`, products cannot alias into the retained band. A test compares this operation against an independently overresolved physical grid and checks zero inviscid energy production. Two-thirds truncation discards high modes; it does not give a physical-space grid an effective resolution equal to its nominal point count.

Pressure is recovered with spatial mean zero from the divergence of `f − (u·∇)u`. Adding a spatial constant to the exact pressure is harmless. The pressure test removes the exact pressure's mean before comparison. Established pseudospectral approaches and implementation context are described by [Mortensen and Langtangen](https://arxiv.org/abs/1602.03638) and [spectralDNS](https://github.com/spectralDNS/spectralDNS).

## Time stepping and stability

Classical RK4 evaluates forcing at all four stage times. The fixed-step interface is differentiable through initial fields, forcing parameters, and scalar timestep; viscosity and grid configuration are currently constructor values, not trainable inputs. Negative or nonfinite viscosity is rejected.

The convenience driver uses RK4 step doubling: it compares one full step with two half steps, estimates the fine solution's local error as `abs(fine − coarse)/15`, and accepts the two-half-step solution only when every component's estimate is at most `atol + rtol*max(abs(old), abs(fine))`. The factor 15 follows from the fourth-order method's factor-of-16 reduction in accumulated local error when the step is halved. A fifth-root controller shrinks or grows the next attempted step. Defaults are `rtol=1e-4, atol=1e-6` in float32 and `rtol=1e-6, atol=1e-9` in float64; both can be supplied to `Simulation.run`.

Each attempt is also limited by maximum retained wavenumbers, component speeds, and viscous rate, using safety factor 0.4. The driver rechecks this estimate on the candidate midpoint and endpoint to catch force-driven speed growth. It clips attempts to the next saved-frame time and the user `dt` upper bound. Rejected candidates never enter the trajectory; nonfinite candidates trigger smaller retries. `max_steps` limits accepted steps and `max_rejections` limits total rejected attempts, with an error reporting the current time on exhaustion. Metadata records rejected steps, tolerances, and the largest accepted estimated error ratio.

These local estimates are not a stability proof or a global error bound. Forcing must be resolved by the stage times: a narrow pulse or oscillation can fall between all samples and defeat both error and CFL estimates. Set `dt` to resolve known forcing timescales and repeat with a smaller limit. The [force-from-rest regression](https://github.com/james-coder/navier-stokes-blowup/blob/main/tests/test_adaptive.py) checks a rapidly growing exponential force against its exact integral, tolerance convergence, precise frame scheduling, and clear retry exhaustion.

Every step of the convenience driver synchronizes with the host for checks. It prioritizes approachable behavior over throughput. `SpectralSolver.integrate` uses compiled loops, retains only the final state by default, and can save at specified step intervals. Backpropagation can still retain substantial intermediate data. No adjoint checkpointing policy or memory bound is implemented.

## Exact benchmark formulas

These formulas are independently checked by substitution using autodiff and by evolution through the solver; their use does not depend on a blowup theorem.

Let `a(t)=a0 exp(−2νk²t)`. The 2D [Taylor–Green vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex) solution is

$$
u=(a\sin(kx)\cos(ky),-a\cos(kx)\sin(ky)),
\qquad p={a^2\over4}(\cos(2kx)+\cos(2ky)).
$$

For 3D point input this implementation appends zero vertical velocity and remains independent of z. **It is not the fully three-dimensional Taylor–Green initial condition containing `cos(z)`**, whose later nonlinear evolution is not this exact formula.

The 3D ABC solution is

$$
u=e^{-\nu k^2t}
(A\sin(kz)+C\cos(ky),\ B\sin(kx)+A\cos(kz),\ C\sin(ky)+B\cos(kx)).
$$

It satisfies `curl(u)=k u`, `Δu=−k²u`, and has pressure `−|u|²/2`, up to a constant. Its nonlinear acceleration is a gradient, which pressure balances. Tests also exercise a genuinely nontrivial forced flow so that benchmark cancellations cannot conceal a missing nonlinear term.

The manufactured forced test uses `u=0.3 exp(t)(sin y,sin z,sin x)` and `p=sin x sin y sin z`. The forcing is expanded analytically, independently of the solver, and includes its nonlinear acceleration, time derivative, diffusion, and pressure gradient. This verifies force signs, stage timing, pressure, and retained nonlinear dynamics together.

## Diagnostics and interpretation

Energy is `E = 1/2 Σ |u|² ΔV`, not a mean-square velocity or an unnormalized FFT norm. Reported velocity/vorticity maxima are Euclidean magnitudes over the sampled points. Gradient magnitude uses the Frobenius norm. None are certified continuum supremum norms. In 2D vorticity is the signed out-of-plane scalar; in 3D it is a vector.

`FieldDiagnostics` computes time and spatial derivatives with autodiff. `SpectralSolver.divergence` and `.vorticity` use discrete Fourier differentiation. Agreement is informative because these are separate paths. Small residuals at a finite set of points cannot establish regularity, uniqueness, or a theorem about a limiting singular time.

The heat-exterior implementation evaluates finite quadrature sums. Its test reference uses a separate adaptive SciPy integral, which is a higher-accuracy floating-point check, not arbitrary-precision or interval arithmetic. Increasing quadrature order is not guaranteed to improve every derivative uniformly at arbitrarily large `Z`.
