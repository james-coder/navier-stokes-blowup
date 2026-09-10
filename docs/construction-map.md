# From the construction to executable requirements

This is the dependency map for implementing the **complete** smooth forced blowup field. Coordinates and an exterior heat profile do not determine that field. The reference sources are the [166-page paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf), SHA-256 `0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f`, and [Lean commit `8937a8f4`](https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538). Section numbers below refer to that paper. Lean links pin the same commit; names refer to declarations in the linked module, not to a claim of automatic code extraction.

The target is the forced construction in Theorem 1.1, including smooth compactly supported force, initially resting flow, bounded energy and finite-time velocity growth. Checking its formal statements, approximating its components, and evaluating the complete field are separate deliverables.

## Dependency table

“Explicit” means a formula can be evaluated once its inputs are supplied. “Choice” means existence is proved but the numerical witness or a quantitative threshold still must be computed. “Limit” requires a truncation rule and an error bound. A Lean `noncomputable` declaration is not, by itself, evidence that numerical approximation is impossible: real arithmetic, integrals and ordinary derivatives also receive that designation.

| Stage and source | Required input → numerical object | Domain and normalization | Choice or limit still required |
|---|---|---|---|
| 1. Geometry, (4.1)–(4.5); [SimilarityProfile.pullback_smoothOn][similarity] | `h`, physical point, `τ` → `q,η,X`, pullbacks and mixed derivative operators | `τ>0`, `X≥0`, `|η|<1`, `0<h<.01`; unit-viscosity coordinates | Explicit implicit root with branch control; positive viscosity rescaling follows afterward. Implemented with independent reference. |
| 2. Outgoing schedule, A.1–A.5; [OutgoingProfile.exists_outgoing_profile][outgoing], [ScheduledProfileChoice.exists_scheduled_core_below][schedule] | Positive amplitudes, moment targets and reset bounds → piecewise radial profile and axial pressure datum `Π₀(η)` | Full logarithmic radial line; pressure zero at infinity; analytic parameter neighborhood | Choice of reset bounds, small `λ`, transition intervals, exact moment adjustments; infinite pressure integral with controlled tails. The pressure datum must precede the inner solve. |
| 3. Axis preparation, B.1–B.3; [NaturalAxisCoefficients.ideal_prefix_analytic_inputs][axis-input], [NaturalProfile.exists_profileFamily][natural] | The *same* `h,Π₀`, plus `j₀,δ,σ,Λ,C` → analytic `φ,U,Π`, their radial average and derivatives | `0≤ΛX≤4.1`, `η∈[-1,1]` with a common complex neighborhood; prescribed axis values | Choice of `δ,σ`, analytic coefficient radius, contraction threshold `Λ₀`, normalization bound `C`; limit of the coefficient-space fixed point. |
| 4. Inner continuation, B.4–B.7; [NominalProfile.AxisPreparation and Witness][nominal] | Axis endpoint values and derivatives → continuation into a nonzero-stress region and an outgoing matching segment | Positive radial interval adjoining the analytic core; preserve pressure history | Reference path, positive sources, transition length, intermediate amplitude; quantitative endpoint margins and all derivative bounds. |
| 5. Five moment repairs, B.8–B.9 and A.1; [FiveProfileMoments.local_profile_moments][moments], [NominalProfile.exists_assembly_threshold_preserving][nominal] | Actual prefix debt and reserved radial patches → corrected `U,E` with matching five radial moment functions | Compact radial repair support; every `η`, including required derivatives; retain the previously fixed `Π₀` | Construct and invert the finite moment map with controlled conditioning; select scale/amplitude below its neighborhood bound. Matching values at finitely many η samples is insufficient. |
| 6. Heat replacement, A.6–A.7; [RadialHeatProfile.radialProfile_source_formula][heat], [OutgoingProfile.Pi_canonical][outgoing] | Outgoing amplitude `c∞`, `h`, heat profile and repair data → exact heat exterior and compatible stress | Positive radius; `E=c∞X^(-1/2-h)H(2(1−η²)/X)`; pressure zero at infinity; `U=V₀=0` in the exterior | Heat integral and pressure integral are implemented. Their attachment and preservation of all matching constraints are not. |
| 7. Annular cone, Theorem 4.6 and Appendix C; [ModulatedProfileAssembly.exists_modulated_profile][modulated], [AlignedProfileSpectralCone.modulated_positive_reference_cone][cone] | Matched nominal profile and annular intervals → modulated leading profiles, stress and admissible directions | Stress zero inside the core and outside the annulus; flat edges; reserved positive-order and mean patches | Finite modulation frequency, cone margin `κ>0`, positivity and derivative bounds uniformly on the entire annulus. |
| 8. Higher-order slow coefficients, §5.1–§5.3; [GlobalSlowProfiles.exists_step, profiles][slow] | Leading profiles and all previous coefficients → each new slow coefficient, pressure and flux correction | Common radial supports; smooth at axis; required exterior forms and zero moments | Recursive analytic solves and moment repairs for every order; finite implementation needs coefficient/derivative error budgets. |
| 9. Slow Borel sum, Lemma 5.4 and Proposition 5.5; [SlowBorelBase.powered_fixed_prefix_bound][borel] | Full coefficient sequence and derivative bounds → divergence-free base, annular stress and flat remainder | `q>0`; streamfunctions are cut off and summed before taking derivatives | Limit with shrinking cutoffs `a_j`, doubling schedule and bounds for every derivative order. “Take the first N terms” is not an error certificate. |
| 10. Bands and auxiliary torus, §6; [ActualCycleParameters.fixedParameters and parameters_indexBounds][parameters] | Common base, supports and band floor → labels, nonoverlapping slow supports, phases, torus charts and carrier frequencies | Band-localized shells and a common auxiliary torus for interacting labels | Large-enough band threshold, centers, finite-overlap bounds and compatible reindexing. Physical time differentiation must include auxiliary phase derivatives. |
| 11. Primary waves and signed covariance, §7; [ActualSignedNativeProfiles.nativeProfiles and potential_profiles_jets][waves] | Background shear, cone directions, prescribed stress and phases → transverse pulse amplitudes, positive squared weights, potentials and pressure | Nonzero angular harmonics, annular supports, transverse leading velocities | Supported amplitude-equation solves and covariance inverses with positive margins. Take actual curls and retain their remainders; leading covariance alone is not a velocity solution. |
| 12. Mean corrections, §8; [TorusInverse.directionalInverse_solves][torus] and [ActualMeanStageData.cycleRankData][mean] | Updated residual, auxiliary means and five radial debts → compact mean velocity and pressure increments | Zero torus mean for fast-time inversion; reserved interior radial patch | Torus small-divisor inversion, five-moment inverse and pressure reconstruction; all discarded-mode/tail errors must be carried forward. |
| 13. Full residual cycle, Proposition 9.6; [ActualCycleResidualBounds.finite_residual_rates][cycle] | Complete current fields, including curl/cutoff terms → next velocity, pressure and residual | Common domain `0<q<q_big`; supports and preserved integrals maintained at each step | Recursive correction; exponent `σ_j=1/5+j/10` improves, but constants and logarithmic losses depend on stage/derivative order. Requires quantitative bounds to select a finite stage. |
| 14. Final local field, Lemma 9.7 and Proposition 9.9; [MixedDiagonalSchedule.exists_three_component_local_schedule][diagonal], [SolenoidalDiagonal.exists_diagonal_scales_smooth_solenoidal][solenoidal] | All correction potentials, direct angular increments, pressures and bounds → locally finite smooth field with flat residual | `q>0`; residual flatness for every Cartesian derivative on bounded X regions | Infinite diagonal schedule preserving solenoidality; effective tail bounds and a common evaluation domain. |
| 15. Compact localization and force, §10; [R3CompactCandidate.of_localized_fields][compact], [R3CompactEnergy.energy_balance][energy] | Local field and vector potential, spatial/time cutoffs → whole-space velocity, pressure and smooth compact force | Initially resting flow; compact support; smooth extension of residual at the singular event | Choose localization neighborhoods and cutoffs; derive uniform derivative and energy bounds. Defining force as an arbitrary residual does not establish smooth extension. |
| 16. Viscosity, periodic consequence, theorem statements; [ComparatorSolution.navier_stokes_breakdown_R3 and navier_stokes_breakdown_periodic][solution] | Completed unit-viscosity whole-space field → every positive viscosity and periodic consequence | Scaling, period and support must be mutually compatible | Explicit rescaling once the completed object exists; theorem checking does not certify any Python translation. |

## Inner equations that must actually be solved

Write `A=1/2+h`, `D=1/2−h`, `d=1−η²`, `L=1−2hη²`, `E=sqrt(2X)φ/C`, and `B=(1/X)∫₀ˣ U(s,η)ds`. Equations (4.7)–(4.13) give

$$W=1-2D\eta B-d B_\eta,\qquad H_c=D\eta+dU,$$
$$2L(X\phi_{XX}+2\phi_X)=W(\phi+X\phi_X)+h(1-2\eta U)\phi+H_c\phi_\eta,$$
$$2L(XU_{XX}+U_X)=WXU_X+A(1-2\eta U)U+H_cU_\eta+d\Pi_\eta-4A\eta\Pi-2\eta X\Pi_X,$$
$$\Pi_X=\phi^2/C^2.$$

These equations contain parameter derivatives. Solving unrelated radial ODEs at independently sampled η values would omit those terms. Pressure is integrated **forward from the fixed axis datum**, while its overall normalization is determined by the outgoing profile.

The regular axis data in B.1 are `U(0,η)=4η+j₀`, `Π(0,η)=Π₀(η)`, and `φ(0,η)=exp(Λ∫₀^η ζ*(w)dw)`, where `ζ*=−LH*/(H*²+σ²)` and `H*=Dη+d(4η+j₀)`. The factor `C` converts φ to physical angular amplitude. It cannot be independently reset after pressure matching. The pressure datum itself is `Π₀(η)=−(1/2)∫ E_id,sched(y,η)²dy` over the full logarithmic radial line (A.21), not an arbitrary negative function.

A useful executable recurrence follows directly from these equations: for a series in X, the radial inverses divide the order-n angular and axial right-hand-side coefficients by `(n+1)(n+2)` and `(n+1)²`, respectively. The average divides `U_n` by `n+1`, and pressure integration divides the coefficient of `φ²/C²` by `n+1`. This identifies an implementation route, but convergence radii and derivative remainder estimates still have to be supplied.

The Appendix B scaled representation uses `Y=ΛX`, `φ=φ*Φ`, `U=U*+u/Λ` and `Π=Π₀+∫g²Φ²/Λ`, with `g=φ*/C`. Its leading profile is `f₀(Yχ)=Σ(-Yχ/2)^n/[n!(n+1)!]`, where `χ=H*²/(H*²+σ²)`. This leading approximation is **not** a replacement for the nonlinear solution. On `0≤Y≤4.1`, it provides the positive comparison profile needed by the fixed-point estimates.

Downstream evaluators need mixed X/η derivatives to the requested order, not just values. Physical derivatives use

$$T_b f={-bf+D\eta f_\eta+Xf_X\over L},\qquad Z_b f={2b\eta f+df_\eta-2\eta Xf_X\over L}.$$

With `V₀=(X/L)[2ηU−2DηB−dBη]` and `v₀=V₀/X`, the Cartesian leading velocity is

$$u_1={v_0x_1\over2q}-q^{-A-1/2}{\phi\over C}x_2,\quad
u_2={v_0x_2\over2q}+q^{-A-1/2}{\phi\over C}x_1,\quad
u_3=q^{-A}U.$$

The axis limit must be evaluated through the regular scalar coefficients; direct division by X or cylindrical radius at zero is invalid.

## Parameter order and admissibility

B.40 fixes outer data `(M_d,T_d,P*,λ,h)` before matching tolerance and `j₀`; then `δ,σ,Λ`; then transition derivative bounds and length; then `C,X_R`; then final cone and transition choices. Appendix C selects modulation after the nominal profile. The Lean schedule refines the outgoing step: obtain reset bounds before choosing λ, fix the core before h, and use the exact outgoing pressure datum in the axis construction. Neither source permits all parameters to be chosen independently.

The admissibility ledger must include, at minimum:

- Geometry and schedule: `0<h<.01`, the outgoing `λ<1/120`, `2h<λ`, all caller/reset/rate caps, and the scheduled wait `60 log(1/λ)` from [ScheduledProfileChoice][schedule]. Its explicit `heightCap` also includes the terminal release budget and caller's additional height cap.
- Axis: `0<j₀≤.05`, the sign/separation properties of `Z*` and `H*`, positive δ and σ, and `χ>.99` everywhere `|Z*|≤δ`; nonvanishing `L` and `H*²+σ²` on the chosen complex neighborhood; analytic pressure bounds; `Λ≥Λ₀`; `C` above the common complex amplitude bound.
- Continuation and matching: endpoint alternatives, positive continuation sources, admissible transition widths, invertible five-moment map, debt within its inverse neighborhood, and exact restoration of all five moment functions and their required derivatives.
- Leading theorem: `C>1`, `0<X_a<X_b`, pressure normalized at infinity, zero inner/exterior stress, positive angular amplitude, flat stress edges, preserved moments, reserved disjoint patches, and uniform cone margins. In the paper's notation, `nθ+t_s n_z≥κ` and `(v_s−2)(n_z−t_s nθ)²≤(2−κ)(nθ+t_s n_z)²` throughout the active region.
- Infinite stages: all coefficient/jet bounds, common supports/domain, positive pulse weights, small-divisor estimates, residual-improvement constants, and summation cutoffs with their differentiated tail bounds.

A reproducible selector must record each selected value, source location, dependencies, checked domain, arithmetic precision, rigorous margin or empirical estimate, and every unavailable bound. A finite sampled margin is not a uniform certificate. An unavailable bound must yield **unresolved**, never “admissible.” Halving λ until the easy scalar inequalities pass does not check the profile, matching, cone or infinite-stage conditions.

Two specific numerical blockers are now identified for [parameter selection #13](https://github.com/james-coder/navier-stokes-blowup/issues/13) and [inner profiles #14](https://github.com/james-coder/navier-stokes-blowup/issues/14):

1. `NaturalProfile.exists_profileFamily` obtains `Λ₀` from `AxisReference.exists_positive_scaled_profiles`. This is not an entirely unspecified constant: [AxisContraction.contractionThreshold](https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/AxisContraction.lean#L520) explicitly uses `1 + remainderBound + remainderLip`, on the ball of radius `norm(referencePair)+1`; the fixed-point error is at most `remainderBound/(2Λ)`. The positivity/slope condition adds a stability threshold. The missing numerical inputs are effective operator and analytic coefficient norm bounds on the actual pressure datum's complex neighborhood. A caller-supplied unexplained Lipschitz constant would merely move that obligation out of the implementation.
2. `NominalProfile` constructs the inner solution from `F.axisDatum` of the chosen outgoing schedule and then repairs its actual moment debt. The current library supplies neither that outgoing schedule nor its quantitative matching certificate. Substituting a convenient pressure and declaring the result the theorem's inner profile would be incorrect.

These are concrete missing numerical translations, not claims that the upstream existence proofs are incomplete or that constructive approximation is impossible. The [multiprecision reference](multiprecision.md) supplies arithmetic and independent checks for the currently explicit components; it does not resolve these choices.

[similarity]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/SimilarityProfile.lean#L327
[outgoing]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/OutgoingProfile.lean#L557
[schedule]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ScheduledProfileChoice.lean#L19
[axis-input]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/NaturalAxisCoefficients.lean#L567
[natural]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/NaturalProfile.lean#L603
[nominal]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/NominalProfile.lean
[moments]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/FiveProfileMoments.lean#L1137
[heat]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/RadialHeatProfile.lean#L695
[modulated]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ModulatedProfileAssembly.lean#L1126
[cone]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/AlignedProfileSpectralCone.lean#L665
[slow]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/GlobalSlowProfiles.lean
[borel]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/SlowBorelBase.lean#L1668
[parameters]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ActualCycleParameters.lean
[waves]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ActualSignedNativeProfiles.lean#L206
[cycle]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ActualCycleResidualBounds.lean#L1190
[diagonal]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/MixedDiagonalSchedule.lean#L301
[solenoidal]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/SolenoidalDiagonal.lean#L297
[compact]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/R3CompactCandidate.lean#L248
[energy]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/R3CompactEnergy.lean#L186
[solution]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ComparatorSolution.lean

[torus]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/TorusInverse.lean#L323
[mean]: https://github.com/openai/NavierStokesAndEuler/blob/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538/NavierStokes/ActualMeanStageData.lean#L384
