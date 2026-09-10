# Independent multiprecision reference

`ComponentReference` evaluates the supported similarity coordinates and heat exterior on the CPU using [mpmath](https://mpmath.org/doc/current/). It imports no production numerical routines. Each instance has a separate precision context; use decimal strings for inputs so Python does not round them to binary64 first.

```bash
python -m pip install 'ns-blowup[reference]'
```

```python
from ns_blowup.reference import ComponentReference

reference = ComponentReference(dps=75, h="0.005", nu="1", c="1")
print(reference.heat_factor("1"))
print(reference.velocity(["1.5", "0", "0"], tau="0.2"))
print(reference.pressure("1.5", "0.2"))
print(reference.coordinates("1.5", "0.5", "1e-12"))
```

The formulas come from equation (4.1) and Lemma A.6 of the [pinned paper](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf), whose hash is recorded in the [source manifest](source-manifest.json). This reference uses three independently evaluable representations:

| Component | Reference representation | Check |
|---|---|---|
| Similarity root `q − z²q^(2h) = τ` | Bracketed bisection above `max(τ, abs(z)^(1/(1/2−h)))` | Substitution into the implicit equation at increasing precision |
| Heat factor `H(Z)` | `Z^(-1-h) U(1+h, 2, 1/Z)` for `Z>0`, `H(0)=1` | Adaptive integration of the defining gamma-weighted improper integral |
| Azimuthal velocity | Heat factor with explicit viscosity scaling | Independently differentiated azimuthal heat-equation residual |
| Pressure | Adaptive integration of `−K(s)²/s` from radius to infinity | Radial pressure balance by centered differences; decay toward zero at large radius |

The heat-factor identity follows by substituting `t=Zv` into [Tricomi U's defining integral](https://mpmath.org/doc/current/functions/bessel.html#hyperu). Unlike the production fixed Gaussian quadrature, mpmath uses its special-function implementation or [adaptive quadrature](https://mpmath.org/doc/current/calculus/integration.html). Both representations remain numerical approximations.

The [decimal reference dataset](multiprecision.json) records results at 25, 50, and 75 decimal digits, inputs, versions, source pins, representation discrepancies, and changes with increasing precision. Reproduce it with [the export script](https://github.com/james-coder/navier-stokes-blowup/blob/main/scripts/export_reference.py):

```bash
python scripts/export_reference.py --output outputs/multiprecision.json
```

Precision convergence and independent representation agreement are **empirical error estimates**, not certified interval bounds. The tests exercise small/large `Z`, positive and negative axial coordinates, small positive time remaining, nonunit viscosity, and the radial pressure identity. The supported domain is finite `0<h<0.01`, `nu>0`, `c>0`, `tau>0`, `Z>=0`, and positive radius for exterior fields. Coordinates additionally allow radius zero. The exterior is singular on the axis even before the proposed singular time.

**Not Implemented:** the complete assembled smooth blowup field, parameter certification, and its multiprecision dataset. Future assembled fields should use this same decimal representation and per-instance precision, but must add their actual profiles, matching conditions, truncation errors and domain before exporting reference values. No exterior-only dataset may be described as that full field.
