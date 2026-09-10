"""Export decimal multiprecision values and empirical convergence evidence."""
import argparse
import json
from pathlib import Path
import platform

import mpmath
from ns_blowup.reference import ComponentReference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/multiprecision.json')
    args = parser.parse_args()
    cases = [
        ('H_small', 'heat_factor', ['.001']),
        ('H_middle', 'heat_factor', ['1']),
        ('H_large', 'heat_factor', ['1000']),
        ('swirl', 'swirl', ['1.5', '.2']),
        ('pressure', 'pressure', ['1.5', '.2']),
        ('heat_residual', 'heat_residual', ['1.5', '.2']),
    ]
    runs = []
    for dps in [25, 50, 75]:
        ref = ComponentReference(dps=dps)
        m = ref.mp
        rows = []
        for name, operation, arguments in cases:
            value = getattr(ref, operation)(*arguments)
            row = {'name': name, 'operation': operation, 'decimal_arguments': arguments,
                   'value': m.nstr(value, dps)}
            if operation == 'heat_factor':
                other = ref.heat_factor(*arguments, method='quadrature')
                row['independent_quadrature_absolute_difference'] = m.nstr(abs(value-other), dps)
            rows.append(row)
        coords = ref.coordinates('1.5', '.5', '1e-12')
        runs.append({'decimal_digits': dps, 'values': rows,
                     'coordinates': {'radius': '1.5', 'z': '.5', 'tau': '1e-12',
                                     **{k: m.nstr(v, dps) for k, v in coords.items()}}})
        print(f'{dps} decimal digits complete', flush=True)
    m = ComponentReference(dps=90).mp
    for low, high in zip(runs, runs[1:]):
        for before, after in zip(low['values'], high['values']):
            after['absolute_change_from_previous_precision'] = m.nstr(
                abs(m.mpf(after['value'])-m.mpf(before['value'])), high['decimal_digits'])
    report = {'schema_version': 1, 'python': platform.python_version(), 'mpmath': mpmath.__version__,
              'parameters': {'h': '.005', 'nu': '1', 'c': '1'},
              'domain': 'tau>0, radius>0, Z>=0; coordinates also allow radius=0',
              'error_interpretation': 'Precision differences and representation agreement are empirical estimates, not certified bounds.',
              'not_implemented': 'Complete assembled smooth blowup field and its reference values.',
              'source': {'paper_sha256': '0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f',
                         'equations': '(4.1), Lemma A.6',
                         'lean_commit': '8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538'},
              'runs': runs}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(path)


if __name__ == '__main__':
    main()
