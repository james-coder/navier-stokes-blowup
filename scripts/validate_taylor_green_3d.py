"""Reproduce the independent Re=20 3D Taylor–Green comparison (CPU, float64).

Requires scipy in addition to the package. Run with JAX_ENABLE_X64=1 and
OPENBLAS_NUM_THREADS=1. The reference uses NumPy rotational-form advection and
SciPy DOP853; it does not call the library's grid, initial data, RHS or integrator.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.signal import resample


def reference(n, times, nu=.05):
    axes = (0, 1, 2)
    fft = lambda u: np.fft.fftn(u, axes=axes)
    ifft = lambda u: np.fft.ifftn(u, axes=axes).real
    x, y, z = np.meshgrid(*(np.arange(n)*2*np.pi/n for _ in axes), indexing='ij')
    u0 = np.stack((np.sin(x)*np.cos(y)*np.cos(z),
                   -np.cos(x)*np.sin(y)*np.cos(z), np.zeros_like(z)), axis=-1)
    frequencies = np.fft.fftfreq(n)*n
    k = np.stack(np.meshgrid(frequencies, frequencies, frequencies, indexing='ij'), axis=-1)
    k2 = np.sum(k*k, axis=-1, keepdims=True)
    mask = np.all(np.abs(k) < n/3, axis=-1, keepdims=True)
    def rhs(t, flat):
        u = flat.reshape(u0.shape)
        uhat = fft(u)*mask
        omega = ifft(1j*np.cross(k, uhat))
        acceleration = fft(np.cross(ifft(uhat), omega))*mask
        longitudinal = np.sum(k*acceleration, axis=-1, keepdims=True)/np.where(k2 == 0, 1, k2)
        return ifft(acceleration-k*longitudinal-nu*k2*uhat).ravel()
    solution = solve_ivp(rhs, (times[0], times[-1]), u0.ravel(), method='DOP853',
                         t_eval=times, rtol=1e-11, atol=1e-13, max_step=.02)
    if not solution.success:
        raise RuntimeError(solution.message)
    velocities = solution.y.T.reshape((len(times),)+u0.shape)
    metrics = statistics(velocities)
    return velocities, {**metrics, 'resolution': n, 'rhs_evaluations': solution.nfev,
                        'method': 'NumPy rotational form + SciPy DOP853',
                        'rtol': 1e-11, 'atol': 1e-13, 'max_dt': .02}


def statistics(velocities):
    n = velocities.shape[1]
    frequency = np.fft.fftfreq(n)*n
    k = np.stack(np.meshgrid(frequency, frequency, frequency, indexing='ij'), axis=-1)
    omega = np.fft.ifftn(1j*np.cross(k, np.fft.fftn(velocities, axes=(1, 2, 3))), axes=(1, 2, 3)).real
    return {'mean_energy': (.5*np.mean(np.sum(velocities**2, axis=-1), axis=(1, 2, 3))).tolist(),
            'mean_enstrophy': (.5*np.mean(np.sum(omega**2, axis=-1), axis=(1, 2, 3))).tolist()}


def main():
    import jax
    from ns_blowup import Simulation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/taylor-green-3d.json')
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise RuntimeError('Set JAX_ENABLE_X64=1 for this benchmark')
    times = np.linspace(0, 1, 5)
    refs = []
    for n in [32, 48]:
        fields, metrics = reference(n, times)
        refs.append(metrics)
        print(f'Independent reference {n}³ complete', flush=True)
    runs = []
    for n in [12, 18, 24]:
        result = Simulation((n, n, n), nu=.05, dtype='float64', device='cpu').run(
            'taylor-green-3d', t_end=1, frames=len(times), dt=.005, rtol=1e-9, atol=1e-11)
        sampled = fields[-1]
        for axis in range(3):
            sampled = resample(sampled, n, axis=axis)
        metrics = statistics(np.asarray(result.velocity))
        runs.append({**metrics, 'resolution': n, 'metadata': result.metadata,
                     'final_velocity_rms_error': float(np.sqrt(np.mean((np.asarray(result.final)-sampled)**2))),
                     'max_energy_error': float(np.max(np.abs(np.array(metrics['mean_energy'])-refs[-1]['mean_energy']))),
                     'max_enstrophy_error': float(np.max(np.abs(np.array(metrics['mean_enstrophy'])-refs[-1]['mean_enstrophy'])))})
        print(f'JAX {n}³ complete: RMS error {runs[-1]["final_velocity_rms_error"]:.3g}', flush=True)
    report = {'schema_version': 1, 'reynolds_number': 20, 'nu': .05, 'amplitude': 1,
              'wavenumber': 1, 'domain': '[0,2*pi)^3 periodic', 'times': times.tolist(),
              'precision': 'float64', 'python': platform.python_version(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'jax': jax.__version__,
              'reference_runs': refs, 'jax_runs': runs,
              'scope': 'Resolved short-time Re=20 validation; no high-Re DNS claim.'}
    for key in ['mean_energy', 'mean_enstrophy']:
        report[f'reference_refinement_max_{key}_difference'] = float(np.max(np.abs(
            np.array(refs[0][key])-refs[1][key])))
    errors = [run['final_velocity_rms_error'] for run in runs]
    assert errors[2] < errors[1] < errors[0], errors
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(path)


if __name__ == '__main__':
    main()
