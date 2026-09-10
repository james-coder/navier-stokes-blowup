# Actual installation transcript

CPU, PyPI ns-blowup 0.1.0. Captured by ScreenCLI 0.3.12.

## Create a clean Python environment

```console
$ python3 -m venv venv

Exit status: 0
```

## Install ns-blowup 0.1.0 from PyPI — CPU

```console
$ venv/bin/python -m pip install --quiet --disable-pip-version-check "ns-blowup[plot]==0.1.0"

Exit status: 0
```

## Verify Python, package and device

```console
$ venv/bin/python -c "import sys, ns_blowup, jax; print(sys.version); print(\"ns-blowup\", ns_blowup.__version__); print(\"JAX\", jax.__version__); print(\"Devices:\", jax.devices())"
3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]
ns-blowup 0.1.0
JAX 0.11.1
Devices: [CpuDevice(id=0)]

Exit status: 0
```

## Simulate the 2D Taylor–Green vortex and export results

```console
$ venv/bin/python demo.py
Velocity shape: (21, 32, 32, 2)
Device: cpu:0
Final energy: 8.080427
NPZ: flow.npz
VTK: flow.vtk
Plot: vorticity.png

Exit status: 0
```

## Generate the offline Observatory

```console
$ venv/bin/python -m ns_blowup.observatory --output observatory.html
observatory.html

Exit status: 0
```
