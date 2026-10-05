# Tests

Tests mirror the supported solver families:

- `fdtd_1d`: 1D solver and dispersion.
- `fdtd_2d/cartesian`: established Ez/Hz, compiled execution, and PEC cut cells.
- `fdtd_2d/gr`: Schwarzschild propagation and CUDA parity.
- `fdtd_3d`: 3D solver, dispersion, and CUDA parity.
- `common`: shared constitutive utilities and cross-solver checks.

Run from the repository root with plotting disabled:

```powershell
$env:MPLBACKEND = 'Agg'
python -m unittest discover -s tests -t .
```

Build optional kernels before testing compiled execution:

```powershell
python setup_cython.py build_ext --inplace
```

CUDA simulator checks run in separate processes. Set `FDTD_TEST_REAL_CUDA=1`
to enable tests requiring CUDA hardware.

The general 2D and GUI tests are available on the `dev` branch.
