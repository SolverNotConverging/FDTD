# General 2D FDTD development solver

This geometry-first, nonuniform conformal solver is developed on the `dev` branch.
The established `FDTD_2D_Ez` and `FDTD_2D_Hz` packages remain at the root
on both `main` and `dev`.
It supports both TE/Hz and TM/Ez polarization through one public API.

```python
from FDTD_2D import FDTD2D, Scene, MeshPolicy
```

Install dependencies and run examples from the repository root:

```powershell
python -m pip install -r FDTD_2D/requirements.txt
python -m FDTD_2D.examples.example_general
python -m FDTD_2D.examples.example_waveguide
python -m FDTD_2D.examples.example_scattering
```

- [Solver guide](docs/general_2d.md)
- [FDTD Studio GUI](gui/README.md)
- [Schwarzschild GR solver](../FDTD_2D_GR/README.md)
- [Established CPML Ez solver](../FDTD_2D_Ez/README.md)
- [Established CPML Hz solver](../FDTD_2D_Hz/README.md)
- [Historical UPML implementation](../legacy/README.md)

The general solver remains a CPU reference implementation. The established Cartesian
CPML solvers retain their separate Cython and CUDA acceleration.
