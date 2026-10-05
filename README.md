# FDTD

This is the `dev` branch. It includes the [general 2D solver](FDTD_2D/README.md) and [FDTD Studio GUI](FDTD_2D/gui/README.md), alongside the established solvers.

Finite-Difference Time-Domain solvers for computational electromagnetics.
The `main` branch provides the established Cartesian 1D, 2D Ez, 2D Hz, and 3D
solvers with their stable top-level Python imports.

## Choose a solver

| Solver | Public import | Documentation |
|---|---|---|
| 1D: Ey, Hx | `from FDTD_1D import FDTD_1D` | [1D guide](FDTD_1D/README.md) |
| 2D TMz: Ez, Hx, Hy | `from FDTD_2D_Ez import FDTD_2D_Ez` | [Ez guide](FDTD_2D_Ez/README.md) |
| 2D TEz: Hz, Ex, Ey | `from FDTD_2D_Hz import FDTD_2D_Hz` | [Hz guide](FDTD_2D_Hz/README.md) |
| 3D: all six components | `from FDTD_3D import FDTD_3D` | [3D guide](FDTD_3D/README.md) |

The Cartesian solvers support material-first geometry, anisotropy and loss,
Debye/Drude/Lorentz dispersion, PEC/PMC boundaries, sources, monitors, and
scattering analysis. Ez and Hz use CPML and conformal PEC cut cells.

The separate [Schwarzschild GR solver](FDTD_2D_GR/README.md) remains available
through `from FDTD_2D_GR import FDTD_2D_GR`. It uses geometric units and a
dedicated polar API. The earlier UPML Ez implementation is retained under
[legacy](legacy/README.md).

## Install and build

Run from the repository root:

```powershell
python -m pip install -r requirements-dev.txt
python setup_cython.py build_ext --inplace
```

The Cython build requires a supported C compiler. Cartesian Ez/Hz
`config("cpu")` and hardware `config("gpu")` require the compiled 2D runtime.
Use `config("python")` explicitly for their reference loops. GPU execution
additionally requires a working CUDA runtime.

See [shared conventions](FDTD_common/docs/general.md) and the
[compiled 2D runtime guide](FDTD_common/docs/compiled_2d/compiled_2d.md).

## Examples and tests

```powershell
python -m FDTD_1D.examples.FDTD_1D_example
python -m FDTD_2D_Ez.examples.Example_1_Simple_Source
python -m FDTD_2D_Hz.examples.Example_1_Simple_Source
python -m FDTD_3D.examples.Example_3D --no-show

$env:MPLBACKEND = 'Agg'
python -m unittest discover -s tests -t .
```

Each solver contains its own `docs/` and `examples/`. Tests are organized under
[tests](tests/README.md), and benchmark scripts and recorded reports under
[benchmarks](benchmarks/README.md). Generated results are kept in ignored
`output/` directories. Supplied lecture PDFs are indexed under
[reference literature](FDTD_common/docs/references/README.md).

## Development branch

The geometry-first general `FDTD_2D` solver and Qt/VTK FDTD Studio GUI are
maintained on the separate `dev` branch. The established solvers above are also
available there.

```powershell
git switch dev
python -m pip install -r FDTD_2D/requirements.txt
python -m FDTD_2D.examples.example_general
```

On `dev`, see `FDTD_2D/README.md` and `FDTD_2D/gui/README.md` for the general
solver and GUI. Return to the production checkout with `git switch main`.

## License

See [LICENSE](LICENSE).
