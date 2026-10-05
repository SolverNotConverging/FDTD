# Benchmarks

Cartesian CPML benchmarks and recorded reports live in `fdtd_2d`.
Run from the repository root:

```powershell
python benchmarks/fdtd_2d/benchmark_2d_compiled.py --backends cpu
python benchmarks/fdtd_2d/benchmark_2d_progress.py --backends cpu
```

Build the Cython extensions before running these benchmarks. Recorded reports
describe the revision and environment in which their timings were collected.
