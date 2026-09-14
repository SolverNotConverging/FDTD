# Native 2D progress measurements

Measured on Windows with an RTX 4070 Laptop GPU. Each case warms both modes,
then alternates progress off/on over five repetitions. The table reports median
stepping time, including native terminal reporting, with stderr redirected to a
file. Grids are 120², with one line monitor and no field histories. Setup,
event/timer creation, and downloads are outside these stepping timings.

| Solver | Backend | Material | Steps | Off (ms) | On (ms) | Change |
|---|---|---|---:|---:|---:|---:|
| Ez | cpu | ordinary | 2000 | 519.06 | 513.64 | -1.05% |
| Ez | cpu | mixed | 2000 | 794.63 | 795.06 | +0.05% |
| Hz | cpu | ordinary | 2000 | 524.90 | 513.82 | -2.11% |
| Hz | cpu | mixed | 2000 | 1118.47 | 1118.06 | -0.04% |
| Ez | gpu | ordinary | 2000 | 51.09 | 52.20 | +2.18% |
| Ez | gpu | mixed | 2000 | 70.74 | 71.86 | +1.59% |
| Hz | gpu | ordinary | 2000 | 54.85 | 55.39 | +0.99% |
| Hz | gpu | mixed | 2000 | 99.42 | 100.62 | +1.20% |
| Ez | gpu | ordinary | 20000 | 495.34 | 496.00 | +0.13% |
| Ez | gpu | mixed | 20000 | 691.14 | 691.96 | +0.12% |
| Hz | gpu | ordinary | 20000 | 528.54 | 529.33 | +0.15% |
| Hz | gpu | mixed | 20000 | 974.07 | 974.86 | +0.08% |

CPU differences are within the observed run-to-run variation; negative values
are measurement variation, not a speedup from reporting. Short GPU runs add
approximately 0.5–1.2 ms; longer GPU runs add 0.08–0.15% in these cases.
Terminal implementations and workloads can change the cost. Progress is off
by default for quiet execution and minimum overhead.

The native Windows polling wait uses a preallocated high-resolution timer.
The initial ordinary `Sleep(1)` implementation added up to a coarse Windows
scheduler tick to short runs and was replaced before these measurements.

The captured long-run output contains 66 intermediate GPU updates
and 24 completed bars. Completed counts are monotonic within each run.
CPU/GPU tests also verify state parity, no Python callbacks or Numba transfers/
allocations during native stepping, quiet defaults, and event cleanup on failure.

[Raw samples](compiled_2d_progress_benchmark.json). Reproduce with:

```powershell
python benchmarks/benchmark_2d_progress.py --cells 120 --steps 2000 --repeats 5
python benchmarks/benchmark_2d_progress.py --cells 120 --steps 20000 --repeats 5 --backends gpu
```
