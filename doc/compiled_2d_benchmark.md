# Cartesian 2D compiled-runtime measurements

Measured on 2026-09-14 in `RF_Engineering_env`: Python 3.12.11, NumPy 2.3.2, Numba 0.65.1, Windows WDDM, NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB).

Each result is the median of three runs after one warm-up, using 100 timesteps. The scene has anisotropic dielectric material with electric loss, an eight-cell PML, a Gaussian point source, and a line monitor. Mixed material adds one Debye, one Drude, and one Lorentz pole. Geometry construction is outside these `run()` timings.

The baseline is revision `f41e5d596d407779fb9cf05263430d95380a788e`. Its CPU backend compiles curls but retains Python/NumPy stepping. Its dispersive GPU selection falls back to host loops; the large dispersive GPU speedups below reflect moving that work to the device. These are machine/workload measurements, not guaranteed speedups.

Unchanged cases use the full matrix measurement. Nondispersive TE cases use the latest paired baseline/compiled pass after finalization was fused. This pass also checks the small-grid result under the same current machine conditions.

## Stepping comparison

Times are milliseconds for all 100 steps. Each entry shows baseline → compiled (speedup).

| Solver | Grid | Material | CPU stepping ms | GPU stepping ms |
|---|---:|---|---:|---:|
| Ez | 120² | ordinary | 28.40 → 25.75 (1.10×) | 44.38 → 4.52 (9.82×) |
| Ez | 120² | mixed | 51.23 → 39.41 (1.30×) | 1672.27 → 5.70 (293.32×) |
| Ez | 256² | ordinary | 170.37 → 115.86 (1.47×) | 70.07 → 14.01 (5.00×) |
| Ez | 256² | mixed | 374.39 → 184.66 (2.03×) | 7726.99 → 18.48 (418.15×) |
| Ez | 512² | ordinary | 2310.93 → 550.51 (4.20×) | 95.90 → 57.33 (1.67×) |
| Ez | 512² | mixed | 4405.78 → 824.52 (5.34×) | 34400.24 → 71.41 (481.74×) |
| Hz | 120² | ordinary | 25.53 → 25.74 (0.99×) | 38.95 → 4.23 (9.21×) |
| Hz | 120² | mixed | 75.01 → 54.11 (1.39×) | 1714.27 → 6.50 (263.91×) |
| Hz | 256² | ordinary | 161.62 → 121.67 (1.33×) | 39.15 → 34.99 (1.12×) |
| Hz | 256² | mixed | 629.28 → 264.88 (2.38×) | 7899.71 → 25.20 (313.52×) |
| Hz | 512² | ordinary | 2150.26 → 551.89 (3.90×) | 89.56 → 60.53 (1.48×) |
| Hz | 512² | mixed | 6244.51 → 1202.55 (5.19×) | 36165.09 → 94.57 (382.40×) |

## Complete run phases

The smallest nondispersive TE CPU case is effectively unchanged: 25.53 ms versus
25.74 ms for 100 steps in the paired check. Larger CPU cases and the GPU stepping
cases improve. Setup and cleanup can still dominate short GPU runs.

All times are milliseconds. Setup includes packing and GPU upload/graph preparation. Total also includes validation and resource cleanup. Independently measured medians need not add to the total. GPU buffer size counts resident simulation/source/output arrays, excluding driver overhead and the memory reserve; it is not process peak memory.

| Solver/grid/material | Backend | Setup | Step | Download | Post-process | Total | Baseline total | GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ez/120²/ordinary | cpu | 1.20 | 25.75 | 0.00 | 0.05 | 27.05 | 29.70 | — |
| Ez/120²/ordinary | gpu | 9.74 | 4.52 | 0.95 | 0.04 | 15.98 | 56.29 | 4.43 |
| Ez/120²/mixed | cpu | 1.92 | 39.41 | 0.00 | 0.14 | 41.47 | 53.08 | — |
| Ez/120²/mixed | gpu | 11.34 | 5.70 | 0.89 | 0.19 | 19.01 | 1674.21 | 5.66 |
| Ez/256²/ordinary | cpu | 8.17 | 115.86 | 0.00 | 0.10 | 125.13 | 177.45 | — |
| Ez/256²/ordinary | gpu | 18.13 | 14.01 | 2.04 | 0.09 | 35.76 | 105.14 | 19.39 |
| Ez/256²/mixed | cpu | 12.60 | 184.66 | 0.00 | 0.30 | 198.29 | 384.11 | — |
| Ez/256²/mixed | gpu | 23.27 | 18.48 | 2.35 | 0.32 | 46.29 | 7737.95 | 24.93 |
| Ez/512²/ordinary | cpu | 58.36 | 550.51 | 0.00 | 0.19 | 609.31 | 2371.78 | — |
| Ez/512²/ordinary | gpu | 77.67 | 57.33 | 4.63 | 0.16 | 145.76 | 187.52 | 76.15 |
| Ez/512²/mixed | cpu | 87.46 | 824.52 | 0.00 | 0.94 | 912.89 | 4495.66 | — |
| Ez/512²/mixed | gpu | 111.53 | 71.41 | 5.41 | 1.08 | 195.80 | 34486.85 | 98.24 |
| Hz/120²/ordinary | cpu | 0.95 | 25.74 | 0.00 | 0.06 | 26.78 | 26.35 | — |
| Hz/120²/ordinary | gpu | 8.21 | 4.23 | 0.70 | 0.05 | 13.88 | 48.67 | 4.30 |
| Hz/120²/mixed | cpu | 2.51 | 54.11 | 0.00 | 0.19 | 56.88 | 77.67 | — |
| Hz/120²/mixed | gpu | 11.18 | 6.50 | 1.22 | 0.18 | 19.97 | 1716.56 | 6.73 |
| Hz/256²/ordinary | cpu | 5.87 | 121.67 | 0.00 | 0.13 | 127.84 | 167.50 | — |
| Hz/256²/ordinary | gpu | 18.93 | 34.99 | 2.46 | 0.10 | 58.11 | 58.69 | 18.82 |
| Hz/256²/mixed | cpu | 12.56 | 264.88 | 0.00 | 0.64 | 278.40 | 639.63 | — |
| Hz/256²/mixed | gpu | 26.39 | 25.20 | 2.40 | 0.55 | 56.86 | 7909.33 | 29.87 |
| Hz/512²/ordinary | cpu | 45.57 | 551.89 | 0.00 | 0.21 | 597.81 | 2195.45 | — |
| Hz/512²/ordinary | gpu | 67.13 | 60.53 | 4.41 | 0.17 | 136.31 | 157.80 | 74.02 |
| Hz/512²/mixed | cpu | 85.01 | 1202.55 | 0.00 | 1.80 | 1289.91 | 6323.44 | — |
| Hz/512²/mixed | gpu | 212.10 | 94.57 | 10.78 | 4.18 | 332.83 | 36247.20 | 118.11 |

## Large histories

512² grids, 100 timesteps, recording every five steps (20 full-field frames), plus the line monitor. Outputs remain resident until the run completes.

| Solver | Material | GPU MiB | Step ms | Download ms | Total ms |
|---|---|---:|---:|---:|---:|
| Ez | ordinary | 236.62 | 61.13 | 63.01 | 307.78 |
| Ez | mixed | 258.71 | 73.57 | 51.16 | 317.93 |
| Hz | ordinary | 194.18 | 59.60 | 35.11 | 203.55 |
| Hz | mixed | 238.27 | 95.61 | 41.73 | 335.30 |

## Verification and reproduction

The full suite passed: **111 tests**, including real CUDA graph tests and CUDA simulator checks. Tests compare final fields, flux/displacement, CPML, ADE histories, monitors, recording, and checkpoint continuation. Independent driver/dispatcher hooks verify no Numba allocations, transfers, or Python launches during native graph replay. CPU profiling verifies no Python calls inside the compiled stepping entry point. Graph validation permits only kernel nodes.

Performance review found uncoalesced GPU reads and redundant nondispersive TE finalization passes. CUDA x threads now traverse contiguous columns, and TE finalization is fused into the nondispersive E update. The affected cases were remeasured after parity and regression checks.

Use `benchmarks/benchmark_2d_compiled.py` to reproduce or change grid sizes, step count, backend, polarization, material, and recording cadence. See the [runtime guide](compiled_2d.md) and [raw measurements](compiled_2d_benchmark.json).
