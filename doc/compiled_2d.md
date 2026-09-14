# Compiled Cartesian 2D execution

`FDTD_2D_Ez` and `FDTD_2D_Hz` keep their material, geometry, source, monitor,
plotting, and checkpoint APIs. Only execution and backend selection change.

```python
sim.config("cpu")  # complete Cython loop
# or sim.config("gpu")  # Numba CUDA kernels + Cython CUDA graph driver
sim.run(record_stride=8, is_include_history=False, progress=True)
print(sim._runtime_stats)
```

Build with `python setup_cython.py build_ext --inplace`. Both accelerated
backends require `FDTD_common._compiled_2d`. The CPU extension builds without
CUDA headers or libraries. GPU execution additionally needs Numba CUDA, its
compatible CUDA compilation/runtime dependencies, and a driver exposing CUDA
graph APIs (CUDA 11.4 or newer). The driver is loaded dynamically. This path
uses ordinary kernels and works with Windows WDDM; it needs no cooperative grid.

Construction and checkpoint loading do not require acceleration. CPU remains
the default requested backend; explicit `config("cpu")`, `config("gpu")`, and
accelerated `run()` validate availability. Errors do not change the requested
backend to NumPy. Select `config("python")` explicitly for reference execution
or individual curl/update methods. Custom Python substep/waveform overrides
are not called by the compiled loop; use the supported source setup APIs.

## Execution boundary

Python builds coefficients, sparse source events, modal drive tables, packed
ADE channels, and monitor/history buffers. Cython binds typed buffers once,
then executes every timestep under `nogil`. GPU setup specializes Numba kernels
before capturing one timestep on a dedicated stream. A device counter drives
source timing, monitor windows, and recording cadence. Cython replays the graph
and waits for completion without Python calls.

`run(..., progress=True)` enables a native terminal bar showing completed steps,
percentage, and elapsed stepping time. It is off by default. Updates are limited
to five per second, plus the initial and final display; CPU timing is checked
every 64 steps. GPU progress queries at most 128 preallocated, timing-disabled
[CUDA completion events](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html),
spaced at least 1024 steps apart, outside the timestep graph. It reports completed
work and introduces no field copies, Python callbacks, or batch synchronization.
Events are released with the graph, including on failure. The bar starts after
setup/compilation and reaches 100% after stepping; downloads and post-processing
follow. Updates can be coarser when individual steps are expensive.

Native output goes to process stderr (a terminal or redirected file);
`contextlib.redirect_stderr` and notebook output widgets do not capture C stdio.
Reference mode and the CUDA simulator use throttled `tqdm` when enabled.
Use `progress=False` for quiet execution and minimum overhead, especially for
very short runs. Measure your workload with `benchmarks/benchmark_2d_progress.py`.
The [progress measurements](compiled_2d_progress_benchmark.md) report CPU/GPU
overhead separately, including short runs and longer runs with live updates.

Graph instantiation rejects any captured node that is not a CUDA kernel,
including memory copies, allocation nodes, and host callbacks.

All field updates, CPML memories, loss, PEC/PMC masks, Debye/Drude/Lorentz poles,
source injection, monitor interpolation, and history recording execute in the
selected backend. Updates retain float64 precision and existing Yee/source
ordering. Mixed poles are retained separately, and electric soft sources enter
the trial displacement before the single coupled ADE solve.

Final fields, derivatives, flux/displacement, CPML, ADE state, monitor results,
and requested histories are synchronized before `run()` returns. Checkpoints
contain host arrays and no CUDA handles. Repeated runs retain existing field
and material history; their source clock and recording indices restart at zero,
as before this migration. Changing setup between completed runs is supported.

## Recording and memory

The complete GPU run must fit in device memory, including source tables,
monitor samples, and all requested field histories. Preflight sums buffer sizes
and reserves the greater of 64 MiB or 5% for runtime overhead. All buffers are
allocated before the first timestep; memory errors leave fields unadvanced.
Use a larger `record_stride`, shorter monitor windows, or
`is_include_history=False` to reduce recording. There is no automatic streaming
or host fallback. Host output arrays also require sufficient system memory.

`_runtime_stats` reports host setup, device setup, stepping, download, and
post-processing durations. `_gpu_transfer_stats` counts operations at the
runtime's allocation/copy boundary and includes buffer bytes and source/monitor
counts. Separate real-GPU tests patch the actual Numba driver/dispatcher hooks
during replay to enforce the zero-operation boundary independently.

## Validation and measurement

```powershell
python setup_cython.py build_ext --inplace
$env:FDTD_TEST_REAL_CUDA = "1"
python -m unittest discover -s tests -v
python benchmarks/benchmark_2d_compiled.py --steps 100 --repeats 3
```

The CUDA simulator uses an explicit Python launch harness for kernel correctness;
it cannot exercise driver graphs. Its statistics set `simulator=True` and report
the Python dispatches. Real GPU tests cover graphs, resource cleanup, actual
transfer/dispatch hooks, memory failure, and checkpoint continuation.

The benchmark warms each case and reports medians for 120², 256², and 512² grids,
both polarizations, nondispersive and mixed-pole media. By default it records a
line monitor; `--history-stride 5` also measures full-field recording. Optional
`--baseline-dir` accepts the three saved pre-migration solver/runtime files.
For the baseline, the final phase includes both downloading and post-processing;
the new runtime reports these separately. Timings are workload and machine dependent.

The [measured results](compiled_2d_benchmark.md) include the original implementation,
the final compiled runtimes, phase timings, and large-history memory costs.
