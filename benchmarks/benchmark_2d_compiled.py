"""Benchmark complete 2D runtimes, optionally against saved pre-migration files.

Run from the repository root, for example:
    python benchmarks/benchmark_2d_compiled.py --steps 100 --repeats 3 \
        --baseline-dir build/compiled2d_baseline --output build/compiled2d_benchmark.json

The baseline directory must contain FDTD_2D_Ez.py, FDTD_2D_Hz.py and cuda_2d.py
from the same old revision. Each case warms once and reports median timings.
"""
import argparse
import contextlib
import copy
import importlib
import importlib.util
import importlib.metadata
import io
import json
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import cuda
from FDTD_2D_Ez import FDTD_2D_Ez
from FDTD_2D_Hz import FDTD_2D_Hz
from FDTD_common import runtime_2d


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_sim(solver, backend, cells, steps, dispersive):
    sim = solver(12e-3, 12e-3, cells, cells, 100e9, steps,
                 dt=1e-14, subpixel=1).config(backend)
    poles = dict(debye=dict(delta_epsilon=0.5, tau=4e-12),
                 drude=dict(omega_p=2e11, gamma=1e10),
                 lorentz=dict(delta_epsilon=0.3, omega_0=3e11, gamma=2e10)) if dispersive else {}
    sim.add_material('medium', epsilon_r=(2.0, 2.5, 3.0), sigma_e=0.01, **poles)
    sim.add_rectangle(material='medium', x_position=(cells // 4, 3 * cells // 4),
                       y_position=(cells // 4, 3 * cells // 4))
    sim.add_PML(8)
    sim.add_source('point', x=cells // 2, y=cells // 2,
                   t0=5e-14, tw=5e-14, is_show=False)
    sim.add_line_monitor(x=(0, cells), y=cells // 2, index=1)
    return sim


def benchmark_case(solver, module, old_cuda, backend, cells, steps, dispersive, repeats, history_stride):
    samples = []
    template = make_sim(solver, backend, cells, steps, dispersive)
    initial = template.state_dict()
    for repetition in range(repeats + 1):
        # Repaint geometry once; restore independent identical initial fields
        # for every timed run. Model construction is outside these run timings.
        sim = solver.__new__(solver)
        sim.load_state_dict(copy.deepcopy(initial))
        timing = {}
        if module is not None:
            def timed_steps(items, **kwargs):
                if backend == 'gpu':
                    cuda.synchronize()
                timing['loop_start'] = time.perf_counter()
                for item in items:
                    yield item
                if backend == 'gpu':
                    cuda.synchronize()
                timing['loop_end'] = time.perf_counter()
            module.tqdm = old_cuda.tqdm = timed_steps
        with contextlib.ExitStack() as stack:
            if module is not None:
                import FDTD_common
                current = importlib.import_module('FDTD_common.cuda_2d')
                sys.modules['FDTD_common.cuda_2d'] = old_cuda
                FDTD_common.cuda_2d = old_cuda
                def restore():
                    sys.modules['FDTD_common.cuda_2d'] = current
                    FDTD_common.cuda_2d = current
                stack.callback(restore)
            with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                start = time.perf_counter()
                sim.run(record_stride=history_stride or 1,
                        is_include_history=bool(history_stride))
                end = time.perf_counter()
        for name in ('Ez', 'Hx', 'Hy') if hasattr(sim, 'Ez') else ('Hz', 'Ex', 'Ey'):
            if not np.isfinite(getattr(sim, name)).all():
                raise RuntimeError(f'Non-finite {name} in benchmark case.')
        if module is None:
            stats = sim._runtime_stats
            sample = dict(setup_seconds=stats['setup_seconds'] + stats.get('device_setup_seconds', 0),
                          stepping_seconds=stats['stepping_seconds'],
                          download_seconds=stats['download_seconds'],
                          postprocess_seconds=stats['postprocess_seconds'],
                          total_seconds=end - start,
                          device_buffer_bytes=stats.get('buffer_bytes', 0))
        else:
            sample = dict(setup_seconds=timing['loop_start'] - start,
                          stepping_seconds=timing['loop_end'] - timing['loop_start'],
                          download_seconds=end - timing['loop_end'],
                          postprocess_seconds=0.0, total_seconds=end - start,
                          device_buffer_bytes=None)
        if repetition:
            samples.append(sample)
    return {key: (None if samples[0][key] is None else statistics.median(s[key] for s in samples))
            for key in samples[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cells', type=int, nargs='+', default=[120, 256, 512])
    parser.add_argument('--backends', nargs='+', default=['cpu', 'gpu'], choices=['cpu', 'gpu'])
    parser.add_argument('--solvers', nargs='+', default=['ez', 'hz'], choices=['ez', 'hz'])
    parser.add_argument('--materials', nargs='+', default=['ordinary', 'mixed'], choices=['ordinary', 'mixed'])
    parser.add_argument('--history-stride', type=int, default=0, help='0 records line monitors only')
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/compiled2d_benchmark.json')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('steps and repeats must be positive')
    old_cuda = (load_module('FDTD_common._benchmark_old_cuda', args.baseline_dir / 'cuda_2d.py')
                if args.baseline_dir else None)
    variants = []
    for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
        if ('ez' if solver is FDTD_2D_Ez else 'hz') not in args.solvers:
            continue
        name = solver.__name__
        if args.baseline_dir:
            module = load_module(name + '._benchmark_old', args.baseline_dir / (name + '.py'))
            variants.append(('baseline', getattr(module, name), module))
        variants.append(('compiled', solver, None))
    result = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                  numpy=np.__version__, numba=importlib.metadata.version('numba'),
                  gpu=str(cuda.get_current_device().name) if 'gpu' in args.backends else None,
                  steps=args.steps, repeats=args.repeats, history_stride=args.history_stride, cases=[])
    for variant, solver, module in variants:
        for cells in args.cells:
            for dispersive in (False, True):
                if ('mixed' if dispersive else 'ordinary') not in args.materials:
                    continue
                for backend in args.backends:
                    case = dict(variant=variant, solver=solver.__name__, cells=cells,
                                material='mixed' if dispersive else 'ordinary', backend=backend)
                    case.update(benchmark_case(solver, module, old_cuda, backend, cells,
                                               args.steps, dispersive, args.repeats, args.history_stride))
                    result['cases'].append(case)
                    print(json.dumps(case), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
