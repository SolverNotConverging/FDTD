"""Compare warmed native stepping with progress off/on in alternating order.

Run from the repository root. Terminal output goes to stderr as in normal runs.
For reproducible file-output measurements, redirect stderr to a log file.
"""
import argparse
import copy
import json
import statistics
from pathlib import Path

from benchmark_2d_compiled import make_sim, FDTD_2D_Ez, FDTD_2D_Hz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, nargs='+', default=[120, 256])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--output', type=Path, default=Path('build/progress_benchmark.json'))
    options = parser.parse_args()
    if options.steps < 1 or options.repeats < 1 or min(options.cells) < 16:
        parser.error('steps/repeats must be positive and grids must be at least 16 cells')
    results = []
    for cells in options.cells:
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            for backend in options.backends:
                for dispersive in (False, True):
                    template = make_sim(solver, backend, cells, options.steps, dispersive)
                    initial = template.state_dict()
                    samples = {False: [], True: []}
                    for repetition in range(options.repeats + 1):
                        for progress in ((False, True) if repetition % 2 else (True, False)):
                            sim = solver.__new__(solver)
                            sim.load_state_dict(copy.deepcopy(initial))
                            sim.run(is_include_history=False, progress=progress)
                            if repetition:
                                samples[progress].append(sim._runtime_stats['stepping_seconds'])
                    off, on = (statistics.median(samples[key]) for key in (False, True))
                    row = dict(solver=solver.__name__, backend=backend, cells=cells,
                               material='mixed' if dispersive else 'ordinary', steps=options.steps,
                               off_seconds=off, on_seconds=on, change_percent=100 * (on / off - 1),
                               off_samples=samples[False], on_samples=samples[True])
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    options.output.parent.mkdir(parents=True, exist_ok=True)
                    options.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
