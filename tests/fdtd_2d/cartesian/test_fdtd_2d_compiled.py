"""Full-state parity and execution-boundary tests for compiled Cartesian 2D."""
import contextlib
import io
import os
import sys
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from FDTD_2D_Ez import FDTD_2D_Ez
from FDTD_2D_Hz import FDTD_2D_Hz
from FDTD_common import runtime_2d


@contextlib.contextmanager
def capture_native_stderr():
    """C stdio uses fd 2, independently of Python's sys.stderr wrapper."""
    with tempfile.TemporaryFile() as output:
        original = os.dup(2)
        try:
            os.dup2(output.fileno(), 2)
            yield output
        finally:
            os.dup2(original, 2)
            os.close(original)


def make_sim(solver, backend, material='mixed', source='point', periodic='', steps=5, grid=(8, 6)):
    if source.endswith('-broadband'):
        steps = 200  # Resolve the modal anchor band on the temporal FFT grid.
    nx, ny = grid
    sim = solver(nx * 1e-3, ny * 1e-3, nx, ny, 100e9, steps, dt=1e-13, subpixel=1).config(backend)
    options = dict(epsilon_r=(2.0, 2.5, 3.0), mu_r=(1.2, 1.3, 1.4),
                   sigma_e=(0.01, 0.02, 0.03), sigma_m=(0.02, 0.03, 0.04))
    poles = dict(debye=dict(delta_epsilon=(0.3, 0.5, 0.7), tau=4e-12),
                 drude=dict(omega_p=2e11, gamma=1e10),
                 lorentz=dict(delta_epsilon=0.4, omega_0=3e11, gamma=2e10))
    if material == 'mixed':
        options.update(poles)
        options['debye'] = [poles['debye'], dict(delta_epsilon=0.2, tau=9e-12)]
        options['lorentz'] = [poles['lorentz'], dict(delta_epsilon=0.1, omega_0=5e11, gamma=3e10)]
    elif material in poles:
        options[material] = poles[material]
    sim.add_material('medium', **options)
    sim.add_rectangle(material='medium', x_position=(1, 6), y_position=(1, 5))
    sim.add_rectangle(material='PEC', x_position=(5, 6), y_position=(2, 3))
    sim.add_rectangle(material='PMC', x_position=(0, 1), y_position=(4, 5))
    sim.periodic = periodic
    if not periodic:
        sim.add_PML(1)
    if source == 'point':
        sim.add_source('point', x=2, y=2, t0=2e-13, tw=3e-13, is_show=False)
        # Overlapping events must accumulate without lost writes.
        sim.add_source('line-soft', x=(1, 5), y=2, t0=2e-13, tw=3e-13, is_show=False)
        sim.add_source('line-soft', x=2, y=(1, 5), t0=2e-13, tw=3e-13, is_show=False)
    elif source == 'sftf':
        sim.add_source('sftf', x=(2, 5), y=(2, 4), angle=0.3,
                       t0=1e-13, tw=2e-13, is_show=False)
    elif source.startswith('waveguide'):
        broadband = source.endswith('-broadband')
        kind = source.removesuffix('-broadband')
        def modes(lo, hi, line, frequency, num_modes, guess, amplitude):
            first = np.stack([np.full(hi - lo, index + 1.0) for index in range(num_modes)])
            return first, -0.5 * first, 1.5 + 0.1 * np.arange(num_modes)
        sim._wg_modes_x = sim._wg_modes_y = modes
        options = dict(broadband=True, frequency_mode_pairs=[(30e9, 0), (55e9, 1), (90e9, 0)]) if broadband else {}
        sim.add_source(kind, x=3 if kind == 'waveguide-x' else (2, 6),
                       y=(1, 5) if kind == 'waveguide-x' else 3,
                       modes_to_show=2, t0=3e-13, tw=2e-13, is_show=False, **options)
    sim.add_line_monitor(x=(0, nx), y=1, index=9)
    sim.add_line_monitor(x=3, y=(0, ny), index=10)
    if steps > 2:
        sim.monitors[1]['it0'] = 1
        sim.monitors[1]['it1'] = steps - 1
    # Exercise every cell and ADE initial endpoint, not just the source region.
    rng = np.random.default_rng(22)
    names = ('Ez', 'Hx', 'Hy') if solver is FDTD_2D_Ez else ('Hz', 'Ex', 'Ey')
    for name in names:
        getattr(sim, name)[:] = rng.normal(scale=1e-4, size=getattr(sim, name).shape)
    return sim


def assert_state(test, actual, expected, histories=True):
    for name, value in vars(expected).items():
        if isinstance(value, np.ndarray) and (name.startswith(('Psi_', 'd_', 'Ca', 'Cb'))
                or name in {'Ez','Dz','Hx','Hy','Bx','By','Ex','Ey','Dx','Dy','Hz','Bz'}
                or (histories and name.endswith('_history'))):
            np.testing.assert_allclose(getattr(actual, name), value, rtol=2e-11, atol=2e-13,
                                       err_msg=name)
    for name in ('_ade_Ez', '_ade_Ex', '_ade_Ey'):
        ade = getattr(expected, name, None)
        if ade is not None:
            for actual_pole, pole in zip(getattr(actual, name).poles, ade.poles):
                for key in ('q', 'v'):
                    if key in pole:
                        np.testing.assert_allclose(actual_pole[key], pole[key],
                            rtol=5e-11, atol=2e-13 if key == 'q' else 1e-3, err_msg=name + key)
    for actual_monitor, monitor in zip(actual.monitor_results, expected.monitor_results):
        for key, value in monitor.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(actual_monitor[key], value, rtol=2e-11, atol=2e-13)
    test.assertEqual(len(actual.monitor_results), len(expected.monitor_results))
    test.assertEqual(actual.Nt_rec, expected.Nt_rec)


class TestCompiledCPU(unittest.TestCase):
    def test_progress_completion_and_state(self):
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            for steps in (0, 1, 130):
                with self.subTest(solver=solver.__name__, steps=steps):
                    actual = make_sim(solver, self.backend, steps=max(steps, 1))
                    reference = make_sim(solver, 'python', steps=max(steps, 1))
                    actual.Nt = reference.Nt = steps
                    if not steps:
                        actual.monitors.clear()
                        reference.monitors.clear()
                    reference.run(record_stride=3)
                    with capture_native_stderr() as output:
                        actual.run(record_stride=3, progress=True)
                        output.seek(0)
                        display = output.read().decode()
                    self.assertIn(f'100.00% {steps}/{steps} steps', display)
                    self.assertTrue(display.endswith('\n'))
                    assert_state(self, actual, reference)
                    with capture_native_stderr() as output:
                        actual.run(is_include_history=False, progress=False)
                        output.seek(0)
                        self.assertEqual(output.read(), b'')

    backend = 'cpu'
    @classmethod
    def setUpClass(cls):
        try:
            runtime_2d.compiled_extension()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc))

    def compare(self, backend, **options):
        results = []
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            with self.subTest(solver=solver.__name__, backend=backend, **options):
                reference = make_sim(solver, 'python', **options)
                actual = make_sim(solver, backend, **options)
                with contextlib.redirect_stderr(io.StringIO()):
                    reference.run(record_stride=2)
                    actual.run(record_stride=2)
                assert_state(self, actual, reference)
                results.append((actual, reference))
        return results

    def test_materials_and_boundaries(self):
        for material in ('ordinary', 'debye', 'drude', 'lorentz', 'mixed'):
            list(self.compare(self.backend, material=material))
        for periodic in ('x', 'y', 'xy'):
            list(self.compare(self.backend, periodic=periodic))

    def test_all_sources(self):
        for source in ('sftf', 'waveguide-x', 'waveguide-y',
                       'waveguide-x-broadband', 'waveguide-y-broadband'):
            list(self.compare(self.backend, source=source))

    def test_multiple_cuda_blocks_and_partial_edge_tiles(self):
        list(self.compare(self.backend, grid=(37, 45), steps=9))

    def test_continuous_and_band_limited_waveforms(self):
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            for f_min in (20e9, 100e9):
                actual = make_sim(solver, self.backend, steps=80)
                reference = make_sim(solver, 'python', steps=80)
                for sim in (actual, reference):
                    for source in sim.sources:
                        source.update(f_min=f_min, f_max=100e9, t0=1e-12, tw=2e-12)
                with contextlib.redirect_stderr(io.StringIO()):
                    reference.run(record_stride=3)
                    actual.run(record_stride=3)
                assert_state(self, actual, reference)

    def test_repeated_run_and_persistence(self):
        for actual, reference in self.compare(self.backend):
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'sim.pkl')
                actual.save(path, include_histories=False)
                restored = type(actual).load(path)
            with contextlib.redirect_stderr(io.StringIO()):
                actual.run(is_include_history=False)
                restored.run(is_include_history=False)
                reference.run(is_include_history=False)
            assert_state(self, actual, reference, histories=False)
            assert_state(self, restored, actual, histories=False)
            self.assertEqual(actual.Nt_rec, 0)

    def test_no_python_numerical_callbacks(self):
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = make_sim(solver, self.backend)
            with patch.object(sim, '_g', side_effect=AssertionError('Python waveform')), \
                 patch.object(sim, 'update_D', side_effect=AssertionError('Python update')):
                sim.run(is_include_history=False)

    def test_manual_substeps_rejected(self):
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = make_sim(solver, self.backend)
            for method in ('calculate_Curl_E', 'calculate_Curl_H', 'calcualte_Psi_B',
                           'calcualte_Psi_D', 'update_B', 'update_H', 'update_D', 'update_E'):
                with self.assertRaisesRegex(RuntimeError, 'Individual 2D'):
                    getattr(sim, method)()

    def test_invalid_record_stride(self):
        sim = make_sim(FDTD_2D_Ez, self.backend)
        for stride in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                sim.run(record_stride=stride)

    def test_empty_and_single_step_runs(self):
        for steps in (0, 1):
            for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
                reference = solver(1, 1, 2, 3, 1, steps).config('python')
                actual = solver(1, 1, 2, 3, 1, steps).config(self.backend)
                with contextlib.redirect_stderr(io.StringIO()):
                    reference.run(record_stride=3)
                    actual.run(record_stride=3)
                assert_state(self, actual, reference)

    def test_bad_field_shape_rejected_before_stepping(self):
        sim = make_sim(FDTD_2D_Ez, self.backend)
        initial = sim.Ez.copy()
        sim.Hx = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'Yee shape'):
            sim.run()
        np.testing.assert_array_equal(sim.Ez, initial)

    def test_no_monitors_or_history(self):
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            actual = make_sim(solver, self.backend)
            reference = make_sim(solver, 'python')
            actual.monitors.clear()
            reference.monitors.clear()
            with contextlib.redirect_stderr(io.StringIO()):
                reference.run(is_include_history=False)
                actual.run(is_include_history=False)
            assert_state(self, actual, reference, histories=False)


class TestBackendContract(unittest.TestCase):
    def test_native_cpu_loop_has_no_python_calls(self):
        try:
            extension = runtime_2d.compiled_extension()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        for solver, polarization in ((FDTD_2D_Ez, 'tm'), (FDTD_2D_Hz, 'te')):
            sim = make_sim(solver, 'cpu')
            prepared = runtime_2d.prepare(sim, polarization, 2, True)
            args = tuple(stage.args for stage in prepared['stages'])
            native_run = getattr(extension, 'run_' + polarization)
            calls = []
            def profile(frame, event, arg):
                if event == 'call':
                    calls.append(frame.f_code.co_name)
            for progress in (False, True):
                with capture_native_stderr():
                    sys.setprofile(profile)
                    try:
                        native_run(args, sim.Nt, sim.Nx, sim.Ny, progress)
                    finally:
                        sys.setprofile(None)
                self.assertEqual(calls, [])

    def test_reference_works_without_extension(self):
        with patch.object(runtime_2d, 'compiled_extension', side_effect=RuntimeError('missing extension')):
            sim = FDTD_2D_Ez(1, 1, 2, 2, 1, 1)
            self.assertEqual(sim.backend_requested, 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'missing extension'):
                sim.run()
            with self.assertRaisesRegex(RuntimeError, 'missing extension'):
                sim.config('cpu')
            sim.config('python')
            with contextlib.redirect_stderr(io.StringIO()):
                sim.run(is_include_history=False)


@unittest.skipUnless(os.environ.get('FDTD_TEST_REAL_CUDA') == '1', 'Set FDTD_TEST_REAL_CUDA=1 for graph tests')
class TestCompiledGPU(TestCompiledCPU):
    backend = 'gpu'

    def test_actual_transfer_allocation_and_dispatch_hooks(self):
        from numba.cuda.cudadrv import driver
        from numba.cuda.dispatcher import CUDADispatcher
        extension = runtime_2d.compiled_extension()
        checked = []
        class AuditedGraph:
            def __init__(self):
                self.inner = extension.CudaGraph()
            def __getattr__(self, name):
                return getattr(self.inner, name)
            def replay(self, steps):
                # Patch the actual Numba driver and dispatcher entry points,
                # independently of runtime_2d's accounting wrappers.
                with contextlib.ExitStack() as stack:
                    for owner, names in ((driver, ('host_to_device', 'device_to_host', 'device_to_device')),
                            (driver.Context, ('memalloc', 'memallocmanaged', 'memhostalloc')),
                            (CUDADispatcher, ('call', '__call__', '__getitem__'))):
                        for name in names:
                            stack.enter_context(patch.object(owner, name,
                                side_effect=AssertionError(f'{name} during stepping')))
                    calls = []
                    def profile(frame, event, arg):
                        if event == 'call':
                            calls.append(frame.f_code.co_name)
                    sys.setprofile(profile)
                    try:
                        self.inner.replay(steps)
                    finally:
                        sys.setprofile(None)
                    if calls:
                        raise AssertionError(f'Python callbacks during native replay: {calls}')
                checked.append(steps)
        proxy = SimpleNamespace(CudaGraph=AuditedGraph, check_cuda_driver=extension.check_cuda_driver)
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            for progress in (False, True):
                sim = make_sim(solver, 'gpu', steps=2050)
                with patch.object(runtime_2d, 'compiled_extension', return_value=proxy), capture_native_stderr():
                    sim.run(record_stride=2, progress=progress)
                for key in ('host_to_device', 'device_to_host', 'allocations', 'python_dispatch'):
                    self.assertEqual(sim._gpu_transfer_stats[key + '_during_steps'], 0)
                self.assertFalse(sim._gpu_transfer_stats['simulator'])
        self.assertEqual(checked, [2050] * 4)

    def test_memory_preflight_does_not_advance_fields(self):
        from numba import cuda
        sim = make_sim(FDTD_2D_Ez, 'gpu')
        initial = sim.Ez.copy()
        with patch.object(type(cuda.current_context()), 'get_memory_info', return_value=(1024, 2048)), \
             patch.object(runtime_2d.DeviceBuffers, 'upload', side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(MemoryError, 'record_stride'):
                sim.run()
        np.testing.assert_array_equal(sim.Ez, initial)

    def test_allocation_failure_does_not_advance_fields(self):
        from numba import cuda
        sim = make_sim(FDTD_2D_Ez, 'gpu')
        initial = sim.Ez.copy()
        with patch.object(cuda, 'device_array', side_effect=MemoryError('injected allocation failure')):
            with self.assertRaisesRegex(MemoryError, 'record_stride'):
                sim.run()
        np.testing.assert_array_equal(sim.Ez, initial)

    def test_capture_failure_closes_graph_and_allows_next_run(self):
        extension = runtime_2d.compiled_extension()
        closed = []
        class FailingGraph:
            def __init__(self):
                self.inner = extension.CudaGraph()
            def __getattr__(self, name):
                return getattr(self.inner, name)
            def end(self):
                raise RuntimeError('injected capture failure')
            def close(self):
                self.inner.close()
                closed.append(True)
        proxy = SimpleNamespace(CudaGraph=FailingGraph, check_cuda_driver=extension.check_cuda_driver)
        sim = make_sim(FDTD_2D_Ez, 'gpu')
        initial = sim.Ez.copy()
        with patch.object(runtime_2d, 'compiled_extension', return_value=proxy):
            with self.assertRaisesRegex(RuntimeError, 'injected capture failure'):
                sim.run()
        self.assertEqual(closed, [True])
        np.testing.assert_array_equal(sim.Ez, initial)
        sim.run()

    def test_progress_setup_failure_closes_events_before_stepping(self):
        extension = runtime_2d.compiled_extension()
        closed = []
        class FailingGraph:
            def __init__(self):
                self.inner = extension.CudaGraph()
            def __getattr__(self, name):
                return getattr(self.inner, name)
            def prepare_progress(self, steps):
                self.inner.prepare_progress(steps)
                raise MemoryError('injected progress setup failure')
            def close(self):
                self.inner.close()
                self.inner.close()  # Cleanup remains idempotent with events.
                closed.append(True)
        proxy = SimpleNamespace(CudaGraph=FailingGraph, check_cuda_driver=extension.check_cuda_driver)
        sim = make_sim(FDTD_2D_Ez, 'gpu', steps=130)
        initial = sim.Ez.copy()
        with patch.object(runtime_2d, 'compiled_extension', return_value=proxy):
            with self.assertRaisesRegex(MemoryError, 'record_stride'):
                sim.run(progress=True)
        self.assertEqual(closed, [True])
        np.testing.assert_array_equal(sim.Ez, initial)
        with capture_native_stderr():
            sim.run(progress=True)


if __name__ == '__main__':
    unittest.main()
