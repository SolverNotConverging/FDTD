import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from FDTD_common.progress import GpuProgress


class FakeEvent:
    def __init__(self):
        self.ready = False
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.ready

    def synchronize(self):
        raise AssertionError("Progress must not synchronize CUDA")


class TestGpuProgress(unittest.TestCase):
    def make_reporter(self, steps=3, enabled=True):
        cuda = SimpleNamespace(config=SimpleNamespace(ENABLE_CUDASIM=False),
                               event=lambda **kwargs: FakeEvent())
        with contextlib.redirect_stderr(io.StringIO()):
            return GpuProgress(cuda, steps, enabled)

    def test_pending_queries_return_without_advancing_and_finish_waits_for_completion(self):
        reporter = self.make_reporter()
        try:
            for step in range(1, 4):
                reporter.submitted(step)
            reporter.poll()
            self.assertEqual(reporter.completed, 0)
            self.assertEqual(reporter.bar.n, 0)
            reporter.events[0].ready = True
            reporter.poll()
            self.assertEqual(reporter.bar.n, 1)

            def complete_on_host_sleep(seconds):
                for event in reporter.events:
                    event.ready = True

            with patch('FDTD_common.progress.sleep', side_effect=complete_on_host_sleep) as pause:
                reporter.finish()
            pause.assert_called_once()
            self.assertEqual(reporter.bar.n, 3)
        finally:
            reporter.close()

    def test_disabled_display_still_checks_the_final_gpu_marker(self):
        reporter = self.make_reporter(enabled=False)
        for step in range(1, 4):
            reporter.submitted(step)
        self.assertEqual(len(reporter.recorded), 1)
        self.assertIsNone(reporter.bar)
        reporter.events[0].ready = True
        reporter.finish()
        self.assertEqual(reporter.completed, 3)

    def test_markers_are_bounded_and_preallocated(self):
        reporter = self.make_reporter(10000, enabled=False)
        self.assertEqual(len(reporter.events), 1)
        reporter = self.make_reporter(10000)
        try:
            self.assertLessEqual(len(reporter.events), 128)
            for step in range(1, 10001):
                reporter.submitted(step)
            self.assertEqual(reporter.recorded[-1][1], 10000)
        finally:
            reporter.close()


class TestTerminalDefaults(unittest.TestCase):
    def test_ez_and_hz_reference_show_progress_by_default_and_allow_quiet_runs(self):
        from FDTD_2D_Ez import FDTD_2D_Ez
        from FDTD_2D_Hz import FDTD_2D_Hz
        for solver in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver(0.002, 0.002, 2, 2, 10e9, 2, subpixel=1).config('python')
            with contextlib.redirect_stderr(io.StringIO()) as output:
                sim.run(is_include_history=False)
            self.assertIn('100%', output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()) as output:
                sim.run(is_include_history=False, progress=False)
            self.assertEqual(output.getvalue(), '')
