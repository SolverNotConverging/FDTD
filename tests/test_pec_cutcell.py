"""PEC geometry and enlarged cut-cell regression checks."""

import unittest
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

from FDTD_2D_Ez import FDTD_2D_Ez
from FDTD_2D_Hz import FDTD_2D_Hz


class TestPecCutCells(unittest.TestCase):
    def test_rectangle_fractions_leave_material_averaging_alone(self):
        for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = cls(4.0, 4.0, 4, 4, 1e8, 1, subpixel=16).config("python")
            before = sim.ERzz.copy() if cls is FDTD_2D_Ez else sim.ERxx.copy()
            sim.add_rectangle(material="PEC", x_position=(1.25, 2.0),
                              y_position=(1.0, 2.0))
            self.assertTrue(sim._has_cutcell)
            self.assertAlmostEqual(sim.pec_fluid_area[1, 1], 0.25)
            self.assertAlmostEqual(sim.pec_fluid_area[2, 1], 1.0)
            after = sim.ERzz if cls is FDTD_2D_Ez else sim.ERxx
            np.testing.assert_array_equal(after, before)

    def test_te_faraday_uses_open_edges_and_merged_area(self):
        sim = FDTD_2D_Hz(4.0, 4.0, 4, 4, 1e8, 1,
                         subpixel=16).config("python")
        sim.add_rectangle(material="PEC", x_position=(1.25, 2.0),
                          y_position=(1.0, 2.0))
        sim.Ex.fill(0)
        sim.Ey.fill(0)
        sim.Ey[1, 1] = 2.0
        sim.calculate_Curl_E()
        self.assertAlmostEqual(sim.d_Ey_x[1, 1], -2.0 / 0.25)
        self.assertTrue(any((1, 1) in group for group in sim._pec_enlarged_groups))

    def test_tm_cut_segment_and_enlargement(self):
        sim = FDTD_2D_Ez(4.0, 4.0, 4, 4, 1e8, 1,
                         subpixel=16).config("python")
        sim.add_rectangle(material="PEC", x_position=(1.25, 2.0),
                          y_position=(1.0, 2.0))
        sim.Ez.fill(0)
        sim.Ez[1, 1] = 2.0
        sim.calculate_Curl_E()
        self.assertAlmostEqual(sim.pec_open_Hy[1, 1], 0.25)
        self.assertAlmostEqual(sim.d_Ez_x[1, 1], -2.0 / 0.25)
        self.assertTrue(any((1, 1) in group for group in sim._pec_enlarged_Hy))

    def test_full_grid_aligned_pec_keeps_original_step(self):
        for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = cls(4.0, 4.0, 4, 4, 1e8, 1).config("python")
            sim.add_rectangle(material="PEC", x_position=(1, 2),
                              y_position=(1, 2))
            self.assertFalse(sim._has_cutcell)

    def test_rectangular_grid_uses_cartesian_courant_limit(self):
        for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = cls(4.0, 8.0, 4, 4, 1e6, 1)
            self.assertAlmostEqual(
                sim.dt, sim.dx * sim.dy
                / (sim.c0 * np.hypot(sim.dx, sim.dy)))

    def test_curved_pec_runs_at_regular_courant_step(self):
        rng = np.random.default_rng(7)
        for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = cls(0.02, 0.02, 20, 20, 10e9, 2000,
                      subpixel=8).config("python")
            sim.add_circle(material="PEC", center=(0.0103, 0.0102),
                           radius=0.0046)
            self.assertTrue(sim._has_cutcell)
            if cls is FDTD_2D_Ez:
                sim.Ez[:] = rng.normal(scale=1e-3, size=sim.Ez.shape)
                sim.Ez[sim.PEC_Ez] = 0
            else:
                sim.Ex[:] = rng.normal(scale=1e-3, size=sim.Ex.shape)
                sim.Ey[:] = rng.normal(scale=1e-3, size=sim.Ey.shape)
                sim.Ex[sim.PEC_Ex] = 0
                sim.Ey[sim.PEC_Ey] = 0
            sim.run(is_include_history=False)
            fields = (sim.Ez, sim.Hx, sim.Hy) if cls is FDTD_2D_Ez else (
                sim.Ex, sim.Ey, sim.Hz)
            self.assertTrue(all(np.all(np.isfinite(field)) for field in fields))
            self.assertLess(max(np.max(np.abs(field)) for field in fields), 0.02)

    @unittest.skipUnless(importlib.util.find_spec("FDTD_common._compiled_2d"),
                         "Cython runtime is not built")
    def test_cpu_cutcell_matches_reference_with_materials_and_pml(self):
        rng = np.random.default_rng(12)
        for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
            pair = []
            for backend in ("python", "cpu"):
                sim = cls(0.016, 0.012, 16, 12, 10e9, 35,
                          subpixel=8).config(backend)
                sim.add_material("glass", epsilon_r=2.5, mu_r=1.2,
                                 sigma_e=0.001, sigma_m=0.001)
                sim.add_rectangle(material="glass", x_position=(2, 5),
                                  y_position=(2, 5))
                sim.add_circle(material="PEC", center=(0.0083, 0.0061),
                               radius=0.0032)
                sim.add_PML(2)
                pair.append(sim)
            reference, native = pair
            names = ("Ez", "Hx", "Hy") if cls is FDTD_2D_Ez else (
                "Ex", "Ey", "Hz")
            for name in names:
                initial = rng.normal(scale=1e-4,
                                     size=getattr(reference, name).shape)
                getattr(reference, name)[:] = initial
                getattr(native, name)[:] = initial
            reference.run(record_stride=2)
            native.run(record_stride=2)
            self.assertEqual(native.backend, "cython")
            for name in names:
                np.testing.assert_allclose(getattr(native, name),
                                           getattr(reference, name),
                                           rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(getattr(native, name + "_history"),
                                           getattr(reference, name + "_history"),
                                           rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec("FDTD_common._compiled_2d"),
                         "Cython runtime is not built")
    def test_cpu_selection_stays_native_for_cut_cells(self):
        sim = FDTD_2D_Hz(4.0, 4.0, 4, 4, 1e8, 2, subpixel=8)
        sim.add_circle(material="PEC", center=(2.3, 2.2), radius=0.8)
        self.assertTrue(sim._has_cutcell)
        backend = sim.backend
        sim.run(is_include_history=False)
        self.assertEqual(sim.backend, backend)

    @unittest.skipUnless(importlib.util.find_spec("numba"),
                         "Numba CUDA is optional")
    def test_cuda_simulator_matches_reference(self):
        script = textwrap.dedent("""
            import numpy as np
            from FDTD_2D_Ez import FDTD_2D_Ez
            from FDTD_2D_Hz import FDTD_2D_Hz

            for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
                sims = []
                for backend in ('python', 'gpu'):
                    sim = cls(.008, .006, 8, 6, 10e9, 5,
                              subpixel=4).config(backend)
                    sim.add_circle(material='PEC', center=(.0042, .0031),
                                   radius=.0017)
                    sims.append(sim)
                reference, gpu = sims
                names = ('Ez', 'Hx', 'Hy') if cls is FDTD_2D_Ez else (
                    'Ex', 'Ey', 'Hz')
                rng = np.random.default_rng(44)
                for name in names:
                    values = rng.normal(scale=1e-4,
                                        size=getattr(reference, name).shape)
                    getattr(reference, name)[:] = values
                    getattr(gpu, name)[:] = values
                reference.run(record_stride=2)
                gpu.run(record_stride=2)
                assert gpu._gpu_transfer_stats['host_to_device_during_steps'] == 0
                for name in names:
                    np.testing.assert_allclose(
                        getattr(gpu, name), getattr(reference, name),
                        rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(
                        getattr(gpu, name + '_history'),
                        getattr(reference, name + '_history'),
                        rtol=1e-10, atol=1e-10)
        """)
        environment = os.environ.copy()
        environment["NUMBA_ENABLE_CUDASIM"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unresolved_subcell_is_reported(self):
        for cls in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = cls(4.0, 4.0, 4, 4, 1e8, 1, subpixel=16)
            with self.assertRaisesRegex(ValueError, "refine the grid"):
                sim.add_circle(material="PEC", center=(1.5, 1.5), radius=0.2)
            self.assertFalse(sim._pec_shapes)
            self.assertFalse(np.any(sim.PEC_cells))


if __name__ == "__main__":
    unittest.main()
