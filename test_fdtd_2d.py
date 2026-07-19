import unittest
import warnings

import numpy as np
from matplotlib import pyplot as plt

from FDTD_2D_Ez import FDTD_2D_Ez, Material
from FDTD_2D_Hz import FDTD_2D_Hz


class TestYeeMaterials(unittest.TestCase):
    def test_named_lossy_materials_map_to_tmz_and_tez_components(self):
        definition = dict(
            epsilon_r=(2.0, 3.0, 4.0), mu_r=(5.0, 6.0, 7.0),
            sigma_e=(0.1, 0.2, 0.3), sigma_m=(0.4, 0.5, 0.6))

        tm = FDTD_2D_Ez(2.0, 2.0, 2, 2, 1.0, 1, subpixel=1)
        material = tm.add_material("lossy", **definition)
        self.assertIsInstance(material, Material)
        tm.add_rectangle(material="lossy", x_position=(0, 1), y_position=(0, 1))
        self.assertEqual(tm.ERzz[0, 0], 4.0)
        self.assertEqual(tm.MRxx[0, 0], 5.0)
        self.assertEqual(tm.MRyy[0, 0], 6.0)
        self.assertEqual(tm.SIGEzz[0, 0], 0.3)
        self.assertEqual(tm.SIGMxx[0, 0], 0.4)
        self.assertEqual(tm.SIGMyy[0, 0], 0.5)

        te = FDTD_2D_Hz(2.0, 2.0, 2, 2, 1.0, 1, subpixel=1)
        te.add_material("lossy", **definition)
        te.add_rectangle(material="lossy", x_position=(0, 1), y_position=(0, 1))
        self.assertEqual(te.ERxx[0, 0], 2.0)
        self.assertEqual(te.ERyy[0, 0], 3.0)
        self.assertEqual(te.MRzz[0, 0], 7.0)
        self.assertEqual(te.SIGExx[0, 0], 0.1)
        self.assertEqual(te.SIGEyy[0, 0], 0.2)
        self.assertEqual(te.SIGMzz[0, 0], 0.6)

    def test_loss_coefficients_decay_uniform_tmz_and_tez_fields(self):
        for solver_class in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver_class(2.0, 2.0, 2, 2, 1.0, 1,
                               dt=1e-12, subpixel=1).config("python")
            sim.add_material("lossy", epsilon_r=2.0, mu_r=3.0,
                             sigma_e=0.25, sigma_m=0.5)
            sim.add_rectangle(material="lossy", x_position=(0, 2), y_position=(0, 2))
            sim._init_Coeff()

            if solver_class is FDTD_2D_Ez:
                sim.Ez.fill(2.0)
                sim.Hx.fill(3.0)
                sim.Hy.fill(4.0)
                sim.update_B()
                np.testing.assert_allclose(sim.Hx, 3.0 * sim.CaHx)
                np.testing.assert_allclose(sim.Hy, 4.0 * sim.CaHy)
                sim.Hx.fill(3.0)
                sim.Hy.fill(4.0)
                sim.Ez.fill(2.0)
                sim.update_D()
                np.testing.assert_allclose(sim.Ez, 2.0 * sim.CaEz)
            else:
                sim.Ex.fill(2.0)
                sim.Ey.fill(3.0)
                sim.Hz.fill(4.0)
                sim.update_B()
                np.testing.assert_allclose(sim.Hz, 4.0 * sim.CaHz)
                sim.Hz.fill(4.0)
                sim.Ex.fill(2.0)
                sim.Ey.fill(3.0)
                sim.update_D()
                np.testing.assert_allclose(sim.Ex, 2.0 * sim.CaEx)
                np.testing.assert_allclose(sim.Ey, 3.0 * sim.CaEy)

    def test_tm_shapes_and_subpixel_material(self):
        sim = FDTD_2D_Ez(2.0, 2.0, 2, 2, 1.0, 1, subpixel=16)
        sim.add_rectangle([1.0, 1.0, 4.0], [3.0, 5.0, 1.0],
                          x_position=(0.25, 0.75), y_position=(0.0, 1.0))

        self.assertEqual(sim.Ez.shape, (3, 3))
        self.assertEqual(sim.Hx.shape, (3, 2))
        self.assertEqual(sim.Hy.shape, (2, 3))
        self.assertAlmostEqual(sim.ERzz[0, 0], 2.5)
        self.assertAlmostEqual(sim.MRxx[0, 0], 2.0)
        self.assertAlmostEqual(sim.MRyy[0, 0], 3.0)
        self.assertAlmostEqual(sim.ERzz_Ez[1, 1], 1.375)
        self.assertAlmostEqual(sim.MRxx_Hx[1, 0], 1.5)
        self.assertAlmostEqual(sim.MRyy_Hy[0, 1], 2.0)

    def test_te_shapes_and_subpixel_material(self):
        sim = FDTD_2D_Hz(2.0, 2.0, 2, 2, 1.0, 1, subpixel=16)
        sim.add_rectangle([3.0, 5.0, 1.0], [1.0, 1.0, 4.0],
                          x_position=(0.25, 0.75), y_position=(0.0, 1.0))

        self.assertEqual(sim.Hz.shape, (2, 2))
        self.assertEqual(sim.Ex.shape, (2, 3))
        self.assertEqual(sim.Ey.shape, (3, 2))
        self.assertAlmostEqual(sim.ERxx_Ex[0, 1], 1.5)
        self.assertAlmostEqual(sim.ERyy_Ey[1, 0], 2.0)
        self.assertAlmostEqual(sim.MRzz_Hz[0, 0], 2.5)

    def test_triangle_uses_subpixel_material_averaging(self):
        for solver_class, cell_material in (
                (FDTD_2D_Ez, "ERzz"), (FDTD_2D_Hz, "ERxx")):
            sim = solver_class(2.0, 2.0, 2, 2, 1.0, 1, subpixel=16)
            sim.add_triangle(ER=5.0, MR=1.0,
                             vertices=((0, 0), (1, 0), (0, 1)))
            value = getattr(sim, cell_material)[0, 0]
            self.assertGreater(value, 1.0)
            self.assertLess(value, 5.0)


class TestConductors(unittest.TestCase):
    def test_tm_pec_pmc_masks_and_coefficients(self):
        sim = FDTD_2D_Ez(4.0, 4.0, 4, 4, 1.0, 1).config("python")
        original_er = sim.ERzz.copy()
        sim.add_rectangle(material="PEC", x_position=(1, 2),
                          y_position=(1, 2), subpixel=3)
        self.assertTrue(sim.PEC_cells[1, 1])
        self.assertTrue(np.all(sim.PEC_Ez[1:3, 1:3]))
        self.assertTrue(np.all(sim.Ez_update_coeff[1:3, 1:3] == 0.0))
        np.testing.assert_array_equal(sim.ERzz, original_er)

        sim.add_circle(ER="PMC", center=(2.5, 2.5), radius=0.6,
                       subpixel=2)
        self.assertTrue(sim.PMC_cells[2, 2])
        self.assertTrue(sim.PMC_Hx[2, 2] and sim.PMC_Hx[3, 2])
        self.assertTrue(sim.PMC_Hy[2, 2] and sim.PMC_Hy[2, 3])

        sim.Dz.fill(1.0)
        sim.Bx.fill(1.0)
        sim.By.fill(1.0)
        sim.update_E()
        sim.update_H()
        self.assertTrue(np.all(sim.Ez[sim.PEC_Ez] == 0.0))
        self.assertTrue(np.all(sim.Hx[sim.PMC_Hx] == 0.0))
        self.assertTrue(np.all(sim.Hy[sim.PMC_Hy] == 0.0))

    def test_te_triangle_pec_and_rectangle_pmc(self):
        sim = FDTD_2D_Hz(4.0, 4.0, 4, 4, 1.0, 1).config("python")
        original_mr = sim.MRzz.copy()
        sim.add_triangle(material="PEC", vertices=((0, 0), (2, 0), (0, 2)),
                         subpixel=1)
        self.assertTrue(sim.PEC_cells[0, 0])
        self.assertTrue(sim.PEC_Ex[0, 0] and sim.PEC_Ex[0, 1])
        self.assertTrue(sim.PEC_Ey[0, 0] and sim.PEC_Ey[1, 0])

        sim.add_rectangle(MR="PMC", x_position=(2, 3), y_position=(2, 3),
                          subpixel=128)
        self.assertTrue(sim.PMC_Hz[2, 2])
        self.assertEqual(sim.Hz_update_coeff[2, 2], 0.0)
        np.testing.assert_array_equal(sim.MRzz, original_mr)

        sim.Dx.fill(1.0)
        sim.Dy.fill(1.0)
        sim.Bz.fill(1.0)
        sim.update_E()
        sim.update_H()
        self.assertTrue(np.all(sim.Ex[sim.PEC_Ex] == 0.0))
        self.assertTrue(np.all(sim.Ey[sim.PEC_Ey] == 0.0))
        self.assertTrue(np.all(sim.Hz[sim.PMC_Hz] == 0.0))

    def test_reduced_fdfd_modes_eliminate_conductor_cells(self):
        for solver_class in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver_class(8e-3, 4e-3, 8, 4, 100e9, 1).config("python")
            sim.add_rectangle(material="PEC", x_position=(2, 3),
                              y_position=(0, 4))
            sim.add_rectangle(material="PMC", x_position=(5, 6),
                              y_position=(0, 4))
            scalar_modes, _, n_eff = sim._wg_modes_y(
                0, 8, 2, 90e9, num_modes=2)
            self.assertEqual(scalar_modes.shape[1], 8)
            self.assertTrue(np.all(scalar_modes[:, [2, 5]] == 0.0))
            self.assertTrue(np.all(np.isfinite(n_eff)))

    def test_conductor_plot_colors_and_dash_styles(self):
        for solver_class in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver_class(4.0, 4.0, 4, 4, 1.0, 1).config("python")
            sim.add_rectangle(material="PEC", x_position=(1, 2), y_position=(1, 2))
            sim.add_rectangle(material="PMC", x_position=(2, 3), y_position=(2, 3))
            fig, ax = plt.subplots()
            contours = sim._draw_conductor_regions(ax, add_legend=True)
            self.assertEqual(len(contours), 2)
            legend = ax.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual({text.get_text() for text in legend.get_texts()}, {"PEC", "PMC"})
            plt.close(fig)


class TestBackends(unittest.TestCase):
    def _compare(self, solver_class, fields, derivatives, backend):
        py = solver_class(1.0, 1.0, 7, 6, 1.0, 1).config("python")
        accelerated = solver_class(1.0, 1.0, 7, 6, 1.0, 1).config(backend)
        rng = np.random.default_rng(42)
        for field in fields:
            values = rng.normal(size=getattr(py, field).shape)
            getattr(py, field)[:] = values
            getattr(accelerated, field)[:] = values

        py.calculate_Curl_E()
        accelerated.calculate_Curl_E()
        py.calculate_Curl_H()
        accelerated.calculate_Curl_H()
        for derivative in derivatives:
            np.testing.assert_allclose(getattr(accelerated, derivative), getattr(py, derivative))
        return accelerated.backend

    def test_cython_matches_python(self):
        if not FDTD_2D_Ez(1, 1, 2, 2, 1, 1)._use_cython_kernel:
            self.skipTest("optional Cython extensions are not built")
        self.assertEqual(self._compare(
            FDTD_2D_Ez, ("Ez", "Hx", "Hy"),
            ("d_Ez_y", "d_Ez_x", "d_Hx_y", "d_Hy_x"), "cpu"), "cython")
        self.assertEqual(self._compare(
            FDTD_2D_Hz, ("Hz", "Ex", "Ey"),
            ("d_Ex_y", "d_Ey_x", "d_Hz_y", "d_Hz_x"), "cpu"), "cython")

    def test_numba_cuda_matches_python_when_available(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = self._compare(
                FDTD_2D_Ez, ("Ez", "Hx", "Hy"),
                ("d_Ez_y", "d_Ez_x", "d_Hx_y", "d_Hy_x"), "gpu")
        if backend != "numba_cuda":
            self.skipTest("Numba-CUDA is unavailable")


class TestSources(unittest.TestCase):
    def test_tm_tfsf_has_low_corner_leakage(self):
        sim = FDTD_2D_Ez(0.05, 0.05, 50, 50, 10e9, 140).config("python")
        sim.add_PML(6)
        sim.add_source("sftf", x=(12, 38), y=(12, 38),
                       angle=np.deg2rad(25.0), is_show=False)
        sim.run(is_include_history=False)

        magnitude = np.abs(sim.Ez)
        inside_peak = np.max(magnitude[12:39, 12:39])
        outside = np.ones_like(magnitude, dtype=bool)
        outside[11:40, 11:40] = False
        self.assertGreater(inside_peak, 0.1)
        self.assertLess(np.max(magnitude[outside]) / inside_peak, 0.01)

    def test_point_and_staggered_fdfd_sources_run(self):
        for solver_class in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver_class(3e-3, 3e-3, 24, 24, 100e9, 2,
                               f_min=80e9).config("python")
            sim.add_rectangle(4.0, 1.0, (0.0, 3e-3), (1.1e-3, 1.9e-3))
            sim.add_PML(2)
            sim.add_source("waveguide-x", x=4, y=(4, 20), modes_to_show=1,
                           mode_index=0, is_show=False)
            self.assertTrue(np.isfinite(sim.sources[0]["n_eff"]))
            sim.run(is_include_history=False)


class TestMonitorIndicesAndPower(unittest.TestCase):
    def test_explicit_monitor_indices_are_stable_and_unique(self):
        for solver_class in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver_class(1.0, 1.0, 4, 4, 1.0, 2).config("python")
            self.assertEqual(sim.add_line_monitor(x=(0, 3), y=2, index=41), 41)
            self.assertEqual(sim.add_line_monitor(x=2, y=(0, 3)), 0)
            with self.assertRaises(ValueError):
                sim.add_line_monitor(x=(0, 3), y=1, index=41)

    def test_requested_frequency_power_and_partial_nf2ff(self):
        for solver_class in (FDTD_2D_Ez, FDTD_2D_Hz):
            sim = solver_class(1.0, 1.0, 4, 4, 100e9, 64,
                               dt=1e-12).config("python")
            frequency = 1.0 / (sim.Nt * sim.dt)
            time = np.arange(sim.Nt) * sim.dt
            tone = np.cos(2.0 * np.pi * frequency * time)[:, None]
            repeated = np.repeat(tone, 3, axis=1)
            zeros = np.zeros_like(repeated)
            common = {
                "index": 37, "ix0": 0, "ix1": 3,
                "iy0": 2, "iy1": 2,
                "it0": 0, "it1": sim.Nt,
                "orientation": "horizontal",
            }
            if solver_class is FDTD_2D_Ez:
                common.update(Ez=repeated, Hx=repeated, Hy=zeros)
            else:
                common.update(Hz=repeated, Ex=-repeated, Ey=zeros)
            sim.monitor_results = [common]
            sim.sources = [{
                "kind": "point", "amplitude": 1.0,
                "t0": 20e-12, "tw": 5e-12,
                "f_min": None, "f_max": 100e9,
                "ix0": 1, "ix1": 1, "iy0": 1, "iy1": 1,
            }]

            spectrum = sim.power_spectrum(37, [frequency], 0)
            self.assertEqual(spectrum["power"].shape, (1,))
            self.assertTrue(spectrum["normalized"])
            self.assertTrue(np.all(spectrum["source_power"] > 0.0))
            np.testing.assert_allclose(
                spectrum["power"], spectrum["raw_power"] / spectrum["source_power"])
            self.assertTrue(np.all(spectrum["power"] > 0.0))
            self.assertTrue(np.all(np.isfinite(spectrum["power"])))

            fig, ax = sim.plot_power_spectrum(spectrum)
            self.assertEqual(len(ax.lines), 1)
            np.testing.assert_allclose(ax.lines[0].get_xdata(), [frequency / 1e9])
            np.testing.assert_allclose(ax.lines[0].get_ydata(), spectrum["power"])
            plt.close(fig)

            ff = sim.NF2FF(top=37, bottom=None, left=None, right=None,
                           freqs=[frequency], nphi=8, src_index=0)
            self.assertEqual(ff["monitor_indices"]["top"], 37)
            self.assertEqual(ff["phi"].shape, (8,))


if __name__ == "__main__":
    unittest.main()
