import unittest

import numpy as np
from matplotlib import pyplot as plt

from FDTD_1D.FDTD_1D import FDTD_1D


class TestYeeGrid(unittest.TestCase):
    def test_field_shapes_and_locations(self):
        sim = FDTD_1D(z_range=2.0, Nz=2, f_max=1.0, Nt=2)

        self.assertEqual(sim.Ey.shape, (3,))
        self.assertEqual(sim.Hx.shape, (2,))
        np.testing.assert_allclose(sim.z_Ey, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(sim.z_Hx, [0.5, 1.5])

    def test_subpixel_cell_assignment_then_yee_average(self):
        sim = FDTD_1D(z_range=2.0, Nz=2, f_max=1.0, Nt=2, subpixel=16)
        sim.add_object(ER=4.0, MR=3.0, region=(0.25, 0.75))

        np.testing.assert_allclose(sim.ER, [2.5, 1.0])
        np.testing.assert_allclose(sim.MR, [2.0, 1.0])
        np.testing.assert_allclose(sim.ER_Ey, [2.5, 1.75, 1.0])
        np.testing.assert_allclose(sim.MR_Hx, [2.0, 1.0])

    def test_python_update_loops_use_all_yee_differences(self):
        sim = FDTD_1D(z_range=3.0, Nz=3, f_max=1.0, Nt=2, dt=1.0)
        sim._cython_kernel = None
        sim.mHx[:] = 1.0
        sim.Ey[:] = [0.0, 1.0, 3.0, 6.0]
        sim.H_Update()
        np.testing.assert_allclose(sim.Hx, [1.0, 2.0, 3.0])

        sim.mEy[:] = 1.0
        sim.left_absorbing_boundary = False
        sim.right_absorbing_boundary = False
        sim.Ey[:] = 0.0
        sim.E_Update()
        np.testing.assert_allclose(sim.Ey, [1.0, 1.0, 1.0, -3.0])

    def test_pec_pmc_objects_constrain_yee_updates_without_changing_material(self):
        sim = FDTD_1D(z_range=4.0, Nz=4, f_max=1.0, Nt=2)
        er_before = sim.ER.copy()
        mr_before = sim.MR.copy()

        sim.add_object(material="PEC", region=slice(1, 2), subpixel=0)
        sim.add_object(ER="PMC", region=(2.0, 3.0), subpixel=0)
        self.assertTrue(sim.PEC_cells[1])
        np.testing.assert_array_equal(sim.PEC_Ey, [False, True, True, False, False])
        self.assertTrue(sim.PMC_cells[2])
        self.assertTrue(sim.PMC_Hx[2])
        np.testing.assert_array_equal(sim.ER, er_before)
        np.testing.assert_array_equal(sim.MR, mr_before)

        sim._init_mEy_mHx()
        self.assertTrue(np.all(sim.mEy[sim.PEC_Ey] == 0.0))
        self.assertTrue(np.all(sim.mHx[sim.PMC_Hx] == 0.0))
        sim.Ey.fill(1.0)
        sim.Hx.fill(1.0)
        sim.H_Update()
        sim.E_Update()
        self.assertTrue(np.all(sim.Ey[sim.PEC_Ey] == 0.0))
        self.assertTrue(np.all(sim.Hx[sim.PMC_Hx] == 0.0))

    def test_pec_pmc_boundaries_use_masks_and_plot_styles(self):
        sim = FDTD_1D(z_range=4.0, Nz=4, f_max=1.0, Nt=2)
        sim.set_boundary(left="PEC", right="PMC")
        self.assertTrue(sim.PEC_Ey[0])
        self.assertTrue(sim.PMC_Hx[-1])
        np.testing.assert_array_equal(sim.ER, np.ones(4))
        np.testing.assert_array_equal(sim.MR, np.ones(4))

        sim.add_object(material="PEC", region=slice(1, 2))
        sim.add_object(material="PMC", region=slice(2, 3))
        fig, ax = plt.subplots()
        artists = sim._draw_conductor_regions(ax)
        self.assertEqual(len(artists), 4)
        self.assertTrue(all(artist.get_linestyle() == "--" for artist in artists))
        labels = set(ax.get_legend_handles_labels()[1])
        self.assertEqual(labels, {"PEC", "PMC"})
        plt.close(fig)

    @unittest.skipUnless(FDTD_1D(1.0, 2, 1.0, 1)._use_cython_kernel,
                         "optional Cython extension is not built")
    def test_cython_and_python_updates_match(self):
        python_sim = FDTD_1D(1.0, 4, 1.0, 1)
        cython_sim = FDTD_1D(1.0, 4, 1.0, 1)
        python_sim._cython_kernel = None

        for sim in (python_sim, cython_sim):
            sim.Ey[:] = [0.0, 0.5, -0.25, 1.0, 0.75]
            sim.Hx[:] = [0.1, -0.2, 0.3, -0.4]
            sim.mEy[:] = np.linspace(0.1, 0.5, 5)
            sim.mHx[:] = np.linspace(0.2, 0.8, 4)
            sim.H_Update()
            sim.E_Update()

        np.testing.assert_allclose(cython_sim.Hx, python_sim.Hx)
        np.testing.assert_allclose(cython_sim.Ey, python_sim.Ey)


if __name__ == "__main__":
    unittest.main()
