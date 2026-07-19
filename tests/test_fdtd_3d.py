import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from FDTD_3D import FDTD_3D, Material
from FDTD_3D.FDTD_3D import _cython_kernel


def simulation(backend="python", steps=6):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = FDTD_3D(8e-3, 9e-3, 10e-3, 8, 9, 10, 5e9, steps, subpixel=1)
    sim.config(backend)
    return sim


class TestYeeGridAndMaterials(unittest.TestCase):
    def test_exact_component_shapes(self):
        sim = simulation()
        self.assertEqual(sim.Ex.shape, (8, 10, 11))
        self.assertEqual(sim.Ey.shape, (9, 9, 11))
        self.assertEqual(sim.Ez.shape, (9, 10, 10))
        self.assertEqual(sim.Hx.shape, (9, 9, 10))
        self.assertEqual(sim.Hy.shape, (8, 10, 10))
        self.assertEqual(sim.Hz.shape, (8, 9, 11))
        self.assertLessEqual(sim.dt, sim.dt_cfl)

    def test_named_anisotropic_material_and_block(self):
        sim = simulation()
        material = sim.add_material("crystal", epsilon_r=(2, 3, 4),
                                    mu_r=(1.1, 1.2, 1.3), sigma_e=(0.1, 0.2, 0.3))
        self.assertIsInstance(material, Material)
        sim.add_block("crystal", (2, 5), (3, 7), (4, 8))
        np.testing.assert_allclose(sim.ERxx[2:5, 3:7, 4:8], 2)
        np.testing.assert_allclose(sim.ERyy[2:5, 3:7, 4:8], 3)
        np.testing.assert_allclose(sim.ERzz[2:5, 3:7, 4:8], 4)
        self.assertEqual(sim.epsilon_Ex.shape, sim.Ex.shape)
        self.assertEqual(sim.mu_Hz.shape, sim.Hz.shape)
        direct = Material("direct", epsilon_r=2.5)
        self.assertEqual(direct.epsilon_r, (2.5, 2.5, 2.5))

    def test_native_pec_and_pmc_masks(self):
        sim = simulation()
        sim.add_block("PEC", (2, 4), (2, 5), (2, 6))
        sim.add_sphere("PMC", (6e-3, 6e-3, 6e-3), 0.7e-3)
        self.assertTrue(sim.PEC_cells[2:4, 2:5, 2:6].all())
        self.assertTrue(np.all(sim.CbEz[sim.PEC_Ez] == 0))
        self.assertTrue(np.all(sim.CbHx[sim.PMC_Hx] == 0))
        # Perfect conductors do not corrupt the ordinary material arrays.
        self.assertTrue(np.all(sim.ERxx == 1.0))
        self.assertTrue(np.all(sim.MRxx == 1.0))

    def test_cylinder_rasterization(self):
        sim = simulation()
        sim.add_material("dielectric", epsilon_r=5)
        sim.add_cylinder("dielectric", center=(4e-3, 4.5e-3, 5e-3),
                         radius=2e-3, height=4e-3, axis="z")
        self.assertGreater(np.count_nonzero(sim.ERxx > 1), 0)
        self.assertLess(np.count_nonzero(sim.ERxx > 1), sim.Nx*sim.Ny*sim.Nz)


class TestPMLSourceAndMonitor(unittest.TestCase):
    def test_cpml_grading(self):
        sim = simulation()
        sim.add_PML(2, direction="xz")
        self.assertGreater(sim._pml["x"]["node_k"][0], 1.0)
        self.assertGreater(sim._pml["z"]["node_k"][-1], 1.0)
        np.testing.assert_array_equal(sim._pml["y"]["node_k"], 1.0)
        self.assertTrue(np.any(sim._pml["x"]["node_c"] != 0.0))

    def test_cpml_absorbs_a_band_limited_pulse(self):
        backend = "cython" if _cython_kernel is not None else "python"
        def final_electric_energy(use_pml):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = FDTD_3D(0.12, 0.12, 0.12, 12, 12, 12,
                              f_min=2e9, f_max=4e9, Nt=220)
            sim.config(backend)
            if use_pml:
                sim.add_PML(2)
            sim.add_source("point", 6, 6, 6, polarization="z",
                           t0=0.35e-9, tw=0.1e-9)
            sim.run(progress=False)
            return sum(np.sum(getattr(sim, name)**2) for name in ("Ex", "Ey", "Ez"))
        reflected = final_electric_energy(False)
        absorbed = final_electric_energy(True)
        self.assertLess(absorbed, 1e-3 * reflected)

    def test_all_soft_source_types_and_monitor_shape(self):
        sim = simulation(steps=5)
        sim.add_source("point", 3, 4, 5, polarization="z", t0=0, tw=1e-10)
        sim.add_source("line", (2, 6), 3, 4, polarization="x", t0=0, tw=1e-10)
        sim.add_source("plane", 3, (2, 7), (2, 8), polarization="y", t0=0, tw=1e-10)
        monitor = sim.add_plane_monitor("x", 5, (1, 8), (1, 9), index=20)
        results = sim.run(progress=False)
        self.assertEqual(monitor, 20)
        self.assertEqual(results[0]["fields"].shape, (5, 7, 8, 6))
        self.assertTrue(np.isfinite(results[0]["fields"]).all())
        self.assertGreater(max(np.max(np.abs(sim.Ex)), np.max(np.abs(sim.Ey)),
                               np.max(np.abs(sim.Ez))), 0)

    def test_power_and_nf2ff_outputs(self):
        sim = simulation(steps=10)
        sim.add_PML(2)
        sim.add_source("point", 4, 4, 5, polarization="z", t0=0, tw=1e-10)
        box = sim.add_nf2ff_box((2, 7), (2, 7), (2, 8), start_index=10)
        sim.run(progress=False)
        power = sim.power_spectrum(box["x_max"], [1e9, 2e9])
        self.assertEqual(power["power"].shape, (2,))
        self.assertTrue(np.isfinite(power["power"]).all())
        ff = sim.NF2FF(box, [1e9], theta=np.linspace(0,np.pi,5),
                       phi=np.linspace(0,2*np.pi,8,endpoint=False),
                       source_index=0)
        self.assertEqual(ff["Etheta"].shape, (1, 5, 8))
        self.assertTrue(np.isfinite(ff["radiation_intensity"]).all())
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=sim.plot_nf2ff(ff,db=True,db_floor=-35)
        self.assertEqual(ax.name,"3d")
        plt.close(fig)

    def test_plane_monitor_hdf5_save_load_plot_and_animation(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with TemporaryDirectory() as directory:
            target=Path(directory)/"saved_plane"
            sim=simulation(steps=8)
            sim.add_source("point",3,4,5,polarization="z",t0=0,tw=1e-10)
            sim.add_plane_monitor("x",5,(1,8),(1,9),index=7,save_path=target)
            sim.run(progress=False)
            saved=target.with_suffix(".h5")
            self.assertTrue(saved.is_file())
            import h5py
            with h5py.File(saved,"r") as handle:
                self.assertEqual(handle.attrs["format"],"FDTD_3D_plane_monitor")
                self.assertTrue(handle["fields"].compression=="gzip")
            loaded=sim.load_plane_monitor(saved,index=71,register=True)
            np.testing.assert_allclose(loaded["fields"],sim.monitor_results[0]["fields"])
            self.assertIs(sim._monitor_by_id(71),loaded)
            fig,_=sim.plot_plane_monitor(saved,component="Ez",time_index=-1)
            plt.close(fig)
            fig,_=sim.plot_plane_monitor(loaded,component="E",frequency=1e9,window="hann")
            plt.close(fig)
            gif=Path(directory)/"monitor.gif"
            fig,animation=sim.animate_plane_monitor(saved,component="Ez",frame_stride=2,
                                                     interval=10,save_path=gif)
            self.assertTrue(gif.is_file())
            plt.close(fig)
            handle=getattr(animation,"_fdtd_h5_handle",None)
            if handle is not None and handle.id.valid:
                handle.close()

    def test_tqdm_chunking_preserves_fields_and_monitor_stride(self):
        from unittest.mock import patch
        backend="cython" if _cython_kernel is not None else "python"
        def make_case():
            sim=simulation(backend,steps=8)
            sim.add_source("point",3,4,5,polarization="z",t0=0,tw=1e-10)
            sim.add_plane_monitor("x",5,(1,8),(1,9))
            return sim
        reference=make_case(); reference.run(record_stride=3,progress=False)
        updates=[]
        class ProgressBar:
            def __init__(self,**kwargs):
                updates.append(("total",kwargs["total"]))
            def update(self,count): updates.append(("update",count))
            def close(self): updates.append(("closed",True))
        shown=make_case()
        with patch("tqdm.auto.tqdm",ProgressBar):
            shown.run(record_stride=3,progress=True)
        self.assertEqual(sum(value for kind,value in updates if kind=="update"),8)
        self.assertIn(("closed",True),updates)
        for name in ("Ex","Ey","Ez","Hx","Hy","Hz"):
            np.testing.assert_allclose(getattr(shown,name),getattr(reference,name),rtol=0,atol=0)
        np.testing.assert_allclose(shown.monitor_results[0]["fields"],
                                   reference.monitor_results[0]["fields"],rtol=0,atol=0)


@unittest.skipIf(_cython_kernel is None, "3D Cython extension is not built")
class TestCythonKernel(unittest.TestCase):
    def make_case(self, backend):
        sim = simulation(backend, steps=8)
        sim.add_material("lossy", epsilon_r=(2, 3, 4), mu_r=(1.1, 1.2, 1.3),
                         sigma_e=0.01)
        sim.add_block("lossy", (2, 5), (2, 6), (2, 7))
        sim.add_block("PMC", (6, 7), (4, 6), (4, 6))
        sim.add_PML(2)
        sim.add_source("point", 3, 4, 5, polarization="z", t0=0, tw=1e-10)
        sim.add_source("line", (2, 6), 3, 4, polarization="x", t0=0, tw=1e-10)
        sim.add_plane_monitor("x", 5, (1, 8), (1, 9))
        return sim

    def test_whole_run_matches_numpy_reference(self):
        python_sim = self.make_case("python")
        cython_sim = self.make_case("cython")
        python_sim.run(progress=False)
        cython_sim.run(progress=False)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                     "Psi_Hx_y", "Psi_Hy_z", "Psi_Ez_x"):
            np.testing.assert_allclose(getattr(cython_sim, name),
                                       getattr(python_sim, name), rtol=0, atol=0)
        np.testing.assert_allclose(cython_sim.monitor_results[0]["fields"],
                                   python_sim.monitor_results[0]["fields"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
