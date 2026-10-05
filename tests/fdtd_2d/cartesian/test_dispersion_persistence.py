import tempfile
import unittest
from pathlib import Path

import numpy as np

from FDTD_2D_Ez import FDTD_2D_Ez
from FDTD_2D_Hz import FDTD_2D_Hz


def assert_ade_equal(testcase, first, second):
    testcase.assertEqual(len(first.poles), len(second.poles))
    for actual, expected in zip(first.poles, second.poles):
        testcase.assertEqual(actual["kind"], expected["kind"])
        np.testing.assert_array_equal(actual["q"], expected["q"])
        if "v" in actual:
            np.testing.assert_array_equal(actual["v"], expected["v"])


class TestDispersivePersistence(unittest.TestCase):
    def test_tm_pickle_continues_live_ade_state_exactly(self):
        sim = FDTD_2D_Ez(
            2e-3, 2e-3, 2, 2, 10e9, 2, dt=1e-12, subpixel=1,
        ).config("python")
        sim.add_material(
            "mixed", epsilon_r=2.0,
            debye={"delta_epsilon": 1.0, "tau": 1e-11},
            drude={"omega_p": 2e10, "gamma": 1e9},
            lorentz={"delta_epsilon": 0.5, "omega_0": 3e10,
                     "gamma": 2e9},
        )
        sim.add_rectangle(
            material="mixed", x_position=(0, 2), y_position=(0, 2))
        sim._init_Coeff()
        sim.Ez.fill(0.2)
        sim.d_Hy_x.fill(0.75)
        sim.update_D()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tm.pkl"
            sim.save(path, include_histories=False)
            restored = FDTD_2D_Ez.load(path)

        np.testing.assert_array_equal(restored.Ez, sim.Ez)
        np.testing.assert_array_equal(restored.Dz, sim.Dz)
        assert_ade_equal(self, restored._ade_Ez, sim._ade_Ez)
        restored.update_E()
        assert_ade_equal(self, restored._ade_Ez, sim._ade_Ez)

        for item in (sim, restored):
            item.d_Hy_x.fill(-0.25)
            item.d_Hx_y.fill(0.1)
            item.update_D()
        np.testing.assert_array_equal(restored.Ez, sim.Ez)
        np.testing.assert_array_equal(restored.Dz, sim.Dz)
        assert_ade_equal(self, restored._ade_Ez, sim._ade_Ez)

    def test_te_pickle_continues_both_live_ade_states_exactly(self):
        sim = FDTD_2D_Hz(
            2e-3, 2e-3, 2, 2, 10e9, 2, dt=1e-12, subpixel=1,
        ).config("python")
        sim.add_material(
            "mixed", epsilon_r=(2.0, 2.5, 1.0),
            debye={"delta_epsilon": (1.0, 1.5, 0.0), "tau": 1e-11},
            drude={"omega_p": (2e10, 2.5e10, 0.0), "gamma": 1e9},
            lorentz={"delta_epsilon": (0.5, 0.75, 0.0),
                     "omega_0": 3e10, "gamma": 2e9},
        )
        sim.add_rectangle(
            material="mixed", x_position=(0, 2), y_position=(0, 2))
        sim._init_Coeff()
        sim.Ex.fill(0.2)
        sim.Ey.fill(-0.3)
        sim.d_Hz_y.fill(0.75)
        sim.d_Hz_x.fill(-0.5)
        sim.update_D()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "te.pkl"
            sim.save(path, include_histories=False)
            restored = FDTD_2D_Hz.load(path)

        for name in ("Ex", "Ey", "Dx", "Dy"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(sim, name))
        assert_ade_equal(self, restored._ade_Ex, sim._ade_Ex)
        assert_ade_equal(self, restored._ade_Ey, sim._ade_Ey)
        restored.update_E()
        assert_ade_equal(self, restored._ade_Ex, sim._ade_Ex)
        assert_ade_equal(self, restored._ade_Ey, sim._ade_Ey)

        for item in (sim, restored):
            item.d_Hz_y.fill(-0.25)
            item.d_Hz_x.fill(0.1)
            item.update_D()
        for name in ("Ex", "Ey", "Dx", "Dy"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(sim, name))
        assert_ade_equal(self, restored._ade_Ex, sim._ade_Ex)
        assert_ade_equal(self, restored._ade_Ey, sim._ade_Ey)


if __name__ == "__main__":
    unittest.main()
