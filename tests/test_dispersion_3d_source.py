import unittest

import numpy as np

from FDTD_3D import FDTD_3D


class TestDispersive3DSourceEndpoint(unittest.TestCase):
    def test_sparse_soft_sources_update_all_pole_endpoints(self):
        sim = FDTD_3D(
            4e-3, 4e-3, 4e-3, 4, 4, 4, 10e9, 1,
            dt=1e-12, subpixel=1,
        ).config("python")
        material = sim.add_material(
            "mixed",
            epsilon_r=(2.0, 2.5, 3.0),
            debye={"delta_epsilon": (1.0, 1.5, 2.0), "tau": 1e-11},
            drude={"omega_p": (1e10, 2e10, 3e10), "gamma": 1e9},
            lorentz={
                "delta_epsilon": (0.5, 0.75, 1.0),
                "omega_0": 4e10,
                "gamma": 2e9,
            },
        )
        sim.add_block(material, (0, 4), (0, 4), (0, 4))

        amplitudes = (0.25, -0.5, 0.75)
        for polarization, amplitude in zip("xyz", amplitudes):
            sim.add_source(
                "point", 1, 1, 1, polarization=polarization,
                amplitude=amplitude, t0=sim.dt, tw=sim.dt,
            )
        # Exercise sequential overlapping sparse-source additions as well.
        sim.add_source(
            "point", 1, 1, 1, polarization="z",
            amplitude=-0.125, t0=sim.dt, tw=sim.dt,
        )

        sim.run(steps=1, progress=False)

        index = (1, 1, 1)
        expected_fields = (0.25, -0.5, 0.625)
        for field, state, expected_field in zip(
                (sim.Ex, sim.Ey, sim.Ez),
                (sim.ade_Ex, sim.ade_Ey, sim.ade_Ez),
                expected_fields):
            with self.subTest(component=field.shape):
                self.assertEqual(field[index], expected_field)
                self.assertEqual(len(state.debye), 1)
                self.assertEqual(len(state.oscillators), 2)
                np.testing.assert_allclose(
                    state.debye[0]["q"][index],
                    state.debye[0]["r"][index] * expected_field,
                    rtol=2e-15, atol=0.0,
                )
                for pole in state.oscillators:
                    expected_q = pole["r"][index] * expected_field
                    np.testing.assert_allclose(
                        pole["q"][index], expected_q,
                        rtol=2e-15, atol=0.0,
                    )
                    np.testing.assert_allclose(
                        pole["v"][index], 2.0 * expected_q / sim.dt,
                        rtol=2e-15, atol=0.0,
                    )


if __name__ == "__main__":
    unittest.main()
