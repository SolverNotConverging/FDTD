import unittest

import numpy as np

from FDTD_2D_Ez import DebyePole, DrudePole, FDTD_2D_Ez, LorentzPole


class TestTMzDispersion(unittest.TestCase):
    @staticmethod
    def simulation():
        return FDTD_2D_Ez(
            2e-3, 2e-3, 2, 2, 10e9, 2, dt=1e-12,
            subpixel=1,
        ).config("python")

    def test_nondispersive_initialization_does_not_allocate_ade_arrays(self):
        sim = self.simulation()
        sim.add_material("lossy", epsilon_r=2.5, sigma_e=0.2)
        sim.add_rectangle(
            material="lossy", x_position=(0, 2), y_position=(0, 2))
        sim._init_Coeff()
        self.assertIsNone(sim._ade_Ez)
        ratio = sim.SIGEzz_Ez * sim.dt / (2 * sim.eps0 * sim.ERzz_Ez)
        expected_ca = sim.Ez_update_coeff * (1 - ratio) / (1 + ratio)
        expected_cb = (sim.Ez_update_coeff * sim.M
                       / (sim.ERzz_Ez * (1 + ratio)))
        np.testing.assert_array_equal(sim.CaEz, expected_ca)
        np.testing.assert_array_equal(sim.CbEz, expected_cb)

        rng = np.random.default_rng(17)
        sim.Ez[:] = rng.normal(size=sim.Ez.shape)
        sim.d_Hy_x[:] = rng.normal(size=sim.d_Hy_x.shape)
        sim.d_Hx_y[:] = rng.normal(size=sim.d_Hx_y.shape)
        sim.Psi_Dz_x[:] = rng.normal(size=sim.Psi_Dz_x.shape)
        sim.Psi_Dz_y[:] = rng.normal(size=sim.Psi_Dz_y.shape)
        curl = (sim.d_Hy_x / sim.kappa_x_Ez
                - sim.d_Hx_y / sim.kappa_y_Ez
                + sim.Psi_Dz_x - sim.Psi_Dz_y)
        expected = sim.CaEz * sim.Ez + sim.CbEz * curl
        sim.update_D()
        np.testing.assert_array_equal(sim.Ez, expected)

    def test_z_poles_rasterize_and_ade_history_survives_coefficient_init(self):
        sim = self.simulation()
        sim.add_material(
            "multipole",
            epsilon_r=(7.0, 8.0, 2.0),
            debye=DebyePole((0.0, 0.0, 4.0), 4e-12),
            drude=DrudePole((0.0, 0.0, 2e11), 1e10),
            lorentz=LorentzPole((0.0, 0.0, 3.0), 3e11, 2e10),
        )
        sim.add_rectangle(
            material="multipole", x_position=(0, 2), y_position=(0, 2))

        self.assertEqual([pole.kind for pole in sim._pole_fields_Ez],
                         ["debye", "drude", "lorentz"])
        np.testing.assert_allclose(sim._pole_fields_Ez[0].strength, 4.0)
        np.testing.assert_allclose(sim._pole_fields_Ez[1].strength, (2e11) ** 2)
        np.testing.assert_allclose(
            sim._pole_fields_Ez[2].strength, 3.0 * (3e11) ** 2)

        sim._init_Coeff()
        ade = sim._ade_Ez
        sim.Ez.fill(1.0)
        sim.d_Hx_y.fill(0.0)
        sim.d_Hy_x.fill(0.0)
        sim.update_D()
        histories = [pole["q"].copy() for pole in ade.poles]
        self.assertTrue(any(np.any(history != 0.0) for history in histories))

        sim._init_Coeff()
        self.assertIs(sim._ade_Ez, ade)
        for pole, expected in zip(ade.poles, histories):
            np.testing.assert_array_equal(pole["q"], expected)

    def test_soft_displacement_is_included_before_one_coupled_ade_solve(self):
        sim = self.simulation()
        sim.add_material(
            "debye", epsilon_r=2.0,
            debye={"delta_epsilon": 4.0, "tau": 4e-12})
        sim.add_rectangle(
            material="debye", x_position=(0, 2), y_position=(0, 2))
        sim._init_Coeff()
        sim.Ez.fill(0.0)
        sim.d_Hx_y.fill(0.0)
        sim.d_Hy_x.fill(0.0)

        sim.update_D(finalize=False)
        sim.Dz[1, 1] += 1.0
        expected_e = 1.0 / sim._ade_Ez.denominator[1, 1]
        sim.update_E()

        self.assertAlmostEqual(sim.Ez[1, 1], expected_e)
        self.assertAlmostEqual(
            sim._ade_Ez.debye[0]["q"][1, 1],
            sim._ade_Ez.debye[0]["r"][1, 1] * expected_e)
        np.testing.assert_allclose(
            sim.Dz, sim._ade_Ez.displacement(sim.Ez), rtol=0.0, atol=1e-15)

    def test_update_e_skips_duplicate_finalize_but_detects_changed_d(self):
        sim = self.simulation()
        sim.add_material(
            "debye", epsilon_r=2.0,
            debye={"delta_epsilon": 4.0, "tau": 4e-12})
        sim.add_rectangle(
            material="debye", x_position=(0, 2), y_position=(0, 2))
        sim._init_Coeff()
        sim.Ez.fill(0.2)
        sim.d_Hy_x.fill(1.0)
        sim.update_D()

        field_before = sim.Ez.copy()
        q_before = sim._ade_Ez.debye[0]["q"].copy()
        sim.update_E()
        np.testing.assert_array_equal(sim.Ez, field_before)
        np.testing.assert_array_equal(sim._ade_Ez.debye[0]["q"], q_before)

        sim.Dz[1, 1] += 0.125
        sim.update_E()
        self.assertNotEqual(sim.Ez[1, 1], field_before[1, 1])
        self.assertFalse(np.array_equal(sim._ade_Ez.debye[0]["q"], q_before))

    def test_direct_material_repaint_clears_dispersive_channels(self):
        painters = (
            (lambda sim: sim.add_rectangle(
                ER=3.0, MR=1.0, x_position=(0, 2), y_position=(0, 2)), True),
            (lambda sim: sim.add_circle(
                ER=3.0, MR=1.0, center=(1, 1), radius=2e-3), True),
            (lambda sim: sim.add_triangle(
                ER=3.0, MR=1.0,
                vertices=((0, 0), (2, 0), (0, 2))), False),
        )
        for paint, clears_all in painters:
            sim = self.simulation()
            sim.add_material(
                "debye", debye={"delta_epsilon": 4.0, "tau": 4e-12})
            sim.add_rectangle(
                material="debye", x_position=(0, 2), y_position=(0, 2))
            self.assertTrue(sim.has_dispersion)
            paint(sim)
            # Circle covers the full domain; the right triangle covers half.
            # In either case the directly painted cells contain no new poles.
            self.assertTrue(all(np.all(channel.strength >= 0.0)
                                for channel in sim._pole_fields_Ez))
            if clears_all:
                self.assertFalse(sim.has_dispersion)
            else:
                self.assertTrue(sim.has_dispersion)
                self.assertTrue(np.any(sim._pole_fields_Ez[0].strength == 0.0))

    def test_live_repaint_to_nondispersive_rebuilds_before_manual_update(self):
        sim = self.simulation()
        sim.add_material(
            "debye", epsilon_r=2.0,
            debye={"delta_epsilon": 4.0, "tau": 4e-12})
        sim.add_rectangle(
            material="debye", x_position=(0, 2), y_position=(0, 2))
        sim._init_Coeff()
        sim.Ez.fill(0.2)
        sim.d_Hy_x.fill(1.0)
        sim.add_rectangle(
            material="vacuum", x_position=(0, 2), y_position=(0, 2))
        self.assertFalse(sim.has_dispersion)

        sim.update_D()

        np.testing.assert_array_equal(sim.Ez, 0.2 + sim.M)
        self.assertIsNone(sim._ade_Ez)


if __name__ == "__main__":
    unittest.main()
