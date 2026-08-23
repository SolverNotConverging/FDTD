import unittest

import numpy as np

from FDTD_common import (
    ADEState,
    DebyePole,
    DrudePole,
    LorentzPole,
    Material,
    PoleField,
)
from FDTD_3D import FDTD_3D


class TestDispersiveMaterial(unittest.TestCase):
    def test_multipole_aliases_and_anisotropic_normalization(self):
        material = Material(
            "mixed", epsilon_r=(2, 3, 4),
            debye={"strength": (1, 2, 3), "relaxation_time": 2e-12},
            drude={"plasma_frequency": (1e12, 2e12, 3e12),
                   "collision_frequency": 4e10},
            lorentz=[
                {"delta_eps": 0.5, "resonance_frequency": 5e12,
                 "damping": 2e10},
                LorentzPole(0.25, 8e12, 3e10),
            ],
        )
        self.assertTrue(material.has_dispersion)
        self.assertEqual(material.epsilon_r, (2.0, 3.0, 4.0))
        self.assertEqual(material.debye[0].delta_epsilon, (1.0, 2.0, 3.0))
        self.assertEqual(material.drude[0].omega_p[2], 3e12)
        self.assertEqual(len(material.lorentz), 2)

        omega = np.array((1e11, 2e11))
        epsilon = material.relative_permittivity(omega)
        self.assertEqual(epsilon.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(epsilon)))

    def test_pole_validation_and_conductor_rejection(self):
        invalid = (
            lambda: DebyePole(1, 0),
            lambda: DrudePole(-1, 0),
            lambda: LorentzPole(1, 0, 0),
            lambda: LorentzPole(-1, 1, 0),
            lambda: Material("bad", debye={"delta_epsilon": 1, "unknown": 2}),
            lambda: Material("pec", kind="PEC", drude=(1, 0)),
        )
        for constructor in invalid:
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()

    def test_three_positional_lorentz_entries_are_three_poles(self):
        material = Material(
            "three-lorentz",
            lorentz=[
                (0.5, 1e11, 1e9),
                (0.7, 2e11, 2e9),
                (0.9, 3e11, 3e9),
            ],
        )
        self.assertEqual(len(material.lorentz), 3)
        self.assertEqual(
            [pole.omega_0[0] for pole in material.lorentz],
            [1e11, 2e11, 3e11],
        )

        anisotropic = Material("anisotropic-lorentz", lorentz={
            "delta_epsilon": (0.5, 0.7, 0.9),
            "omega_0": (1e11, 2e11, 3e11),
            "gamma": (1e9, 2e9, 3e9),
        })
        self.assertEqual(len(anisotropic.lorentz), 1)
        self.assertEqual(anisotropic.lorentz[0].omega_0,
                         (1e11, 2e11, 3e11))


class TestADEState(unittest.TestCase):
    def test_first_step_combines_debye_drude_lorentz_and_conductivity(self):
        dt = 2e-13
        eps0 = 8.8541878128e-12
        shape = (3,)
        channels = [
            PoleField("debye", (3e-12,), np.full(shape, 1.5)),
            PoleField("drude", (2e10,), np.full(shape, (4e12) ** 2)),
            PoleField("lorentz", (5e12, 3e10),
                      np.full(shape, 0.75 * (5e12) ** 2)),
        ]
        state = ADEState(np.full(shape, 2.5), np.full(shape, 0.2), eps0,
                         dt, curl_scale=3.0, pole_fields=channels)
        field = np.ones(shape)
        curl = np.array((0.25, -0.5, 0.0))

        expected = state.ca + state.cb * curl
        displacement = state.advance(field, curl)

        np.testing.assert_allclose(field, expected, rtol=1e-14, atol=0)
        np.testing.assert_allclose(
            displacement, state.epsilon_r * field
            + sum(pole["q"] for pole in state.poles), rtol=1e-14, atol=0)
        for pole in state.poles:
            self.assertGreater(np.max(np.abs(pole["q"])), 0.0)
        state.reset()
        for pole in state.poles:
            np.testing.assert_array_equal(pole["q"], 0.0)

    def test_zero_poles_reproduce_centered_conductive_coefficients(self):
        epsilon_r = np.array((2.0, 3.0))
        sigma = np.array((0.1, 0.2))
        eps0 = 8.85e-12
        dt = 1e-12
        scale = 7.0
        state = ADEState(epsilon_r, sigma, eps0, dt, scale)
        ratio = sigma * dt / (2 * eps0 * epsilon_r)
        np.testing.assert_allclose(state.ca, (1 - ratio) / (1 + ratio))
        np.testing.assert_allclose(
            state.cb, scale / (epsilon_r * (1 + ratio)))

    def test_history_copy_preserves_unchanged_and_scales_repainted_strength(self):
        shape = (3,)
        old = ADEState(
            np.ones(shape), np.zeros(shape), 1.0, 0.1, 0.1,
            [PoleField("debye", (1.0,), np.array((1.0, 2.0, 0.0)))],
        )
        old.debye[0]["q"][:] = (3.0, 4.0, 5.0)
        new = ADEState(
            np.ones(shape), np.zeros(shape), 1.0, 0.1, 0.1,
            [PoleField("debye", (1.0,), np.array((1.0, 1.0, 2.0)))],
            mask=np.array((False, False, True)),
        ).copy_history_from(old)
        np.testing.assert_allclose(new.debye[0]["q"], (3.0, 2.0, 0.0))

    def test_scalar_masked_index_is_safe(self):
        field = np.ones(2)
        state = ADEState(
            np.ones(2), np.zeros(2), 1.0, 0.1, 0.1,
            [PoleField("debye", (1.0,), np.ones(2))],
            mask=np.array((True, False)),
        )
        displacement = state.solve_displacement(field, 2.0, index=0)
        self.assertEqual(field[0], 0.0)
        self.assertEqual(state.debye[0]["q"][0], 0.0)
        self.assertEqual(displacement, 0.0)


class TestDispersive3D(unittest.TestCase):
    @staticmethod
    def simulation():
        return FDTD_3D(
            4e-3, 4e-3, 4e-3, 4, 4, 4, 10e9, 4,
            dt=1e-12, subpixel=1,
        ).config("python")

    def test_anisotropic_poles_map_to_all_three_electric_components(self):
        sim = self.simulation()
        material = sim.add_material(
            "dispersive", epsilon_r=(2, 3, 4),
            debye={"delta_epsilon": (1, 2, 3), "tau": (1e-11, 2e-11, 3e-11)},
            drude={"omega_p": (1e10, 2e10, 3e10), "gamma": 1e9},
            lorentz={"delta_epsilon": (0.5, 0.75, 1.0),
                     "omega_0": (2e10, 3e10, 4e10), "gamma": 2e9},
        )
        sim.add_block(material, (0, 4), (0, 4), (0, 4))
        self.assertEqual([len(state.poles) for state in
                          (sim.ade_Ex, sim.ade_Ey, sim.ade_Ez)], [3, 3, 3])
        np.testing.assert_allclose(sim._dispersion[0][0].strength, 1.0)
        np.testing.assert_allclose(sim._dispersion[1][0].strength, 2.0)
        np.testing.assert_allclose(sim._dispersion[2][0].strength, 3.0)

    def test_numpy_e_update_advances_and_reset_clears_ade(self):
        sim = self.simulation()
        sim.add_material("relaxing", epsilon_r=2,
                         debye={"delta_epsilon": 3, "tau": 1e-11},
                         drude={"omega_p": 2e10, "gamma": 1e9},
                         lorentz={"delta_epsilon": 1, "omega_0": 3e10,
                                  "gamma": 2e9})
        sim.add_block("relaxing", (0, 4), (0, 4), (0, 4))
        sim.Ex[:, 1:-1, 1:-1] = 1.0
        sim._update_e_numpy()
        self.assertGreater(max(np.max(np.abs(pole["q"]))
                               for pole in sim.ade_Ex.poles), 0.0)
        self.assertTrue(np.all(np.isfinite(sim.Ex)))
        sim.reset_fields()
        for state in (sim.ade_Ex, sim.ade_Ey, sim.ade_Ez):
            for pole in state.poles:
                np.testing.assert_array_equal(pole["q"], 0.0)

    def test_nondispersive_coefficients_keep_original_arithmetic(self):
        sim = self.simulation()
        sim.add_material("lossy", epsilon_r=(2.0, 3.0, 4.0),
                         sigma_e=(0.1, 0.2, 0.3))
        sim.add_block("lossy", (0, 4), (0, 4), (0, 4))
        self.assertIsNone(sim.ade_Ex)
        self.assertIsNone(sim.ade_Ey)
        self.assertIsNone(sim.ade_Ez)
        for component, epsilon, sigma, mask, actual_ca, actual_cb in (
            (0, sim.epsilon_Ex, sim._edge_average(sim._sigma_e[0], 0),
             sim.PEC_Ex, sim.CaEx, sim.CbEx),
            (1, sim.epsilon_Ey, sim._edge_average(sim._sigma_e[1], 1),
             sim.PEC_Ey, sim.CaEy, sim.CbEy),
            (2, sim.epsilon_Ez, sim._edge_average(sim._sigma_e[2], 2),
             sim.PEC_Ez, sim.CaEz, sim.CbEz),
        ):
            with self.subTest(component=component):
                expected_ca, expected_cb = sim._loss_coeff(epsilon, sigma, mask)
                np.testing.assert_array_equal(actual_ca, expected_ca)
                np.testing.assert_array_equal(actual_cb, expected_cb)


if __name__ == "__main__":
    unittest.main()
