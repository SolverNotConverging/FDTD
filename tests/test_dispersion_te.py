import unittest

import numpy as np

from FDTD_2D_Hz import (
    DebyePole,
    DrudePole,
    FDTD_2D_Hz,
    LorentzPole,
)


def make_sim(nx=3, ny=3):
    return FDTD_2D_Hz(
        float(nx), float(ny), nx, ny, f_max=10e9, Nt=2,
        dt=1e-12, subpixel=1,
    ).config("python")


def add_full_dispersive_material(sim):
    material = sim.add_material(
        "multipole",
        epsilon_r=(2.0, 3.0, 1.0),
        sigma_e=(0.02, 0.03, 0.0),
        debye=DebyePole((1.0, 1.5, 0.0), (2e-11, 3e-11, 1.0)),
        drude=DrudePole((2e10, 3e10, 0.0), (4e9, 5e9, 0.0)),
        lorentz=LorentzPole(
            (0.5, 0.75, 0.0), (5e10, 6e10, 1.0), (2e9, 3e9, 0.0)),
    )
    sim.add_rectangle(
        material=material, x_position=(0, sim.Nx),
        y_position=(0, sim.Ny), subpixel=1,
    )
    return material


class TestTEzDispersiveGeometry(unittest.TestCase):
    def test_all_shapes_convexly_map_x_and_y_poles_to_faces(self):
        sim = make_sim(4, 4)
        material = sim.add_material(
            "relaxing",
            epsilon_r=(2.0, 3.0, 1.0),
            debye={"delta_epsilon": (2.0, 3.0, 0.0), "tau": 1e-9},
            drude={"omega_p": (2e9, 3e9, 0.0), "gamma": 1e8},
            lorentz={
                "delta_epsilon": (0.5, 0.75, 0.0),
                "omega_0": 4e9,
                "gamma": 2e8,
            },
        )

        sim.add_rectangle(
            material=material, x_position=(0, 1), y_position=(0, 1))
        sim.add_circle(
            material=material, center=(2.5, 2.5), radius=0.49, subpixel=1)
        sim.add_triangle(
            material=material, vertices=((1, 0), (2, 0), (1, 1)),
            subpixel=1)

        self.assertEqual(len(sim._pole_fields_Ex), 3)
        self.assertEqual(len(sim._pole_fields_Ey), 3)
        debye_x = next(p for p in sim._pole_fields_Ex if p.kind == "debye")
        debye_y = next(p for p in sim._pole_fields_Ey if p.kind == "debye")
        self.assertEqual(debye_x.strength[0, 0], 2.0)
        self.assertEqual(debye_y.strength[0, 0], 3.0)
        self.assertEqual(debye_x.strength[2, 2], 2.0)
        self.assertEqual(debye_y.strength[1, 0], 3.0)

        # Repainting half a cell with nondispersive vacuum attenuates every
        # existing pole strength instead of averaging its time constants.
        sim.add_rectangle(
            material="vacuum", x_position=(0.0, 0.5), y_position=(0, 1),
            subpixel=2)
        self.assertEqual(debye_x.strength[0, 0], 1.0)
        self.assertEqual(debye_y.strength[0, 0], 1.5)

        sim._average_material_to_yee()
        mapped_x = next(p for p in sim._pole_fields_Ex_yee if p.kind == "debye")
        mapped_y = next(p for p in sim._pole_fields_Ey_yee if p.kind == "debye")
        self.assertEqual(mapped_x.strength.shape, sim.Ex.shape)
        self.assertEqual(mapped_y.strength.shape, sim.Ey.shape)
        self.assertEqual(mapped_x.strength[0, 0], 1.0)
        self.assertEqual(mapped_x.strength[0, 1], 0.5)
        self.assertEqual(mapped_y.strength[0, 0], 1.5)


class TestTEzADEUpdate(unittest.TestCase):
    def test_one_step_updates_both_components_and_every_pole(self):
        sim = make_sim(2, 2)
        add_full_dispersive_material(sim)
        sim._init_Coeff()

        sim.Ex.fill(0.25)
        sim.Ey.fill(-0.4)
        sim.d_Hz_y.fill(2.0)
        sim.d_Hz_x.fill(-3.0)
        old_ex = sim.Ex.copy()
        old_ey = sim.Ey.copy()
        expected_ex = sim._ade_Ex.ca * old_ex + sim._ade_Ex.cb * 2.0
        expected_ey = sim._ade_Ey.ca * old_ey + sim._ade_Ey.cb * 3.0

        sim.update_D()

        np.testing.assert_allclose(sim.Ex, expected_ex, rtol=2e-15, atol=0.0)
        np.testing.assert_allclose(sim.Ey, expected_ey, rtol=2e-15, atol=0.0)
        for state, old_field, new_field in (
                (sim._ade_Ex, old_ex, sim.Ex),
                (sim._ade_Ey, old_ey, sim.Ey)):
            field_sum = old_field + new_field
            for pole in state.debye:
                np.testing.assert_allclose(pole["q"], pole["r"] * field_sum)
            for pole in state.oscillators:
                expected_q = pole["r"] * field_sum
                np.testing.assert_allclose(pole["q"], expected_q)
                np.testing.assert_allclose(pole["v"], 2.0 * expected_q / sim.dt)

        np.testing.assert_allclose(
            sim.Dx, sim._ade_Ex.displacement(sim.Ex), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            sim.Dy, sim._ade_Ey.displacement(sim.Ey), rtol=0.0, atol=0.0)

    def test_deferred_and_immediate_finalize_match_and_state_persists(self):
        immediate = make_sim(2, 2)
        deferred = make_sim(2, 2)
        for sim in (immediate, deferred):
            add_full_dispersive_material(sim)
            sim._init_Coeff()
            sim.Ex.fill(0.2)
            sim.Ey.fill(0.3)
            sim.d_Hz_y.fill(1.5)
            sim.d_Hz_x.fill(0.75)

        immediate.update_D()
        deferred.update_D(finalize=False)
        deferred.update_E()

        for name in ("Ex", "Ey", "Dx", "Dy"):
            np.testing.assert_allclose(
                getattr(deferred, name), getattr(immediate, name),
                rtol=0.0, atol=0.0)
        for first, second in zip(
                deferred._ade_Ex.poles, immediate._ade_Ex.poles):
            np.testing.assert_allclose(first["q"], second["q"], rtol=0.0, atol=0.0)
            if "v" in first:
                np.testing.assert_allclose(first["v"], second["v"], rtol=0.0, atol=0.0)

        ade_ex = deferred._ade_Ex
        q_before = [pole["q"].copy() for pole in ade_ex.poles]
        deferred._init_Coeff()
        self.assertIs(deferred._ade_Ex, ade_ex)
        for pole, expected in zip(deferred._ade_Ex.poles, q_before):
            np.testing.assert_array_equal(pole["q"], expected)

    def test_update_e_skips_duplicate_finalize_but_detects_changed_d(self):
        sim = make_sim(2, 2)
        add_full_dispersive_material(sim)
        sim._init_Coeff()
        sim.Ex.fill(0.2)
        sim.Ey.fill(0.3)
        sim.d_Hz_y.fill(1.0)
        sim.d_Hz_x.fill(0.5)
        sim.update_D()

        ex_before = sim.Ex.copy()
        ey_before = sim.Ey.copy()
        q_before = [pole["q"].copy() for pole in sim._ade_Ex.poles]
        qy_before = [pole["q"].copy() for pole in sim._ade_Ey.poles]
        sim.update_E()
        np.testing.assert_array_equal(sim.Ex, ex_before)
        np.testing.assert_array_equal(sim.Ey, ey_before)
        for pole, expected in zip(sim._ade_Ex.poles, q_before):
            np.testing.assert_array_equal(pole["q"], expected)

        sim.Dx[0, 0] += 0.125
        sim.update_E()
        self.assertNotEqual(sim.Ex[0, 0], ex_before[0, 0])
        np.testing.assert_array_equal(sim.Ey, ey_before)
        for pole, expected in zip(sim._ade_Ey.poles, qy_before):
            np.testing.assert_array_equal(pole["q"], expected)
        self.assertTrue(any(
            not np.array_equal(pole["q"], expected)
            for pole, expected in zip(sim._ade_Ex.poles, q_before)))

    def test_live_pec_repaint_clears_fields_and_ade_memory(self):
        sim = make_sim(2, 2)
        add_full_dispersive_material(sim)
        sim._init_Coeff()
        sim.Ex.fill(1.0)
        sim.Ey.fill(1.0)
        sim.update_D()
        sim.add_rectangle(
            material="PEC", x_position=(0, 1), y_position=(0, 1))

        sim.update_E()

        np.testing.assert_array_equal(sim.Ex[sim.PEC_Ex], 0.0)
        np.testing.assert_array_equal(sim.Ey[sim.PEC_Ey], 0.0)
        for state, mask in ((sim._ade_Ex, sim.PEC_Ex),
                            (sim._ade_Ey, sim.PEC_Ey)):
            for pole in state.poles:
                np.testing.assert_array_equal(pole["q"][mask], 0.0)
                if "v" in pole:
                    np.testing.assert_array_equal(pole["v"][mask], 0.0)

    def test_nondispersive_update_keeps_original_arithmetic(self):
        sim = make_sim(3, 2)
        sim.add_material(
            "lossy", epsilon_r=(2.0, 3.0, 1.0),
            sigma_e=(0.2, 0.3, 0.0))
        sim.add_rectangle(
            material="lossy", x_position=(0, 3), y_position=(0, 2))
        sim._init_Coeff()
        self.assertIsNone(sim._ade_Ex)
        self.assertIsNone(sim._ade_Ey)
        rng = np.random.default_rng(41)
        sim.Ex[:] = rng.normal(size=sim.Ex.shape)
        sim.Ey[:] = rng.normal(size=sim.Ey.shape)
        sim.d_Hz_y[:] = rng.normal(size=sim.d_Hz_y.shape)
        sim.d_Hz_x[:] = rng.normal(size=sim.d_Hz_x.shape)
        sim.Psi_Dx_y[:] = rng.normal(size=sim.Psi_Dx_y.shape)
        sim.Psi_Dy_x[:] = rng.normal(size=sim.Psi_Dy_x.shape)

        expected_ex = sim.CaEx * sim.Ex + sim.CbEx * (
            sim.d_Hz_y / sim.kappa_y_Ex + sim.Psi_Dx_y)
        expected_ey = sim.CaEy * sim.Ey - sim.CbEy * (
            sim.d_Hz_x / sim.kappa_x_Ey + sim.Psi_Dy_x)
        sim.update_D()
        np.testing.assert_array_equal(sim.Ex, expected_ex)
        np.testing.assert_array_equal(sim.Ey, expected_ey)

    def test_dispersive_resident_cuda_selection_warns_and_runs_on_host(self):
        sim = make_sim(2, 2)
        sim.Nt = 1
        add_full_dispersive_material(sim)
        # Exercise backend routing without requiring a CUDA driver: the ADE
        # decision must happen before importing or launching the resident loop.
        sim.backend = "numba_cuda"
        sim._use_numba_cuda = True
        with self.assertWarnsRegex(RuntimeWarning, "host update loops"):
            sim.run(is_include_history=False)
        self.assertTrue(sim._ade_cuda_host_fallback)
        self.assertTrue(np.isfinite(sim.Ex).all())
        self.assertTrue(np.isfinite(sim.Ey).all())


if __name__ == "__main__":
    unittest.main()
