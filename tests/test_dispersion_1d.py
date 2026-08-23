import unittest

import numpy as np

from FDTD_1D import DebyePole, DrudePole, FDTD_1D, LorentzPole
from FDTD_common import ADEState


class TestDispersion1D(unittest.TestCase):
    @staticmethod
    def _material(sim):
        return sim.add_material(
            "multipole",
            epsilon_r=(2.0, 3.0, 4.0),
            debye={
                "delta_epsilon": (1.0, 2.0, 3.0),
                "tau": (1e-9, 2e-9, 3e-9),
            },
            drude={
                "omega_p": (1e9, 2e9, 3e9),
                "gamma": (1e7, 2e7, 3e7),
            },
            lorentz={
                "delta_epsilon": (1.0, 2.0, 3.0),
                "omega_0": (1e9, 2e9, 3e9),
                "gamma": (1e7, 2e7, 3e7),
            },
        )

    def test_nondispersive_initialization_does_not_allocate_ade_arrays(self):
        sim = FDTD_1D(3.0, 3, 1.0, 1, dt=1e-12, subpixel=1)
        sim.add_material("lossy", epsilon_r=2.5, sigma_e=0.2)
        sim.add_object(material="lossy", region=slice(None))
        sim._init_mEy_mHx()
        self.assertIsNone(sim._ade_Ey)
        electric_ratio = (sim.sigma_e_Ey * sim.dt
                          / (2 * sim.eps0 * sim.ER_Ey))
        expected_ca = (sim.Ey_update_coeff * (1 - electric_ratio)
                       / (1 + electric_ratio))
        expected_m = (sim.Ey_update_coeff * sim.c0 * sim.dt
                      / (sim.ER_Ey * (1 + electric_ratio)))
        np.testing.assert_array_equal(sim.CaEy, expected_ca)
        np.testing.assert_array_equal(sim.mEy, expected_m)

        sim._cython_kernel = None
        sim.Ey[:] = (0.1, -0.2, 0.3, -0.4)
        sim.Hx[:] = (0.25, -0.5, 0.75)
        expected = sim.Ey.copy()
        for index in range(1, sim.Nz):
            expected[index] = (sim.CaEy[index] * expected[index]
                               + sim.mEy[index]
                               * (sim.Hx[index] - sim.Hx[index - 1]) / sim.dz)
        expected[0] = (sim.CaEy[0] * expected[0]
                       + sim.mEy[0] * sim.Hx[0] / sim.dz)
        expected[-1] = (sim.CaEy[-1] * expected[-1]
                        - sim.mEy[-1] * sim.Hx[-1] / sim.dz)
        sim.E_Update()
        np.testing.assert_array_equal(sim.Ey, expected)

    def test_poles_are_exported_and_y_component_is_rasterized_to_ey(self):
        self.assertTrue(all(cls is not None for cls in (
            DebyePole, DrudePole, LorentzPole)))
        sim = FDTD_1D(2.0, 2, 1.0, 1, subpixel=16)
        material = self._material(sim)

        # Half of the first cell is filled; the second remains vacuum.
        sim.add_object(material=material, region=(0.25, 0.75))

        self.assertEqual(sim.ER[0], 2.0)
        channels = {channel.kind: channel for channel in sim._pole_fields_Ey_cell}
        np.testing.assert_allclose(channels["debye"].strength, [1.0, 0.0])
        np.testing.assert_allclose(channels["drude"].strength, [2e18, 0.0])
        np.testing.assert_allclose(channels["lorentz"].strength, [4e18, 0.0])
        self.assertEqual(channels["debye"].parameters, (2e-9,))
        self.assertEqual(channels["drude"].parameters, (2e7,))
        self.assertEqual(channels["lorentz"].parameters, (2e9, 2e7))

        yee = {channel.kind: channel for channel in sim._pole_fields_Ey}
        np.testing.assert_allclose(yee["debye"].strength, [1.0, 0.5, 0.0])
        np.testing.assert_allclose(yee["drude"].strength, [2e18, 1e18, 0.0])
        np.testing.assert_allclose(yee["lorentz"].strength, [4e18, 2e18, 0.0])

    def test_direct_ordinary_object_convexly_replaces_pole_strength(self):
        sim = FDTD_1D(2.0, 2, 1.0, 1, subpixel=1)
        material = self._material(sim)
        sim.add_object(material=material, region=slice(None))
        sim._init_mEy_mHx()
        for pole in sim._ade_Ey.poles:
            pole["q"][:] = [1.0, 2.0, 3.0]
            if "v" in pole:
                pole["v"][:] = [4.0, 5.0, 6.0]
        sim.add_object(ER=5.0, MR=1.0, region=slice(0, 1))

        channels = {channel.kind: channel for channel in sim._pole_fields_Ey_cell}
        np.testing.assert_allclose(channels["debye"].strength, [0.0, 2.0])
        np.testing.assert_allclose(channels["drude"].strength, [0.0, 4e18])
        np.testing.assert_allclose(channels["lorentz"].strength, [0.0, 8e18])
        yee = {channel.kind: channel for channel in sim._pole_fields_Ey}
        np.testing.assert_allclose(yee["debye"].strength, [0.0, 1.0, 2.0])

        sim._init_mEy_mHx()
        for pole in sim._ade_Ey.poles:
            np.testing.assert_array_equal(pole["q"], [0.0, 1.0, 3.0])
            if "v" in pole:
                np.testing.assert_array_equal(pole["v"], [0.0, 2.5, 6.0])

    def test_dispersive_e_update_uses_ade_and_bypasses_cython(self):
        sim = FDTD_1D(3.0, 3, 1.0, 1, dt=1e-12, subpixel=1)
        material = self._material(sim)
        sim.add_object(material=material, region=slice(None))
        sim._init_mEy_mHx()

        class FailIfCalled:
            @staticmethod
            def update_e(*_args):
                raise AssertionError("the direct-E Cython kernel cannot update ADE state")

        sim._cython_kernel = FailIfCalled()
        sim.Ey[:] = [0.1, -0.2, 0.3, -0.4]
        sim.Hx[:] = [0.25, -0.5, 0.75]

        expected_field = sim.Ey.copy()
        expected_ade = ADEState(
            sim.ER_Ey, sim.sigma_e_Ey, sim.eps0, sim.dt,
            sim.c0 * sim.dt, sim._pole_fields_Ey, mask=sim.PEC_Ey,
        )
        curl_h = np.empty_like(sim.Ey)
        curl_h[0] = sim.Hx[0] / sim.dz
        curl_h[1:-1] = (sim.Hx[1:] - sim.Hx[:-1]) / sim.dz
        curl_h[-1] = -sim.Hx[-1] / sim.dz
        expected_ade.advance(expected_field, curl_h)

        sim.E_Update()

        np.testing.assert_allclose(sim.Ey, expected_field)
        for actual, expected in zip(sim._ade_Ey.poles, expected_ade.poles):
            np.testing.assert_allclose(actual["q"], expected["q"])
            if "v" in expected:
                np.testing.assert_allclose(actual["v"], expected["v"])

    def test_impressed_e_source_is_part_of_the_same_ade_endpoint(self):
        sim = FDTD_1D(3.0, 3, 1.0, 1, dt=1e-12, subpixel=1)
        sim.add_material(
            "debye", epsilon_r=2.0,
            debye={"delta_epsilon": 3.0, "tau": 1e-9},
        )
        sim.add_object(material="debye", region=slice(None))
        sim._init_mEy_mHx()
        sim.Ey.fill(0.25)
        sim.Hx.fill(0.0)
        source_index = 2
        source_curl = 1.75
        expected_e = (sim._ade_Ey.ca[source_index] * 0.25
                      + sim._ade_Ey.cb[source_index] * source_curl)

        sim.E_Update(source_index=source_index, source_curl=source_curl)

        self.assertAlmostEqual(sim.Ey[source_index], expected_e)
        pole = sim._ade_Ey.debye[0]
        self.assertAlmostEqual(
            pole["q"][source_index],
            pole["r"][source_index] * (0.25 + expected_e),
        )

    def test_absorbing_boundary_keeps_endpoint_polarization_synchronized(self):
        sim = FDTD_1D(3.0, 3, 1.0, 1, dt=1e-12, subpixel=1)
        sim.add_material(
            "debye", epsilon_r=2.0,
            debye={"delta_epsilon": 3.0, "tau": 1e-9},
        )
        sim.add_object(material="debye", region=slice(None))
        sim._init_mEy_mHx()
        sim.left_absorbing_boundary = True
        sim.Ey[:] = (0.5, 0.25, 0.0, 0.0)
        old_endpoint = sim.Ey[0]

        sim.E_Update()

        pole = sim._ade_Ey.debye[0]
        self.assertAlmostEqual(
            pole["q"][0],
            pole["r"][0] * (old_endpoint + sim.Ey[0]),
        )

    def test_rebuild_preserves_memory_and_pec_clears_it(self):
        sim = FDTD_1D(3.0, 3, 1.0, 1, dt=1e-12, subpixel=1)
        material = self._material(sim)
        sim.add_object(material=material, region=slice(None))
        sim._init_mEy_mHx()

        for offset, pole in enumerate(sim._ade_Ey.poles, start=1):
            pole["q"][:] = np.arange(sim.Nz + 1, dtype=float) + offset
            if "v" in pole:
                pole["v"][:] = 10.0 * np.arange(sim.Nz + 1) + offset
        old = [
            {name: value.copy() for name, value in pole.items()
             if name in {"q", "v"}}
            for pole in sim._ade_Ey.poles
        ]

        sim._init_mEy_mHx()
        for pole, memory in zip(sim._ade_Ey.poles, old):
            np.testing.assert_array_equal(pole["q"], memory["q"])
            if "v" in memory:
                np.testing.assert_array_equal(pole["v"], memory["v"])

        sim.add_object(material="PEC", region=slice(1, 2))
        sim._init_mEy_mHx()
        np.testing.assert_array_equal(sim.PEC_Ey, [False, True, True, False])
        for pole, memory in zip(sim._ade_Ey.poles, old):
            expected_q = memory["q"].copy()
            expected_q[sim.PEC_Ey] = 0.0
            np.testing.assert_array_equal(pole["q"], expected_q)
            if "v" in memory:
                expected_v = memory["v"].copy()
                expected_v[sim.PEC_Ey] = 0.0
                np.testing.assert_array_equal(pole["v"], expected_v)


if __name__ == "__main__":
    unittest.main()
