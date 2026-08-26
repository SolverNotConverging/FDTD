from pathlib import Path
import tempfile
import unittest

import numpy as np

from FDTD_2D_GR import FDTD_2D_GR, SchwarzschildGeometry


class TestSchwarzschildGeometry(unittest.TestCase):
    def test_landmarks_and_optical_index(self):
        geometry = SchwarzschildGeometry(2.0)
        self.assertAlmostEqual(geometry.horizon_isotropic_radius, 1.0)
        self.assertAlmostEqual(geometry.horizon_areal_radius, 4.0)
        self.assertAlmostEqual(geometry.photon_sphere_areal_radius, 6.0)

        rho_ph = geometry.photon_sphere_isotropic_radius
        self.assertAlmostEqual(geometry.areal_radius(rho_ph), 6.0, places=13)
        self.assertAlmostEqual(
            geometry.optical_circumference_radius(rho_ph),
            geometry.critical_impact_parameter,
            places=13,
        )
        self.assertAlmostEqual(
            geometry.photon_orbit_period,
            6.0 * np.pi * np.sqrt(3.0) * geometry.mass,
            places=13,
        )

    def test_horizon_is_rejected(self):
        geometry = SchwarzschildGeometry()
        with self.assertRaisesRegex(ValueError, "rho > M/2"):
            geometry.refractive_index(0.5)


class TestGRYeeGrid(unittest.TestCase):
    @staticmethod
    def make_closed_sim(**overrides):
        settings = dict(
            rho_min=0.8,
            rho_max=6.0,
            Nr=72,
            Nphi=144,
            courant=0.5,
            inner_sponge_width=0.0,
            outer_sponge_width=0.0,
            radial_boundary="pec",
        )
        settings.update(overrides)
        return FDTD_2D_GR(**settings)

    def test_shapes_and_explicit_cfl_validation(self):
        sim = self.make_closed_sim()
        self.assertEqual(sim.Hz.shape, (72, 144))
        self.assertEqual(sim.Er.shape, (72, 144))
        self.assertEqual(sim.Ephi.shape, (73, 144))
        with self.assertRaisesRegex(ValueError, "CFL"):
            self.make_closed_sim(dt=1.01 * sim.dt_cfl)

    def test_constraint_clean_launch_and_lossless_invariant(self):
        sim = self.make_closed_sim()
        sim.initialize_orbiting_packet(
            phi0=1.2,
            azimuthal_mode=6,
            radial_width=0.2,
            angular_width=0.4,
        )
        initial_divergence = np.max(np.abs(sim.electric_divergence()))
        initial_invariant = sim.conserved_energy()
        self.assertLess(initial_divergence, 1.0e-11)

        for _ in range(200):
            sim.step()

        final_divergence = np.max(np.abs(sim.electric_divergence()))
        final_invariant = sim.conserved_energy()
        self.assertLess(final_divergence, 2.0e-10)
        self.assertLess(
            abs(final_invariant / initial_invariant - 1.0),
            2.0e-13,
        )

    def test_rotational_equivariance_across_periodic_seam(self):
        first = self.make_closed_sim(Nr=56, Nphi=128)
        second = self.make_closed_sim(Nr=56, Nphi=128)
        cell_shift = 19
        phi0 = 0.37
        launch = dict(
            azimuthal_mode=8,
            radial_width=0.22,
            angular_width=0.36,
        )
        first.initialize_orbiting_packet(phi0=phi0, **launch)
        second.initialize_orbiting_packet(
            phi0=phi0 + cell_shift * first.dphi, **launch
        )

        for _ in range(30):
            first.step()
            second.step()

        np.testing.assert_allclose(
            second.Hz, np.roll(first.Hz, cell_shift, axis=1), rtol=0.0, atol=2e-13
        )
        np.testing.assert_allclose(
            second.Er, np.roll(first.Er, cell_shift, axis=1), rtol=0.0, atol=2e-13
        )
        np.testing.assert_allclose(
            second.Ephi,
            np.roll(first.Ephi, cell_shift, axis=1),
            rtol=0.0,
            atol=2e-13,
        )

    def test_launch_direction_controls_angular_motion(self):
        positive = self.make_closed_sim(Nr=64, Nphi=160)
        negative = self.make_closed_sim(Nr=64, Nphi=160)
        launch = dict(
            phi0=np.pi,
            azimuthal_mode=8,
            radial_width=0.24,
            angular_width=0.38,
        )
        positive.initialize_orbiting_packet(direction=1, **launch)
        negative.initialize_orbiting_packet(direction=-1, **launch)
        positive.step(40)
        negative.step(40)
        self.assertGreater(positive.diagnostics()["phi_mean"], np.pi)
        self.assertLess(negative.diagnostics()["phi_mean"], np.pi)

    def test_run_history_and_active_constraint(self):
        sim = FDTD_2D_GR(
            rho_min=0.7,
            rho_max=6.0,
            Nr=64,
            Nphi=128,
            courant=0.5,
            inner_sponge_width=0.3,
            outer_sponge_width=1.0,
        )
        sim.initialize_orbiting_packet(
            azimuthal_mode=6,
            radial_width=0.2,
            angular_width=0.4,
        )
        history = sim.run(steps=7, record_stride=3, store_snapshots=True)
        self.assertEqual(history["time"].shape, (4,))
        self.assertEqual(history["phi_coherence"].shape, (4,))
        self.assertEqual(len(history["snapshots"]), 4)
        self.assertEqual(history["snapshots"][0].shape, (64, 128))
        self.assertTrue(np.all(np.isfinite(history["energy"])))
        self.assertTrue(np.all((history["phi_coherence"] >= 0.0)))
        self.assertTrue(np.all((history["phi_coherence"] <= 1.0)))
        self.assertLess(history["divergence_linf"][0], 1.0e-10)

    def test_physical_scaling(self):
        sim = self.make_closed_sim()
        one_mass_length = sim.to_physical_length(sim.mass, 1.0)
        one_mass_time = sim.to_physical_time(sim.mass, 1.0)
        self.assertAlmostEqual(
            one_mass_length,
            sim.G_SI * sim.SOLAR_MASS_KG / sim.C_SI**2,
            places=12,
        )
        self.assertAlmostEqual(
            one_mass_time,
            sim.G_SI * sim.SOLAR_MASS_KG / sim.C_SI**3,
            places=18,
        )

    def test_mp4_animation_when_ffmpeg_is_available(self):
        from matplotlib.animation import writers

        if not writers.is_available("ffmpeg"):
            self.skipTest("FFmpeg is not installed")
        sim = self.make_closed_sim(Nr=16, Nphi=32, rho_max=4.0)
        sim.initialize_orbiting_packet(
            azimuthal_mode=4,
            radial_width=0.2,
            angular_width=0.4,
        )
        history = sim.run(steps=2, record_stride=1, store_snapshots=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "photon_packet.mp4"
            result = sim.save_animation(output, history, fps=5)
            self.assertEqual(result, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "fps"):
                sim.save_animation(output, history, fps=0)

    def test_cython_backend_matches_numpy_when_built(self):
        for radial_boundary, inner_width, outer_width in (
            ("pec", 0.0, 0.0),
            ("characteristic", 0.25, 0.6),
        ):
            options = dict(
                rho_min=0.8,
                rho_max=5.0,
                Nr=28,
                Nphi=56,
                courant=0.45,
                inner_sponge_width=inner_width,
                outer_sponge_width=outer_width,
                radial_boundary=radial_boundary,
            )
            reference = FDTD_2D_GR(**options).config("python")
            accelerated = FDTD_2D_GR(**options).config("cpu")
            if accelerated.backend != "cython":
                self.skipTest("optional GR Cython extension is not built")

            launch = dict(
                phi0=0.43,
                direction=-1,
                azimuthal_mode=6,
                radial_width=0.22,
                angular_width=0.40,
            )
            reference.initialize_orbiting_packet(**launch)
            accelerated.initialize_orbiting_packet(**launch)
            reference.step(13)
            accelerated.step(5)
            accelerated.step(8)

            self.assertEqual(accelerated.step_count, reference.step_count)
            self.assertAlmostEqual(accelerated.time, reference.time, places=13)
            for field in ("Hz", "Er", "Ephi"):
                np.testing.assert_allclose(
                    getattr(accelerated, field),
                    getattr(reference, field),
                    rtol=1e-12,
                    atol=2e-13,
                )


if __name__ == "__main__":
    unittest.main()
