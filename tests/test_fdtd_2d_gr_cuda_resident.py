import os
import importlib.util
import subprocess
import sys
import textwrap
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestGRDeviceResidentCuda(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numba") is not None,
        "Numba is an optional CUDA-backend dependency",
    )
    def test_cuda_simulator_matches_numpy_without_step_transfers(self):
        script = textwrap.dedent(r"""
            import numpy as np

            from FDTD_2D_GR import FDTD_2D_GR
            from FDTD_2D_GR import cuda_gr

            cases = (
                (np.complex128, "characteristic", 0.25, 0.6, 1e-12, 2e-13),
                (np.complex64, "pec", 0.0, 0.0, 3e-5, 3e-6),
            )
            for dtype, boundary, inner, outer, rtol, atol in cases:
                options = dict(
                    rho_min=0.8,
                    rho_max=5.0,
                    Nr=24,
                    Nphi=48,
                    courant=0.45,
                    inner_sponge_width=inner,
                    outer_sponge_width=outer,
                    radial_boundary=boundary,
                    dtype=dtype,
                )
                reference = FDTD_2D_GR(**options).config("python")
                accelerated = FDTD_2D_GR(**options).config("gpu")
                assert accelerated.backend == "numba_cuda"
                launch = dict(
                    phi0=0.43,
                    direction=1,
                    azimuthal_mode=6,
                    radial_width=0.22,
                    angular_width=0.40,
                )
                reference.initialize_orbiting_packet(**launch)
                accelerated.initialize_orbiting_packet(**launch)

                reference.step(11)
                accelerated.step(1)
                accelerated.step(3)
                accelerated.step(7)
                assert cuda_gr.host_is_dirty(accelerated)
                stats = cuda_gr.transfer_stats(accelerated)
                assert stats["host_to_device_during_steps"] == 0
                assert stats["device_to_host_during_steps"] == 0
                assert stats["host_syncs"] == 0
                accelerated.sync_fields()
                assert not cuda_gr.host_is_dirty(accelerated)
                assert accelerated.step_count == reference.step_count == 11
                np.testing.assert_allclose(
                    accelerated.time, reference.time, rtol=0.0, atol=1e-14
                )
                for field in ("Hz", "Er", "Ephi"):
                    np.testing.assert_allclose(
                        getattr(accelerated, field),
                        getattr(reference, field),
                        rtol=rtol,
                        atol=atol,
                    )

                # Raw-field plots must synchronize device-resident state too;
                # otherwise real_hz/abs_hz would silently show the previous step.
                reference.step(1)
                accelerated.step(1)
                assert cuda_gr.host_is_dirty(accelerated)
                ax, _ = accelerated.plot_snapshot(
                    quantity="real_hz", colorbar=False
                )
                assert not cuda_gr.host_is_dirty(accelerated)
                for field in ("Hz", "Er", "Ephi"):
                    np.testing.assert_allclose(
                        getattr(accelerated, field),
                        getattr(reference, field),
                        rtol=rtol,
                        atol=atol,
                    )
                import matplotlib.pyplot as plt
                plt.close(ax.figure)

            # The high-level run loop chunks device work at history samples and
            # synchronizes only at those requested samples.
            options = dict(
                rho_min=0.8,
                rho_max=5.0,
                Nr=20,
                Nphi=40,
                courant=0.4,
                inner_sponge_width=0.2,
                outer_sponge_width=0.5,
            )
            reference = FDTD_2D_GR(**options).config("python")
            accelerated = FDTD_2D_GR(**options).config("gpu")
            for simulation in (reference, accelerated):
                simulation.initialize_orbiting_packet(
                    phi0=1.1,
                    azimuthal_mode=5,
                    radial_width=0.22,
                    angular_width=0.42,
                )
            reference_history = reference.run(
                steps=8, record_stride=3, store_snapshots=True)
            accelerated_history = accelerated.run(
                steps=8, record_stride=3, store_snapshots=True)
            for field in ("Hz", "Er", "Ephi"):
                np.testing.assert_allclose(
                    getattr(accelerated, field), getattr(reference, field),
                    rtol=1e-12, atol=2e-13)
            for key in (
                "time", "energy", "rho_mean", "phi_mean", "phi_coherence", "rho_peak",
                "phi_peak", "divergence_linf", "divergence_linf_global",
            ):
                np.testing.assert_allclose(
                    accelerated_history[key], reference_history[key],
                    rtol=1e-11, atol=2e-12)
            for gpu_frame, reference_frame in zip(
                accelerated_history["snapshots"], reference_history["snapshots"]
            ):
                np.testing.assert_allclose(
                    gpu_frame, reference_frame, rtol=2e-6, atol=2e-7)
            stats = cuda_gr.transfer_stats(accelerated)
            assert stats["host_to_device_during_steps"] == 0
            assert stats["device_to_host_during_steps"] == 0
            assert stats["steps"] == 8
        """)
        environment = os.environ.copy()
        environment["NUMBA_ENABLE_CUDASIM"] = "1"
        environment["MPLBACKEND"] = "Agg"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode:
            self.fail(
                "GR CUDA-simulator parity process failed.\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
