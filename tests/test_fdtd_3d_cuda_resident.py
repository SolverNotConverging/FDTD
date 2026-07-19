import os
import subprocess
import sys
import textwrap
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDeviceResidentCuda3D(unittest.TestCase):
    def test_cuda_simulator_matches_numpy_without_step_transfers(self):
        script = textwrap.dedent(r"""
            import numpy as np
            from FDTD_3D import FDTD_3D

            def make_case(backend):
                sim = FDTD_3D(
                    x_range=5e-3, y_range=5e-3, z_range=5e-3,
                    Nx=5, Ny=5, Nz=5, f_max=100e9, Nt=4,
                    dt=1e-13, subpixel=1,
                ).config(backend)
                sim.add_material(
                    "lossy", epsilon_r=(2.0, 3.0, 4.0),
                    mu_r=(1.2, 1.3, 1.4),
                    sigma_e=(0.01, 0.02, 0.03),
                    sigma_m=(0.02, 0.03, 0.04),
                )
                sim.add_block("lossy", (1, 4), (1, 4), (1, 4))
                sim.add_block("PEC", (0, 1), (0, 1), (0, 1))
                sim.add_block("PMC", (4, 5), (4, 5), (4, 5))
                sim.add_PML(1)
                sim.add_source(
                    "point", 2, 2, 2, polarization="z",
                    t0=2e-13, tw=1e-13)
                sim.add_source(
                    "line", (1, 4), 3, 2, polarization="x",
                    t0=2e-13, tw=1e-13)
                sim.add_source(
                    "plane", 3, (1, 4), (1, 4), polarization="y",
                    t0=2e-13, tw=1e-13)
                sim.add_plane_monitor("x", 2, (0, 5), (0, 5), index=9)
                return sim

            reference = make_case("python")
            gpu = make_case("gpu")
            assert gpu.backend == "numba_cuda"
            reference.run(record_stride=2, progress=False)
            gpu.run(record_stride=2, progress=False)

            names = (
                "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                "Psi_Hx_y", "Psi_Hx_z", "Psi_Hy_x", "Psi_Hy_z",
                "Psi_Hz_x", "Psi_Hz_y", "Psi_Ex_y", "Psi_Ex_z",
                "Psi_Ey_x", "Psi_Ey_z", "Psi_Ez_x", "Psi_Ez_y",
            )
            for name in names:
                np.testing.assert_allclose(
                    getattr(gpu, name), getattr(reference, name),
                    rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(
                gpu.monitor_results[0]["fields"],
                reference.monitor_results[0]["fields"],
                rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(
                gpu.monitor_results[0]["time"],
                reference.monitor_results[0]["time"], rtol=0, atol=0)
            np.testing.assert_allclose(
                gpu.last_source_waveforms, reference.last_source_waveforms,
                rtol=0, atol=0)
            assert gpu.current_step == reference.current_step == 4
            assert gpu._gpu_state is None
            assert gpu._gpu_transfer_stats["host_to_device_during_steps"] == 0
            assert gpu._gpu_transfer_stats["device_to_host_during_steps"] == 0
            assert gpu._gpu_transfer_stats["source_points"] == 13
            assert gpu._gpu_transfer_stats["monitor_points"] == 25
            assert gpu._gpu_transfer_stats["record_count"] == 2
        """)
        environment = os.environ.copy()
        environment["NUMBA_ENABLE_CUDASIM"] = "1"
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
                "3D CUDA-simulator parity process failed.\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
