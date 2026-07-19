import os
import subprocess
import sys
import textwrap
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDeviceResidentCuda(unittest.TestCase):
    def test_cuda_simulator_matches_python_without_step_transfers(self):
        script = textwrap.dedent(r"""
            import numpy as np
            from FDTD_2D_Ez import FDTD_2D_Ez
            from FDTD_2D_Hz import FDTD_2D_Hz

            configurations = (
                (FDTD_2D_Ez, ("Ez", "Hx", "Hy")),
                (FDTD_2D_Hz, ("Ex", "Ey", "Hz")),
            )
            for solver_class, fields in configurations:
                options = dict(
                    x_range=4e-3, y_range=4e-3, Nx=4, Ny=4,
                    f_max=100e9, Nt=3, dt=1e-13, subpixel=1,
                )
                python = solver_class(**options).config("python")
                gpu = solver_class(**options).config("gpu")
                assert gpu.backend == "numba_cuda"
                for simulation in (python, gpu):
                    simulation.add_material(
                        "lossy", epsilon_r=(2.0, 3.0, 4.0),
                        mu_r=(1.2, 1.3, 1.4),
                        sigma_e=(0.01, 0.02, 0.03),
                        sigma_m=(0.02, 0.03, 0.04),
                    )
                    simulation.add_rectangle(
                        material="lossy", x_position=(1, 3), y_position=(0, 3))
                    simulation.add_rectangle(
                        material="PEC", x_position=(2, 3), y_position=(1, 2))
                    simulation.add_rectangle(
                        material="PMC", x_position=(0, 1), y_position=(2, 3))
                    simulation.add_PML(1, direction="xy")
                    simulation.add_source(
                        "point", x=2, y=1, t0=2e-13, tw=1e-13,
                        is_show=False)
                    simulation.add_source(
                        "line-soft", x=(0, 3), y=2, t0=2e-13,
                        tw=1e-13, is_show=False)
                    simulation.add_line_monitor(x=(0, 4), y=1, index=9)
                    simulation.run(record_stride=2, is_include_history=True)

                for name in fields:
                    np.testing.assert_allclose(
                        getattr(gpu, name), getattr(python, name),
                        rtol=1e-12, atol=1e-13)
                    np.testing.assert_allclose(
                        getattr(gpu, name + "_history"),
                        getattr(python, name + "_history"),
                        rtol=1e-12, atol=1e-13)
                    np.testing.assert_allclose(
                        gpu.monitor_results[0][name],
                        python.monitor_results[0][name],
                        rtol=1e-12, atol=1e-13)
                assert gpu._gpu_transfer_stats["host_to_device_during_steps"] == 0
                assert gpu._gpu_transfer_stats["device_to_host_during_steps"] == 0
                assert gpu._gpu_transfer_stats["source_events"] > 0
                assert gpu._gpu_transfer_stats["monitor_points"] == 4
                assert "_gpu_state" not in gpu.state_dict()

            # Exercise delayed sparse TF/SF events separately.
            for solver_class, fields in configurations:
                options = dict(
                    x_range=8e-3, y_range=8e-3, Nx=8, Ny=8,
                    f_max=100e9, Nt=2, dt=1e-13, subpixel=1,
                )
                python = solver_class(**options).config("python")
                gpu = solver_class(**options).config("gpu")
                for simulation in (python, gpu):
                    simulation.add_source(
                        "sftf", x=(2, 6), y=(2, 6), angle=0.3,
                        t0=1e-13, tw=2e-13, is_show=False)
                    simulation.run(is_include_history=False)
                for name in fields:
                    np.testing.assert_allclose(
                        getattr(gpu, name), getattr(python, name),
                        rtol=1e-12, atol=1e-13)
                assert gpu._gpu_transfer_stats["source_events"] > 0
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
                "CUDA-simulator parity process failed.\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
