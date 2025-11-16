"""Minimal example showing how to run :class:`FDTD_2D_Hz_GPU` on a GPU."""

import torch

from FDTD_2D_Hz_GPU import FDTD_2D_Hz_GPU


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 2D Hz GPU solver on {device}…")

    sim = FDTD_2D_Hz_GPU(
        x_range=12e-3,
        y_range=12e-3,
        Nx=120,
        Ny=120,
        f_min=50e9,
        f_max=90e9,
        Nt=800,
        device=device,
    )

    # Periodic BC disabled for this example.
    sim.periodic = []

    sim.add_PML(pml_width=15, order=3, direction='xy', kappa_max=6, alpha_max=0.03)

    # Two simple dielectric inclusions to make the fields interesting.
    sim.add_rectangle(ER=[4.0, 4.0, 1.0], MR=1.0,
                      x_position=(3e-3, 4e-3), y_position=(6e-3, 9e-3))
    sim.add_circle(ER=5.5, MR=1.0, center=(8.5e-3, 4.0e-3), radius=1.2e-3)

    # Soft point source near the bottom-left corner
    sim.add_source('point', x=2.5e-3, y=2.0e-3, amplitude=2.5, t0=0, tw=100e-12, f_max=80e9)
    # Line source feeding a vertical segment to excite more modes
    sim.add_source('line-soft', x=6.0e-3, y=(2.0e-3, 5.0e-3), amplitude=1.0, t0=0, tw=120e-12, f_max=80e9)

    # Optional field monitor that samples along the center line
    sim.add_monitor(
        orientation='horizontal',
        ix0=0,
        ix1=sim.Nx,
        iy0=sim.Ny // 2,
        iy1=sim.Ny // 2 + 1,
        it0=0,
        it1=sim.Nt,
    )

    sim.run(record_stride=5)

    print("Simulation finished. Recorded history shape:", sim.Ex_history.shape)
    if sim.monitor_results:
        hz_monitor = sim.monitor_results[0]["Hz"]
        print("Monitor Hz data shape:", hz_monitor.shape)

    # Uncomment to visualize the run (requires matplotlib)
    # sim.show_animation(fps=60, dynamic_clim=True)


if __name__ == "__main__":
    main()
