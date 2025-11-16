"""Minimal example for running :class:`FDTD_2D_Ez_GPU` on the GPU."""

import torch

from FDTD_2D_Ez_GPU import FDTD_2D_Ez_GPU


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 2D Ez GPU solver on {device}…")

    sim = FDTD_2D_Ez_GPU(
        x_range=12e-3,
        y_range=12e-3,
        Nx=120,
        Ny=120,
        f_min=40e9,
        f_max=85e9,
        Nt=800,
        device=device,
    )

    sim.periodic = []
    sim.add_PML(pml_width=15, order=3, direction='xy', kappa_max=6, alpha_max=0.03)

    sim.add_rectangle(ER=[3.6, 3.6, 1.0], MR=[1.0, 1.0, 1.0],
                      x_position=(4e-3, 6e-3), y_position=(5e-3, 7.5e-3))
    sim.add_circle(ER=5.0, MR=1.0, center=(8.0e-3, 3.0e-3), radius=1.0e-3)

    sim.add_source('point', x=2.0e-3, y=3.0e-3, amplitude=2.0, t0=0, tw=90e-12, f_max=70e9)
    sim.add_source('line-soft', x=(5.5e-3, 9.0e-3), y=9.5e-3,
                   amplitude=1.2, t0=0, tw=110e-12, f_max=70e9)

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

    print("Simulation finished. Recorded Ez history shape:", sim.Ez_history.shape)
    if sim.monitor_results:
        ez_monitor = sim.monitor_results[0]["Ez"]
        print("Monitor Ez data shape:", ez_monitor.shape)

    # Optional visualization (requires matplotlib)
    # sim.show_animation(fps=60, dynamic_clim=True)


if __name__ == "__main__":
    main()
