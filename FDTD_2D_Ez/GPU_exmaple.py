"""Run the unified TMz solver with its Numba-CUDA backend."""

from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(
    x_range=12e-3,
    y_range=12e-3,
    Nx=120,
    Ny=120,
    f_min=50e9,
    f_max=90e9,
    Nt=4000,
)
sim.config(backend='gpu')
print(f"Requested GPU; selected backend: {sim.backend}")

# Periodic BC disabled for this example.
sim.periodic = ''

sim.add_PML(pml_width=15, order=3, direction='xy', kappa_max=6, alpha_max=0.03)

# Two simple dielectric inclusions to make the fields interesting.
sim.add_rectangle(ER=[4.0, 4.0, 3.0], MR=1.0,
                  x_position=(3e-3, 4e-3), y_position=(6e-3, 9e-3))
sim.add_circle(ER=5.5, MR=1.0, center=(8.5e-3, 8e-3), radius=1.2e-3)

# Soft point source near the bottom-left corner
sim.add_source('point', x=2.5e-3, y=2.0e-3, amplitude=2.5)
# Line source feeding a vertical segment to excite more modes
sim.add_source('line-soft', x=6.0e-3, y=(2.0e-3, 5.0e-3), amplitude=1.0)

sim.run(record_stride=5)

sim.show_animation(fps=120, dynamic_clim=False)
