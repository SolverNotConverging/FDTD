from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz(x_range=14e-3, y_range=14e-3, Nx=160, Ny=160, f_min=180e9, f_max=260e9, Nt=2400)
sim.add_PML(pml_width=18, sigma_max=1.0, order=3, direction='xy')

# Scatterer with higher permittivity
sim.add_circle(ER=1.0, MR=[3.4, 3.4, 3.4], center=(7e-3, 7e-3), radius=1.2e-3)

# Total-field/scattered-field boundaries
sim.add_source('sftf-x', x=40, y=(40, 120), amplitude=1.0)
sim.add_source('sftf-x', x=120, y=(40, 120), amplitude=1.0)
sim.add_source('sftf-y', x=(40, 120), y=40, amplitude=1.0)
sim.add_source('sftf-y', x=(40, 120), y=120, amplitude=1.0)

sim.run(record_stride=5)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)
