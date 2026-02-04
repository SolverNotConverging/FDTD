from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=50e9, f_max=100e9, Nt=2000)
sim.periodic = ''
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_rectangle(ER=[5, 6, 7], MR=1, x_position=(5e-3, 6e-3), y_position=(4e-3, 5e-3))
sim.add_circle(ER=5, MR=1, center=(10e-3, 10e-3), radius=1.25e-3, nsub=6)

sim.add_source('point', x=8e-3, y=4e-3, amplitude=3)
sim.add_source('line-soft', x=3e-3, y=(4e-3, 10e-3), amplitude=1)

sim.run(record_stride=1)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)
