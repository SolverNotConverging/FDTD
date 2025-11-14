from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_max=100e9, Nt=1000)
sim.periodic = ['y']
sim.add_PML(pml_width=20, order=3, direction='x', kappa_max=7, alpha_max=0.025)

sim.add_circle(ER=5, MR=1, center=(8e-3, 8e-3), radius=1.25e-3, nsub=6)

sim.add_source('sftf-x', x=30, y=(0, 140))

sim.run(record_stride=4)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=60, dynamic_clim=False, clim_smooth=0.25)
