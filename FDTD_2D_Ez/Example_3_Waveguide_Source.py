from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=50e9, f_max=70e9, Nt=1000)
sim.periodic = ''
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_rectangle(ER=[3, 3, 3], MR=1, y_position=(6e-3, 8e-3), x_position=(0e-3, 14e-3))
sim.add_rectangle(ER=[4, 4, 4], MR=1, y_position=(0e-3, 14e-3), x_position=(6e-3, 8e-3))

sim.add_source('waveguide-x', y=(4e-3, 10e-3), x=22, amplitude=1, mode_index=1, modes_to_show=3, is_show=1)
sim.add_source('waveguide-y', x=(4e-3, 10e-3), y=22, amplitude=2, mode_index=0, modes_to_show=3, is_show=1)

sim.run(record_stride=4)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=30, dynamic_clim=False, clim_smooth=0.25)
