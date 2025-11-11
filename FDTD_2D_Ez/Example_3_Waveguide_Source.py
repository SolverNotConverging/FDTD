from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=250e9, f_max=300e9, Nt=1000)
sim.periodic = ['']
sim.add_PML(pml_width=20, sigma_max=1, order=3, direction='xy')

sim.add_rectangle(ER=[3, 3, 3], MR=1, y_position=(7e-3, 9e-3), x_position=(0e-3, 14e-3))
sim.add_rectangle(ER=[5, 5, 5], MR=1, y_position=(0e-3, 14e-3), x_position=(7e-3, 9e-3))

sim.add_source('waveguide-x', y=(3e-3, 12e-3), x=22, amplitude=2, mode_index=1, modes_to_show=3)
sim.add_source('waveguide-y', x=(3e-3, 12e-3), y=22, amplitude=2, mode_index=2, modes_to_show=3)

sim.run(record_stride=4)

# Save for later
sim.save_npz("fdtd_run.npz", include_histories=True)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=60, dynamic_clim=False, clim_smooth=0.25)
