from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=100e9, f_max=120e9, Nt=4000)
sim.config(backend='cpu')
sim.periodic = ''
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_rectangle(ER=[7, 7, 7], MR=1, y_position=(6e-3, 8e-3), x_position=(0e-3, 14e-3))
sim.add_rectangle(ER=[8, 8, 8], MR=1, y_position=(0e-3, 14e-3), x_position=(6e-3, 8e-3))

sim.add_source('waveguide-x', y=(3e-3, 11e-3), x=22, amplitude=1, mode_index=0, modes_to_show=3, is_show=1)
sim.add_source('waveguide-y', x=(3e-3, 11e-3), y=22, amplitude=1.5, mode_index=1, modes_to_show=3, is_show=1)

sim.run(record_stride=4)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=30, dynamic_clim=False, clim_smooth=0.25)
