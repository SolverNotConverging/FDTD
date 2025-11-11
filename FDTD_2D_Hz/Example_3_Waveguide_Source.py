from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz(x_range=16e-3, y_range=12e-3, Nx=200, Ny=150, f_min=190e9, f_max=240e9, Nt=2400)
sim.add_PML(pml_width=18, sigma_max=1.0, order=3, direction='xy')

# Ridge waveguide sections (relative permittivity map supplied via MR argument)
sim.add_rectangle(ER=1.0, MR=[2.5, 2.5, 2.5], x_position=(4e-3, 12e-3), y_position=(4e-3, 8e-3))
sim.add_rectangle(ER=1.0, MR=[4.0, 4.0, 4.0], x_position=(0e-3, 6e-3), y_position=(8e-3, 10e-3))

# Launch the second mode of the vertical waveguide port propagating to +x
sim.add_source('waveguide-x', x=40, y=(45, 105), amplitude=1.5, mode_index=2, modes_to_show=3)

sim.run(record_stride=4)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)
