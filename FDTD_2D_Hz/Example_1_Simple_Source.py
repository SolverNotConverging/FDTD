from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=180e9, f_max=220e9, Nt=2000)
sim.add_PML(pml_width=20, sigma_max=1.0, order=3, direction='xy')

# Simple dielectric inclusion (relative permittivity 2.5) surrounded by vacuum
sim.add_rectangle(ER=1.0, MR=[2.5, 2.5, 2.5], x_position=(4e-3, 10e-3), y_position=(4e-3, 10e-3))

# Point source exciting Hz at the centre of the grid
sim.add_source('point', x=70, y=70, amplitude=1.0)

sim.run(record_stride=4)

# Electric fields (Ex/Ey) are stored as normalized quantities (E/eta0)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)
