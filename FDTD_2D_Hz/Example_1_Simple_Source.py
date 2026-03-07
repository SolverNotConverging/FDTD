from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=50e9, f_max=100e9, Nt=4000)
sim.periodic = ''
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_rectangle(ER=[5, 6, 7], MR=1, x_position=(5e-3, 6e-3), y_position=(4e-3, 5e-3))
sim.add_circle(ER=5, MR=1, center=(10e-3, 10e-3), radius=1.25e-3, nsub=6)

sim.add_source('point', x=8e-3, y=4e-3, amplitude=3)
sim.add_source('line-soft', x=3e-3, y=(4e-3, 10e-3), amplitude=1)
sim.add_line_monitor(x=4e-3, y=(2e-3, 12e-3))

sim.run(record_stride=1)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

power_results = sim.calculate_line_monitor_power_fft(monitor_index=0, window=None)
source_results = sim.calculate_source_power_fft(source_index=1)
sim.plot_fft_results((power_results, source_results))

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=120, dynamic_clim=True, clim_smooth=0.25)
