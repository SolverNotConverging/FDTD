from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=50e9, f_max=100e9, Nt=4000)
sim.config(backend='cpu')
sim.periodic = 'xy'
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_material("anisotropic", epsilon_r=(5, 6, 7), sigma_e=(0, 0, 0.02))
sim.add_material("high_index", epsilon_r=5)
sim.add_material("triangle", epsilon_r=3.5)
sim.add_rectangle(material="anisotropic", x_position=(5e-3, 6e-3), y_position=(4e-3, 5e-3), subpixel=16)
sim.add_circle(material="high_index", center=(10e-3, 10e-3), radius=1.25e-3, subpixel=16)
sim.add_triangle(material="triangle",
                 vertices=((2e-3, 9e-3), (4e-3, 9e-3), (3e-3, 11e-3)))
sim.add_rectangle(material="PEC", x_position=(11.5e-3, 11.7e-3),
                  y_position=(3e-3, 6e-3))

sim.add_source('point', x=8e-3, y=4e-3, amplitude=3)
sim.add_source('line-soft', x=3e-3, y=(4e-3, 10e-3), amplitude=1)
sim.add_line_monitor(x=4e-3, y=(2e-3, 12e-3), index=10)

sim.run(record_stride=1)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

power_results = sim.calculate_line_monitor_power_fft(monitor_index=10)
source_results = sim.calculate_source_power_fft(source_index=1)
sim.plot_fft_results((power_results, source_results))

requested_freqs = [60e9, 80e9, 100e9]
plane_power = sim.power_spectrum(10, requested_freqs, source_index=1)
sim.plot_power_spectrum(plane_power)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)
