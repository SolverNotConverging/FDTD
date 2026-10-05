from matplotlib import pyplot as plt

from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=50e9, f_max=120e9, Nt=2000)
sim.config(backend='cpu')
sim.periodic = ''
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_rectangle(ER=[3, 3, 3], MR=1, y_position=(6e-3, 8e-3), x_position=(0e-3, 14e-3))

sim.add_source('waveguide-x', x=22, y=(4e-3, 10e-3), amplitude=1, mode_index=0, modes_to_show=3, is_show=0)

sim.add_line_monitor(x=8e-3, y=(3e-3, 11e-3))
sim.add_line_monitor(x=10e-3, y=(3e-3, 11e-3))

sim.run(record_stride=4)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

results_1 = sim.calculate_line_monitor_power_fft(monitor_index=0, window=None)
results_2 = sim.calculate_line_monitor_power_fft(monitor_index=1, window=None)
source_results = sim.calculate_source_power_fft(source_index=0)

sim.plot_fft_results((results_1, results_2, source_results))

plt.show()

# Dynamic color scaling in the animation (smoothed to reduce flicker)
# sim.show_animation(fps=30, dynamic_clim=False, clim_smooth=0.25)
