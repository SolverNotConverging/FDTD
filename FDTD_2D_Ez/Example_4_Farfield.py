import numpy as np

from FDTD_2D_Ez import FDTD_2D_Ez

f_min = 80e9
f_max = 120e9

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=f_min, f_max=f_max, Nt=1200)
sim.config(backend='cpu')
sim.periodic = ''
sim.add_PML(pml_width=20, direction='xy')

sim.add_rectangle(ER=[4, 4, 4], MR=1, x_position=(6e-3, 8e-3), y_position=(6e-3, 7e-3))
for i in np.arange(7e-3, 10e-3, 0.1e-3):
    sim.add_rectangle(ER=4, MR=1, x_position=(6e-3 - (i - 7e-3) / 2, 8e-3 + (i - 7e-3) / 2), y_position=(i, i + 0.1e-3))

sim.add_source('waveguide-y', x=(2e-3, 12e-3), y=6.1e-3, amplitude=2, mode_index=0, modes_to_show=3)

sim.add_line_monitor(x=(21, 119), y=119, index=10)  # top
sim.add_line_monitor(x=(21, 119), y=21, index=20)   # bottom
sim.add_line_monitor(y=(21, 119), x=21, index=30)   # left
sim.add_line_monitor(y=(21, 119), x=119, index=40)  # right

sim.run(record_stride=4, is_include_history=True)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

freqs = np.linspace(f_min, f_max, 3)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)

ff = sim.NF2FF(top=10, bottom=20, left=30, right=40,
               freqs=freqs, nphi=3600, src_index=0)

power_spectrum = sim.power_spectrum(monitor_index=10, freqs=freqs, source_index=0)
sim.plot_power_spectrum(power_spectrum)

# plot at a few frequencies
sim.show_FF(ff, freq_idx=np.arange(0, 3), component='Etheta')  # first frequency
