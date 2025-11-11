from FDTD_2D_Ez import FDTD_2D_Ez
import numpy as np

sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=200e9, f_max=300e9, Nt=1000)
sim.periodic = ['']
sim.add_PML(pml_width=20, sigma_max=1, order=3, direction='xy')

sim.add_rectangle(ER=[4, 4, 4], MR=1, x_position=(6e-3, 8e-3), y_position=(6e-3, 8e-3))

sim.add_source('waveguide-y', x=(2e-3, 12e-3), y=6.1e-3, amplitude=2, mode_index=2, modes_to_show=4)

sim.add_line_monitor(x=(21, 119), y=119)
sim.add_line_monitor(x=(21, 119), y=21)
sim.add_line_monitor(y=(21, 119), x=21)
sim.add_line_monitor(y=(21, 119), x=119)

sim.run(record_stride=4, is_include_history=True)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

freqs = np.linspace(200e9, 300e9, 3)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=120, dynamic_clim=True, clim_smooth=0.25)

ff = sim.NF2FF(top=0, bottom=1, left=2, right=3, freqs=freqs, nphi=3600)

# plot at a few frequencies
sim.show_FF(ff, freq_idx=np.arange(0, 3), component='Etheta')  # first frequency
