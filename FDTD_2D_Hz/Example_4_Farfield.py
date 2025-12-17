import numpy as np

from FDTD_2D_Hz import FDTD_2D_Hz

f_min = 30e9
f_max = 50e9

sim = FDTD_2D_Hz(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=f_min, f_max=f_max, Nt=2000)
sim.periodic = ['']
sim.add_PML(pml_width=20, direction='xy')

sim.add_rectangle(ER=[4, 4, 4], MR=1, y_position=(6e-3, 8e-3), x_position=(6e-3, 7e-3))
for i in np.arange(7e-3, 10e-3, 0.1e-3):
    sim.add_rectangle(ER=4, MR=1, y_position=(6e-3 - (i - 7e-3) / 2, 8e-3 + (i - 7e-3) / 2), x_position=(i, i + 0.1e-3))

sim.add_rectangle(ER=1e8, MR=1, y_position=(3e-3, 11e-3), x_position=(5.9e-3, 6e-3))

sim.add_source('waveguide-x', y=(2e-3, 12e-3), x=6.5e-3, amplitude=2, mode_index=1, modes_to_show=3)

sim.add_line_monitor(x=(21, 119), y=119)
sim.add_line_monitor(x=(21, 119), y=21)
sim.add_line_monitor(y=(21, 119), x=21)
sim.add_line_monitor(y=(21, 119), x=119)

sim.run(record_stride=4, is_include_history=True)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

freqs = np.linspace(f_min, f_max, 10)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=120, dynamic_clim=False, clim_smooth=0.25)

ff = sim.NF2FF(top=0, bottom=1, left=2, right=3, freqs=freqs, nphi=3600)

# plot at a few frequencies
sim.show_FF(ff, freq_idx=np.arange(0, 10), component='Ephi')  # first frequency
