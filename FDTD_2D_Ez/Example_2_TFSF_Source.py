import numpy as np

from FDTD_2D_Ez import FDTD_2D_Ez

f_min, f_max = 50e9, 100e9
sim = FDTD_2D_Ez(x_range=14e-3, y_range=14e-3, Nx=140, Ny=140, f_min=f_min, f_max=f_max, Nt=10000)
sim.config(backend='cpu')
sim.periodic = ''
sim.add_PML(pml_width=20, order=3, direction='xy', kappa_max=7, alpha_max=0.025)

sim.add_circle(material='PEC', center=(7e-3, 7e-3), radius=2e-3)

theta = np.deg2rad(30)
sim.add_source(
    kind='sftf',
    x=(30, 110),
    y=(30, 110),
    amplitude=1.0,
    f_min=f_min,  # Gaussian (optional)
    f_max=f_max,  # used for kx, ky if f_min is None
    angle=theta,
    is_show=1,
)

sim.add_line_monitor(x=(21, 119), y=119)
sim.add_line_monitor(x=(21, 119), y=21)
sim.add_line_monitor(y=(21, 119), x=21)
sim.add_line_monitor(y=(21, 119), x=119)

sim.run(record_stride=4)

# Save for later
sim.save("fdtd_run.pkl", include_histories=True)

# Dynamic color scaling in the animation (smoothed to reduce flicker)
sim.show_animation(fps=60, dynamic_clim=False, clim_smooth=0.25)
freqs = np.linspace(f_min, f_max, 10)
ff = sim.NF2FF(top=0, bottom=1, left=2, right=3, freqs=freqs, nphi=3600, src_index=0)

# plot at a few frequencies
sim.show_FF(ff, freq_idx=np.arange(0, 10), component='Etheta')  # first frequency
