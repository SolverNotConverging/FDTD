"""Small 3D PEC-scattering example with power and NF2FF post-processing."""

import matplotlib.pyplot as plt
import numpy as np

from FDTD_3D import FDTD_3D


sim = FDTD_3D(
    x_range=100e-3, y_range=100e-3, z_range=100e-3,
    Nx=100, Ny=100, Nz=100,
    f_min=6e9, f_max=10e9, Nt=2000,
)
sim.config(backend="cpu")  # whole-run Cython kernel when it has been built
sim.add_PML(15)

# Materials are defined once and then referenced by name or by returned object.
dielectric = sim.add_material("dielectric", epsilon_r=2.2, sigma_e=2e-4)
sim.add_block(dielectric, x=(13e-3, 17e-3), y=(13e-3, 17e-3), z=(9e-3, 12e-3))
sim.add_sphere("PEC", center=(15e-3, 15e-3, 17e-3), radius=3e-3)

# A simple soft x-polarized plane source. Tuples are half-open spans.
sim.add_source(
    "plane", x=(45, 55), y=(45, 55), z=50, polarization="x",
    amplitude=1.0, t0=0.6e-9, tw=0.18e-9, f_min=6e9, f_max=10e9,
)

# Six cell-centered planes form the closed equivalence surface.
nf_box = sim.add_nf2ff_box(x=(20, 80), y=(20, 80), z=(20, 80), start_index=10)
# This additional plane is automatically persisted when run() completes.
saved_monitor = sim.add_plane_monitor(
    "z", position=24, first=(0, 100), second=(0, 100), index=1, normal="+",
    save_path="output/monitor_data/z24_plane.h5",
)
sim.run(record_stride=1)
monitor_animation_fig, monitor_animation = sim.animate_plane_monitor(
    saved_monitor, component="Ex", frame_stride=4,
)

frequencies = np.linspace(6e9, 10e9, 41)
power = sim.power_spectrum(nf_box["z_max"], frequencies, source_index=0, window=None)
sim.plot_power_spectrum(power, db=True)

far_field = sim.NF2FF(
    nf_box, freqs=[8e9], theta=np.linspace(0, np.pi, 91),
    phi=np.linspace(0, 2 * np.pi, 181, endpoint=False),
    source_index=0, window="hann",
)
sim.plot_nf2ff(far_field, db=True, db_floor=-40)

# In a later session, the saved data can be loaded and plotted without rerunning:
# loaded = sim.load_plane_monitor("output/monitor_data/z24_plane.h5", register=False)
# sim.plot_plane_monitor(loaded, component="Ex", frequency=8e9, window="hann")
plt.show()
