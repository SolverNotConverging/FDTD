from FDTD_2D_Hz import FDTD_2D_Hz

sim = FDTD_2D_Hz.load("fdtd_run.pkl")
sim.show_animation(fps=120, dynamic_clim=False)
