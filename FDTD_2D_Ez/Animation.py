from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez.load("fdtd_run.pkl")
sim.show_animation(fps=100, dynamic_clim=False)
