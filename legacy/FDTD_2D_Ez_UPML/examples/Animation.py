from legacy.FDTD_2D_Ez_UPML import FDTD_2D_Ez

sim = FDTD_2D_Ez.load("fdtd_run.pkl")
sim.show_animation(fps=30, dynamic_clim=False)
