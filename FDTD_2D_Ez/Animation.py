from FDTD_2D_Ez import FDTD_2D_Ez

sim = FDTD_2D_Ez.load_npz("fdtd_run.npz")
sim.show_animation(fps=100, dynamic_clim=False)
