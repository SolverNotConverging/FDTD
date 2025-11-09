from FDTD_1D import FDTD_1D

z_range = 20e-3
Nz = 500
f_max = 100e9
Nt = 5000

sim = FDTD_1D(z_range=z_range, Nz=Nz, f_max=f_max, Nt=Nt)
sim.add_source(src_position=1e-3, amplitude=1.0, is_show=True)
sim.left_perfect_boundary = True
sim.right_perfect_boundary = True
sim.add_object(3.13, 1, (8e-3, 9e-3))
sim.add_object(3.13, 1, (10.3e-3, 11.3e-3))
sim.add_object(3.13, 1, (12.6e-3, 13.6e-3))
sim.run()
sim.show_animation(fps=1)
