from FDTD_1D import FDTD_1D

z_range = 20e-3
Nz = 500
f_max = 100e9
Nt = 5000

sim = FDTD_1D(z_range=z_range, Nz=Nz, f_max=f_max, Nt=Nt)
sim.add_object(3, 1, (0e-3, 5e-3))
sim.add_object(1, 3, (5e-3, 7e-3))
sim.add_object(1, 1, (7e-3, 10e-3))
sim.add_object(1, 3, (10e-3, 20e-3))
sim.set_boundary("absorbing", "absorbing")
sim.add_source(src_position=2e-3, amplitude=1.0, is_show=True)
sim.run()
sim.show_animation(fps=100)
