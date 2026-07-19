from FDTD_1D import FDTD_1D

z_range = 20e-3
Nz = 500
f_max = 100e9
Nt = 5000

sim = FDTD_1D(z_range=z_range, Nz=Nz, f_max=f_max, Nt=Nt)
sim.add_material("dielectric", epsilon_r=3.0, sigma_e=1)
sim.add_material("magnetic", mu_r=3.0, sigma_m=0.0)
sim.add_object(material="dielectric", region=(3e-3, 5e-3))
sim.add_object(material="magnetic", region=(5e-3, 7e-3))
sim.add_object(material="vacuum", region=(7e-3, 10e-3))
sim.add_object(material="magnetic", region=(10e-3, 20e-3))
sim.set_boundary("absorbing", "absorbing")
sim.add_source(src_position=2e-3, amplitude=1.0, is_show=True)
sim.run()
sim.show_animation(fps=100)
