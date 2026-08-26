from FDTD_1D import FDTD_1D

z_range = 20e-3
Nz = 500
f_max = 100e9
Nt = 5000

sim = FDTD_1D(z_range=z_range, Nz=Nz, f_max=f_max, Nt=Nt)
material = sim.add_material(
    "multipole",
    epsilon_r=(2.0, 2.2, 2.4),
    sigma_e=1e-3,
    debye=[
        {"delta_epsilon": (1.0, 1.1, 1.2), "tau": 10e-12},
        {"delta_epsilon": 0.3, "tau": 40e-12},
    ],
)

sim.add_object(material="multipole", region=(5e-3, 15e-3))
sim.set_boundary("absorbing", "absorbing")
sim.add_source(src_position=2e-3, amplitude=1.0, is_show=True)
sim.run()
sim.show_animation(fps=100)
