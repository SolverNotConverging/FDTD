"""Run from the repository root: python -m FDTD_2D.examples.example_general."""
import numpy as np
from .. import (Scene, Material, MeshPolicy, FDTD2D, RunControl, GaussianPulse,
                SurfaceImpedance, ThinSheet, LumpedPort)


def main():
    scene = Scene()
    scene.rectangle("high index", (.016, .024), (.012, .018), Material("n3", epsilon_r=9), rank=20)
    metal = SurfaceImpedance.good_conductor(5.8e7, (1e9, 15e9))
    scene.polygon("metal tip", [(.011, .010), (.014, .015), (.011, .020)], metal, rank=40)
    scene.sheet("resistive film", (.027, .010), (.030, .020), ThinSheet.resistive(75), rank=30)
    scene.add_port(LumpedPort("feed", (.015, .015), rank=100, invariant_length=.001))
    scene.add_port(LumpedPort("receive", (.026, .015), rank=100, invariant_length=.001))
    policy = MeshPolicy(max_step=.002, f_max=15e9, f_min=5e9, min_step=.00025,
                        min_dt=2e-13, growth=2)
    sim = FDTD2D(scene, mesh_policy=policy, polarization="TM",
                 frequencies=np.linspace(5e9, 15e9, 51))
    control = RunControl(max_time=4e-9, min_time=1e-9,
                         field_tolerance=1e-5, dft_tolerance=1e-4, check_steps=100)
    result = sim.scattering(control, GaussianPulse(10e9, 8e-11))
    result.save("general_2d_sparameters.npz")
    print("mesh", sim.mesh.shape, "dt", sim.dt)
    print("generated domain", sim.scene.generated)
    print("rejected anchors", len(sim.mesh.report.rejected))
    print("limited resolution", sim.mesh.report.limited_resolution)
    print("topology", sim.topology.report)
    print("valid S bins", result.valid.sum(), "of", len(result.valid))
    print("runs", [(r.steps, r.stop_reason) for r in result.runs])


if __name__ == "__main__":
    main()
