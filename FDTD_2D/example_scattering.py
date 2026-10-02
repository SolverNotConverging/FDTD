"""Plane-wave scattering with automatic domain, closed TF/SF and closed NTFF."""
import numpy as np
from . import Scene,PEC,PlaneWave,MeshPolicy,FDTD2D,RunControl,GaussianPulse


def main():
    scene=Scene()
    scene.circle("cylinder",(0.,0.),.003,PEC,rank=40)
    scene.add_plane_wave(PlaneWave("illumination",axis="x",direction=1))
    sim=FDTD2D(scene,mesh_policy=MeshPolicy(.001,20e9,min_step=.0002,growth=2),
               polarization="TM",frequencies=np.linspace(10e9,20e9,21))
    result=sim.run(RunControl(1.5e-9,field_tolerance=1e-5,dft_tolerance=1e-4),
                   {"illumination":GaussianPulse(15e9,6e-11)})
    far=result.far_field(np.linspace(0,2*np.pi,181))
    np.savez_compressed("cylinder_far_field.npz",frequencies=far.frequencies,angles=far.angles,
                        amplitude=far.scalar_amplitude,power=far.power_per_radian)
    print("mesh",sim.mesh.shape,"dt",sim.dt,"stop",result.stop_reason)
    print("boxes",sim.scene.generated)
    print("15 GHz scattering amplitude magnitude",abs(far.scalar_amplitude[10]).max())


if __name__=="__main__":
    main()
