"""Broadband two-port guide: python -m FDTD_2D.examples.example_waveguide."""
import numpy as np
from .. import Scene,MeshPolicy,FDTD2D,WaveguidePort,PEC,GaussianPulse,RunControl


def main():
    scene=Scene()
    scene.rectangle("lower wall",(0,.024),(-.002,0),PEC,rank=40)
    scene.rectangle("upper wall",(0,.024),(.012,.014),PEC,rank=40)
    scene.add_port(WaveguidePort("left","x",.006,(0,.012),normal=-1))
    scene.add_port(WaveguidePort("right","x",.018,(0,.012),normal=1))
    policy=MeshPolicy(.001,40e9,min_step=.00025,cells_per_wavelength=10,growth=2)
    sim=FDTD2D(scene,mesh_policy=policy,polarization="TM",frequencies=np.linspace(20e9,40e9,41))
    result=sim.scattering(RunControl(2e-9),GaussianPulse(30e9,1.5e-10))
    result.save("waveguide_sparameters.npz")
    k=20
    print("30 GHz S matrix:\n",result.s[k])
    print("Expected S21:",np.exp(-1j*sim.waveguides["left"].beta[k,0]*.012))


if __name__=="__main__":
    main()
