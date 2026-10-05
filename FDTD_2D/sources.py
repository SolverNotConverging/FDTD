"""Axis-aligned plane-wave TF/SF with a mesh-matched 1D incident grid."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix
from shapely.geometry import box, Point
from .pml import PML, CPML
from .topology import EPS0,MU0
from .mesh import C0


@dataclass(frozen=True)
class PlaneWave:
    name: str
    axis: str = "x"
    direction: int = 1

    def __post_init__(self):
        if not self.name or self.axis not in {"x","y"} or self.direction not in {-1,1}:
            raise ValueError("PlaneWave needs a name, axis x/y and direction +/-1")


class IncidentPlane:
    def __init__(self,source,topology,bounds,pml,dt,waveform):
        self.source,self.waveform,self.dt=source,waveform,dt
        self.t=topology; axis=0 if source.axis=="x" else 1
        b=np.asarray(bounds,dtype=float)
        if b.shape!=(4,) or b[2]<=b[0] or b[3]<=b[1]:
            raise ValueError("TF/SF must be a closed rectangle (xmin,ymin,xmax,ymax)")
        self.box=box(*b)
        if not topology.scene.domain.contains(self.box):
            raise ValueError("TF/SF box must lie strictly inside the domain")
        for geom in topology.scene.geometry:
            if not self.box.contains(geom.shape):
                raise ValueError("TF/SF box must enclose the physical geometry")
        for port in topology.scene.ports:
            if not self.box.contains(port.shape):
                raise ValueError("TF/SF box must enclose physical port apertures")
        for a,nodes in enumerate((topology.mesh.x,topology.mesh.y)):
            if any(np.min(abs(nodes-value))>4*topology.tol for value in b[[a,a+2]]):
                raise ValueError("TF/SF boundaries must be mesh anchors")
        material=topology.scene.background
        if material.has_dispersion or any(material.sigma_e) or any(material.sigma_m):
            raise ValueError("Plane-wave incident grid requires lossless nondispersive background")
        epsilon=material.epsilon_r[0]*EPS0; mu=material.mu_r[0]*MU0
        eta=np.sqrt(mu/epsilon)
        self.scalar_coefficient=mu if topology.polarization=="TE" else epsilon
        self.vector_coefficient=epsilon if topology.polarization=="TE" else mu
        self.amplitude=1/eta if topology.polarization=="TE" else 1.
        self.source_flux=2*(1. if topology.polarization=="TE" else 1/eta)
        nodes=topology.mesh.x if axis==0 else topology.mesh.y
        widths=np.diff(nodes); centers=(nodes[1:]+nodes[:-1])/2
        nq=len(widths); self.q=np.zeros(nq); self.p=np.zeros(nq+1)
        self.mq=self.scalar_coefficient*widths
        self.mp=self.vector_coefficient*np.r_[widths[0]/2,(widths[:-1]+widths[1:])/2,widths[-1]/2]
        self.b=coo_matrix((np.tile([-1.,1.],nq),(np.repeat(np.arange(nq),2),
                          np.column_stack((np.arange(nq),np.arange(nq)+1)).ravel())),shape=(nq,nq+1)).tocsr()
        configuration=pml or PML(max(widths[0],widths[-1])*4,sides=(source.axis+"min",source.axis+"max"))
        profile=object.__new__(CPML); profile.config=configuration; profile.dt=dt
        profile.topology=SimpleNamespace(scene=topology.scene)
        profile.speed=C0/np.sqrt(material.epsilon_r[0]*material.mu_r[0])
        self.pk,self.pa,self.pb=profile._profile(nodes,source.axis)
        self.qk,self.qa,self.qb=profile._profile(centers,source.axis)
        self.pm=np.zeros_like(self.p); self.qm=np.zeros_like(self.q)
        inside_min,inside_max=b[axis],b[axis+2]
        candidates=np.where((centers>nodes[0]+configuration.width)&(centers<inside_min))[0] if source.direction==1 else np.where((centers< nodes[-1]-configuration.width)&(centers>inside_max))[0]
        if len(candidates)<2:
            raise ValueError("Need two background cells between TF/SF and incident-grid PML")
        self.source_cell=int(candidates[0] if source.direction==1 else candidates[-1])
        self.end_time=waveform.end_time+abs((inside_max if source.direction==1 else inside_min)-centers[self.source_cell])/profile.speed
        self.qindices=np.clip(np.searchsorted(nodes,topology.centers[:,axis],side="right")-1,0,nq-1)
        self.maskq=np.array([self.box.contains(Point(c)) for c in topology.centers],float)
        # Virtual-guide coordinates are not physical: keep their commutator zero.
        physical_nq=len(topology.host.groups) if hasattr(topology,"host") else len(topology.groups)
        self.maskq[physical_nq:]=1.
        self.pindices=[]; self.sign=[]; self.maskp=[]
        physical_np=topology.host.incidence.shape[1] if hasattr(topology,"host") else topology.incidence.shape[1]
        for e,(line,owners,normal) in enumerate(topology.edge_geometry):
            midpoint=np.array(line.interpolate(.5,normalized=True).coords[0])
            self.pindices.append(np.argmin(abs(nodes-midpoint[axis])))
            self.sign.append(normal[axis] if normal is not None else 0.)
            self.maskp.append(float(self.box.contains(Point(midpoint))) if e<physical_np else 1.)
        self.pindices=np.asarray(self.pindices); self.sign=np.asarray(self.sign); self.maskp=np.asarray(self.maskp)
        # No cut or material-changing cells on the TF/SF shell.
        for fid,face in enumerate(topology.faces):
            if face.polygon.distance(self.box.boundary)<4*topology.tol:
                if not np.allclose([face.epsilon,face.mu,face.sigma_e,face.sigma_m],
                                   [material.epsilon_r[0],material.mu_r[0],0.,0.]) or len(topology.groups[topology.face_group[fid]])!=1:
                    raise ValueError("TF/SF shell must be uncut homogeneous background")

    def vector_correction(self):
        raw=self.b.T@self.q
        self.pm=self.pb*self.pm+self.pa*raw
        self.p+=self.dt*(raw/self.pk+self.pm)/self.mp
        q=self.q[self.qindices]
        return self.maskp*(self.t.incidence.T@q)-self.t.incidence.T@(self.maskq*q)

    def scalar_correction(self,step):
        p=self.p[self.pindices]*self.sign
        correction=self.t.incidence@(self.maskp*p)-self.maskq*(self.t.incidence@p)
        raw=-self.b@self.p
        self.qm=self.qb*self.qm+self.qa*raw
        raw=raw/self.qk+self.qm
        raw[self.source_cell]+=self.source_flux*float(self.waveform((step+.5)*self.dt))
        self.q+=self.dt*raw/self.mq
        return correction

    def memory_norm(self):
        return np.linalg.norm(self.pm)**2+np.linalg.norm(self.qm)**2
