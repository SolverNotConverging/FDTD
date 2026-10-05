"""Generate a physical domain, nested closed measurement boxes and PML from objects."""
from dataclasses import dataclass
import numpy as np
from .mesh import C0, Anchor
from .pml import PML


@dataclass(frozen=True)
class DomainPolicy:
    pml_cells: int = 8
    clearance_wavelengths: float = .25
    contour_cells: int = 3
    tfsf_cells: int = 3

    def generate(self,scene,mesh_policy):
        shapes=[g.shape for g in scene.geometry]+[p.shape for p in scene.ports]
        if not shapes:
            raise ValueError("Automatic domain generation needs geometry or a port")
        if (any(not isinstance(v,int) for v in (self.pml_cells,self.contour_cells,self.tfsf_cells))
                or self.pml_cells<2 or self.contour_cells<2 or self.tfsf_cells<2
                or not np.isfinite(self.clearance_wavelengths) or self.clearance_wavelengths<0):
            raise ValueError("Invalid automatic-domain policy")
        bounds=np.array([s.bounds for s in shapes])
        physical=(bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max())
        h=mesh_policy.max_step
        # A degenerate line/point scene still receives a finite interior extent.
        physical=tuple(v+(-h if k<2 else h) for k,v in enumerate(physical))
        def expand(b,d):
            return b[0]-d,b[1]-d,b[2]+d,b[3]+d
        tfsf=expand(physical,self.tfsf_cells*h) if scene.plane_waves else None
        ntff=expand(tfsf or physical,self.contour_cells*h)
        f=mesh_policy.f_min or mesh_policy.f_max
        index=np.sqrt(scene.background.epsilon_r[0]*scene.background.mu_r[0])
        clearance=max(self.contour_cells*h,self.clearance_wavelengths*C0/(f*index))
        pml_width=self.pml_cells*h
        domain=expand(ntff,clearance+pml_width)
        scene.x=(domain[0],domain[2]); scene.y=(domain[1],domain[3])
        scene.generated=DomainReport(physical,tfsf,ntff,domain,pml_width)
        return PML(pml_width)


@dataclass(frozen=True)
class DomainReport:
    physical_bounds: tuple
    tfsf_bounds: tuple
    ntff_bounds: tuple
    domain_bounds: tuple
    pml_width: float

    def anchors(self):
        result=[]
        for name,bounds in (("NTFF",self.ntff_bounds),("TF/SF",self.tfsf_bounds)):
            if bounds is not None:
                for axis,values in (("x",(bounds[0],bounds[2])),("y",(bounds[1],bounds[3]))):
                    result.extend(Anchor(axis,v,150,name,"closed contour") for v in values)
        return result
