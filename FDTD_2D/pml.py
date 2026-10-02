"""Unsplit CFS-CPML for the uncut Cartesian collar of the reference mesh."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags
from .mesh import C0


@dataclass(frozen=True)
class PML:
    width: float  # physical width on each enabled side
    sides: tuple = ("xmin","xmax","ymin","ymax")
    reflection: float = 1e-8
    order: int = 3
    kappa_max: float = 7.0
    alpha_fraction: float = .05

    def __post_init__(self):
        if (not np.isfinite(self.width) or self.width<=0 or not 0<self.reflection<1
                or self.order<1 or self.kappa_max<1 or self.alpha_fraction<0
                or any(s not in {"xmin","xmax","ymin","ymax"} for s in self.sides)):
            raise ValueError("Invalid PML configuration")

    def compile(self,topology,dt):
        return CPML(self,topology,dt)


class CPML:
    def __init__(self,configuration,topology,dt,uniform_waveguide=False):
        self.config,self.topology,self.dt=configuration,topology,dt
        scene=topology.scene
        if any(configuration.width*sum(s.startswith(axis) for s in configuration.sides)>=span[1]-span[0]
               for axis,span in (("x",scene.x),("y",scene.y))):
            raise ValueError("PML collar leaves no interior region")
        # Cut faces, material changes, and sheet networks inside the collar need
        # a separate conformal PML derivation; reject rather than use a sponge.
        for geom in scene.geometry:
            if self._in_collar(geom.shape.bounds) and not uniform_waveguide:
                raise ValueError("Reference CPML requires a homogeneous, uncut collar")
        for port in scene.ports:
            if self._in_collar(port.shape.bounds):
                raise ValueError("Ports must lie outside PML")
        if scene.background.has_dispersion or len(set(scene.background.epsilon_r))>1 or len(set(scene.background.mu_r))>1:
            raise ValueError("CPML collar requires an isotropic nondispersive background")
        n=np.sqrt(scene.background.epsilon_r[0]*scene.background.mu_r[0])
        self.speed=C0/n
        weights=[]; positions=[]
        for line,owners,normal in topology.edge_geometry:
            point=np.array(line.interpolate(.5,normalized=True).coords[0])
            positions.append(point)
            if normal is None:
                # All surface-network edges are outside the collar by validation above.
                p=np.asarray(line.coords)
                tangent=(p[-1]-p[0])/line.length
                normal=np.array([-tangent[1],tangent[0]])
            weights.append(normal**2)
        weights=np.array(weights)
        self.bx=topology.incidence@diags(weights[:,0])
        self.by=topology.incidence@diags(weights[:,1])
        positions=np.array(positions)
        kx,ax,bx=self._profile(positions[:,0],"x")
        ky,ay,by=self._profile(positions[:,1],"y")
        # Within PML, edges are Cartesian; there is one derivative per vector DOF.
        self.pk=weights[:,0]*kx+weights[:,1]*ky
        self.pa=weights[:,0]*ax+weights[:,1]*ay
        self.pb=weights[:,0]*bx+weights[:,1]*by
        self.qk=[]; self.qa=[]; self.qb=[]
        for axis,k in (("x",0),("y",1)):
            kk,aa,bb=self._profile(topology.centers[:,k],axis)
            self.qk.append(kk); self.qa.append(aa); self.qb.append(bb)
        self.p_memory=np.zeros(topology.incidence.shape[1])
        self.q_memory=np.zeros((2,len(topology.groups)))

    def _in_collar(self,bounds):
        x0,y0,x1,y1=bounds
        scene=self.topology.scene; width=self.config.width
        return any({"xmin":x0<scene.x[0]+width,"xmax":x1>scene.x[1]-width,
                    "ymin":y0<scene.y[0]+width,"ymax":y1>scene.y[1]-width}[s] for s in self.config.sides)

    def _profile(self,coordinate,axis):
        span=self.topology.scene.x if axis=="x" else self.topology.scene.y
        width=self.config.width
        fraction=np.zeros_like(coordinate)
        if axis+"min" in self.config.sides:
            fraction=np.maximum(fraction,(span[0]+width-coordinate)/width)
        if axis+"max" in self.config.sides:
            fraction=np.maximum(fraction,(coordinate-span[1]+width)/width)
        fraction=np.clip(fraction,0,1)
        peak=-(self.config.order+1)*np.log(self.config.reflection)*self.speed/(2*width)
        sigma=peak*fraction**self.config.order
        kappa=1+(self.config.kappa_max-1)*fraction**self.config.order
        alpha=self.config.alpha_fraction*peak*(1-fraction)*(fraction>0)
        b=np.exp(-(sigma/kappa+alpha)*self.dt)
        a=np.zeros_like(b)
        np.divide(sigma*(b-1),kappa*(sigma+kappa*alpha),out=a,where=sigma>0)
        return kappa,a,b

    def vector_drive(self,raw):
        self.p_memory=self.pb*self.p_memory+self.pa*raw
        return raw/self.pk+self.p_memory

    def scalar_drive(self,vector):
        result=np.zeros(len(self.topology.groups))
        for k,b in enumerate((self.bx,self.by)):
            raw=-b@vector
            self.q_memory[k]=self.qb[k]*self.q_memory[k]+self.qa[k]*raw
            result+=raw/self.qk[k]+self.q_memory[k]
        return result

    def memory_norm(self):
        return np.linalg.norm(self.p_memory)**2+np.linalg.norm(self.q_memory)**2

    @classmethod
    def combine(cls,topology,parts):
        """Join physical and disjoint virtual-guide CPML memories on one graph."""
        result=object.__new__(cls)
        result.topology=topology
        for attr in ("pk","pa","pb"):
            values=[]
            for local,pml in parts:
                values.append(getattr(pml,attr) if pml is not None else
                              np.full(local.incidence.shape[1],1. if attr in {"pk","pb"} else 0.))
            setattr(result,attr,np.concatenate(values))
        for attr in ("qk","qa","qb"):
            setattr(result,attr,np.array([np.concatenate([
                getattr(pml,attr)[k] if pml is not None else np.full(len(local.groups),1. if attr in {"qk","qb"} else 0.)
                for local,pml in parts]) for k in range(2)]))
        weights=[]
        for line,_,normal in topology.edge_geometry:
            if normal is None:
                points=np.array(line.coords); tangent=(points[-1]-points[0])/line.length
                normal=np.array([-tangent[1],tangent[0]])
            weights.append(normal**2)
        weights=np.array(weights)
        result.bx=topology.incidence@diags(weights[:,0])
        result.by=topology.incidence@diags(weights[:,1])
        result.p_memory=np.zeros(topology.incidence.shape[1])
        result.q_memory=np.zeros((2,len(topology.groups)))
        return result
