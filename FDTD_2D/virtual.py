"""Independent matched guide branches, joined by a shared aperture flux.

The auxiliary field graph occupies no physical volume. Its complete transverse
grid (including unmonitored modes) continues the device cross-section into CPML.
"""
from copy import copy
import numpy as np
from scipy.sparse import block_diag
from .geometry import Scene
from .mesh import Mesh
from .topology import Topology, EdgeBlock, EPS0, MU0
from .surfaces import PEC, PMC
from FDTD_common.material import Material
from .pml import PML, CPML
from .ports import WaveguidePort


class VirtualGuides:
    def __init__(self,host,guides):
        self.host=host; self.locals=[]; self.entries={}
        used=set()
        for name,guide in guides.items():
            config=guide.port.virtual_waveguide
            if config is None:
                continue
            if used.intersection(guide.vector):
                raise ValueError("Matched waveguide apertures cannot share flux unknowns")
            used.update(guide.vector)
            dl,dr=guide.distances.T
            if not np.allclose(dl,dr,rtol=1e-6,atol=host.tol):
                raise ValueError("Matched aperture needs equal longitudinal cells on both sides; use MeshPolicy or mesh_step")
            h=float(dl[0]+dr[0]); length=config.length_cells*h
            soft=PEC if host.polarization=="TE" else PMC
            scene=Scene((0.,length),(guide.transverse[0],guide.transverse[-1]),
                        boundaries={"xmin":soft,"ymin":guide.walls[0],"ymax":guide.walls[1]})
            for k,(eps,mu,se,sm) in enumerate(guide.materials):
                mat=Material(f"{name}:row{k}",epsilon_r=eps/EPS0,mu_r=mu/MU0,sigma_e=se,sigma_m=sm)
                scene.rectangle(f"row{k}",(0.,length),tuple(guide.transverse[k:k+2]),mat)
            mesh=Mesh(np.arange(config.length_cells+1)*h,guide.transverse)
            # Preserve exact endpoints even when multiplication rounds differently.
            scene.x=tuple(mesh.x[[0,-1]])
            local=Topology(scene,mesh,host.polarization,enlargement=0)
            source_position=(config.length_cells-config.pml_cells-config.source_clearance_cells)*h
            source=WaveguidePort(name,"x",source_position,tuple(scene.y),normal=1,
                                 modes=guide.port.modes,tracking_floor=guide.port.tracking_floor,
                                 invariant_length=guide.port.invariant_length,virtual_waveguide=None)
            self.locals.append(local)
            self.entries[name]=(guide,local,source,h,config)
        self.topology=self._join()

    def _join(self):
        if not self.locals:
            return self.host
        t=copy(self.host); t.host=self.host; parts=[self.host,*self.locals]
        t.incidence=block_diag([p.incidence for p in parts],format="lil")
        t.vector_mass=block_diag([p.vector_mass for p in parts],format="lil")
        for attr in ("scalar_mass","scalar_loss","areas","centers"):
            setattr(t,attr,np.concatenate([getattr(p,attr) for p in parts],axis=0))
        t.groups=list(self.host.groups); t.blocks=list(self.host.blocks); t.edge_geometry=list(self.host.edge_geometry)
        qo=len(self.host.groups); po=self.host.incidence.shape[1]
        for name,(guide,local,source,h,config) in self.entries.items():
            t.groups.extend([] for _ in local.groups)
            t.blocks.extend(EdgeBlock(b.indices+po,b.mass.copy(),b.loss.copy(),b.laws) for b in local.blocks)
            t.edge_geometry.extend((line,tuple(v+qo for v in owners),normal) for line,owners,normal in local.edge_geometry)
            cells={f.cell:local.face_group[k]+qo for k,f in enumerate(local.faces)}
            inside=guide.left if guide.port.normal==1 else guide.right
            outside=guide.right if guide.port.normal==1 else guide.left
            replacement=np.array([cells[(0,k)] for k in range(len(guide.widths))])
            for k,(i,o,a,e) in enumerate(zip(inside,outside,replacement,guide.vector)):
                coefficient=float(t.incidence[o,e])
                t.incidence[o,e]=0.; t.incidence[a,e]=coefficient
                # Both half cells now contain the cloned device-side material.
                eps,mu,se,sm=guide.materials[k]
                mass=guide.widths[k]*h*(eps if t.polarization=="TE" else mu)
                loss=guide.widths[k]*h*(se if t.polarization=="TE" else sm)
                t.vector_mass[e,e]=mass
                block_index,block=next((j,b) for j,b in enumerate(t.blocks) if e in b.indices)
                if len(block.indices)!=1 or block.laws!=(None,):
                    raise ValueError("Matched aperture cannot cross a surface-network edge")
                # Replace without modifying the physical topology used for CPML validation.
                t.blocks[block_index]=EdgeBlock(block.indices.copy(),np.array([[mass]]),np.array([[loss]]),(None,))
                line,owners,normal=t.edge_geometry[e]
                t.edge_geometry[e]=(line,tuple(a if v==o else v for v in owners),normal)
            if guide.port.normal==1:
                guide.right=replacement
            else:
                guide.left=replacement
            self.entries[name]=(guide,local,source,h,config,qo,po)
            qo+=len(local.groups); po+=local.incidence.shape[1]
        t.incidence=t.incidence.tocsr(); t.vector_mass=t.vector_mass.tocsr()
        return t

    def compile_sources(self,frequencies,dt):
        result={}
        for name,(guide,local,port,h,config,qo,po) in self.entries.items():
            source=port.compile(local,frequencies,dt)
            source.left+=qo; source.right+=qo; source.vector+=po
            result[name]=source
        return result

    def cpml(self,physical_pml,dt):
        parts=[(self.host,physical_pml.compile(self.host,dt) if physical_pml is not None else None)]
        for _,local,_,h,config,*_ in self.entries.values():
            configuration=PML(config.pml_cells*h,sides=("xmax",),kappa_max=1.)
            parts.append((local,CPML(configuration,local,dt,uniform_waveguide=True)))
        return CPML.combine(self.topology,parts) if self.locals else parts[0][1]
