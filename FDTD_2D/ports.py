"""Lumped terminals, tracked waveguide modes, and multi-run scattering studies."""
from dataclasses import dataclass,field
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, LineString
from .topology import EPS0, MU0
from .surfaces import SurfaceImpedance, PEC


@dataclass(frozen=True)
class GaussianPulse:
    frequency: float
    width: float
    delay: float = None
    amplitude: float = 1.0

    def __post_init__(self):
        if self.frequency < 0 or self.width <= 0 or not np.all(np.isfinite([self.frequency,self.width,self.amplitude])):
            raise ValueError("Invalid pulse")
        if self.delay is None:
            object.__setattr__(self,"delay",6*self.width)

    @property
    def end_time(self):
        return self.delay+6*self.width

    def __call__(self,t):
        u = t-self.delay
        return self.amplitude*np.exp(-(u/self.width)**2)*np.cos(2*np.pi*self.frequency*u)


@dataclass
class LumpedPort:
    name: str
    start: tuple
    end: tuple = None
    resistance: float = 50.0
    rank: int = 100
    invariant_length: float = 1.0

    def __post_init__(self):
        if not self.name or self.resistance <= 0 or self.invariant_length <= 0:
            raise ValueError("Lumped port requires a name, positive resistance and invariant length")
        if not np.all(np.isfinite([self.resistance,self.invariant_length,*self.start])):
            raise ValueError("Nonfinite port parameters")
        if self.end is not None and self.start == self.end:
            raise ValueError("Lumped terminal line must have nonzero length")

    @property
    def shape(self):
        return Point(self.start) if self.end is None else LineString([self.start,self.end])

    def anchor_coordinates(self):
        return [(a,v) for p in ([self.start] if self.end is None else [self.start,self.end])
                for a,v in zip(("x","y"),p)]

    def compile(self, topology):
        te = topology.polarization == "TE"
        vector = np.zeros(topology.incidence.shape[1] if te else len(topology.groups))
        if te:
            if self.end is None:
                raise ValueError("TE lumped port needs an in-plane voltage line")
            direction = np.asarray(self.end)-self.start
            if np.count_nonzero(direction) != 1:
                raise ValueError("Reference lumped ports currently need an axis-aligned line")
            # Integrate the electric field along the terminal line using cell crossings.
            axis = int(direction[1] != 0)
            covered=0.
            for k,(line,owners,normal) in enumerate(topology.edge_geometry):
                if normal is None:
                    continue
                coordinates=np.asarray(line.coords)
                if np.max(abs(coordinates[:,1-axis]-self.start[1-axis]))>3*topology.tol:
                    continue
                lo,hi=sorted((self.start[axis],self.end[axis]))
                length=max(0,min(hi,coordinates[:,axis].max())-max(lo,coordinates[:,axis].min()))
                if length <= topology.tol:
                    continue
                # E is tangent to the counterclockwise circulation around a face.
                electric=np.array([-normal[1],normal[0]])
                vector[k]=electric[axis]*np.sign(direction[axis])*length
                covered+=length
            if abs(covered-self.shape.length) > topology.tol*10:
                raise ValueError("Lumped voltage line must be an anchored, fully retained electric path")
        else:
            if self.end is not None:
                raise ValueError("TM lumped ports use an out-of-plane terminal at start; omit end")
            # A mesh anchor may coincide with four dual cells; deposit through nearest cell.
            hits = [(np.linalg.norm(c-np.asarray(self.start)),k) for k,c in enumerate(topology.centers)]
            _,k = min(hits)
            if topology.scene.material_at(self.start).__class__ is SurfaceImpedance:
                raise ValueError("Lumped terminal is inside an opaque conductor")
            vector[k] = self.invariant_length
        if not np.any(vector):
            raise ValueError("Empty lumped terminal")
        return vector


@dataclass(frozen=True)
class VirtualWaveguide:
    length_cells: int = 32
    pml_cells: int = 12
    source_clearance_cells: int = 6

    def __post_init__(self):
        if (any(not isinstance(v,int) for v in (self.length_cells,self.pml_cells,self.source_clearance_cells))
                or self.pml_cells<2 or self.source_clearance_cells<1 or self.length_cells<self.pml_cells+self.source_clearance_cells+3):
            raise ValueError("Virtual guide requires length >= PML + clearance + 3 cells")


@dataclass
class WaveguidePort:
    name: str
    axis: str
    position: float
    span: tuple
    normal: int = 1  # outward from device; incoming wave travels opposite normal
    modes: int = 1
    rank: int = 100
    tracking_floor: float = .6
    invariant_length: float = 1.0
    virtual_waveguide: object = field(default_factory=VirtualWaveguide)
    mesh_step: float = None

    def __post_init__(self):
        if (not self.name or self.axis not in {"x","y"} or self.normal not in {-1,1}
                or self.span[1] <= self.span[0] or self.modes < 1 or self.invariant_length <= 0):
            raise ValueError("Invalid waveguide port")
        if (not np.all(np.isfinite([self.position,*self.span,self.invariant_length,self.tracking_floor]))
                or not 0<self.tracking_floor<=1 or not isinstance(self.modes,int)
                or (self.virtual_waveguide is not None and not isinstance(self.virtual_waveguide,VirtualWaveguide))
                or (self.mesh_step is not None and (not np.isfinite(self.mesh_step) or self.mesh_step<=0))):
            raise ValueError("Invalid waveguide continuation or modal parameters")

    @property
    def shape(self):
        return LineString([(self.position,v) if self.axis=="x" else (v,self.position) for v in self.span])

    def anchor_coordinates(self):
        return [(self.axis,self.position),*(('y' if self.axis=='x' else 'x',v) for v in self.span)]

    def compile(self, topology, frequencies, dt):
        return CompiledWaveguide(self,topology,frequencies,dt)


def track_modes(previous, current, weights, floor=.6):
    """Global overlap assignment and phase transport; reject ambiguous matches."""
    a = previous*np.sqrt(weights[:,None]); b = current*np.sqrt(weights[:,None])
    a /= np.linalg.norm(a,axis=0); b /= np.linalg.norm(b,axis=0)
    overlap = a.conj().T @ b
    rows,cols = linear_sum_assignment(-abs(overlap))
    order = cols[np.argsort(rows)]
    quality = abs(overlap[np.arange(len(order)),order])
    if np.any(quality < floor):
        raise ValueError("Ambiguous modal tracking; add frequency anchors or reduce mode count")
    phase = np.exp(-1j*np.angle(overlap[np.arange(len(order)),order]))
    return order,phase,quality


class CompiledWaveguide:
    def __init__(self, port, topology, frequencies, dt):
        self.port,self.topology,self.frequencies,self.dt = port,topology,np.asarray(frequencies),dt
        grid = topology.mesh.x if port.axis=="x" else topology.mesh.y
        transverse = topology.mesh.y if port.axis=="x" else topology.mesh.x
        plane = np.argmin(abs(grid-port.position))
        if abs(grid[plane]-port.position) > topology.tol*4 or not 0 < plane < len(grid)-1:
            raise ValueError("Waveguide plane must be an interior mesh anchor")
        active = np.where((transverse[:-1]>=port.span[0]-topology.tol)
                          & (transverse[1:]<=port.span[1]+topology.tol))[0]
        if not len(active) or abs(transverse[active[0]]-port.span[0]) > topology.tol*4 or abs(transverse[active[-1]+1]-port.span[1])>topology.tol*4:
            raise ValueError("Waveguide aperture endpoints must be mesh anchors")
        self.widths = np.diff(transverse)[active]
        self.left,self.right,self.vector,self.sign = [],[],[],[]
        self.distances = []
        materials = []
        face_by_cell = {}
        for k,f in enumerate(topology.faces):
            face_by_cell.setdefault(f.cell,[]).append(k)
        retained = []
        for j in active:
            cells = [(plane-1,j),(plane,j)] if port.axis=="x" else [(j,plane-1),(j,plane)]
            fids = [face_by_cell.get(c,[]) for c in cells]
            if not fids[0] and not fids[1]:
                continue
            if any(len(v)!=1 for v in fids):
                raise ValueError("Waveguide plane must have an unsplit, propagation-invariant aperture")
            ids = [topology.face_group[v[0]] for v in fids]
            fs = [topology.faces[v[0]] for v in fids]
            if any(len(topology.groups[k]) != 1 for k in ids):
                raise ValueError("Move waveguide plane away from enlarged cells")
            if port.virtual_waveguide is None and not np.allclose([fs[0].epsilon,fs[0].mu,fs[0].sigma_e,fs[0].sigma_m],
                               [fs[1].epsilon,fs[1].mu,fs[1].sigma_e,fs[1].sigma_m]):
                raise ValueError("Waveguide materials vary through the port plane")
            entries = [(k,n) for k,(line,owners,n) in enumerate(topology.edge_geometry)
                       if n is not None and set(owners)==set(ids)]
            if len(entries)!=1:
                raise ValueError("Waveguide plane crosses a boundary or a split edge")
            k,n = entries[0]
            self.left.append(ids[0]); self.right.append(ids[1]); self.vector.append(k)
            self.sign.append(n[0 if port.axis=="x" else 1]*port.normal)
            self.distances.append([abs(topology.centers[k,0 if port.axis=="x" else 1]-port.position) for k in ids])
            inside=fs[0] if port.normal==1 else fs[1]
            materials.append([inside.epsilon*EPS0,inside.mu*MU0,inside.sigma_e,inside.sigma_m])
            retained.append(j)
        if not retained or np.any(np.diff(retained)!=1):
            raise ValueError("Reference waveguide requires one connected transverse aperture")
        self.widths = np.diff(transverse)[retained]
        self.transverse = transverse[np.r_[retained,retained[-1]+1]]
        self.left,self.right,self.vector = map(lambda v:np.asarray(v,int),(self.left,self.right,self.vector))
        self.sign = np.array(self.sign); self.distances=np.array(self.distances)
        self.materials=np.array(materials)
        # Determine physical side-wall laws by sampling just outside the retained aperture.
        walls = []
        for value,side in ((self.transverse[0],-1),(self.transverse[-1],1)):
            wall_point=Point((port.position,value) if port.axis=="x" else (value,port.position))
            sheet_walls=[g.material for g in topology.scene.geometry if g.sheet and g.shape.distance(wall_point)<topology.tol*4]
            point = (port.position,value+side*topology.tol*10) if port.axis=="x" else (value+side*topology.tol*10,port.position)
            if sheet_walls:
                wall=sheet_walls[-1]
                if not isinstance(wall,SurfaceImpedance):
                    raise ValueError("Transmissive sheet side walls need an exterior-coupled modal problem")
            elif not topology.scene.domain.covers(Point(point)):
                name = ("ymin" if side<0 else "ymax") if port.axis=="x" else ("xmin" if side<0 else "xmax")
                wall = topology.scene.boundaries[name]
            else:
                wall = topology.scene.material_at(point)
                if not isinstance(wall,SurfaceImpedance):
                    raise ValueError("Waveguide aperture endpoints must terminate on physical surface walls")
            walls.append(wall)
        self.walls=tuple(walls)
        self.q_modes=[]; self.p_modes=[]; self.beta=[]; self.valid=[]; self.overlaps=[]
        previous=None
        te = topology.polarization=="TE"
        for frequency in self.frequencies:
            omega=2*np.pi*frequency
            symbol=2/dt*np.sin(omega*dt/2); cosine=np.cos(omega*dt/2)
            eps,mu,se,sm=self.materials.T
            eps=eps+se*cosine/(1j*symbol); mu=mu+sm*cosine/(1j*symbol)
            coefficient = eps if te else mu
            scalar = mu if te else eps
            n=len(self.widths)
            stiffness=np.zeros((n,n),complex)
            for k in range(n-1):
                g=1/(coefficient[k]*self.widths[k]/2+coefficient[k+1]*self.widths[k+1]/2)
                stiffness[k,k]+=g; stiffness[k+1,k+1]+=g
                stiffness[k,k+1]-=g; stiffness[k+1,k]-=g
            for k,wall in zip((0,n-1),walls):
                hard = wall.is_pmc if te else wall.is_pec
                soft = wall.is_pec if te else wall.is_pmc
                if hard:
                    stiffness[k,k]+=2/(coefficient[k]*self.widths[k])
                elif not soft:
                    z=complex(wall.impedance(frequency,dt))
                    dual=z if te else 1/z
                    stiffness[k,k]+=1/(coefficient[k]*self.widths[k]/2+1/(1j*symbol*cosine*dual))
            mass=self.widths/coefficient
            values,vecs=eig(np.diag(symbol**2*scalar*self.widths)-stiffness,np.diag(mass))
            order=np.argsort(values.real)[::-1][:port.modes]
            values,vecs=values[order],vecs[:,order]
            if len(values)<port.modes:
                raise ValueError("Not enough resolved waveguide modes")
            if previous is not None:
                order,phase,quality=track_modes(previous,vecs,self.widths,port.tracking_floor)
                values,vecs=values[order],vecs[:,order]*phase
            else:
                quality=np.ones(port.modes)
                for k in range(port.modes):
                    pivot=np.argmax(abs(vecs[:,k])); vecs[:,k]*=np.exp(-1j*np.angle(vecs[pivot,k]))
            previous=vecs.copy()
            wave=np.sqrt(values.astype(complex))
            # exp(+i omega t - i beta x): a passive forward mode has Im(beta)<=0.
            wave=np.where(wave.imag>1e-10*max(abs(wave).max(),1),-wave,wave)
            admittance=wave[None,:]/(symbol*coefficient[:,None])
            # Normalize the transported power of this staggered discrete system.
            # Both temporal centering and the scalar average at the normal face
            # enter its energy flux. This matters when mixing with circuit ports.
            spacing=np.sum(self.distances,axis=1)
            if not np.allclose(spacing,spacing[0],rtol=1e-6):
                raise ValueError("Waveguide continuation spacing varies across the aperture")
            beta=2/spacing[0]*np.arcsin(wave*spacing[0]/2)
            dl,dr=self.distances.T
            face_factor=(np.exp(1j*dl[:,None]*beta)+np.exp(-1j*dr[:,None]*beta))/2
            power=.5*cosine*np.sum(self.widths[:,None]*(face_factor*vecs*np.conj(admittance*vecs)).real,axis=0)
            valid=(wave.real>0)&(power>1e-12)&(values.real>0)
            norm=np.sqrt(np.where(valid,power*self.port.invariant_length,1))
            q=vecs/norm; p=admittance*q
            if te:
                # Fix electric orientation across opposing port normals. The
                # scalar here is magnetic, and reverses under time reversal.
                q*=port.normal; p*=port.normal
            self.q_modes.append(q); self.p_modes.append(p); self.beta.append(beta)
            self.valid.append(valid); self.overlaps.append(quality)
        self.q_modes,self.p_modes,self.beta,self.valid,self.overlaps=map(np.asarray,
            (self.q_modes,self.p_modes,self.beta,self.valid,self.overlaps))

    def measure(self, q, p):
        return q[self.left],q[self.right],p[self.vector]*self.sign

    def decompose(self, spectra):
        left,right,p=spectra
        incoming=np.full((len(self.frequencies),self.port.modes),np.nan+0j)
        outgoing=incoming.copy()
        for k in range(len(self.frequencies)):
            valid=self.valid[k]
            if not np.any(valid):
                continue
            beta=self.beta[k,valid]
            qmode=self.q_modes[k][:,valid]; pmode=self.p_modes[k][:,valid]
            dl,dr=self.distances.T
            # Distances are oriented along outward normal; use actual scalar sampling positions.
            offsets=np.column_stack([-dl,dr])*self.port.normal
            plus=np.vstack([qmode*np.exp(-1j*offsets[:,0,None]*beta),
                            qmode*np.exp(-1j*offsets[:,1,None]*beta),pmode])
            minus=np.vstack([qmode*np.exp(1j*offsets[:,0,None]*beta),
                             qmode*np.exp(1j*offsets[:,1,None]*beta),-pmode])
            if self.topology.polarization=="TE":
                minus=-minus  # incoming E keeps its sign; incoming H reverses
            weights=np.sqrt(np.tile(self.widths,3))
            matrix=np.column_stack([minus,plus])*weights[:,None]
            field=np.r_[left[k],right[k],p[k]]*weights
            coefficients,_,_,_=np.linalg.lstsq(matrix,field,rcond=1e-10)
            count=valid.sum()
            incoming[k,valid]=coefficients[:count]; outgoing[k,valid]=coefficients[count:]
        return incoming,outgoing
