"""Ranked, geometry-first rectilinear mesh generation with an explicit CFL budget."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog
from shapely.ops import nearest_points

C0 = 299792458.0


@dataclass(frozen=True)
class Anchor:
    axis: str
    position: float
    rank: int
    owner: str
    reason: str


@dataclass
class MeshReport:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # (request, winning request)
    limited_resolution: list = field(default_factory=list)


@dataclass(frozen=True)
class Mesh:
    x: np.ndarray
    y: np.ndarray
    report: MeshReport = field(default_factory=MeshReport, compare=False)

    def __post_init__(self):
        for axis in ("x", "y"):
            data = np.array(getattr(self, axis), dtype=float, copy=True)
            if data.ndim != 1 or len(data) < 3 or not np.all(np.isfinite(data)) or np.any(np.diff(data) <= 0):
                raise ValueError("Each mesh axis needs at least two positive cells")
            data.flags.writeable = False
            object.__setattr__(self, axis, data)

    @property
    def dx(self):
        return np.diff(self.x)

    @property
    def dy(self):
        return np.diff(self.y)

    @property
    def shape(self):
        return len(self.dx), len(self.dy)

    def cfl(self, courant=.9):
        return courant / (C0*np.sqrt(self.dx.min()**-2+self.dy.min()**-2))


@dataclass(frozen=True)
class MeshPolicy:
    max_step: float
    f_max: float
    cells_per_wavelength: int = 20
    min_step: float = 0.0
    min_dt: float = 0.0
    courant: float = .9
    growth: float = 1.5
    feature_cells: int = 3
    max_axis_cells: int = 5000
    f_min: float = None

    def __post_init__(self):
        if (not np.all(np.isfinite([self.max_step, self.f_max, self.min_step, self.min_dt,
                                    self.courant, self.growth]))
                or self.max_step <= 0 or self.f_max <= 0 or self.min_step < 0 or self.min_dt < 0
                or not 0 < self.courant < 1 or self.growth <= 1
                or self.cells_per_wavelength < 4 or self.feature_cells < 1):
            raise ValueError("Invalid mesh policy")
        if self.f_min is not None and not 0 < self.f_min <= self.f_max:
            raise ValueError("f_min must be positive and no larger than f_max")

    @property
    def floor(self):
        # Vacuum is a conservative speed bound for the supported eps,mu >= 1 media.
        return max(self.min_step, C0*self.min_dt*np.sqrt(2)/self.courant,
                   self.max_step*1e-7)

    def build(self, scene, extra_anchors=()):
        report = MeshReport()
        requests = list(extra_anchors)
        regions = []
        background_step = self._wavelength_step(scene.background)
        for geom in scene.geometry:
            x0, y0, x1, y1 = geom.bounds
            for axis, span in (("x", (x0, x1)), ("y", (y0, y1))):
                for position in span:
                    requests.append(Anchor(axis, position, geom.rank, geom.name, "bounding box"))
            rings=[np.array(geom.shape.coords)] if geom.sheet else [np.array(r.coords)[:-1]
                   for r in (geom.shape.exterior,*geom.shape.interiors)]
            # Anchor sharp turns, including hole boundaries, but not every
            # tessellation vertex of a smooth circle.
            for coords in rings:
                if not geom.sheet:
                    before=coords-np.roll(coords,1,axis=0)
                    after=np.roll(coords,-1,axis=0)-coords
                    cosine=np.sum(before*after,axis=1)/(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
                    coords=coords[np.arccos(np.clip(cosine,-1,1))>np.pi/12]
                for point in coords:
                    for axis, position in zip(("x", "y"), point):
                        requests.append(Anchor(axis, position, geom.rank, geom.name, "tip/vertex"))
            target = self.max_step if geom.conductor else self._wavelength_step(geom.material)
            # Resolve narrow extents; sheets retain their zero-thickness topology instead.
            widths = [v for v in (x1-x0, y1-y0) if v > 0]
            if widths:
                target = min(target, min(widths)/self.feature_cells)
            regions.append((geom.bounds, target, geom.name))
        conductors = [g for g in scene.geometry if g.conductor]
        for i, first in enumerate(conductors):
            for second in conductors[i+1:]:
                distance = first.shape.distance(second.shape)
                if not 0 < distance < self.feature_cells*self.max_step:
                    continue
                a, b = nearest_points(first.shape, second.shape)
                owner = first.name + "/" + second.name
                for p in (a, b):
                    for axis, value in zip(("x", "y"), p.coords[0]):
                        requests.append(Anchor(axis, value, max(first.rank, second.rank), owner, "conductor gap"))
                bounds = (min(a.x,b.x), min(a.y,b.y), max(a.x,b.x), max(a.y,b.y))
                regions.append((bounds, distance/self.feature_cells, owner))
        for port in scene.ports:
            for axis, position in port.anchor_coordinates():
                requests.append(Anchor(axis, position, port.rank, port.name, "port"))
        collars=[]
        for port in scene.ports:
            if getattr(port,"virtual_waveguide",None) is None:
                continue
            axis=port.axis; idx=0 if axis=="x" else 1
            h=min(self.max_step,background_step)
            for bounds,target,_ in regions:
                if bounds[idx]<=port.position<=bounds[idx+2]:
                    h=min(h,target)
            h=max(self.floor,h if port.mesh_step is None else port.mesh_step)
            span=scene.x if axis=="x" else scene.y
            for value in (port.position-h,port.position+h):
                if not span[0]<value<span[1]:
                    raise ValueError("Matched port needs a full cell on either side of its plane")
                requests.append(Anchor(axis,value,port.rank,port.name,"uniform port collar"))
            collars.extend([(axis,port.position-h,port.position,h),(axis,port.position,port.position+h,h)])
        result = []
        for axis, span in (("x", scene.x), ("y", scene.y)):
            ends = [Anchor(axis, v, 2**63-1, "domain", "boundary") for v in span]
            accepted = ends[:]
            candidates = sorted((r for r in requests if r.axis == axis),
                                key=lambda r: (-r.rank, 0 if r.reason=="port" else 1, r.position, r.owner, r.reason))
            tolerance = 1e-12*(span[1]-span[0])
            for request in candidates:
                if not span[0] <= request.position <= span[1]:
                    raise ValueError("Anchor outside domain")
                nearest = min(accepted, key=lambda a: abs(a.position-request.position))
                separation = abs(nearest.position-request.position)
                if separation <= tolerance:
                    report.accepted.append(request)
                elif separation < self.floor*(1-1e-12):
                    report.rejected.append((request, nearest))
                else:
                    accepted.append(request)
                    report.accepted.append(request)
            accepted.sort(key=lambda a: a.position)
            positions = np.array([a.position for a in accepted])
            targets = []
            idx = 0 if axis == "x" else 1
            for lo, hi in zip(positions[:-1], positions[1:]):
                h = min(self.max_step, background_step)
                for bounds, target, owner in regions:
                    if hi >= bounds[idx] and lo <= bounds[idx+2]:
                        h = min(h, target)
                        if target < self.floor and owner not in report.limited_resolution:
                            report.limited_resolution.append(owner)
                for a,c0,c1,spacing in collars:
                    if a==axis and abs(lo-c0)<tolerance and abs(hi-c1)<tolerance:
                        h=spacing
                targets.append(max(h, self.floor))
            result.append(self._axis(positions, np.array(targets)))
            report.accepted.extend(ends)
        mesh = Mesh(*result, report)
        if self.min_dt and mesh.cfl(self.courant) < self.min_dt*(1-1e-9):
            raise RuntimeError("Mesh failed the requested CFL budget")
        return mesh

    def _wavelength_step(self, material):
        if min(material.epsilon_r) < 1 or min(material.mu_r) < 1:
            raise ValueError("Automatic meshing currently requires epsilon_r, mu_r >= 1")
        # Resolve the largest real refractive index across the operating band.
        # epsilon_r is epsilon_inf for dispersive Material objects.
        low=self.f_max/1000 if self.f_min is None else self.f_min
        frequencies=list(np.geomspace(low,self.f_max,257))
        # Sample narrow Lorentz resonances as well as the logarithmic band grid.
        for pole in material.lorentz:
            for resonance,damping in zip(pole.omega_0,pole.gamma):
                center=resonance/(2*np.pi); width=damping/(2*np.pi)
                frequencies.extend(center+offset*max(width,center*1e-8)
                                   for offset in np.linspace(-4,4,129))
        frequencies=np.array(sorted({f for f in frequencies if low<=f<=self.f_max}))
        eps = material.relative_permittivity(2*np.pi*frequencies)
        eps += 1j*np.asarray(material.sigma_e)/(2*np.pi*frequencies[:,None]*8.8541878128e-12)
        mu = np.asarray(material.mu_r)+1j*np.asarray(material.sigma_m)/(2*np.pi*frequencies[:,None]*1.25663706212e-6)
        index = np.max(np.sqrt(eps*mu).real)
        return C0/(self.f_max*index*self.cells_per_wavelength)

    def _axis(self, anchors, targets):
        spans = np.diff(anchors)
        if self.floor > self.max_step:
            raise ValueError("CFL floor exceeds max_step")
        counts = np.ceil(spans/targets-1e-10).astype(int)
        counts = np.maximum(1, np.minimum(counts, np.floor(spans/self.floor+1e-9).astype(int)))
        if counts.sum()<2:
            if spans.sum()<2*self.floor:
                raise ValueError("Domain is too small for two cells at the CFL spacing floor")
            counts[np.argmax(spans)]=2
        # Solve positive widths with exact anchor positions and adjacent growth bounds.
        # Coarse intervals may gain cells to make grading feasible without moving anchors.
        for _ in range(80):
            total = counts.sum()
            if total > self.max_axis_cells:
                raise ValueError("Mesh exceeds max_axis_cells")
            from scipy.sparse import lil_matrix
            eq = lil_matrix((len(spans), total))
            start = 0
            for row, count in enumerate(counts):
                eq[row, start:start+count] = 1
                start += count
            ub = lil_matrix((2*(total-1), total))
            for j in range(total-1):
                ub[2*j,j], ub[2*j,j+1] = 1, -self.growth
                ub[2*j+1,j], ub[2*j+1,j+1] = -self.growth, 1
            # Scale to unit length: SI millimetres otherwise upset LP tolerances.
            scale = spans.sum()
            upper = np.repeat(np.maximum(targets, spans/counts), counts)/scale
            solved = linprog(np.zeros(total), A_ub=ub.tocsr(), b_ub=np.zeros(ub.shape[0]),
                             A_eq=eq.tocsr(), b_eq=spans/scale,
                             bounds=list(zip(np.full(total, self.floor/scale), upper)), method="highs")
            if solved.success:
                values = [anchors[0]]
                start = 0
                for k, count in enumerate(counts):
                    lines = anchors[k]+np.cumsum(solved.x[start:start+count])*scale
                    lines[-1] = anchors[k+1]
                    values.extend(lines)
                    start += count
                return np.array(values)
            candidates = np.where((counts+1)*self.floor <= spans*(1+1e-12))[0]
            if not len(candidates):
                break
            k = candidates[np.argmax(spans[candidates]/counts[candidates])]
            counts[k] += 1
        raise ValueError("Anchors, min_step and growth are incompatible; relax growth or the CFL floor")
