"""Split conformal cells and conservative enlargement on a nonuniform mesh.

The field incidence matrix and its transpose are shared by both curl updates.
Enlargement merges connected fluid faces only, never opposite sides of a sheet.
This is a finite-integration reference discretization, not the legacy Yee layout.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.sparse import coo_matrix, diags
from shapely import set_precision
from shapely.geometry import Point, LineString, Polygon, box
from shapely.ops import split, unary_union
from shapely.strtree import STRtree
from .surfaces import SurfaceImpedance, ThinSheet, PEC

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


def polygons(shape):
    if isinstance(shape, Polygon):
        return [shape] if shape.area > 0 else []
    return [p for g in getattr(shape, "geoms", ()) for p in polygons(g)]


def segments(shape):
    if isinstance(shape, LineString):
        c = list(shape.coords)
        return [LineString([a, b]) for a, b in zip(c[:-1], c[1:]) if a != b]
    return [p for g in getattr(shape, "geoms", ()) for p in segments(g)]


@dataclass
class Face:
    polygon: object
    cell: tuple
    epsilon: float
    mu: float
    sigma_e: float
    sigma_m: float
    reference_area: float


@dataclass
class TopologyReport:
    split_cells: int = 0
    extra_scalar_dofs: int = 0
    enlarged_groups: list = field(default_factory=list)
    staircase_fallbacks: list = field(default_factory=list)


@dataclass
class EdgeBlock:
    indices: np.ndarray
    mass: np.ndarray
    loss: np.ndarray
    laws: tuple


class UnsupportedCells(ValueError):
    def __init__(self,cells):
        self.cells=set(cells)
        super().__init__("Isolated conformal sliver cannot be enlarged")


class Topology:
    def __init__(self, scene, mesh, polarization="TE", enlargement=.3, conformal=True,
                 fallback=True, min_dt=0., courant=.85):
        if polarization not in {"TE", "TM"} or not 0 <= enlargement <= 1:
            raise ValueError("Use TE or TM and enlargement in [0,1]")
        if not (np.array_equal(mesh.x[[0,-1]], scene.x) and np.array_equal(mesh.y[[0,-1]], scene.y)):
            raise ValueError("Mesh endpoints must equal the scene domain")
        self.scene, self.mesh, self.polarization = scene, mesh, polarization
        self.report = TopologyReport()
        self.tol = min(mesh.dx.min(), mesh.dy.min())*1e-8
        self._staircase_cells=set()
        self._staircase_sheets=set()
        records=[]
        for _ in range(8):
            self.report=TopologyReport(staircase_fallbacks=records.copy())
            try:
                self._compile(conformal, enlargement)
                actual=self.cfl(courant)
                if not min_dt or actual >= min_dt*(1-1e-10):
                    return
                threshold=(2*courant/min_dt)**2
                bad=np.where(self.cfl_rates>threshold)[0]
                cells={self.faces[k].cell for g in bad for k in self.groups[g]}
                reason="conformal CFL bound below min_dt"
            except UnsupportedCells as error:
                cells=error.cells
                reason="isolated fluid fragment cannot be enlarged"
            if not fallback or not conformal:
                raise ValueError(reason+"; refine the mesh or enable fallback")
            new=cells-self._staircase_cells
            if new:
                self._staircase_cells.update(new)
                records.append((tuple(sorted(new)),reason))
                continue
            sheets=[g for g in scene.geometry if g.sheet and g.name not in self._staircase_sheets
                    and any(g.shape.intersects(box(mesh.x[i],mesh.y[j],mesh.x[i+1],mesh.y[j+1])) for i,j in cells)]
            if sheets:
                self._staircase_sheets.update(g.name for g in sheets)
                records.extend((g.name,reason+"; sheet snapped to mesh faces") for g in sheets)
                continue
            raise ValueError(reason+"; staircase fallback also cannot meet the CFL budget")
        raise ValueError("Local staircase fallback did not resolve topology within eight passes")

    def _compile(self, conformal, enlargement):
        scene, mesh = self.scene, self.mesh
        volumes = [g for g in scene.geometry if not g.sheet]
        conductor_parts = []
        for k, g in enumerate(volumes):
            if g.conductor:
                later = unary_union([v.shape for v in volumes[k+1:]])
                visible = set_precision(g.shape.difference(later), self.tol)
                if not visible.is_empty:
                    conductor_parts.append((visible, g.material, g.name))
        metal = unary_union([v[0] for v in conductor_parts])
        sheets = [replace(g,shape=self._staircase_line(g.shape)) if g.name in self._staircase_sheets else g
                  for g in scene.geometry if g.sheet]
        if not conformal:
            sheets=[replace(g,shape=self._staircase_line(g.shape)) for g in sheets]
        # Intersecting sheets need unambiguous junction ownership; fail explicitly.
        for k, a in enumerate(sheets):
            for b in sheets[k+1:]:
                if a.shape.intersects(b.shape):
                    raise ValueError("Intersecting sheets require a junction model; separate endpoints")
        faces = []
        for i, (x0,x1) in enumerate(zip(mesh.x[:-1], mesh.x[1:])):
            for j, (y0,y1) in enumerate(zip(mesh.y[:-1], mesh.y[1:])):
                cell = set_precision(box(x0,y0,x1,y1), self.tol)
                if conformal and (i,j) not in self._staircase_cells:
                    pieces = polygons(cell.difference(metal))
                else:
                    pieces = [] if metal.covers(cell.centroid) else [cell]
                for sheet in sheets:
                    if not sheet.shape.intersects(cell):
                        continue
                    for sheet_piece in segments(sheet.shape):
                        if not sheet_piece.intersects(cell):
                            continue
                        start, end = np.asarray(sheet_piece.coords)
                        direction = (end-start)/np.linalg.norm(end-start)
                        extent = 4*max(scene.x[1]-scene.x[0], scene.y[1]-scene.y[0])
                        cutter = LineString([start-extent*direction, end+extent*direction])
                        new = []
                        for p in pieces:
                            new.extend(polygons(split(p, cutter)))
                        pieces = [set_precision(p, self.tol) for p in new]
                pieces = [p for p in pieces if p.area > self.tol**2]
                if len(pieces) > 1:
                    self.report.split_cells += 1
                    self.report.extra_scalar_dofs += len(pieces)-1
                for p in pieces:
                    eps, mu, se, sm = self._material(p)
                    faces.append(Face(p, (i,j), eps, mu, se, sm, (x1-x0)*(y1-y0)))
        if not faces:
            raise ValueError("No retained field region")
        self.faces = faces
        tree = STRtree([f.polygon for f in faces])
        network = unary_union([f.polygon.boundary for f in faces])
        raw = []
        for line in segments(network):
            if line.length < self.tol:
                continue
            midpoint = line.interpolate(.5, normalized=True)
            near = tree.query(midpoint.buffer(self.tol*2))
            owners = [int(k) for k in near if faces[k].polygon.boundary.distance(midpoint) < self.tol*2]
            if not owners or len(owners) > 2:
                raise ValueError("Ambiguous conformal edge ownership")
            law = None
            for sheet in sheets:
                if sheet.shape.distance(midpoint) < self.tol*2:
                    law = sheet.material
                    break
            if len(owners) == 1 and law is None:
                side = self._domain_side(midpoint)
                if side:
                    law = scene.boundaries[side]
                else:
                    near_metal = [v for v in conductor_parts if v[0].boundary.distance(midpoint) < self.tol*3]
                    if near_metal:
                        law = near_metal[-1][1]
                    elif not conformal or self._staircase_cells:
                        closest = min(conductor_parts, key=lambda v:v[0].distance(midpoint), default=None)
                        if closest:
                            law = closest[1]
                    if law is None:
                        raise ValueError("Unowned interior boundary")
            raw.append((line, owners, law))
        parent = np.arange(len(faces))
        separated=[owners for _,owners,law in raw if law is not None and len(owners)==2]
        def root(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k
        # Connectivity is exclusively through ordinary fluid edges. Sheets block merging.
        for _ in range(len(faces)):
            groups = {}
            for k in range(len(faces)):
                groups.setdefault(root(k), []).append(k)
            small = {r for r, ks in groups.items() if sum(faces[k].polygon.area for k in ks)
                     < enlargement*max(faces[k].reference_area for k in ks)}
            if not small:
                break
            changed = False
            for r in sorted(small):
                if root(r) != r:
                    continue
                choices = []
                for line, owners, law in raw:
                    if len(owners) == 2 and law is None:
                        a,b = map(root, owners)
                        if a != b and r in (a,b):
                            other = b if a == r else a
                            # A path around a sheet tip must not merge the two
                            # boundary traces back into one scalar unknown.
                            if not any({root(v) for v in pair}=={r,other} for pair in separated):
                                choices.append((line.length, other))
                if choices:
                    parent[r] = max(choices)[1]
                    changed = True
            if not changed:
                raise UnsupportedCells(faces[k].cell for r in small for k in groups[r])
        groups = {}
        for k in range(len(faces)):
            groups.setdefault(root(k), []).append(k)
        self.groups = list(groups.values())
        self.report.enlarged_groups = [g for g in self.groups if len(g)>1]
        face_group = np.empty(len(faces), int)
        for k, members in enumerate(self.groups):
            face_group[members] = k
        self.face_group = face_group
        self.centers = np.array([[sum(faces[k].polygon.area*faces[k].polygon.centroid.x for k in g),
                                  sum(faces[k].polygon.area*faces[k].polygon.centroid.y for k in g)]
                                 for g in self.groups])
        self.areas = np.array([sum(faces[k].polygon.area for k in g) for g in self.groups])
        self.centers /= self.areas[:,None]
        te = self.polarization == "TE"
        self.scalar_mass = np.array([sum((f.mu*MU0 if te else f.epsilon*EPS0)*f.polygon.area
                                        for f in (faces[k] for k in g)) for g in self.groups])
        self.scalar_loss = np.array([sum((f.sigma_m if te else f.sigma_e)*f.polygon.area
                                        for f in (faces[k] for k in g)) for g in self.groups])
        rows, cols, data, self.blocks, self.edge_geometry = [], [], [], [], []
        nedge = 0
        for line, owners, law in raw:
            ids = [face_group[k] for k in owners]
            if len(ids) == 2 and ids[0] == ids[1] and law is None:
                continue
            p = np.asarray(line.coords)
            tangent = (p[-1]-p[0])/line.length
            normal = np.array([-tangent[1], tangent[0]])
            middle = np.array(line.interpolate(.5, normalized=True).coords[0])
            # Cartesian cells reproduce Yee half widths. Cut faces use centroid distances.
            distances = [max(abs(np.dot(self.centers[k]-middle, normal)), self.tol) for k in ids]
            eps = [(faces[o].epsilon*EPS0 if te else faces[o].mu*MU0) for o in owners]
            losses = [(faces[o].sigma_e if te else faces[o].sigma_m) for o in owners]
            length = line.length
            if law is None:
                mass = np.array([[sum(e*d for e,d in zip(eps,distances))*length]])
                loss = np.array([[sum(e*d for e,d in zip(losses,distances))*length]])
                incidence = [(ids[0],1), (ids[1],-1)]
                for k, sign in incidence:
                    rows.append(k); cols.append(nedge); data.append(sign*length)
                self.blocks.append(EdgeBlock(np.array([nedge]), mass, loss, (None,)))
                # Flux is positive away from the first owner.
                n = normal*np.sign(np.dot(middle-self.centers[ids[0]], normal))
                self.edge_geometry.append((line, tuple(ids), n))
                nedge += 1
                continue
            if isinstance(law, ThinSheet):
                if len(ids) != 2:
                    raise ValueError("A transmissive sheet must have retained fields on both sides")
                transform = np.array([[1,1],[1,-1]])/np.sqrt(2)
                laws = (law.odd,law.even) if te else (law.even,law.odd)
            else:
                transform = np.eye(len(ids))
                laws = (law,)*len(ids)
            keep = [k for k, z in enumerate(laws) if not (z.is_pec if te else z.is_pmc)]
            if not keep:
                continue
            u = transform[:,keep]
            mass = u.T @ np.diag(np.array(eps)*distances*length) @ u
            loss = u.T @ np.diag(np.array(losses)*distances*length) @ u
            indices = np.arange(nedge,nedge+len(keep))
            for local, column in enumerate(u.T):
                for k, weight in zip(ids,column):
                    rows.append(k); cols.append(nedge+local); data.append(weight*length)
                self.edge_geometry.append((line, tuple(ids), None))
            self.blocks.append(EdgeBlock(indices,mass,loss,tuple(laws[k] for k in keep)))
            nedge += len(keep)
        self.incidence = coo_matrix((data,(rows,cols)),shape=(len(self.groups),nedge)).tocsr()
        if not nedge:
            raise ValueError("Topology has no active vector field degrees of freedom")
        mrows,mcols,mdata = [],[],[]
        for block in self.blocks:
            for i,k in enumerate(block.indices):
                for j,l in enumerate(block.indices):
                    mrows.append(k); mcols.append(l); mdata.append(block.mass[i,j])
        self.vector_mass = coo_matrix((mdata,(mrows,mcols)),shape=(nedge,nedge)).tocsr()
        if not conformal:
            self.report.staircase_fallbacks = [(g.name,"explicit staircase mesh") for g in scene.geometry if g.conductor]

    def _staircase_line(self,line):
        endpoints=np.asarray(line.coords)[[0,-1]]
        indices=[np.array([np.argmin(abs(self.mesh.x-p[0])),np.argmin(abs(self.mesh.y-p[1]))]) for p in endpoints]
        a,b=indices
        if np.array_equal(a,b):
            axis=np.argmax(abs(endpoints[1]-endpoints[0]))
            b=b.copy(); b[axis]+=1 if b[axis]<self.mesh.shape[axis] else -1
        sign=np.sign(b-a).astype(int)
        points=[]
        while True:
            points.append((self.mesh.x[a[0]],self.mesh.y[a[1]]))
            if np.array_equal(a,b):
                break
            choices=[]
            for axis in range(2):
                if a[axis]!=b[axis]:
                    candidate=a.copy(); candidate[axis]+=sign[axis]
                    p=Point(self.mesh.x[candidate[0]],self.mesh.y[candidate[1]])
                    choices.append((line.distance(p),axis,candidate))
            _,_,a=min(choices,key=lambda v:(v[0],v[1]))
        return LineString(points)

    def _domain_side(self, point):
        for name,value,coord in (("xmin",self.scene.x[0],point.x),("xmax",self.scene.x[1],point.x),
                                 ("ymin",self.scene.y[0],point.y),("ymax",self.scene.y[1],point.y)):
            if abs(value-coord) < self.tol*3:
                return name

    def _material(self, polygon):
        remaining = polygon
        totals = np.zeros(4)
        for geom in reversed(self.scene.geometry):
            if geom.sheet or geom.conductor:
                continue
            portion = remaining.intersection(geom.shape)
            self._accumulate_material(totals, geom.material, portion.area)
            remaining = remaining.difference(geom.shape)
        self._accumulate_material(totals, self.scene.background, remaining.area)
        return totals/polygon.area

    def _accumulate_material(self, totals, material, area):
        if area == 0:
            return
        if material.has_dispersion:
            raise ValueError("Bulk ADE dispersion is not yet supported by the conformal reference solver")
        if len(set(material.epsilon_r)) > 1 or len(set(material.mu_r)) > 1:
            raise ValueError("Conformal reference solver currently requires isotropic bulk materials")
        totals += area*np.array([material.epsilon_r[0],material.mu_r[0],
                                 material.sigma_e[0],material.sigma_m[0]])

    def locate(self, point):
        hits = [self.face_group[k] for k,f in enumerate(self.faces) if f.polygon.contains(Point(point))]
        if len(set(hits)) != 1:
            raise ValueError("Sample point must lie strictly in one retained region")
        return hits[0]

    def scalar_grid(self, scalar):
        result = np.full(self.mesh.shape,np.nan)
        weight = np.zeros(self.mesh.shape)
        sums = np.zeros(self.mesh.shape)
        for k,f in enumerate(self.faces):
            sums[f.cell] += scalar[self.face_group[k]]*f.polygon.area
            weight[f.cell] += f.polygon.area
        np.divide(sums,weight,out=result,where=weight>0)
        return result

    def cfl(self, courant=.9):
        """Conservative spectral upper bound; includes split cells and enlargement."""
        from scipy.sparse import block_diag
        inv = block_diag([coo_matrix(np.linalg.inv(b.mass)) for b in self.blocks],format="csr")
        # Blocks are contiguous, so the block-diagonal order equals field order.
        q = diags(1/np.sqrt(self.scalar_mass))
        operator = q @ self.incidence @ inv @ self.incidence.T @ q
        self.cfl_rates=np.asarray(abs(operator).sum(axis=1)).ravel()
        upper = self.cfl_rates.max()
        return 2*courant/np.sqrt(upper)
