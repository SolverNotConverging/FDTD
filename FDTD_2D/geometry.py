"""Physical, mesh-independent geometry. All coordinates are metres.

Rank controls mesh requests only; insertion order controls material overlays.
Sheets have zero area and never become a one-cell-thick opaque volume.
"""
from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Point, Polygon, LineString, box
from FDTD_common.material import Material
from .surfaces import SurfaceImpedance, ThinSheet, PEC, PMC


@dataclass(frozen=True)
class Geometry:
    name: str
    shape: object
    material: object
    rank: int = 0
    sheet: bool = False

    def __post_init__(self):
        if not self.name or not self.shape.is_valid or self.shape.is_empty:
            raise ValueError("Geometry needs a name and a valid nonempty shape")
        if not isinstance(self.rank, int):
            raise ValueError("Geometry rank must be an integer; larger ranks win")
        if self.sheet:
            if not isinstance(self.shape, LineString) or not isinstance(self.material, (SurfaceImpedance, ThinSheet)):
                raise ValueError("A sheet needs a line and a surface impedance")
        elif not isinstance(self.shape, Polygon):
            raise ValueError("Volume geometry must be a polygon")

    @property
    def bounds(self):
        return self.shape.bounds

    @property
    def conductor(self):
        return isinstance(self.material, (SurfaceImpedance, ThinSheet))


@dataclass
class Scene:
    x: tuple = None
    y: tuple = None
    background: Material = field(default_factory=lambda: Material("vacuum"))
    geometry: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    boundaries: dict = field(default_factory=dict)
    plane_waves: list = field(default_factory=list)
    generated: object = None

    def __post_init__(self):
        for span in (self.x, self.y):
            if span is None:
                continue
            if len(span) != 2 or not np.all(np.isfinite(span)) or span[1] <= span[0]:
                raise ValueError("Domain ranges must be finite and increasing")
        if self.background.kind != "ordinary":
            raise ValueError("Background must be an ordinary dielectric")
        self.boundaries = {side: self.boundaries.get(side, PEC)
                           for side in ("xmin", "xmax", "ymin", "ymax")}
        if not all(isinstance(v, SurfaceImpedance) for v in self.boundaries.values()):
            raise ValueError("Outer boundaries must be SurfaceImpedance objects")

    @property
    def domain(self):
        if self.x is None or self.y is None:
            raise ValueError("Domain has not yet been generated from the objects")
        return box(self.x[0], self.y[0], self.x[1], self.y[1])

    def _check_bounds(self,shape):
        bounds=shape.bounds
        for axis,span in enumerate((self.x,self.y)):
            if span is not None and (bounds[axis]<span[0] or bounds[axis+2]>span[1]):
                raise ValueError("Geometry or port lies outside the explicit domain")

    def add(self, name, shape, material, rank=0, sheet=False):
        if any(g.name == name for g in self.geometry):
            raise ValueError(f"Duplicate geometry name {name!r}")
        if isinstance(material, str):
            material = {"PEC": PEC, "PMC": PMC}.get(material.upper())
            if material is None:
                raise ValueError("Use a Material or SurfaceImpedance object")
        if isinstance(material, Material) and material.kind != "ordinary":
            material = PEC if material.kind == "PEC" else PMC
        if not isinstance(material, (Material, SurfaceImpedance, ThinSheet)):
            raise TypeError("Invalid material")
        if isinstance(material, ThinSheet) and not sheet:
            raise ValueError("ThinSheet is only valid on zero-thickness geometry")
        geom = Geometry(name, shape, material, rank, sheet)
        self._check_bounds(shape)
        self.geometry.append(geom)
        return geom

    def rectangle(self, name, x, y, material, rank=0):
        if x[1] <= x[0] or y[1] <= y[0]:
            raise ValueError("Use sheet() for zero-thickness geometry")
        return self.add(name, box(x[0], y[0], x[1], y[1]), material, rank)

    def circle(self, name, center, radius, material, rank=0, segments=128):
        if not np.isfinite(radius) or radius <= 0 or segments < 16:
            raise ValueError("Circle requires positive radius and at least 16 segments")
        return self.add(name, Point(center).buffer(radius, quad_segs=int(segments) // 4), material, rank)

    def polygon(self, name, vertices, material, rank=0):
        return self.add(name, Polygon(vertices), material, rank)

    def sheet(self, name, start, end, material=PEC, rank=0):
        return self.add(name, LineString([start, end]), material, rank, sheet=True)

    def add_port(self, port):
        if any(p.name == port.name for p in self.ports):
            raise ValueError(f"Duplicate port name {port.name!r}")
        self._check_bounds(port.shape)
        self.ports.append(port)
        return port

    def add_plane_wave(self,source):
        from .sources import PlaneWave
        if not isinstance(source,PlaneWave) or any(s.name==source.name for s in self.plane_waves):
            raise ValueError("Plane-wave sources need unique names")
        if any(p.name==source.name for p in self.ports):
            raise ValueError("Source and port names must be distinct")
        self.plane_waves.append(source)
        return source

    def material_at(self, point):
        material = self.background
        p = Point(point)
        for geom in self.geometry:
            if not geom.sheet and geom.shape.covers(p):
                material = geom.material
        return material
