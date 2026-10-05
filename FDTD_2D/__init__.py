"""Geometry-first, nonuniform conformal 2D FDTD reference solver."""
from FDTD_common.material import Material
from .geometry import Scene, Geometry
from .mesh import Mesh, MeshPolicy, Anchor
from .surfaces import SurfaceImpedance, ThinSheet, PEC, PMC
from .ports import LumpedPort, WaveguidePort, VirtualWaveguide, GaussianPulse
from .solver import FDTD2D, RunControl, RunResult, ScatteringResult
from .pml import PML
from .domain import DomainPolicy, DomainReport
from .contours import ClosedContour, FarField
from .sources import PlaneWave
from .monitors import FieldMonitor

__all__ = ["Material","Scene","Geometry","Mesh","MeshPolicy","Anchor",
           "SurfaceImpedance","ThinSheet","PEC","PMC","LumpedPort",
           "WaveguidePort","GaussianPulse","FDTD2D","RunControl",
           "RunResult","ScatteringResult","PML","VirtualWaveguide","DomainPolicy",
           "DomainReport","ClosedContour","FarField","PlaneWave","FieldMonitor"]
