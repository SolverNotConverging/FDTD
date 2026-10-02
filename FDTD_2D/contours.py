"""Closed near-to-far contours and the outgoing cylindrical 2D Green function."""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from scipy.sparse import coo_matrix
from .topology import EPS0, MU0


@dataclass(frozen=True)
class ClosedContour:
    vertices: tuple

    def __post_init__(self):
        points=np.asarray(self.vertices,dtype=float)
        if (points.ndim!=2 or points.shape[1]!=2 or len(points)<5 or not np.all(np.isfinite(points))
                or not np.array_equal(points[0],points[-1])):
            raise ValueError("NTFF contour must explicitly close: repeat its first vertex at the end")
        polygon=Polygon(points)
        if not polygon.is_valid or polygon.area<=0 or not polygon.exterior.is_simple:
            raise ValueError("NTFF contour must be simple with positive enclosed area")
        differences=np.diff(points,axis=0)
        if np.any(np.count_nonzero(differences,axis=1)!=1):
            raise ValueError("Reference NTFF contours require axis-aligned segments")
        if not polygon.exterior.is_ccw:
            object.__setattr__(self,"vertices",tuple(map(tuple,points[::-1])))

    @classmethod
    def rectangle(cls,bounds):
        if len(bounds)!=4 or not np.all(np.isfinite(bounds)) or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
            raise ValueError("Rectangle bounds must be finite and increasing")
        x0,y0,x1,y1=bounds
        return cls(((x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)))

    def compile(self,topology,pml=None):
        return CompiledContour(self,topology,pml)

    def anchors(self,rank=150,owner="NTFF"):
        from .mesh import Anchor
        return [Anchor(axis,value,rank,owner,"closed contour")
                for point in self.vertices[:-1] for axis,value in zip(("x","y"),point)]


@dataclass
class FarField:
    frequencies: np.ndarray
    angles: np.ndarray
    scalar_amplitude: np.ndarray  # q(r,phi) ~ amplitude(phi)*exp(-ikr)/sqrt(r)
    power_per_radian: np.ndarray  # W/radian for phasors; W*s^2/radian for raw time DFTs


def cylindrical_far_field(frequencies,angles,positions,normals,lengths,q,p,epsilon,mu,polarization,invariant_length=1.):
    """Continuous homogeneous-background equivalence integral, exp(+i omega t).

    q/p are colocated physical phasors, with p positive along outward energy
    flux. No open-surface or missing-side approximation is permitted by callers.
    """
    frequencies=np.asarray(frequencies); angles=np.asarray(angles,dtype=float)
    direction=np.column_stack((np.cos(angles),np.sin(angles)))
    omega=2*np.pi*frequencies; k=omega*np.sqrt(epsilon*mu)
    coefficient=epsilon if polarization=="TE" else mu
    derivative=-1j*omega[:,None]*coefficient*p
    projected=np.asarray(normals)@direction.T
    phase=np.exp(1j*k[:,None,None]*(np.asarray(positions)@direction.T)[None,:,:])
    integrand=(1j*k[:,None,None]*q[:,:,None]*projected[None,:,:]-derivative[:,:,None])
    amplitude=(-1j/4)*np.sqrt(2/(np.pi*k))*np.exp(1j*np.pi/4)
    amplitude=amplitude[:,None]*np.sum(integrand*phase*np.asarray(lengths)[None,:,None],axis=1)
    eta=np.sqrt(mu/epsilon)
    power=.5*(eta if polarization=="TE" else 1/eta)*abs(amplitude)**2*invariant_length
    return FarField(frequencies.copy(),angles.copy(),amplitude,power)


class CompiledContour:
    def __init__(self,contour,t,pml):
        polygon=Polygon(contour.vertices); self.contour=contour
        if not t.scene.domain.contains(polygon):
            raise ValueError("Closed NTFF contour must lie strictly inside the simulation domain")
        for g in t.scene.geometry:
            if not polygon.contains(g.shape):
                raise ValueError(f"NTFF contour must enclose geometry {g.name!r}")
        for port in t.scene.ports:
            if not polygon.contains(port.shape):
                raise ValueError("NTFF contour must enclose every physical port aperture")
        if pml is not None:
            x0,y0,x1,y1=polygon.bounds
            ranges={"xmin":x0-t.scene.x[0],"xmax":t.scene.x[1]-x1,
                    "ymin":y0-t.scene.y[0],"ymax":t.scene.y[1]-y1}
            if any(ranges[s]<=pml.width for s in pml.sides):
                raise ValueError("NTFF contour must lie outside PML")
        material=t.scene.background
        if material.has_dispersion or material.sigma_e!=(0.,0.,0.) or material.sigma_m!=(0.,0.,0.):
            raise ValueError("Reference NTFF requires a lossless nondispersive background")
        self.epsilon=material.epsilon_r[0]*EPS0; self.mu=material.mu_r[0]*MU0
        self.polarization=t.polarization
        qr=[]; qc=[]; qv=[]; pr=[]; pc=[]; pv=[]
        positions=[]; normals=[]; lengths=[]
        for start,end in zip(contour.vertices[:-1],contour.vertices[1:]):
            side=LineString([start,end]); tangent=(np.array(end)-start)/side.length
            normal=np.array([tangent[1],-tangent[0]])
            covered=0.
            for e,(line,owners,n) in enumerate(t.edge_geometry):
                if n is None or len(owners)!=2 or side.distance(line)>4*t.tol:
                    continue
                coords=np.asarray(line.coords)
                if np.max(abs((coords-np.array(start))@normal))>4*t.tol:
                    continue
                midpoint=np.array(line.interpolate(.5,normalized=True).coords[0])
                if side.distance(Point(midpoint))>4*t.tol:
                    continue
                lo,hi=sorted(((coords-np.array(start))@tangent).tolist())
                length=max(0.,min(hi,side.length)-max(lo,0.))
                if length<=t.tol:
                    continue
                if abs(length-line.length)>4*t.tol:
                    raise ValueError("NTFF corners must be mesh anchors")
                index=len(lengths)
                distance=abs((t.centers[list(owners)]-midpoint)@normal)
                for j,g in enumerate(owners):
                    if len(t.groups[g])!=1:
                        raise ValueError("NTFF shell must be Cartesian and uncut")
                    face=t.faces[t.groups[g][0]]
                    if not np.allclose([face.epsilon,face.mu,face.sigma_e,face.sigma_m],
                                       [material.epsilon_r[0],material.mu_r[0],0.,0.]):
                        raise ValueError("NTFF contour must sample the homogeneous background")
                    qr.append(index); qc.append(g); qv.append(distance[1-j]/distance.sum())
                pr.append(index); pc.append(e); pv.append(float(n@normal))
                positions.append(midpoint); normals.append(normal); lengths.append(length); covered+=length
            if abs(covered-side.length)>max(20*t.tol,side.length*1e-7):
                raise ValueError("Every NTFF side must be fully resolved by retained mesh faces")
        self.qsample=coo_matrix((qv,(qr,qc)),shape=(len(lengths),len(t.groups))).tocsr()
        self.psample=coo_matrix((pv,(pr,pc)),shape=(len(lengths),t.incidence.shape[1])).tocsr()
        self.positions=np.asarray(positions); self.normals=np.asarray(normals); self.lengths=np.asarray(lengths)

    def spectra(self,nf):
        return np.zeros((2,nf,len(self.lengths)),complex)

    def accumulate(self,spectra,q,p,qe,pe):
        # Auxiliary guides never occupy physical contour samples.
        spectra[0]+=qe[:,None]*(self.qsample@q[:self.qsample.shape[1]])
        spectra[1]+=pe[:,None]*(self.psample@p[:self.psample.shape[1]])

    def far_field(self,frequencies,spectra,angles,invariant_length=1.):
        return cylindrical_far_field(frequencies,angles,self.positions,self.normals,self.lengths,
                                     spectra[0],spectra[1],self.epsilon,self.mu,self.polarization,invariant_length)
