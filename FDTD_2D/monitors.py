"""Regional scalar-field DFTs on retained conformal degrees of freedom."""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import box


@dataclass(frozen=True)
class FieldMonitor:
    name: str
    bounds: tuple
    frequencies: tuple

    def compile(self, topology, dt):
        bounds=np.asarray(self.bounds,float); frequencies=np.asarray(self.frequencies,float)
        if (not self.name or bounds.shape!=(4,) or not np.all(np.isfinite(bounds))
                or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]):
            raise ValueError('Field monitor needs a name and positive rectangular bounds')
        if (frequencies.ndim!=1 or not 1<=len(frequencies)<=201 or not np.all(np.isfinite(frequencies))
                or np.any(frequencies<=0) or np.any(np.diff(frequencies)<=0) or np.any(frequencies*dt>=.5)):
            raise ValueError('Monitor frequencies must be positive, increasing and below Nyquist (1–201 samples)')
        host=getattr(topology,'host',topology)
        region=box(*bounds)
        if not host.scene.domain.covers(region):
            raise ValueError(f'Field monitor {self.name!r} extends outside the generated domain')
        ids=np.unique([host.face_group[k] for k,face in enumerate(host.faces)
                       if face.polygon.intersection(region).area>host.tol**2])
        if not len(ids):
            raise ValueError(f'Field monitor {self.name!r} contains no retained field cells')
        if len(ids)*len(frequencies)>20_000_000:
            raise ValueError('Monitor DFT exceeds 20 million complex samples; reduce region or frequencies')
        return RegionalDFT(self,ids.astype(int),frequencies)


class RegionalDFT:
    def __init__(self, definition, indices, frequencies):
        self.definition,self.indices,self.frequencies=definition,indices,frequencies
        self.spectra=np.zeros((len(frequencies),len(indices)),complex)

    def accumulate(self, scalar, time, dt):
        self.spectra+=np.exp(-2j*np.pi*self.frequencies*time)[:,None]*scalar[self.indices]*dt
