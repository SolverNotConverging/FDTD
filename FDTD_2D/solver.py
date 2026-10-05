"""Combined TE/TM conformal FDTD reference runtime and scattering studies."""
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.sparse.linalg import splu
from .topology import Topology
from .ports import LumpedPort, WaveguidePort


@dataclass(frozen=True)
class RunControl:
    max_time: float
    min_time: float = 0.0
    field_tolerance: float = None  # energy fraction of observed maximum
    dft_tolerance: float = None
    check_steps: int = 100
    consecutive: int = 3

    def __post_init__(self):
        if not np.isfinite(self.max_time) or self.max_time <= 0 or not 0 <= self.min_time <= self.max_time:
            raise ValueError("Require 0 <= min_time <= finite max_time")
        if self.check_steps < 1 or self.consecutive < 1:
            raise ValueError("Convergence checks require positive intervals and counts")
        for tolerance in (self.field_tolerance,self.dft_tolerance):
            if tolerance is not None and not 0 < tolerance < 1:
                raise ValueError("Convergence tolerances must lie in (0,1)")


@dataclass
class RunResult:
    frequencies: np.ndarray
    channels: tuple
    incoming: np.ndarray
    outgoing: np.ndarray
    scalar: np.ndarray
    vector: np.ndarray
    steps: int
    dt: float
    stop_reason: str
    diagnostics: list = field(default_factory=list)
    driven_channels: tuple = ()
    ntff_spectra: object = None
    ntff_contour: object = None
    invariant_length: float = 1.
    field_monitors: tuple = ()

    @property
    def time(self):
        return self.steps*self.dt

    def s_column(self, channel, incident_floor=1e-8):
        if self.driven_channels!=(channel,):
            raise ValueError("S-column normalization requires exactly one driven channel; use scattering() for a complete matrix")
        j=self.channels.index(channel)
        a=self.incoming[:,j]
        valid=np.isfinite(a)&(abs(a)>incident_floor*np.nanmax(abs(a)))&(abs(a)>0)
        result=np.full(self.outgoing.shape,np.nan+0j)
        np.divide(self.outgoing,a[:,None],out=result,where=valid[:,None])
        return result,valid

    def far_field(self,angles):
        if self.ntff_contour is None:
            raise ValueError("No closed NTFF contour was compiled for this run")
        return self.ntff_contour.far_field(self.frequencies,self.ntff_spectra,angles,self.invariant_length)


@dataclass
class ScatteringResult:
    frequencies: np.ndarray
    channels: tuple
    s: np.ndarray  # frequency, outgoing channel, incoming channel
    valid: np.ndarray
    runs: list

    def save(self,path):
        np.savez_compressed(path,frequencies=self.frequencies,channels=np.asarray(self.channels,dtype=str),
                 s=self.s,valid=self.valid)


class FDTD2D:
    def __init__(self,scene,mesh=None,mesh_policy=None,polarization="TE",frequencies=(),
                 dt=None,courant=None,enlargement=.3,conformal=True,fallback=True,pml=None,
                 domain_policy=None,ntff=None,tfsf=None):
        scene=deepcopy(scene)  # compiled geometry and port definitions form a snapshot
        if scene.x is None or scene.y is None:
            if mesh is not None or mesh_policy is None or (scene.x is None)!=(scene.y is None):
                raise ValueError("Automatic domain generation requires both axes omitted and a mesh_policy")
            from .domain import DomainPolicy
            generated_pml=(domain_policy or DomainPolicy()).generate(scene,mesh_policy)
            pml=generated_pml if pml is None else pml
        if mesh is None:
            if mesh_policy is None:
                raise ValueError("Supply a mesh or mesh_policy")
            anchors=scene.generated.anchors() if scene.generated else []
            if ntff is not None and ntff is not False:
                anchors.extend(ntff.anchors())
            if tfsf is not None:
                from .contours import ClosedContour
                anchors.extend(ClosedContour.rectangle(tfsf).anchors(owner="TF/SF"))
            mesh=mesh_policy.build(scene,anchors)
        self.scene,self.mesh=scene,mesh
        courant=(mesh_policy.courant if mesh_policy is not None else .85) if courant is None else courant
        if not 0 < courant < 1:
            raise ValueError("courant must lie strictly between zero and one")
        self.topology=Topology(scene,mesh,polarization,enlargement,conformal,fallback,
                               mesh_policy.min_dt if mesh_policy is not None else 0.,courant)
        limit=self.topology.cfl(courant)
        self.dt=limit if dt is None else float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0 or self.dt > limit*(1+1e-12):
            raise ValueError(f"dt exceeds conformal CFL bound {limit:g}")
        if mesh_policy is not None and mesh_policy.min_dt and self.dt < mesh_policy.min_dt:
            raise ValueError("Split topology violates min_dt; refine mesh or enable staircase fallback")
        self.polarization=polarization
        self.pml=pml
        if pml is not None:
            pml.compile(self.topology,self.dt)  # validate collar before any run
        self.frequencies=np.array(frequencies,dtype=float)
        if (self.frequencies.ndim!=1 or np.any(self.frequencies<=0)
                or not np.all(np.isfinite(self.frequencies)) or np.any(np.diff(self.frequencies)<=0)
                or np.any(self.frequencies*self.dt>=.5)):
            raise ValueError("DFT frequencies must be positive, increasing and below Nyquist")
        self.lumped={}; self.waveguides={}; channels=[]
        lengths={port.invariant_length for port in scene.ports}
        if len(lengths)>1:
            raise ValueError("All ports must share invariant_length for compatible power normalization")
        for port in scene.ports:
            if isinstance(port,LumpedPort):
                self.lumped[port.name]=(port,port.compile(self.topology)); channels.append((port.name,0))
            elif isinstance(port,WaveguidePort):
                if not len(self.frequencies):
                    raise ValueError("Waveguide ports require DFT frequencies")
                self.waveguides[port.name]=port.compile(self.topology,self.frequencies,self.dt)
                channels.extend((port.name,m) for m in range(port.modes))
            else:
                raise TypeError("Unknown port type")
        self.channels=tuple(channels)
        from .virtual import VirtualGuides
        host=self.topology
        self.virtual=VirtualGuides(host,self.waveguides)
        self.topology=self.virtual.topology
        limit=self.topology.cfl(courant)
        if dt is not None and self.dt>limit*(1+1e-12):
            raise ValueError(f"dt exceeds coupled virtual-guide CFL bound {limit:g}")
        previous_dt=self.dt
        self.dt=min(self.dt,limit)
        if mesh_policy is not None and mesh_policy.min_dt and self.dt<mesh_policy.min_dt:
            raise ValueError("Virtual guide violates min_dt")
        if self.dt!=previous_dt:
            for name,old in list(self.waveguides.items()):
                new=old.port.compile(host,self.frequencies,self.dt)
                new.left=old.left.copy(); new.right=old.right.copy()
                self.waveguides[name]=new
        self.guide_sources=self.virtual.compile_sources(self.frequencies,self.dt)
        self.ntff=None
        if ntff is not False:
            from .contours import ClosedContour
            if ntff is None and scene.generated is not None:
                ntff=ClosedContour.rectangle(scene.generated.ntff_bounds)
            if ntff is not None:
                self.ntff=ntff.compile(host,pml)
        self.tfsf=tfsf if tfsf is not None else (scene.generated.tfsf_bounds if scene.generated else None)
        if scene.plane_waves and self.tfsf is None:
            raise ValueError("Plane waves need an automatic domain or explicit closed TF/SF box")
        if self.ntff is not None and self.tfsf is not None:
            from shapely.geometry import box,Polygon
            if not Polygon(self.ntff.contour.vertices).contains(box(*self.tfsf)):
                raise ValueError("Closed NTFF contour must surround the TF/SF box in the scattered-field region")
        self.scalar=np.zeros(len(self.topology.groups))
        self.vector=np.zeros(self.topology.incidence.shape[1])
        for name,(port,v) in list(self.lumped.items()):
            self.lumped[name]=(port,np.pad(v,(0,(len(self.scalar) if polarization=="TM" else len(self.vector))-len(v))))

    def _runtime(self):
        t=self.topology; dt=self.dt
        mass=t.vector_mass
        loss=block_diag([csc_matrix(b.loss) for b in t.blocks],format="lil")
        history=[]
        cache={}
        for block in t.blocks:
            for index,law in zip(block.indices,block.laws):
                if law is None or (law.is_pmc if self.polarization=="TE" else law.is_pec):
                    continue
                if law not in cache:
                    a,b,c,d=law.state_space(admittance=self.polarization=="TE")
                    identity=np.eye(len(b))
                    f=np.linalg.solve(identity-dt*a/2,identity+dt*a/2)
                    g=np.linalg.solve(identity-dt*a/2,dt*b)
                    l=c@(identity+f)/2
                    z=d+c@g/2
                    cache[law]=(f,g,l,z)
                f,g,l,z=cache[law]
                length=t.edge_geometry[index][0].length
                loss[index,index]+=length*z
                history.append([index,f,g,l,length,np.zeros(len(g))])
        scalar_loss=diags(t.scalar_loss,format="csc")
        for port,v in self.lumped.values():
            column=csc_matrix(v[:,None])
            loading=(column@column.T)/(port.resistance*port.invariant_length)
            if self.polarization=="TE":
                loss=loss+loading
            else:
                scalar_loss=scalar_loss+loading
        vector_left=mass/dt+loss/2
        vector_right=mass/dt-loss/2
        scalar_mass=diags(t.scalar_mass)
        scalar_left=scalar_mass/dt+scalar_loss/2
        scalar_right=scalar_mass/dt-scalar_loss/2
        return splu(vector_left.tocsc()),vector_right,splu(scalar_left.tocsc()),scalar_right,history

    def _waveguide_source(self,compiled,waveform,mode,steps):
        """Broadband TF/SF synthesis using tracked complex modal profiles."""
        if mode >= compiled.port.modes or mode < 0:
            raise ValueError("Source mode is outside the monitored bank")
        nfft=1 << int(np.ceil(np.log2(max(8,2*(steps+1)))))
        times=np.arange(nfft)*self.dt
        pulse=np.asarray(waveform(times),dtype=float)
        if pulse.shape != times.shape or not np.all(np.isfinite(pulse)):
            raise ValueError("Waveforms must return finite vectorized real samples")
        spectrum=np.fft.rfft(pulse)
        if not np.any(abs(spectrum)>0):
            raise ValueError("Waveguide excitation waveform is zero")
        ff=np.fft.rfftfreq(nfft,self.dt)
        significant=abs(spectrum)>1e-5*abs(spectrum).max()
        inband=(ff>=self.frequencies[0])&(ff<=self.frequencies[-1])
        if np.any(significant&~inband):
            raise ValueError("Significant waveguide pulse spectrum lies outside the solved band")
        bins=np.where(inband)[0]
        n=len(compiled.widths)
        qs=np.zeros((len(ff),n),complex); ps=qs.copy()
        valid=compiled.valid[:,mode]
        if np.any(~valid):
            raise ValueError("Excitation mode crosses cutoff in the solved band; restrict the band")
        def interpolate(values):
            return np.interp(ff[bins],self.frequencies,values.real)+1j*np.interp(ff[bins],self.frequencies,values.imag)
        beta=interpolate(compiled.beta[:,mode])
        polarity=-1 if self.polarization=="TE" else 1
        for k in range(n):
            q=interpolate(compiled.q_modes[:,k,mode])
            p=interpolate(compiled.p_modes[:,k,mode])
            inside=compiled.distances[k,0 if compiled.port.normal==1 else 1]
            qs[bins,k]=polarity*q*np.exp(-1j*beta*inside)*spectrum[bins]
            # Incoming local flux is negative along outward normal, at native half time.
            ps[bins,k]=-polarity*p*spectrum[bins]*np.exp(1j*np.pi*ff[bins]*self.dt)*compiled.sign[k]
        qtime=np.fft.irfft(qs,n=nfft,axis=0)[:steps+1]
        ptime=np.fft.irfft(ps,n=nfft,axis=0)[:steps]
        return qtime,ptime

    def run(self,control,excitations=None,initial_scalar=None,initial_vector=None,probes=(),progress=None,field_monitors=()):
        """Run independently; optional progress(info, scalar, vector) receives read-only views.

        Callbacks run after check_steps and at completion and may raise to cancel
        cooperatively. FieldMonitor definitions accumulate regional scalar DFTs
        every step and contribute to DFT convergence without changing updates.
        """
        if not isinstance(control,RunControl):
            control=RunControl(float(control))
        excitations={} if excitations is None else dict(excitations)
        steps=int(np.floor(control.max_time/self.dt+1e-12))
        if steps < 1:
            raise ValueError("max_time is shorter than one time step")
        t=self.topology; dt=self.dt
        q=np.zeros_like(self.scalar) if initial_scalar is None else np.array(initial_scalar,dtype=float,copy=True)
        p=np.zeros_like(self.vector) if initial_vector is None else np.array(initial_vector,dtype=float,copy=True)
        if q.shape!=self.scalar.shape or p.shape!=self.vector.shape or not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
            raise ValueError("Initial fields do not match compiled degrees of freedom")
        active={}; end=0
        plane_definitions={s.name:s for s in self.scene.plane_waves}
        incident_planes=[]
        for channel,waveform in excitations.items():
            if isinstance(channel,str):
                channel=(channel,0)
            if channel not in self.channels and channel not in [(name,0) for name in plane_definitions]:
                raise ValueError(f"Unknown channel {channel}")
            if not hasattr(waveform,"end_time"):
                raise ValueError("Waveform must declare end_time to gate convergence")
            if not np.isfinite(waveform.end_time) or waveform.end_time<0:
                raise ValueError("Waveform end_time must be finite and nonnegative")
            end=max(end,waveform.end_time)
            active[channel]=waveform
            if channel[0] in plane_definitions:
                from .sources import IncidentPlane
                incident=IncidentPlane(plane_definitions[channel[0]],t,self.tfsf,self.pml,dt,waveform)
                incident_planes.append(incident); end=max(end,incident.end_time)
        wave_sources={}
        for (name,mode),wave in active.items():
            if name in self.waveguides:
                source=self.guide_sources.get(name,self.waveguides[name])
                fields=self._waveguide_source(source,wave,mode,steps)
                if name in self.guide_sources:
                    speed=299792458/np.sqrt(np.max(source.materials[:,0]*source.materials[:,1])/(8.8541878128e-12*1.25663706212e-6))
                    slowness=1/speed
                    if len(self.frequencies)>1:
                        slowness=max(slowness,np.max(np.gradient(source.beta[:,mode].real,2*np.pi*self.frequencies)))
                    # A pulse must reach the physical aperture before early stopping.
                    end=max(end,wave.end_time+source.port.position*slowness)
                if name in self.guide_sources and self.polarization=="TE":
                    fields=tuple(v*self.waveguides[name].port.normal for v in fields)
                if name in wave_sources:
                    wave_sources[name]=tuple(a+b for a,b in zip(wave_sources[name],fields))
                else:
                    wave_sources[name]=fields
        vp,vr,sq,sr,history=self._runtime()
        cpml=self.virtual.cpml(self.pml,dt)
        nf=len(self.frequencies)
        port_dft={name:np.zeros((2,nf),complex) for name in self.lumped}
        guide_dft={name:np.zeros((3,nf,len(g.widths)),complex) for name,g in self.waveguides.items()}
        probe_ids=[t.locate(point) for point in probes]
        probe_dft=np.zeros((nf,len(probe_ids)),complex)
        monitors=tuple(m.compile(t,dt) for m in field_monitors)
        ntff_dft=self.ntff.spectra(nf) if self.ntff is not None else None
        if control.dft_tolerance is not None and not ((nf and (self.channels or probe_ids or self.ntff)) or monitors):
            raise ValueError("DFT convergence requires frequency-resolved ports, probes, NTFF or field monitors")
        previous_dft=None; peak=0.; memory_peak=0.; stable=0; window_peak=0.; memory_window=0.
        # At least two cycles of the lowest DFT bin per comparison window.
        monitored_frequencies=[*self.frequencies,*(f for m in monitors for f in m.frequencies)]
        interval=max(control.check_steps,int(np.ceil(2/(min(monitored_frequencies)*dt))) if monitored_frequencies else 1)
        diagnostics=[]; reason="max_time"; completed=0
        for step in range(steps):
            oldq=q.copy(); oldp=p.copy()
            drive=t.incidence.T@q
            if cpml is not None:
                drive=cpml.vector_drive(drive)
            for index,f,g,l,length,state in history:
                drive[index]-=length*(l@state)
            for incident in incident_planes:
                drive+=incident.vector_correction()
            for (name,mode),wave in active.items():
                if name in self.lumped and self.polarization=="TE":
                    port,v=self.lumped[name]
                    drive+=v*float(wave(step*dt))/(port.resistance*port.invariant_length)
            for name,(qtime,ptime) in wave_sources.items():
                compiled=self.guide_sources.get(name,self.waveguides[name])
                inside=compiled.left if compiled.port.normal==1 else compiled.right
                for k,cell in enumerate(inside):
                    edge=compiled.vector[k]
                    drive[edge]-=t.incidence[cell,edge]*qtime[step,k]
            p=vp.solve(vr@oldp+drive)
            for item in history:
                index,f,g,l,length,state=item
                item[-1]=f@state+g*(oldp[index]+p[index])/2
            drive=cpml.scalar_drive(p) if cpml is not None else -t.incidence@p
            for incident in incident_planes:
                drive+=incident.scalar_correction(step)
            for (name,mode),wave in active.items():
                if name in self.lumped and self.polarization=="TM":
                    port,v=self.lumped[name]
                    drive+=v*float(wave((step+.5)*dt))/(port.resistance*port.invariant_length)
            for name,(qtime,ptime) in wave_sources.items():
                compiled=self.guide_sources.get(name,self.waveguides[name])
                inside=compiled.left if compiled.port.normal==1 else compiled.right
                for k,cell in enumerate(inside):
                    edge=compiled.vector[k]
                    drive[cell]-=t.incidence[cell,edge]*ptime[step,k]
            q=sq.solve(sr@oldq+drive)
            time=(step+1)*dt; completed=step+1
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(p)):
                raise FloatingPointError(f"Nonfinite field at step {completed}")
            qe=np.exp(-2j*np.pi*self.frequencies*time)*dt
            pe=np.exp(-2j*np.pi*self.frequencies*(step+.5)*dt)*dt
            for name,(port,v) in self.lumped.items():
                sample=v@((oldp+p)/2 if self.polarization=="TE" else (oldq+q)/2)
                sample_time=step*dt if self.polarization=="TE" else (step+.5)*dt
                source=active.get((name,0))
                voltage_source=0. if source is None else float(source(sample_time))
                current=(voltage_source-sample)/port.resistance
                phase=np.exp(-2j*np.pi*self.frequencies*sample_time)*dt
                port_dft[name][0]+=sample*phase
                port_dft[name][1]+=current*phase
            for name,compiled in self.waveguides.items():
                left,right,flux=compiled.measure(q,p)
                if name in wave_sources and name not in self.guide_sources:
                    inc=wave_sources[name][0][step+1]
                    if compiled.port.normal==1:
                        left=left-inc
                    else:
                        right=right-inc
                guide_dft[name][0]+=qe[:,None]*left
                guide_dft[name][1]+=qe[:,None]*right
                guide_dft[name][2]+=pe[:,None]*flux
            if probe_ids:
                probe_dft+=qe[:,None]*q[probe_ids]
            for monitor in monitors:
                monitor.accumulate(q,time,dt)
            if self.ntff is not None:
                self.ntff.accumulate(ntff_dft,q,p,qe,pe)
            energy=.5*(q@(t.scalar_mass*q)+p@(t.vector_mass@p))
            peak=max(peak,energy); window_peak=max(window_peak,energy)
            memory=sum(np.linalg.norm(item[-1])**2 for item in history)
            memory+=sum(incident.memory_norm() for incident in incident_planes)
            if cpml is not None:
                memory+=cpml.memory_norm()
            memory_peak=max(memory_peak,memory); memory_window=max(memory_window,memory)
            if progress is not None and (completed%control.check_steps==0 or completed==steps):
                qview,pview=q.view(),p.view()
                qview.flags.writeable=False; pview.flags.writeable=False
                progress(dict(step=completed,steps=steps,time=time,dt=dt,energy=float(energy)),qview,pview)
            if completed%interval==0:
                field_ratio=window_peak/peak if peak>0 else np.inf
                memory_ratio=memory_window/memory_peak if memory_peak>0 else 0
                spectrum=np.concatenate([*(v.ravel() for v in port_dft.values()),
                                         *(v.ravel() for v in guide_dft.values()),probe_dft.ravel(),
                                         ntff_dft.ravel() if ntff_dft is not None else np.array([],complex),
                                         *(m.spectra.ravel() for m in monitors)])
                norm=np.linalg.norm(spectrum)
                dft_ratio=(np.linalg.norm(spectrum-previous_dft)/norm
                           if previous_dft is not None and norm>0 else np.inf)
                # Both requested tests must pass. A zero field before excitation never passes.
                ready=time-interval*dt>=max(control.min_time,end)
                tests=[]
                if control.field_tolerance is not None:
                    tests.append(max(field_ratio,memory_ratio)<control.field_tolerance)
                if control.dft_tolerance is not None:
                    tests.append(dft_ratio<control.dft_tolerance)
                stable=stable+1 if ready and tests and all(tests) else 0
                diagnostics.append(dict(time=time,energy=energy,field_ratio=field_ratio,
                                        memory_ratio=memory_ratio,dft_ratio=dft_ratio,consecutive=stable))
                previous_dft=spectrum.copy(); window_peak=0; memory_window=0
                if stable>=control.consecutive:
                    if progress is not None and completed%control.check_steps:
                        qview,pview=q.view(),p.view()
                        qview.flags.writeable=False; pview.flags.writeable=False
                        progress(dict(step=completed,steps=steps,time=time,dt=dt,energy=float(energy)),qview,pview)
                    reason="converged"; break
        self.scalar,self.vector=q,p
        incoming=np.full((nf,len(self.channels)),np.nan+0j); outgoing=incoming.copy()
        for name,(port,v) in self.lumped.items():
            j=self.channels.index((name,0)); voltage,current=port_dft[name]
            # DFTs store peak phasors; the sqrt(2) puts circuit and modal waves
            # on the same time-average watt normalization.
            incoming[:,j]=(voltage+port.resistance*current)/(2*np.sqrt(2*port.resistance))
            outgoing[:,j]=(voltage-port.resistance*current)/(2*np.sqrt(2*port.resistance))
        for name,compiled in self.waveguides.items():
            a,b=compiled.decompose(guide_dft[name])
            for mode in range(compiled.port.modes):
                j=self.channels.index((name,mode))
                incoming[:,j]=a[:,mode]; outgoing[:,j]=b[:,mode]
                if (name,mode) in active and name not in self.guide_sources:
                    times=np.arange(completed+1)*dt
                    pulse=active[name,mode](times)
                    incoming[:,j]+=np.exp(-2j*np.pi*self.frequencies[:,None]*times)@pulse*dt
        result=RunResult(self.frequencies,self.channels,incoming,outgoing,q.copy(),p.copy(),completed,dt,reason,diagnostics,tuple(active))
        result.probe_spectra=probe_dft
        result.field_monitors=monitors
        result.ntff_spectra=ntff_dft; result.ntff_contour=self.ntff
        result.invariant_length=next(iter({p.invariant_length for p in self.scene.ports}),1.)
        return result

    def scattering(self,control,waveform,channels=None,incident_floor=1e-6,progress=None,field_monitors=()):
        """Independent drives form a complete S matrix, including mixed port types.

        Receiver lumped terminals remain loaded. Incoming waves at waveguide
        receivers are included in A; solve B=S*A rather than discarding them.
        """
        channels=self.channels if channels is None else tuple(channels)
        if not channels or len(channels)!=len(self.channels) or set(channels)!=set(self.channels):
            raise ValueError("A complete scattering study must independently drive every channel")
        runs=[]
        for index,channel in enumerate(channels):
            callback=None
            if progress is not None:
                def callback(info,q,p,index=index,channel=channel):
                    progress(dict(info,run_index=index,run_count=len(channels),channel=channel),q,p)
            runs.append(self.run(control,{channel:waveform},progress=callback,field_monitors=field_monitors))
        a=np.stack([r.incoming for r in runs],axis=-1)
        b=np.stack([r.outgoing for r in runs],axis=-1)
        nf,nc,_=a.shape
        s=np.full((nf,nc,nc),np.nan+0j); valid=np.zeros(nf,bool)
        peak=np.nanmax(abs(a))
        for k in range(nf):
            if len(channels)<nc or not np.all(np.isfinite(a[k])) or not np.all(np.isfinite(b[k])):
                continue
            singular=np.linalg.svd(a[k],compute_uv=False)
            if singular[-1] <= incident_floor*max(peak,singular[0]):
                continue
            s[k]=b[k]@np.linalg.pinv(a[k],rcond=incident_floor); valid[k]=True
        return ScatteringResult(self.frequencies,self.channels,s,valid,runs)
