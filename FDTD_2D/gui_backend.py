"""Versioned JSON bridge used by the native C++ modeller.

Only project data cross the process boundary. No eval, generated Python code,
or Qt/Python GUI dependency is required. stdout is newline-delimited JSON.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from . import (Scene,Material,PEC,PMC,SurfaceImpedance,ThinSheet,LumpedPort,
               WaveguidePort,VirtualWaveguide,PlaneWave,MeshPolicy,DomainPolicy,
               FDTD2D,RunControl,GaussianPulse)
from .monitors import FieldMonitor

SCHEMA_VERSION=1

def pulse_from_settings(settings):
    if settings.get("pulse_mode","manual")=="auto":
        lo=finite(settings["f_min_ghz"],"start frequency")*1e9
        hi=finite(settings["f_max_ghz"],"stop frequency")*1e9
        if not 0<lo<hi: raise ValueError("Automatic pulse requires 0 < fmin < fmax")
        # Leave both spectral tails below the modal synthesis guard (1e-5),
        # while retaining useful nonzero source power at the band endpoints.
        return GaussianPulse((lo+hi)/2,np.sqrt(np.log(5e5))/(np.pi*(hi-lo)/2))
    if settings.get("pulse_mode","manual")!="manual": raise ValueError("Unknown pulse mode")
    return GaussianPulse(finite(settings.get("pulse_ghz",15),"pulse frequency")*1e9,
        finite(settings.get("pulse_width_ps",60),"pulse width")*1e-12,
        delay=finite(settings["pulse_delay_ps"],"pulse delay")*1e-12 if settings.get("pulse_delay_ps") else None)


def finite(value,label):
    result=float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def material(spec,band):
    kind=spec.get("type","PEC")
    if kind=="PEC": return PEC
    if kind=="PMC": return PMC
    if kind=="dielectric":
        return Material(spec.get("name","dielectric"),epsilon_r=finite(spec.get("epsilon_r",1),"epsilon_r"),
                        mu_r=finite(spec.get("mu_r",1),"mu_r"),sigma_e=finite(spec.get("sigma_e",0),"sigma_e"),
                        sigma_m=finite(spec.get("sigma_m",0),"sigma_m"))
    if kind=="SIBC":
        if spec.get("model","resistance")=="conductor":
            return SurfaceImpedance.good_conductor(finite(spec.get("conductivity",5.8e7),"conductivity"),band)
        return SurfaceImpedance(finite(spec.get("resistance",50),"surface resistance"))
    if kind=="film":
        return ThinSheet.resistive(finite(spec.get("resistance",50),"sheet resistance"))
    if kind=="thin_metal":
        return ThinSheet.conductive(finite(spec.get("conductivity",5.8e7),"conductivity"),
                                    finite(spec.get("thickness_um",1),"thickness")*1e-6,band)
    raise ValueError(f"Unknown material type {kind!r}")


def compile_project(project):
    if project.get("version")!=SCHEMA_VERSION or project.get("units")!="mm":
        raise ValueError("Expected a version 1 FDTD project with units 'mm'")
    settings=project.get("settings",{})
    low=finite(settings.get("f_min_ghz",10),"f_min")*1e9
    high=finite(settings.get("f_max_ghz",20),"f_max")*1e9
    count=int(settings.get("frequency_count",41))
    if not 0<low<=high or not 1<=count<=2001 or (low<high and count<2):
        raise ValueError("Frequency band and sample count are inconsistent")
    # Fitting needs an interval even when monitoring a single frequency.
    band=(low,high) if low<high else (low*.99,high*1.01)
    background=dict(project.get("background",{}),type="dielectric")
    scene=Scene(background=material(background,band))
    mm=lambda values:tuple(finite(v,"coordinate")*1e-3 for v in values)
    for item in project.get("objects",[]):
        name=item["name"]; kind=item["kind"]; model=material(item.get("material",{}),band)
        rank=int(item.get("rank",0))
        if kind=="rectangle":
            x,y=finite(item["x"],"x")*1e-3,finite(item["y"],"y")*1e-3
            w,h=finite(item["width"],"width")*1e-3,finite(item["height"],"height")*1e-3
            scene.rectangle(name,(x,x+w),(y,y+h),model,rank)
        elif kind=="circle":
            scene.circle(name,mm((item["x"],item["y"])),finite(item["radius"],"radius")*1e-3,model,rank)
        elif kind=="polygon":
            scene.polygon(name,[mm(p) for p in item["vertices"]],model,rank)
        elif kind=="sheet":
            scene.sheet(name,mm(item["start"]),mm(item["end"]),model,rank)
        else:
            raise ValueError(f"Unknown geometry {kind!r}")
    for item in project.get("ports",[]):
        common=dict(name=item["name"],rank=int(item.get("rank",100)),
                    invariant_length=finite(item.get("depth_mm",1),"invariant depth")*1e-3)
        if item["kind"]=="lumped":
            end=mm(item["end"]) if item.get("end") is not None else None
            scene.add_port(LumpedPort(start=mm(item["start"]),end=end,
                                     resistance=finite(item.get("resistance",50),"port resistance"),**common))
        elif item["kind"]=="waveguide":
            config=VirtualWaveguide(int(item.get("length_cells",32)),int(item.get("pml_cells",12)),
                                   int(item.get("clearance_cells",6)))
            scene.add_port(WaveguidePort(axis=item.get("axis","x"),position=finite(item["position"],"position")*1e-3,
                span=mm(item["span"]),normal=-int(item["inward_normal"]) if "inward_normal" in item else int(item.get("normal",1)),modes=int(item.get("modes",1)),
                virtual_waveguide=config,mesh_step=finite(item["mesh_step_mm"],"port step")*1e-3 if item.get("mesh_step_mm") else None,**common))
        else:
            raise ValueError("Unknown port kind")
    for item in project.get("sources",[]):
        scene.add_plane_wave(PlaneWave(item["name"],item.get("axis","x"),int(item.get("direction",1))))
    policy=MeshPolicy(max_step=finite(settings.get("max_step_mm",1),"max step")*1e-3,f_min=low,f_max=high,
        min_step=finite(settings.get("min_step_mm",.2),"min step")*1e-3,
        min_dt=finite(settings.get("min_dt_ps",0),"min dt")*1e-12,
        cells_per_wavelength=int(settings.get("cells_per_wavelength",20)),growth=finite(settings.get("growth",2),"growth"))
    domain=DomainPolicy(pml_cells=int(settings.get("pml_cells",8)),
                        clearance_wavelengths=finite(settings.get("clearance_wavelengths",.25),"clearance"))
    return FDTD2D(scene,mesh_policy=policy,polarization=settings.get("polarization","TM"),
                  frequencies=np.linspace(low,high,count),domain_policy=domain,
                  enlargement=finite(settings.get("enlargement",.3),"enlargement"),fallback=bool(settings.get("fallback",True)))


def clean_json(value):
    if isinstance(value,np.ndarray): return clean_json(value.tolist())
    if isinstance(value,np.generic): return clean_json(value.item())
    if isinstance(value,(tuple,list)): return [clean_json(v) for v in value]
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,data):
    path=Path(path); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean_json(data),allow_nan=False),encoding="utf-8")
    os.replace(temporary,path)


def emit(event,**payload):
    print(json.dumps(clean_json(dict(event=event,**payload)),allow_nan=False),flush=True)


class FieldWriter:
    """Native retained conformal polygons, including separate sheet-side fields."""
    def __init__(self,topology,bounds=None):
        points=[]; connectivity=[]; offsets=[]
        groups=[]
        from shapely.ops import triangulate
        from .topology import polygons
        from shapely.geometry import box
        from shapely import set_precision
        region=box(*bounds) if bounds is not None else None
        for fid,face in enumerate(topology.faces):
            # Clip display polygons at exact monitor coordinates, without inheriting
            # the topology's fixed precision grid on the new region boundaries.
            shapes=polygons(set_precision(face.polygon,0).intersection(region)) if region is not None else [face.polygon]
            parts=[part for shape in shapes for part in ([shape] if not shape.interiors else
                   [p for triangle in triangulate(shape) for p in polygons(triangle.intersection(shape))])
                   if part.area>topology.tol**2]
            for polygon in parts:
                coordinates=np.asarray(polygon.exterior.coords)[:-1]
                start=len(points); points.extend((float(x)*1e3,float(y)*1e3,0.) for x,y in coordinates)
                connectivity.extend(range(start,len(points))); offsets.append(len(connectivity)); groups.append(topology.face_group[fid])
        self.groups=np.asarray(groups); n=len(offsets)
        numbers=lambda data:" ".join(format(v,".12g") for v in np.asarray(data).ravel())
        self.head=f'<?xml version="1.0"?><VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"><UnstructuredGrid><Piece NumberOfPoints="{len(points)}" NumberOfCells="{n}"><CellData Scalars="field"><DataArray type="Float64" Name="field" format="ascii">'
        self.tail=('</DataArray></CellData><Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'+numbers(points)+
            '</DataArray></Points><Cells><DataArray type="Int64" Name="connectivity" format="ascii">'+numbers(connectivity)+
            '</DataArray><DataArray type="Int64" Name="offsets" format="ascii">'+numbers(offsets)+
            '</DataArray><DataArray type="UInt8" Name="types" format="ascii">'+" ".join(["7"]*n)+
            '</DataArray></Cells></Piece></UnstructuredGrid></VTKFile>')

    def write(self,path,values):
        text=" ".join(format(v,".9g") for v in np.asarray(values)[self.groups])
        path=Path(path); temporary=path.with_suffix(".tmp")
        temporary.write_text(self.head+text+self.tail,encoding="utf-8")
        os.replace(temporary,path)

    def write_complex(self,path,values):
        arrays=dict(real=values.real,imaginary=values.imag,magnitude=abs(values),phase=np.angle(values,deg=True))
        tags=''.join(f'<DataArray type="Float64" Name="{name}" format="ascii">'+
                     ' '.join(format(v,'.9g') for v in data[self.groups])+'</DataArray>'
                     for name,data in arrays.items())
        header=self.head.split('<CellData')[0]+'<CellData Scalars="magnitude">'+tags+'</CellData>'
        tail=self.tail.split('</CellData>',1)[1]
        path=Path(path); temporary=path.with_suffix('.tmp'); temporary.write_text(header+tail,encoding='utf-8')
        os.replace(temporary,path)


class Cancelled(RuntimeError): pass


def port_mode_metadata(sim):
    """Export the actual tracked, staggered modal bank used by the solver."""
    ports=[]
    for name,guide in sim.waveguides.items():
        modes=[]
        for m in range(guide.port.modes):
            beta=guide.beta[:,m]
            uniform=np.allclose(guide.q_modes[:,:,m],guide.q_modes[:,0:1,m],rtol=1e-8,atol=1e-12)
            family='TEM' if sim.polarization=='TE' and uniform and all(w.is_pec for w in guide.walls) else ('TM-like' if sim.polarization=='TE' else 'TE-like')
            modes.append(dict(index=m,family=family,beta_real_rad_m=beta.real,beta_imag_rad_m=beta.imag,
                attenuation_np_m=-beta.imag,effective_index=beta.real/(2*np.pi*sim.frequencies/299792458.),
                valid=guide.valid[:,m],overlap=guide.overlaps[:,m],
                q_real=guide.q_modes[:,:,m].real,q_imag=guide.q_modes[:,:,m].imag,
                p_real=guide.p_modes[:,:,m].real,p_imag=guide.p_modes[:,:,m].imag))
        ports.append(dict(name=name,axis=guide.port.axis,normal=guide.port.normal,inward_normal=-guide.port.normal,
            position_mm=guide.port.position*1e3,depth_mm=guide.port.invariant_length*1e3,
            frequencies_ghz=sim.frequencies/1e9,
            transverse_edges_mm=guide.transverse*1e3,
            transverse_mm=(guide.transverse[:-1]+guide.transverse[1:])*500,
            scalar_label='Hz' if sim.polarization=='TE' else 'Ez',
            scalar_unit='A/m' if sim.polarization=='TE' else 'V/m',
            tangent_label='Et' if sim.polarization=='TE' else 'Ht',
            tangent_unit='V/m' if sim.polarization=='TE' else 'A/m',modes=modes))
    return ports


def run_project(project,output,preview=False,events=emit):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/"cancel").exists(): raise Cancelled("Simulation cancelled")
    events("status",message="Compiling geometry and ranked mesh")
    sim=compile_project(project)
    if (output/"cancel").exists(): raise Cancelled("Simulation cancelled")
    writer=FieldWriter(sim.topology)
    monitors=[]; names=set()
    for spec in project.get('monitors',[]):
        name=spec['name']
        if name in names: raise ValueError('Field monitor names must be unique')
        names.add(name)
        bounds=(finite(spec['x'],'monitor x'),finite(spec['y'],'monitor y'),
                finite(spec['x'],'monitor x')+finite(spec['width'],'monitor width'),
                finite(spec['y'],'monitor y')+finite(spec['height'],'monitor height'))
        frequencies=spec.get('frequencies_ghz',[])
        if not frequencies: raise ValueError('A field monitor needs at least one frequency')
        definition=FieldMonitor(name,tuple(v*1e-3 for v in bounds),tuple(finite(f,'monitor frequency')*1e9 for f in frequencies))
        definition.compile(sim.topology,sim.dt)  # Fail before starting any study.
        monitors.append(definition)
    epsilon=np.array([sum(sim.topology.faces[f].epsilon*sim.topology.faces[f].polygon.area for f in group)/sim.topology.areas[k]
                      if group else 1. for k,group in enumerate(sim.topology.groups)])
    mesh_file=output/"mesh.vtu"; writer.write(mesh_file,epsilon)
    report=sim.scene.generated
    scale=lambda b:[v*1e3 for v in b] if b is not None else None
    metadata=dict(version=1,mesh_x_mm=sim.mesh.x*1e3,mesh_y_mm=sim.mesh.y*1e3,
        dt_ps=sim.dt*1e12,shape=sim.mesh.shape,scalar_dofs=len(sim.scalar),vector_dofs=len(sim.vector),
        domain_mm=scale(report.domain_bounds),ntff_mm=scale(report.ntff_bounds),tfsf_mm=scale(report.tfsf_bounds),
        physical_mm=scale(report.physical_bounds),pml_width_mm=report.pml_width*1e3,
        rejected_anchors=[dict(owner=a.owner,axis=a.axis,position_mm=a.position*1e3,winner=w.owner) for a,w in sim.mesh.report.rejected],
        limited_resolution=sim.mesh.report.limited_resolution,split_dofs=sim.topology.report.extra_scalar_dofs,
        enlarged_groups=len(sim.topology.report.enlarged_groups),fallbacks=sim.topology.report.staircase_fallbacks,
        field_file=str(mesh_file),channels=sim.channels,port_modes=port_mode_metadata(sim),
        monitors=[dict(name=m.name,bounds_mm=np.asarray(m.bounds)*1e3,frequencies_ghz=np.asarray(m.frequencies)/1e9) for m in monitors])
    write_json(output/"mesh.json",metadata)
    events("compiled",**metadata)
    if preview:
        events("complete",preview=True,path=str(output/"mesh.json")); return metadata
    settings=project.get("settings",{})
    pulse=pulse_from_settings(settings)
    events("status",message=f"Pulse: {pulse.frequency/1e9:g} GHz, width {pulse.width*1e12:g} ps, delay {pulse.delay*1e9:g} ns ({settings.get('pulse_mode','manual')})")
    tolerance=lambda name:finite(settings[name],name) if settings.get(name) else None
    control=RunControl(finite(settings.get("max_time_ns",1.5),"max time")*1e-9,
        min_time=finite(settings.get("min_time_ns",0),"min time")*1e-9,
        field_tolerance=tolerance("field_tolerance"),dft_tolerance=tolerance("dft_tolerance"),check_steps=50)
    snapshot_interval=finite(settings.get("time_snapshot_interval_ns",0),"time snapshot interval")*1e-9
    if snapshot_interval<0: raise ValueError("Time snapshot interval cannot be negative")
    last_frame=-np.inf; peaks={}; peak_paths={}; peak_steps={}; snapshots={}; last_saved={}
    def progress(info,q,p):
        nonlocal last_frame
        if (output/"cancel").exists(): raise Cancelled("Simulation cancelled")
        index=info.get("run_index",0)
        if info["energy"]>peaks.get(index,-1):
            peaks[index]=info["energy"]; peak_paths[index]=q.copy(); peak_steps[index]=info["step"]
        if snapshot_interval>0 and info['time']-last_saved.get(index,-np.inf)>=snapshot_interval:
            path=output/f"field_time_{index}_{info['step']}.vtu"; writer.write(path,q)
            snapshots.setdefault(index,[]).append(dict(field_file=str(path),step=info['step'],time_ns=info['time']*1e9))
            last_saved[index]=info['time']
        now=time.monotonic()
        if now-last_frame>=.25 or info["step"]==info["steps"]:
            field_file=output/f"field_live_{index}_{info['step']}.vtu"
            writer.write(field_file,q)
            events("progress",**info,fraction=(index+info["step"]/info["steps"])/info.get("run_count",1),
                   field_file=str(field_file),scalar_label="Hz (A/m)" if sim.polarization=="TE" else "Ez (V/m)")
            # Keep a short live history; persistent peak/final snapshots are saved below.
            old=sorted(output.glob(f"field_live_{index}_*.vtu"),key=lambda path:path.stat().st_mtime)
            for path in old[:-3]:
                try: path.unlink()
                except OSError: pass
            last_frame=now
    mode=settings.get("study","sparameters")
    if mode=="sparameters":
        if not sim.channels: raise ValueError("An S-parameter study needs at least one port")
        study=sim.scattering(control,pulse,progress=progress,field_monitors=monitors); runs=study.runs
        study.save(output/"sparameters.npz")
        scattering=dict(channels=study.channels,s_real=study.s.real,s_imag=study.s.imag,valid=study.valid)
    elif mode=="excitation":
        drives={}
        for item in project.get("excitations",[]):
            channel=(item["name"],int(item.get("mode",0)))
            if channel in drives: raise ValueError("Duplicate excitation channel")
            drives[channel]=GaussianPulse(pulse.frequency,pulse.width,pulse.delay,finite(item.get("amplitude",1),"amplitude"))
        if not drives: raise ValueError("Choose at least one transmitting port or plane wave")
        runs=[sim.run(control,drives,progress=progress,field_monitors=monitors)]
        scattering=dict(channels=sim.channels,s_real=None,s_imag=None,valid=None)
    else:
        raise ValueError("Unknown study type")
    angles=np.linspace(0,2*np.pi,361); records=[]
    for index,run in enumerate(runs):
        monitor_records=[]
        for mi,monitor in enumerate(run.field_monitors):
            mwriter=FieldWriter(sim.topology,bounds=monitor.definition.bounds); files=[]
            for fi,spectrum in enumerate(monitor.spectra):
                values=np.zeros(len(sim.scalar),complex); values[monitor.indices]=spectrum
                monitor_file=output/f'monitor_{index}_{mi}_{fi}.vtu'; mwriter.write_complex(monitor_file,values); files.append(str(monitor_file))
            archive=output/f'monitor_{index}_{mi}.npz'
            np.savez_compressed(archive,frequencies=monitor.frequencies,indices=monitor.indices,spectra=monitor.spectra,
                                bounds=monitor.definition.bounds,centers=sim.topology.centers[monitor.indices])
            monitor_records.append(dict(name=monitor.definition.name,frequencies_ghz=monitor.frequencies/1e9,
                bounds_mm=np.asarray(monitor.definition.bounds)*1e3,field_files=files,archive=str(archive),
                field_label='Hz' if sim.polarization=='TE' else 'Ez',unit='A s/m' if sim.polarization=='TE' else 'V s/m'))
        final_file=output/f"field_final_{index}.vtu"; peak_file=output/f"field_peak_{index}.vtu"
        writer.write(final_file,run.scalar); writer.write(peak_file,peak_paths.get(index,run.scalar))
        frames=snapshots.get(index,[]).copy()
        if not frames:
            for path in output.glob(f"field_live_{index}_*.vtu"):
                step=int(path.stem.rsplit('_',1)[1]); frames.append(dict(field_file=str(path),step=step,time_ns=step*run.dt*1e9))
        peak_step=peak_steps.get(index,run.steps)
        frames.extend([dict(field_file=str(peak_file),step=peak_step,time_ns=peak_step*run.dt*1e9,label='peak energy'),
                       dict(field_file=str(final_file),step=run.steps,time_ns=run.time*1e9,label='final')])
        # One actual field per time instant; prefer the persistent peak/final copy.
        frames=sorted({frame['step']:frame for frame in frames}.values(),key=lambda frame:frame['step'])
        far=run.far_field(angles)
        reference=None
        if len(run.driven_channels)==1 and run.driven_channels[0] in run.channels:
            j=run.channels.index(run.driven_channels[0]); reference=run.incoming[:,j]
        normalized=np.full_like(far.scalar_amplitude,np.nan+0j)
        if reference is not None:
            good=abs(reference)>max(abs(reference).max()*1e-6,1e-300)
            np.divide(far.scalar_amplitude,reference[:,None],out=normalized,where=good[:,None])
        coefficient=.5*(np.sqrt(sim.ntff.mu/sim.ntff.epsilon) if sim.polarization=="TE" else np.sqrt(sim.ntff.epsilon/sim.ntff.mu))*run.invariant_length
        record=dict(driven_channels=run.driven_channels,steps=run.steps,time_ns=run.time*1e9,stop_reason=run.stop_reason,
            incoming_real=run.incoming.real,incoming_imag=run.incoming.imag,outgoing_real=run.outgoing.real,outgoing_imag=run.outgoing.imag,
            far_real=far.scalar_amplitude.real,far_imag=far.scalar_amplitude.imag,power=far.power_per_radian,
            normalized_far_real=normalized.real,normalized_far_imag=normalized.imag,normalized_power=coefficient*abs(normalized)**2,
            peak_field=str(peak_file),peak_step=peak_step,final_field=str(final_file),time_snapshots=frames,diagnostics=run.diagnostics,monitors=monitor_records)
        records.append(record)
    result=dict(version=1,project=project,mesh=metadata,frequencies_ghz=sim.frequencies/1e9,angles_deg=angles*180/np.pi,
                scalar_label="Hz (A/m)" if sim.polarization=="TE" else "Ez (V/m)",runs=records,**scattering)
    write_json(output/"results.json",result)
    np.savez_compressed(output/"fields_and_far_field.npz",frequencies=sim.frequencies,angles=angles,
                        **{f"scalar_{i}":r.scalar for i,r in enumerate(runs)},
                        **{f"vector_{i}":r.vector for i,r in enumerate(runs)},
                        **{f"far_{i}":records[i]["far_real"]+1j*records[i]["far_imag"] for i in range(len(runs))})
    export_csv(output,result)
    events("complete",preview=False,path=str(output/"results.json")); return result


def export_csv(output,result):
    if result["s_real"] is not None:
        with (output/"sparameters.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream); writer.writerow(["frequency_GHz","outgoing","incoming","real","imag","magnitude_dB","phase_deg","valid"])
            labels=[f"{name}:{mode}" for name,mode in result["channels"]]
            for f,frequency in enumerate(result["frequencies_ghz"]):
                for i,out in enumerate(labels):
                    for j,inc in enumerate(labels):
                        z=complex(result["s_real"][f,i,j],result["s_imag"][f,i,j])
                        writer.writerow([frequency,out,inc,z.real,z.imag,20*np.log10(max(abs(z),1e-15)),np.angle(z,deg=True),result["valid"][f]])
    with (output/"far_field.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream); writer.writerow(["run","frequency_GHz","angle_deg","real_raw_DFT","imag_raw_DFT","power_raw_DFT","power_per_incident_watt"])
        for i,run in enumerate(result["runs"]):
            for f,frequency in enumerate(result["frequencies_ghz"]):
                for a,angle in enumerate(result["angles_deg"]):
                    writer.writerow([i,frequency,angle,run["far_real"][f,a],run["far_imag"][f,a],run["power"][f,a],run["normalized_power"][f,a]])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--project",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    try:
        project=json.loads(Path(args.project).read_text(encoding="utf-8-sig"))
        run_project(project,args.output,args.preview)
    except Cancelled as exc:
        emit("cancelled",message=str(exc)); return 2
    except Exception as exc:
        emit("error",message=str(exc)); traceback.print_exc(file=sys.stderr); return 1
    return 0


if __name__=="__main__": sys.exit(main())
