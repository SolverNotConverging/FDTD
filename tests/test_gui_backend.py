"""Process bridge validation, independent of Qt/VTK installations."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from FDTD_2D import Scene,Mesh,FDTD2D,RunControl,Material
from FDTD_2D.gui_backend import compile_project,run_project,FieldWriter,clean_json,port_mode_metadata,pulse_from_settings
from FDTD_2D import FieldMonitor


ROOT=Path(__file__).resolve().parents[1]


class GuiBridgeTests(unittest.TestCase):
    def test_auto_pulse_fits_frequency_band_and_keeps_legacy_manual(self):
        pulse=pulse_from_settings(dict(pulse_mode='auto',f_min_ghz=20,f_max_ghz=40))
        self.assertAlmostEqual(pulse.frequency,30e9)
        self.assertAlmostEqual(pulse.delay,6*pulse.width)
        edge=np.exp(-(np.pi*pulse.width*10e9)**2); self.assertLess(edge,1e-5); self.assertGreater(edge,1e-6)
        legacy=pulse_from_settings(dict(pulse_ghz=15,pulse_width_ps=60)); self.assertEqual(legacy.frequency,15e9); self.assertAlmostEqual(legacy.width,60e-12)
        with self.assertRaisesRegex(ValueError,'fmin'): pulse_from_settings(dict(pulse_mode='auto',f_min_ghz=20,f_max_ghz=20))

    def test_inward_normal_and_tem_mode(self):
        project=json.loads((ROOT/'gui/examples/waveguide.fdtd.json').read_text()); project['settings']['frequency_count']=3
        sim=compile_project(project); guide=sim.waveguides['left']
        self.assertEqual(guide.port.normal,-1); self.assertTrue(np.all(guide.valid[:,0]))
        q=guide.q_modes[:,:,0]
        np.testing.assert_allclose(q/q[:,0,None],1,rtol=1e-10,atol=1e-10)
        # Uniform transverse Hz with beta near the background wavenumber is TEM.
        np.testing.assert_allclose(guide.beta[:,0].real,2*np.pi*sim.frequencies/299792458,rtol=.035)
        self.assertEqual(port_mode_metadata(sim)[0]['modes'][0]['family'],'TEM')

    def test_saved_time_snapshots_have_sorted_real_steps(self):
        project=json.loads((ROOT/'gui/examples/cylinder.fdtd.json').read_text())
        project['settings'].update(frequency_count=3,max_time_ns=.15,time_snapshot_interval_ns=.015,field_tolerance=0,dft_tolerance=0)
        with TemporaryDirectory() as directory:
            result=run_project(project,directory,events=lambda *args,**kwargs:None)
            frames=result['runs'][0]['time_snapshots']; steps=[f['step'] for f in frames]
            self.assertGreater(len(frames),2); self.assertEqual(steps,sorted(set(steps)))
            self.assertTrue(all(Path(f['field_file']).exists() for f in frames)); self.assertEqual(steps[-1],result['runs'][0]['steps'])

    def test_mm_project_converts_geometry_and_generates_closed_boxes(self):
        project=json.loads((ROOT/'gui/examples/cylinder.fdtd.json').read_text())
        project['settings']['frequency_count']=3
        sim=compile_project(project)
        self.assertAlmostEqual(sim.scene.geometry[0].bounds[2],.003)
        self.assertIsNotNone(sim.tfsf)
        self.assertEqual(sim.ntff.contour.vertices[0],sim.ntff.contour.vertices[-1])
        self.assertEqual(sim.scene.plane_waves[0].name,'illumination')

    def test_progress_callback_is_read_only_and_does_not_change_fields(self):
        mesh=Mesh(np.linspace(0,.006,7),np.linspace(0,.006,7))
        sim=FDTD2D(Scene((0,.006),(0,.006)),mesh=mesh,polarization='TM')
        q=np.random.default_rng(11).normal(size=len(sim.scalar))
        control=RunControl(3e-11,check_steps=5)
        reference=sim.run(control,initial_scalar=q)
        events=[]
        def progress(info,scalar,vector):
            self.assertFalse(scalar.flags.writeable); self.assertFalse(vector.flags.writeable)
            events.append(info)
        observed=sim.run(control,initial_scalar=q,progress=progress)
        np.testing.assert_array_equal(reference.scalar,observed.scalar)
        np.testing.assert_array_equal(reference.vector,observed.vector)
        self.assertGreater(len(events),0)
        self.assertEqual(events[-1]['step'],observed.steps)

    def test_field_file_retains_conformal_side_dofs_and_holes(self):
        scene=Scene((0,.006),(0,.006))
        scene.circle('small pec',(.0035,.0035),.0002,'PEC')
        scene.sheet('film',(.001,.0014),(.005,.0044),'PEC')
        mesh=Mesh(np.linspace(0,.006,7),np.linspace(0,.006,7))
        sim=FDTD2D(scene,mesh=mesh,polarization='TM',enlargement=0)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'field.vtu'; writer=FieldWriter(sim.topology)
            values=np.arange(len(sim.scalar),dtype=float); writer.write(path,values)
            xml=ET.parse(path); piece=xml.find('.//Piece')
            samples=np.fromstring(piece.find('./CellData/DataArray').text,sep=' ')
            self.assertEqual(len(samples),int(piece.attrib['NumberOfCells']))
            np.testing.assert_array_equal(samples,values[writer.groups])
            self.assertGreaterEqual(len(samples),len(sim.topology.faces))

    def test_result_json_and_csv_use_null_for_invalid_bins(self):
        self.assertEqual(clean_json(np.array([np.nan,np.inf,1.])),[None,None,1.])
        project=json.loads((ROOT/'gui/examples/cylinder.fdtd.json').read_text())
        project['objects'][0]['radius']=1
        project['settings'].update(frequency_count=3,max_time_ns=.03,field_tolerance=0,dft_tolerance=0)
        seen=[]
        with TemporaryDirectory() as directory:
            result=run_project(project,directory,events=lambda event,**payload:seen.append((event,payload)))
            path=Path(directory)
            loaded=json.loads((path/'results.json').read_text())
            self.assertEqual(loaded['version'],1)
            self.assertIsNone(loaded['s_real'])
            self.assertEqual(loaded['runs'][0]['normalized_power'][0][0],None)
            self.assertTrue((path/'far_field.csv').is_file())
            self.assertTrue((path/'fields_and_far_field.npz').is_file())
            self.assertEqual(seen[0][0],'status'); self.assertEqual(seen[-1][0],'complete')
            self.assertTrue(any(event=='compiled' for event,_ in seen))
            self.assertLessEqual(result['runs'][0]['time_ns'],.03)

    def test_invalid_schema_and_unknown_geometry_are_rejected(self):
        with self.assertRaises(ValueError): compile_project(dict(version=999,units='mm'))
        project=json.loads((ROOT/'gui/examples/cylinder.fdtd.json').read_text())
        project['objects'][0]['kind']='python_code'
        with self.assertRaises(ValueError): compile_project(project)

    def test_regional_dft_matches_every_step_and_is_run_independent(self):
        mesh=Mesh(np.linspace(0,.006,7),np.linspace(0,.006,7))
        sim=FDTD2D(Scene((0,.006),(0,.006)),mesh=mesh,polarization='TM')
        initial=np.random.default_rng(7).normal(size=len(sim.scalar))
        monitor=FieldMonitor('region',(.0012,.0012,.0042,.0042),(11e9,17e9))
        compiled=monitor.compile(sim.topology,sim.dt); expected=np.zeros_like(compiled.spectra)
        def sample(info,q,p):
            expected[:]+=np.exp(-2j*np.pi*np.array(monitor.frequencies)*info['time'])[:,None]*q[compiled.indices]*info['dt']
        control=RunControl(3e-11,check_steps=1)
        observed=sim.run(control,initial_scalar=initial,progress=sample,field_monitors=(monitor,))
        np.testing.assert_allclose(observed.field_monitors[0].spectra,expected,rtol=1e-14,atol=1e-25)
        repeated=sim.run(control,initial_scalar=initial,field_monitors=(monitor,))
        plain=sim.run(control,initial_scalar=initial)
        np.testing.assert_array_equal(repeated.field_monitors[0].spectra,expected)
        np.testing.assert_array_equal(observed.scalar,plain.scalar)
        self.assertLess(len(compiled.indices),len(sim.scalar))
        with self.assertRaisesRegex(ValueError,'outside'):
            FieldMonitor('outside',(-.01,0,.004,.004),(11e9,)).compile(sim.topology,sim.dt)
        with self.assertRaisesRegex(ValueError,'increasing'):
            FieldMonitor('bad',monitor.bounds,(17e9,11e9)).compile(sim.topology,sim.dt)

    def test_monitor_files_are_clipped_and_keep_complex_dft(self):
        project=json.loads((ROOT/'gui/examples/cylinder.fdtd.json').read_text())
        project['settings'].update(frequency_count=3,max_time_ns=.04,field_tolerance=0,dft_tolerance=0)
        project['monitors']=[dict(name='small',kind='monitor',x=-4,y=-4,width=8,height=8,frequencies_ghz=[13,17])]
        with TemporaryDirectory() as directory:
            result=run_project(project,directory,events=lambda *args,**kwargs:None)
            record=result['runs'][0]['monitors'][0]
            self.assertEqual(record['name'],'small'); self.assertEqual(len(record['field_files']),2)
            with np.load(record['archive']) as archive:
                self.assertGreater(np.linalg.norm(archive['spectra']),0)
            xml=ET.parse(record['field_files'][0]); arrays={a.attrib['Name']:np.fromstring(a.text,sep=' ') for a in xml.findall('.//CellData/DataArray')}
            np.testing.assert_allclose(arrays['magnitude'],np.hypot(arrays['real'],arrays['imaginary']),rtol=2e-8,atol=1e-30)
            np.testing.assert_allclose(arrays['phase'],np.angle(arrays['real']+1j*arrays['imaginary'],deg=True),atol=2e-6)
            points=np.fromstring(xml.find('.//Points/DataArray').text,sep=' ').reshape(-1,3)
            self.assertTrue(np.all(abs(points[:,:2])<=4.000000001))
            self.assertEqual(result['mesh']['monitors'][0]['name'],'small')

    def test_exported_tracked_modes_match_solver_bank(self):
        project=json.loads((ROOT/'gui/examples/waveguide.fdtd.json').read_text())
        project['settings']['frequency_count']=3
        sim=compile_project(project); ports=port_mode_metadata(sim)
        self.assertEqual(len(ports),2)
        bank=ports[0]['modes'][0]; guide=sim.waveguides[ports[0]['name']]
        np.testing.assert_array_equal(bank['q_real']+1j*bank['q_imag'],guide.q_modes[:,:,0])
        np.testing.assert_array_equal(bank['beta_real_rad_m']+1j*bank['beta_imag_rad_m'],guide.beta[:,0])
        self.assertTrue(np.all(bank['valid'])); self.assertGreater(np.min(bank['overlap']),.99)
        self.assertGreater(bank['beta_real_rad_m'][-1],bank['beta_real_rad_m'][0])


if __name__=='__main__': unittest.main()
