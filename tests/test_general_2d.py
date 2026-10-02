"""Physical and structural checks for the geometry-first conformal reference."""
import unittest
import numpy as np
from scipy.sparse.csgraph import connected_components
from FDTD_2D import (Scene, Material, Mesh, MeshPolicy, Anchor, FDTD2D, RunControl,
                     PEC, PMC, SurfaceImpedance, ThinSheet, LumpedPort, WaveguidePort,
                     GaussianPulse, PML)
from FDTD_2D import DomainPolicy,ClosedContour,PlaneWave,VirtualWaveguide
from FDTD_2D.topology import Topology
from FDTD_2D.ports import track_modes


def uniform(n=12):
    return Mesh(np.linspace(0,.012,n+1),np.linspace(0,.012,n+1))


class GeometryMeshTests(unittest.TestCase):
    def test_isolated_sliver_falls_back_locally_with_report(self):
        from shapely.geometry import Polygon
        scene=Scene((0,.012),(0,.012))
        ring=Polygon([(.003,.003),(.009,.003),(.009,.009),(.003,.009)],
                     holes=[[ (.0061,.0061),(.0069,.0061),(.0069,.0069),(.0061,.0069) ]])
        scene.add('ring',ring,PEC)
        # A smaller cavity cannot enlarge into a disconnected exterior region.
        ring=Polygon(ring.exterior.coords,holes=[[ (.0064,.0064),(.0066,.0064),(.0066,.0066),(.0064,.0066) ]])
        scene.geometry.clear(); scene.add('ring',ring,PEC)
        topology=Topology(scene,uniform())
        self.assertTrue(topology.report.staircase_fallbacks)
        self.assertEqual(topology._staircase_cells,{(6,6)})
        with self.assertRaises(ValueError):
            Topology(scene,uniform(),fallback=False)

    def test_geometry_precedes_mesh_and_rank_only_changes_requests(self):
        scene=Scene((0,.012),(0,.012))
        first=scene.rectangle('high',(.003,.006),(.003,.008),Material('n4',epsilon_r=16),rank=20)
        scene.rectangle('low',(.00305,.008),(.003,.008),Material('n2',epsilon_r=4),rank=1)
        mesh=MeshPolicy(.001,10e9,min_step=.0002,growth=2).build(scene)
        self.assertTrue(np.any(mesh.x==first.bounds[0]))
        rejected=[a for a,w in mesh.report.rejected]
        self.assertTrue(any(a.owner=='low' and a.axis=='x' for a in rejected))
        self.assertEqual(scene.material_at((.004,.004)).name,'n2')
        self.assertGreaterEqual(min(mesh.dx.min(),mesh.dy.min()),.0002*(1-1e-8))
        self.assertLessEqual(max((mesh.dx[1:]/mesh.dx[:-1]).max(),(mesh.dx[:-1]/mesh.dx[1:]).max()),2+1e-7)

    def test_real_index_includes_permeability(self):
        scene=Scene((0,.012),(0,.012),background=Material('n6',epsilon_r=4,mu_r=9))
        policy=MeshPolicy(.003,10e9,growth=2)
        mesh=policy.build(scene)
        expected=299792458/(10e9*6*20)
        self.assertLessEqual(mesh.dx.max(),expected*(1+1e-7))

    def test_cfl_budget_rejects_close_lower_rank_port_anchor(self):
        scene=Scene((0,.012),(0,.012))
        policy=MeshPolicy(.001,10e9,min_dt=3e-13,growth=2)
        requests=[Anchor('x',.005,10,'important','port'),Anchor('x',.00501,1,'other','tip')]
        mesh=policy.build(scene,requests)
        self.assertTrue(any(a.owner=='other' and winner.owner=='important' for a,winner in mesh.report.rejected))
        self.assertGreaterEqual(mesh.cfl(policy.courant),policy.min_dt*(1-1e-8))

    def test_sheet_has_no_area_and_retains_split_dofs(self):
        scene=Scene((0,.012),(0,.012))
        scene.sheet('diagonal',(0,.0013),(.012,.0107),PEC)
        topology=Topology(scene,uniform(),enlargement=.2)
        self.assertAlmostEqual(topology.areas.sum(),.012**2,places=12)
        self.assertGreater(topology.report.extra_scalar_dofs,0)
        adjacency=topology.incidence@topology.incidence.T
        self.assertEqual(connected_components(adjacency,directed=False)[0],2)
        self.assertTrue(topology.report.enlarged_groups)


class BoundaryAndRuntimeTests(unittest.TestCase):
    def test_curved_pec_cavity_frequency_converges_to_analytic_root(self):
        from scipy.sparse import block_diag,csc_matrix,diags
        from scipy.sparse.linalg import eigsh
        from shapely.geometry import Point
        scene=Scene((0,.012),(0,.012))
        scene.add('circular cavity',scene.domain.difference(Point(.006,.006).buffer(.004,quad_segs=64)),PEC)
        expected=299792458*2.4048255577/(2*np.pi*.004)
        errors=[]
        for n in (12,24,48):
            topology=Topology(scene,uniform(n),polarization='TM')
            inv=block_diag([csc_matrix(np.linalg.inv(b.mass)) for b in topology.blocks])
            scale=diags(1/np.sqrt(topology.scalar_mass))
            operator=scale@topology.incidence@inv@topology.incidence.T@scale
            frequency=np.sqrt(eigsh(operator,k=1,sigma=0,return_eigenvectors=False)[0])/(2*np.pi)
            errors.append(abs(frequency-expected)/expected)
        self.assertGreater(errors[0],errors[1])
        self.assertGreater(errors[1],errors[2])
        self.assertLess(errors[-1],.002)

    def test_nonuniform_conformal_energy_conservation(self):
        scene=Scene((0,.012),(0,.012))
        scene.circle('obstacle',(.006,.006),.0017,PEC)
        mesh=Mesh(.012*np.linspace(0,1,17)**1.3,.012*np.linspace(0,1,15)**1.2)
        for polarization in ('TE','TM'):
            sim=FDTD2D(scene,mesh=mesh,polarization=polarization)
            t=sim.topology
            q=np.random.default_rng(43).normal(size=len(t.groups))
            initial=.5*q@(t.scalar_mass*q)
            result=sim.run(RunControl(3e-10),initial_scalar=q)
            q,p=result.scalar,result.vector
            invariant=.5*(q@(t.scalar_mass*q)+p@(t.vector_mass@p)+sim.dt*q@(t.incidence@p))
            self.assertAlmostEqual(invariant/initial,1,places=10)

    def test_pml_absorbs_and_field_convergence_stops_early(self):
        mesh=Mesh(np.linspace(0,.032,33),np.linspace(0,.032,33))
        sim=FDTD2D(Scene((0,.032),(0,.032)),mesh=mesh,polarization='TM',pml=PML(.008))
        q=np.exp(-np.sum((sim.topology.centers-.016)**2,axis=1)/(2*.001**2))
        result=sim.run(RunControl(4e-9,field_tolerance=1e-3,check_steps=50),initial_scalar=q)
        self.assertEqual(result.stop_reason,'converged')
        self.assertLess(result.time,4e-9)
        self.assertLess(result.diagnostics[-1]['field_ratio'],1e-3)

    def test_dft_convergence_and_source_end_guard(self):
        mesh=Mesh(np.linspace(0,.032,33),np.linspace(0,.032,33))
        scene=Scene((0,.032),(0,.032))
        scene.add_port(LumpedPort('feed',(.016,.016)))
        sim=FDTD2D(scene,mesh=mesh,polarization='TM',pml=PML(.008),frequencies=[15e9,20e9,25e9])
        pulse=GaussianPulse(20e9,3e-11,delay=3e-10)
        result=sim.run(RunControl(4e-9,dft_tolerance=1e-3,check_steps=50),{'feed':pulse})
        self.assertEqual(result.stop_reason,'converged')
        self.assertGreater(result.time,pulse.end_time)

    def test_lossless_modified_energy_is_conserved_both_polarizations(self):
        for polarization in ('TE','TM'):
            scene=Scene((0,.012),(0,.012))
            scene.sheet('diagonal',(0,.0013),(.012,.0107),PEC)
            sim=FDTD2D(scene,mesh=uniform(),polarization=polarization,enlargement=.2)
            t=sim.topology
            q=np.random.default_rng(41).normal(size=len(t.groups))
            p=np.zeros(t.incidence.shape[1])
            initial=.5*q@(t.scalar_mass*q)
            result=sim.run(RunControl(1e-9),initial_scalar=q,initial_vector=p)
            q,p=result.scalar,result.vector
            invariant=.5*(q@(t.scalar_mass*q)+p@(t.vector_mass@p)+sim.dt*q@(t.incidence@p))
            self.assertAlmostEqual(invariant/initial,1,places=10)

    def test_pec_sheet_has_no_cross_sheet_coupling(self):
        scene=Scene((0,.012),(0,.012))
        scene.sheet('diagonal',(0,.0013),(.012,.0107),PEC)
        for polarization in ('TE','TM'):
            sim=FDTD2D(scene,mesh=uniform(),polarization=polarization)
            _,labels=connected_components(sim.topology.incidence@sim.topology.incidence.T,directed=False)
            q=(labels==labels[0]).astype(float)
            result=sim.run(RunControl(3e-10),initial_scalar=q)
            np.testing.assert_array_equal(result.scalar[labels!=labels[0]],0)

    def test_pmc_is_exact_surface_infinite_impedance(self):
        scene=Scene((0,.012),(0,.012),boundaries={s:PMC for s in ('xmin','xmax','ymin','ymax')})
        tm=FDTD2D(scene,mesh=uniform(),polarization='TM')
        q=np.ones(len(tm.topology.groups))
        result=tm.run(RunControl(1e-10),initial_scalar=q)
        np.testing.assert_allclose(result.scalar,q,atol=1e-12)
        te=FDTD2D(scene,mesh=uniform(),polarization='TE')
        result=te.run(RunControl(1e-10),initial_scalar=q)
        self.assertGreater(np.linalg.norm(result.vector),0)

    def test_lossy_sibc_and_memory_remain_passive(self):
        for polarization in ('TE','TM'):
            wall=SurfaceImpedance(20,((30,2e10),(50,1e11)))
            scene=Scene((0,.012),(0,.012),boundaries={s:wall for s in ('xmin','xmax','ymin','ymax')})
            sim=FDTD2D(scene,mesh=uniform(),polarization=polarization)
            q=np.random.default_rng(2).normal(size=len(sim.topology.groups))
            result=sim.run(RunControl(2e-9,field_tolerance=1e-3,check_steps=20),initial_scalar=q)
            self.assertLess(np.linalg.norm(result.scalar),np.linalg.norm(q)*.1)
            self.assertTrue(np.all(np.isfinite(result.vector)))

    def test_thin_transmissive_sheet_couples_two_retained_sides(self):
        scene=Scene((0,.012),(0,.012))
        scene.sheet('film',(0,.0063),(.012,.0063),ThinSheet.resistive(50))
        for polarization in ('TE','TM'):
            sim=FDTD2D(scene,mesh=uniform(),polarization=polarization)
            q=(sim.topology.centers[:,1]<.0063).astype(float)
            result=sim.run(RunControl(2e-10),initial_scalar=q)
            self.assertGreater(np.linalg.norm(result.scalar[sim.topology.centers[:,1]>.0063]),0)

    def test_max_time_and_source_guard(self):
        scene=Scene((0,.012),(0,.012))
        scene.add_port(LumpedPort('feed',(.006,.006)))
        sim=FDTD2D(scene,mesh=uniform(),polarization='TM',frequencies=[10e9])
        pulse=GaussianPulse(10e9,2e-11,delay=3e-10)
        result=sim.run(RunControl(2e-10,field_tolerance=.1,dft_tolerance=.1,check_steps=5),{'feed':pulse})
        self.assertEqual(result.stop_reason,'max_time')
        self.assertLessEqual(result.time,2e-10)

    def test_zero_fields_do_not_satisfy_convergence(self):
        sim=FDTD2D(Scene((0,.012),(0,.012)),mesh=uniform())
        result=sim.run(RunControl(1e-10,field_tolerance=.1,check_steps=5,consecutive=1))
        self.assertEqual(result.stop_reason,'max_time')


class PortTests(unittest.TestCase):
    def test_thin_pec_walls_define_a_matched_virtual_aperture(self):
        scene=Scene((0,.024),(-.006,.018))
        scene.sheet('bottom',(0,0),(.024,0),PEC,rank=40)
        scene.sheet('top',(0,.012),(.024,.012),PEC,rank=40)
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        scene.add_port(WaveguidePort('b','x',.018,(0,.012),normal=1))
        mesh=Mesh(np.linspace(0,.024,25),np.linspace(-.006,.018,25))
        sim=FDTD2D(scene,mesh=mesh,polarization='TE',frequencies=np.linspace(20e9,40e9,21))
        result=sim.run(RunControl(2e-9),{'a':GaussianPulse(30e9,1.5e-10)})
        s,_=result.s_column(('a',0)); k=10
        self.assertEqual(sim.waveguides['a'].walls,(PEC,PEC))
        self.assertLess(abs(s[k,0]),3e-4)
        self.assertLess(abs(s[k,1]-np.exp(-1j*sim.waveguides['a'].beta[k,0]*.012)),3e-4)

    def test_virtual_guide_absorbs_receiver_without_calibration(self):
        scene=Scene((0,.024),(0,.012))  # PEC ends behind disconnected apertures
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        scene.add_port(WaveguidePort('b','x',.018,(0,.012),normal=1))
        mesh=Mesh(np.linspace(0,.024,25),np.linspace(0,.012,13))
        sim=FDTD2D(scene,mesh=mesh,polarization='TM',frequencies=np.linspace(20e9,40e9,21))
        result=sim.run(RunControl(2e-9),{'a':GaussianPulse(30e9,1.5e-10)})
        s,valid=result.s_column(('a',0))
        k=10
        self.assertLess(abs(result.incoming[k,1]/result.incoming[k,0]),3e-4)
        self.assertLess(abs(s[k,0]),3e-4)
        self.assertLess(abs(s[k,1]-np.exp(-1j*sim.waveguides['a'].beta[k,0]*.012)),3e-4)
        self.assertGreater(len(sim.scalar),len(sim.virtual.host.groups))
        # Virtual guides are excluded from the physical display grid.
        self.assertEqual(sim.topology.scalar_grid(result.scalar).shape,mesh.shape)

    def test_te_mixed_ports_use_electric_time_reversal_convention(self):
        scene=Scene((0,.024),(0,.012),boundaries={'xmin':SurfaceImpedance(376.73),'xmax':SurfaceImpedance(376.73)})
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        scene.add_port(WaveguidePort('b','x',.018,(0,.012),normal=1))
        scene.add_port(LumpedPort('feed',(.012,.004),(.012,.008)))
        mesh=Mesh(np.linspace(0,.024,25),np.linspace(0,.012,13))
        sim=FDTD2D(scene,mesh=mesh,polarization='TE',frequencies=np.linspace(20e9,40e9,21))
        result=sim.scattering(RunControl(2e-9),GaussianPulse(30e9,1.5e-10))
        np.testing.assert_allclose(result.s[10],result.s[10].T,atol=2e-6)

    def test_mixed_lumped_modal_scattering_is_reciprocal(self):
        scene=Scene((0,.024),(0,.012),boundaries={'xmin':SurfaceImpedance(415),'xmax':SurfaceImpedance(415)})
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        scene.add_port(WaveguidePort('b','x',.018,(0,.012),normal=1))
        scene.add_port(LumpedPort('feed',(.012,.006)))
        mesh=Mesh(np.linspace(0,.024,25),np.linspace(0,.012,13))
        sim=FDTD2D(scene,mesh=mesh,polarization='TM',frequencies=np.linspace(20e9,40e9,21))
        result=sim.scattering(RunControl(2e-9),GaussianPulse(30e9,1.5e-10))
        np.testing.assert_allclose(result.s[10],result.s[10].T,atol=2e-6)
        # Simultaneous source/receiver operation retains waves but cannot yield
        # an identifiable full S matrix from one dependent drive state.
        simultaneous=sim.run(RunControl(2e-9),{'a':GaussianPulse(30e9,1.5e-10),'feed':GaussianPulse(30e9,1.5e-10)})
        self.assertTrue(np.all(np.isfinite(simultaneous.outgoing[10])))
        with self.assertRaises(ValueError):
            simultaneous.s_column(('a',0))

    def test_transmissive_sheet_agrees_with_analytic_scattering(self):
        scene=Scene((0,.024),(0,.012),boundaries={'xmin':SurfaceImpedance(415),'xmax':SurfaceImpedance(415)})
        scene.sheet('film',(.0125,0),(.0125,.012),ThinSheet.resistive(50))
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        scene.add_port(WaveguidePort('b','x',.018,(0,.012),normal=1))
        mesh=Mesh(np.linspace(0,.024,25),np.linspace(0,.012,13))
        sim=FDTD2D(scene,mesh=mesh,polarization='TM',frequencies=np.linspace(20e9,40e9,21))
        result=sim.scattering(RunControl(2e-9),GaussianPulse(30e9,1.5e-10))
        guide=sim.waveguides['a']; k=10; beta=guide.beta[k,0]
        z=guide.q_modes[k,5,0]/guide.p_modes[k,5,0]
        reflection=-z/(100+z)*np.exp(-2j*beta*.0065)
        transmission=100/(100+z)*np.exp(-1j*beta*.012)
        self.assertLess(abs(result.s[k,0,0]-reflection),.015)
        self.assertLess(abs(result.s[k,1,0]-transmission),.01)

    def test_broadband_straight_guide_s_matrix_matches_discrete_phase(self):
        scene=Scene((0,.024),(0,.012),boundaries={'xmin':SurfaceImpedance(415),'xmax':SurfaceImpedance(415)})
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        scene.add_port(WaveguidePort('b','x',.018,(0,.012),normal=1))
        mesh=Mesh(np.linspace(0,.024,25),np.linspace(0,.012,13))
        sim=FDTD2D(scene,mesh=mesh,polarization='TM',frequencies=np.linspace(20e9,40e9,21))
        result=sim.scattering(RunControl(2e-9),GaussianPulse(30e9,1.5e-10))
        expected=np.exp(-1j*sim.waveguides['a'].beta[:,0]*.012)
        valid=result.valid & (sim.frequencies>26e9) & (sim.frequencies<34e9)
        np.testing.assert_allclose(result.s[valid,1,0],expected[valid],rtol=2e-4,atol=2e-4)
        self.assertLess(abs(result.s[valid,0,0]).max(),2e-4)
        np.testing.assert_allclose(result.s[valid,0,1],result.s[valid,1,0],atol=1e-10)

    def test_tracking_recovers_permutation_and_complex_phase(self):
        rng=np.random.default_rng(1)
        previous,_=np.linalg.qr(rng.normal(size=(20,3)))
        current=previous[:,[2,0,1]]*np.exp(1j*np.array([.2,1.7,-2]))
        order,phase,quality=track_modes(previous,current,np.ones(20))
        np.testing.assert_allclose(current[:,order]*phase,previous,atol=1e-14)
        np.testing.assert_allclose(quality,1)

    def test_waveguide_cutoff_and_power_normalization(self):
        scene=Scene((0,.012),(0,.012))
        scene.add_port(WaveguidePort('guide','x',.006,(0,.012),modes=2))
        frequencies=np.linspace(20e9,30e9,5)
        sim=FDTD2D(scene,mesh=uniform(),polarization='TM',frequencies=frequencies)
        modes=sim.waveguides['guide']
        for k in range(len(frequencies)):
            beta=modes.beta[k]
            dl,dr=modes.distances.T
            face=(np.exp(1j*dl[:,None]*beta)+np.exp(-1j*dr[:,None]*beta))/2
            power=.5*np.cos(np.pi*frequencies[k]*sim.dt)*np.sum(modes.widths[:,None]*(face*modes.q_modes[k]*modes.p_modes[k].conj()).real,axis=0)
            np.testing.assert_allclose(power[modes.valid[k]],1,atol=1e-12)
        cutoff=299792458/(2*.012)
        self.assertTrue(np.all(modes.valid[:,0]))
        self.assertLess(cutoff,frequencies[0])

    def test_lumped_receiver_is_loaded_and_yields_spectra(self):
        scene=Scene((0,.012),(0,.012))
        scene.add_port(LumpedPort('a',(.003,.006)))
        scene.add_port(LumpedPort('b',(.009,.006)))
        sim=FDTD2D(scene,mesh=uniform(),polarization='TM',frequencies=[10e9,15e9])
        result=sim.scattering(RunControl(1e-9),GaussianPulse(12e9,3e-11))
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.all(np.isfinite(result.s)))
        np.testing.assert_allclose(result.s[:,0,1],result.s[:,1,0],rtol=.01,atol=.01)

    def test_foster_good_conductor_fit_and_thin_slab(self):
        model=SurfaceImpedance.good_conductor(5.8e7,(1e9,20e9),order=20,tolerance=.03)
        f=np.geomspace(1e9,20e9,40)
        target=np.sqrt(2j*np.pi*f*1.25663706212e-6/5.8e7)
        self.assertLess(np.max(abs(model.impedance(f)-target)/abs(target)),.03)
        sheet=ThinSheet.conductive(5.8e7,1e-6,(1e9,20e9),order=64,tolerance=.03)
        self.assertTrue(np.all(sheet.even.impedance(f).real>=0))
        self.assertTrue(np.all(sheet.odd.impedance(f).real>=0))


class AutomaticDomainAndRadiationTests(unittest.TestCase):
    def test_radiating_point_source_is_independent_of_closed_contour(self):
        scene=Scene((0,.04),(0,.04))
        scene.add_port(LumpedPort('feed',(.02,.02)))
        mesh=Mesh(np.linspace(0,.04,41),np.linspace(0,.04,41))
        angles=np.linspace(0,2*np.pi,25)
        amplitudes=[]
        for bounds in ((.012,.012,.028,.028),(.010,.010,.030,.030)):
            sim=FDTD2D(scene,mesh=mesh,polarization='TM',frequencies=[15e9],pml=PML(.008),
                       ntff=ClosedContour.rectangle(bounds))
            run=sim.run(RunControl(1.5e-9),{'feed':GaussianPulse(15e9,5e-11)})
            amplitudes.append(run.far_field(angles).scalar_amplitude)
        np.testing.assert_allclose(amplitudes[0],amplitudes[1],rtol=.03,atol=1e-18)

    def test_matched_mesh_policy_collars_survive_roundoff(self):
        scene=Scene()
        scene.rectangle('lower',(0,.024),(-.002,0),PEC,rank=40)
        scene.rectangle('upper',(0,.024),(.012,.014),PEC,rank=40)
        scene.add_port(WaveguidePort('a','x',.006,(0,.012),normal=-1))
        policy=MeshPolicy(.001,40e9,min_step=.00025,cells_per_wavelength=10,growth=2)
        sim=FDTD2D(scene,mesh_policy=policy,polarization='TM',frequencies=[30e9])
        np.testing.assert_allclose(sim.waveguides['a'].distances[:,0],sim.waveguides['a'].distances[:,1],rtol=1e-7)
        self.assertIsNotNone(sim.ntff)

    def test_objects_generate_domain_pml_and_closed_ntff(self):
        scene=Scene()
        scene.circle('pec',(.003,-.002),.001,PEC,rank=30)
        scene.add_port(LumpedPort('feed',(.005,-.002)))
        sim=FDTD2D(scene,mesh_policy=MeshPolicy(.001,10e9,growth=2),polarization='TM',frequencies=[10e9])
        self.assertIsNone(scene.x)  # caller's scene remains a mesh-independent snapshot
        self.assertIsNotNone(sim.pml)
        self.assertIsNotNone(sim.ntff)
        vertices=sim.ntff.contour.vertices
        self.assertEqual(vertices[0],vertices[-1])
        self.assertTrue(sim.scene.domain.contains(scene.geometry[0].shape))
        self.assertIsNone(sim.scene.generated.tfsf_bounds)
        with self.assertRaises(ValueError):
            ClosedContour(((0,0),(1,0),(1,1),(0,1)))

    def test_closed_green_integral_matches_outgoing_cylindrical_wave(self):
        from scipy.special import hankel2
        from FDTD_2D.contours import cylindrical_far_field
        from FDTD_2D.topology import EPS0,MU0
        f=np.array([10e9]); k=2*np.pi*f[0]*np.sqrt(EPS0*MU0)
        angles=np.linspace(0,2*np.pi,31)
        expected=np.sqrt(2/(np.pi*k))*np.exp(1j*np.pi/4)*np.ones((1,len(angles)))
        for radius in (.012,.023):
            vertices=ClosedContour.rectangle((-radius,-radius,radius,radius)).vertices
            points=[]; normals=[]; lengths=[]
            for a,b in zip(vertices[:-1],vertices[1:]):
                tangent=np.subtract(b,a); length=np.linalg.norm(tangent); tangent/=length
                points.extend(np.array(a)+(np.arange(300)+.5)[:,None]/300*np.subtract(b,a))
                normals.extend([np.array([tangent[1],-tangent[0]])]*300)
                lengths.extend([length/300]*300)
            points=np.array(points); normals=np.array(normals); r=np.linalg.norm(points,axis=1)
            q=hankel2(0,k*r)[None,:]
            derivative=-k*hankel2(1,k*r)*np.sum(points/r[:,None]*normals,axis=1)
            for polarization,coefficient in (('TM',MU0),('TE',EPS0)):
                p=-derivative[None,:]/(2j*np.pi*f[:,None]*coefficient)
                far=cylindrical_far_field(f,angles,points,normals,lengths,q,p,EPS0,MU0,polarization)
                np.testing.assert_allclose(far.scalar_amplitude,expected,rtol=2e-5,atol=2e-6)

    def test_mesh_matched_plane_wave_has_no_empty_space_tfsf_leakage(self):
        for polarization,axis,direction in (('TM','x',1),('TE','y',-1)):
            scene=Scene()
            scene.rectangle('background marker',(-.002,.002),(-.002,.002),Material('vacuum'))
            scene.add_plane_wave(PlaneWave('plane',axis,direction))
            policy=MeshPolicy(.001,20e9,growth=2,min_step=.0002)
            sim=FDTD2D(scene,mesh_policy=policy,polarization=polarization,frequencies=[15e9],
                       domain_policy=DomainPolicy(pml_cells=8))
            probe=(.010,.010)
            result=sim.run(RunControl(1.2e-9),{'plane':GaussianPulse(15e9,5e-11)},probes=[(0,0),probe])
            self.assertIsNotNone(sim.scene.generated.tfsf_bounds)
            interior=abs(result.probe_spectra[0,0]); exterior=abs(result.probe_spectra[0,1])
            self.assertGreater(interior,0.)
            self.assertLess(exterior/interior,1e-7)
            far=result.far_field(np.linspace(0,2*np.pi,13))
            self.assertLess(np.max(abs(far.scalar_amplitude))/interior,1e-6)


if __name__=='__main__':
    unittest.main()
