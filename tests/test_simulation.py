import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from zulf_tools import simulation as sim, storage
from zulf_tools.analysis import execute

MODEL = {'isotopes': ['1H', '1H', '13C'],
         'couplings_hz': [[0, 8, 125], [8, 0, 125], [125, 125, 0]]}
SETTINGS = dict(npoints=48, sampling_rate_hz=1200., t1_points=5, t1_step_s=.0019,
                tm_s=.003, pulse_field_ut=[2., 1., .5], pulse_duration_s=.0008,
                field1_ut=.13, field2_ut=.27, t2star_s=.6)


def direct_reference(model, settings):
    """Independent repeated expm propagation, not the optimized eigenbasis formula."""
    h, m, _, _ = sim.build(model)
    h1, h2 = h+settings['field1_ut']*m[2], h+settings['field2_ut']*m[2]
    up = expm(-1j*(h1+sum(b*a for b,a in zip(settings['pulse_field_ut'], m)))*settings['pulse_duration_s'])
    exc = up @ expm(-1j*h1*settings['tm_s']) @ up if settings['sequence'] == 'mq' else up
    rho = exc @ m[0] @ exc.conj().T
    u2 = expm(-1j*h2/settings['sampling_rate_hz'])
    rows = 1 if settings['sequence'] == 'fid' else settings['t1_points']
    out = np.zeros((rows, settings['npoints']), complex)
    for i in range(rows):
        u1 = expm(-1j*h1*i*settings['t1_step_s'])
        r = rho.copy() if settings['sequence'] == 'fid' else up @ u1 @ rho @ u1.conj().T @ up.conj().T
        for k in range(settings['npoints']):
            out[i,k] = np.trace(r @ m[0])*np.exp(-k/settings['sampling_rate_hz']/settings['t2star_s'])
            r = u2 @ r @ u2.conj().T
    return out


class SimulationTests(unittest.TestCase):
    def test_s2_s3_multiplicity_and_degenerate_pulse(self):
        # Six-spin A2B3X regression: every spin multiplicity must survive.
        j = np.zeros((6,6))
        for i in range(2):
            for k in range(2,5): j[i,k] = j[k,i] = 7.032
            j[i,5] = j[5,i] = -2.317
        for i in range(2,5): j[i,5] = j[5,i] = 125.2572
        model = dict(isotopes=['1H']*5+['13C'],couplings_hz=j.tolist(),symmetry_groups=[[0,1],[2,3,4]])
        s = dict(SETTINGS,sequence='2d',npoints=24,t1_points=3,field1_ut=0,field2_ut=0,
                 pulse_field_ut=[0,0,50],pulse_duration_s=.0009337068)
        ref = direct_reference(model,s)
        for backend in ['cpu'] + (['gpu'] if sim.gpu_probe()['available'] else []):
            actual,_,_,report = sim.compute(model,dict(s,backend=backend))
            self.assertEqual(report['sector_dimensions'],[8,8,24,24])
            np.testing.assert_allclose(actual,ref,rtol=1e-10,atol=1e-7)

    def test_eigenbasis_and_sectors_against_direct_propagation(self):
        for sequence in ('fid', '2d', 'mq'):
            s = dict(SETTINGS, sequence=sequence)
            ref = direct_reference(MODEL, s)
            for symmetry in (True, False):
                actual, _, _, report = sim.compute(MODEL, dict(s, symmetry=symmetry))
                np.testing.assert_allclose(actual, ref, rtol=1e-10, atol=1e-7)
                self.assertEqual(sum(report['sector_dimensions']), 8)

    def test_single_spin_analytic_precession(self):
        model = {'isotopes':['1H'], 'couplings_hz':[[0]]}
        s = dict(npoints=100, sampling_rate_hz=1000, field2_ut=1, t2star_s=2)
        fid, _, t, _ = sim.compute(model, s)
        g = 2*np.pi*42.58
        np.testing.assert_allclose(fid[0].real, g*g/2*np.cos(g*t)*np.exp(-t/2), atol=1e-8)

    def test_invalid_symmetry_and_spin(self):
        model = copy.deepcopy(MODEL)
        model['couplings_hz'][0][2] += .01; model['couplings_hz'][2][0] += .01
        self.assertEqual(sim.validate(model)[3], [])
        model['symmetry_groups'] = [[0,1]]
        with self.assertRaises(ValueError): sim.validate(model)
        with self.assertRaises(ValueError): sim.validate({'isotopes':['14N'], 'couplings_hz':[[0]]})
        with self.assertRaises(ValueError): sim.compute(MODEL, {'pulse_duration_ms':1})
        with self.assertRaises(InterruptedError): sim.compute(MODEL, SETTINGS, cancel=lambda:True)

    def test_import_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); models = root/'molecule'; models.mkdir()
            (models/'structure.csv').write_text('1H,1H,13C\n0,8,125\n8,0,125\n125,125,0\n')
            (models/'symmetry.csv').write_text('S2,1 2\n')
            storage.write_json(root/'local.json', {'model_roots':[str(models)]})
            with patch.object(storage,'PROJECT',root), patch.object(storage,'ROOT',root/'.analysis'):
                result = execute('import_spin_model',{'folder':str(models)})
                self.assertEqual(result['model']['symmetry_groups'], [[0,1]])
                r = execute('simulate_spin_dynamics',{'model':result['model'],'settings':dict(SETTINGS,sequence='2d')})
                self.assertTrue(storage.artifact(r['run_id'],'spectrum2d.png').exists())
                with np.load(storage.artifact(r['run_id'],'simulation.npz')) as data:
                    self.assertEqual(data['fid'].shape,(5,48))
                    self.assertTrue(np.isfinite(data['spectrum2d']).all())

    def test_gpu_matches_cpu_if_available(self):
        probe = sim.gpu_probe()
        if not probe['available']: self.skipTest(probe['reason'])
        for sequence in ('fid','2d','mq'):
            cpu = sim.compute(MODEL, dict(SETTINGS,sequence=sequence))[0]
            gpu = sim.compute(MODEL, dict(SETTINGS,sequence=sequence,backend='gpu'))[0]
            np.testing.assert_allclose(gpu,cpu,rtol=1e-10,atol=1e-7)


if __name__ == '__main__': unittest.main()
