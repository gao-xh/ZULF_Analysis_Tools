from pathlib import Path
import hashlib
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from zulf_tools import band_relaxation, storage


class BandBudgetTests(unittest.TestCase):
    def test_cancel_after_candidate_keeps_numeric_prediction(self):
        cancelled=[False]
        def progress(*args):cancelled[0]=True
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            with patch.object(band_relaxation,'load_group_averages',return_value=self.fixture()),patch('zulf_tools.analysis.plot'):
                with self.assertRaises(InterruptedError):
                    band_relaxation.fit_frequency_decay('fixture',[[26,34],[54,68]],[0,1],[2,3],
                        t2_bounds=[.2,2.],components=[1],settings=dict(starts=1),
                        record={},directory=directory,cancel=lambda:cancelled[0],progress=progress)
            rows=storage.read_json(directory/'candidates.json')
            self.assertEqual(len(rows),1)
            checkpoint=directory/rows[0]['checkpoint_artifact']
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),rows[0]['checkpoint_sha256'])
            with np.load(checkpoint) as a:
                error=np.linalg.norm(a['validation']-a['prediction'])/np.linalg.norm(a['validation'])
                self.assertAlmostEqual(error,rows[0]['validation_relative_complex_residual'])
                np.testing.assert_allclose(a['validation'],a['group_spectra'][[2,3]].mean(axis=0))
            self.assertFalse((directory/'band_arrays.npz').exists())
            self.assertFalse(rows[0]['scientifically_validated'])

    def fixture(self):
        fs=256.;n=2048;t=np.arange(n)/fs
        fid=np.exp(-t)*np.cos(2*np.pi*30*t+.3)+.7*np.exp(-t/.8)*np.cos(2*np.pi*61*t-.5)
        means=np.array([fid,fid,fid,fid])
        return means,np.ones(4),dict(sampling_rate_hz=fs,points=n,arrays_sha256='fixture')

    def test_budget_prioritizes_baselines_across_bands_and_keeps_canonical_results(self):
        elapsed=[0.];calls=[];real_fit=band_relaxation.fit_modes
        def timed_fit(*args,**kwargs):
            calls.append((list(args[2]),kwargs['mode_count'],kwargs['shared_decay']))
            result=real_fit(*args,**kwargs)
            elapsed[0]+=1.
            return result
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(band_relaxation,'load_group_averages',return_value=self.fixture()), \
                 patch.object(band_relaxation,'time',SimpleNamespace(perf_counter=lambda:elapsed[0])), \
                 patch.object(band_relaxation,'fit_modes',side_effect=timed_fit),patch('zulf_tools.analysis.plot'):
                r=band_relaxation.fit_frequency_decay('fixture',[[26,34],[54,68]],[0,1],[2,3],
                    t2_bounds=[.2,2.],components=[2,1],settings=dict(starts=1,total_seconds=2.5),
                    record={},directory=Path(temp),cancel=lambda:False,progress=lambda *args:None)
            self.assertTrue(r['total_budget_exhausted'])
            self.assertEqual(r['completed_configurations'],3)
            self.assertEqual(r['requested_configurations'],6)
            self.assertEqual(calls[:2],[([26,34],1,False),([54,68],1,False)])
            self.assertEqual([(c['band_index'],c['mode_count']) for c in r['candidates']],[(0,1),(0,2),(1,1)])
            self.assertEqual(storage.read_json(Path(temp)/'candidates.json'),r['candidates'])
            # Different band widths ensure cached bins are not accidentally reused.
            for c in r['candidates']:
                for row in c['validation_group_errors']:
                    self.assertAlmostEqual(row['frozen_relative_complex_residual'],c['validation_relative_complex_residual'])
                    self.assertLessEqual(row['conditional_gain_relative_complex_residual'],row['frozen_relative_complex_residual']+1e-9)

    def test_invalid_later_band_is_rejected_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(band_relaxation,'load_group_averages',return_value=self.fixture()), \
                 patch.object(band_relaxation,'fit_modes') as fit,patch('zulf_tools.analysis.plot'):
                with self.assertRaisesRegex(ValueError,'too few native observations'):
                    band_relaxation.fit_frequency_decay('fixture',[[26,34],[60,61]],[0,1],[2,3],
                        t2_bounds=[.2,2.],components=[1,8],record={},directory=Path(temp),
                        cancel=lambda:False,progress=lambda *args:None)
                fit.assert_not_called()


if __name__=='__main__':
    unittest.main()
