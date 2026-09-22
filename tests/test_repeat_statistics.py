import unittest
import numpy as np
from zulf_tools.repeat_statistics import weighted_repeat_statistics,accumulation_diagnostic,classify_reproducibility


class RepeatStatisticsTests(unittest.TestCase):
    def test_unequal_group_noise_matches_independent_scan_variance(self):
        rng=np.random.default_rng(612);counts=np.array([20,50,100,200,30,80])
        noise=(rng.normal(size=(6,20000))+1j*rng.normal(size=(6,20000)))/np.sqrt(counts[:,None])
        r=weighted_repeat_statistics(3+noise,counts)
        self.assertAlmostEqual(float(np.mean(r['standard_error']**2)),2/counts.sum(),delta=.0001)
        self.assertGreater(float(np.median(r['repeat_snr'])),30)

    def test_measured_accumulation_scales_without_imposing_law(self):
        rng=np.random.default_rng(10);counts=np.full(32,100)
        noise=(rng.normal(size=(32,1024))+1j*rng.normal(size=(32,1024)))/10
        noise[:,0]+=2
        mask=np.ones(1024,bool);mask[0]=False
        r=accumulation_diagnostic(noise,counts,mask,0,permutations=32)
        self.assertAlmostEqual(r['noise_log_log_slope'],-.5,delta=.08)
        self.assertGreater(r['levels'][-1]['snr'],r['levels'][0]['snr']*3)

    def test_phase_cancellation_and_cautious_labels(self):
        spectra=np.array([[1,2],[-1,2],[1,2],[-1,2]],complex)
        r=weighted_repeat_statistics(spectra,[1,1,1,1])
        self.assertEqual(r['coherence'][0],0)
        self.assertEqual(r['coherence'][1],1)
        self.assertEqual(classify_reproducibility(20,1,True),'insufficient_evidence')
        self.assertEqual(classify_reproducibility(20,20,True,True),'suspected_interference')
        self.assertEqual(classify_reproducibility(20,20,True),'reproducible_signal_candidate')
        self.assertEqual(classify_reproducibility(1,2,False),'noise_compatible')


if __name__=='__main__':
    unittest.main()
