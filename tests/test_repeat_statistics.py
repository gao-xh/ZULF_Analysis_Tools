import unittest
import numpy as np
from zulf_tools.repeat_statistics import weighted_repeat_statistics,accumulation_diagnostic,classify_reproducibility,spectral_coherence


class RepeatStatisticsTests(unittest.TestCase):
    def test_band_phase_cancellation_without_bin_count_snr_inflation(self):
        reference=np.array([1.,2.,3.],complex);phases=np.array([-np.pi/3,np.pi/3])
        spectra=np.exp(1j*phases[:,None])*reference
        r=spectral_coherence(spectra,[1,1],reference,np.full(3,.001))
        self.assertAlmostEqual(r['coherent_to_mean_group_norm'],.5)
        np.testing.assert_allclose([x['relative_phase_rad'] for x in r['group_diagnostics']],phases)
        tiled=spectral_coherence(np.tile(spectra,(1,10)),[1,1],np.tile(reference,10),np.full(30,.001))
        self.assertAlmostEqual(r['group_diagnostics'][0]['group_rms_signal_to_scatter'],tiled['group_diagnostics'][0]['group_rms_signal_to_scatter'])
        low=spectral_coherence(spectra,[1,1],reference,np.full(3,100.))
        self.assertTrue(all(x['relative_phase_rad'] is None for x in low['group_diagnostics']))

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
