import unittest
import numpy as np
from zulf_tools.repeat_statistics import weighted_repeat_statistics,accumulation_diagnostic,classify_reproducibility,spectral_coherence,residual_repeat_diagnostic,directional_residual_diagnostic


class RepeatStatisticsTests(unittest.TestCase):
    def test_directional_check_retains_correlated_bin_scatter(self):
        shape=np.exp(1j*np.linspace(0,2,30));coeff=np.array([.5,1.,1.5,2.,.75,1.25,1.75,.25])
        residuals=coeff[:,None]*shape
        r=directional_residual_diagnostic(shape,residuals,np.ones(8))
        expected=coeff.mean()/(coeff.std(ddof=1)/np.sqrt(8))
        self.assertAlmostEqual(r['signed_projection_to_sem'],expected)
        duplicate=directional_residual_diagnostic(np.tile(shape,20),np.tile(residuals,(1,20)),np.ones(8))
        self.assertAlmostEqual(duplicate['signed_projection_to_sem'],expected)
        opposite=directional_residual_diagnostic(shape,-residuals,np.ones(8))
        self.assertFalse(opposite['requires_review'])
        self.assertLess(opposite['signed_projection_to_sem'],0)

    def test_directional_check_detects_local_structure_amid_orthogonal_scatter(self):
        rng=np.random.default_rng(519);d=rng.normal(size=(8,100));v=rng.normal(size=(8,100))
        # Construct a discovery direction from its own data, not validation.
        d-=d.mean(axis=0);d[:,0]=.5+rng.normal(0,.01,8)
        v[:,0]=.5+rng.normal(0,.01,8)
        r=residual_repeat_diagnostic(d,np.ones(8),v,np.ones(8),np.zeros(100))
        self.assertFalse(r['band_norm_requires_review'])
        self.assertTrue(r['directional_validation']['requires_review'])
        self.assertTrue(r['requires_review'])
        absent=directional_residual_diagnostic(np.zeros(100),v,np.ones(8))
        self.assertFalse(absent['available']);self.assertFalse(absent['requires_review'])

    def test_reproducible_model_error_is_not_hidden_by_repeat_noise(self):
        rng=np.random.default_rng(602);truth=np.exp(1j*np.linspace(0,2,64));counts=np.full(8,10)
        noise=(rng.normal(size=(16,64))+1j*rng.normal(size=(16,64)))*.03
        d=truth+noise[:8];v=truth+noise[8:]
        good=residual_repeat_diagnostic(d,counts,v,counts,truth)
        self.assertFalse(good['requires_review'])
        wrong=truth-.2*np.sin(np.linspace(0,np.pi,64))
        bad=residual_repeat_diagnostic(d,counts,v,counts,wrong)
        self.assertTrue(bad['requires_review']);self.assertTrue(bad['reproducible_residual'])
        self.assertGreater(bad['residual_alignment'],.9)
        tiled=residual_repeat_diagnostic(np.tile(d,(1,5)),counts,np.tile(v,(1,5)),counts,np.tile(wrong,5))
        self.assertAlmostEqual(tiled['validation_residual_to_sem'],bad['validation_residual_to_sem'])

    def test_missing_or_zero_scatter_is_not_a_confidence_claim(self):
        y=np.ones((2,8),complex)
        missing=residual_repeat_diagnostic(y[:1],[1],y,[1,1],np.ones(8))
        self.assertFalse(missing['available'])
        zero=residual_repeat_diagnostic(y,[1,1],y,[1,1],np.zeros(8))
        self.assertFalse(zero['available']);self.assertTrue(zero['requires_review'])
        self.assertIsNone(zero['validation_residual_to_sem'])

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

    def test_accumulation_difference_cannot_measure_common_component(self):
        rng=np.random.default_rng(124)
        noise=(rng.normal(size=(16,128))+1j*rng.normal(size=(16,128)))*.01
        mask=np.ones(128,bool);mask[0]=False
        baseline=accumulation_diagnostic(noise,np.ones(16),mask,0)
        # A shared spectral component changes pooled amplitudes but cancels
        # identically in differences, including in reference noise bins.
        common=np.exp(1j*np.linspace(0,2,128))
        biased=accumulation_diagnostic(noise+common,np.ones(16),mask,0)
        np.testing.assert_allclose(
            [r['median_difference_noise'] for r in baseline['levels']],
            [r['median_difference_noise'] for r in biased['levels']],rtol=1e-12)
        self.assertGreater(biased['levels'][-1]['snr'],baseline['levels'][-1]['snr']*100)


if __name__=='__main__':
    unittest.main()
