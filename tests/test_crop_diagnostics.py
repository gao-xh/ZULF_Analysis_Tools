import unittest
import numpy as np
from zulf_tools.crop_diagnostics import band_blocks,propose_intervals,guarded_start,early_amplitude_diagnostic,display_extrema_indices


class CropTests(unittest.TestCase):
    def test_display_retains_transients_missed_by_regular_stride(self):
        y=np.zeros(65516);y[1]=100.;y[103]=-90.;y[-1]=7.
        original=y.copy();indices=display_extrema_indices(y)
        self.assertEqual(y[::4].max(),0.)
        self.assertTrue({0,1,103,len(y)-1}.issubset(set(indices)))
        self.assertLessEqual(len(indices),15000)
        self.assertTrue(np.all(np.diff(indices)>0))
        np.testing.assert_array_equal(y,original)
        for n in [0,1,4,15,100,101,65516]:
            for budget in [4,5,16,15000]:
                data=np.sin(np.arange(n));selected=display_extrema_indices(data,budget)
                self.assertLessEqual(len(selected),budget)
                if n:
                    self.assertEqual(data[selected].min(),data.min())
                    self.assertEqual(data[selected].max(),data.max())

    def test_fine_blocks_locate_finite_transient_and_preserve_threshold_alternatives(self):
        y=np.ones(1000);y[:40]=1000;y[40:80]=15
        r=early_amplitude_diagnostic(y,1000)
        self.assertEqual(r['reference_interval_s'],[.5,1.])
        self.assertEqual([p['candidate_start_s'] for p in r['thresholds']],[.085,.085,.045])
        # A burst away from acquisition start is not a reason to trim everything.
        y[:80]=1;y[200:240]=1000
        self.assertTrue(all(p['candidate_start_s'] is None for p in early_amplitude_diagnostic(y,1000)['thresholds']))
        y[:500]=1000
        self.assertTrue(all(p['candidate_start_s'] is None for p in early_amplitude_diagnostic(y,1000)['thresholds']))

    def test_slow_baseline_does_not_discard_fast_signal(self):
        self.assertEqual(guarded_start(2.75,.0375,16.379),(.0375,True,False))
        self.assertEqual(guarded_start(None,0.,16.379),(0.,True,False))
        self.assertEqual(guarded_start(.05,.0375,16.379),(.05,False,False))

    def test_window_amplitude_matches_known_tone_and_repeat_noise(self):
        fs=256;n=2048;t=np.arange(n)/fs
        signal=2*np.cos(2*np.pi*40*t)
        # Deterministic opposite perturbations give an independent SEM check.
        noise=.2*np.cos(2*np.pi*40*t)
        groups=np.array([signal+noise,signal-noise])
        r=band_blocks(groups,[1,1],fs,[[39,41]],1.)
        # Hann center and two adjacent bins have amplitudes 2, 1, 1.
        np.testing.assert_allclose(r['amplitude'],np.sqrt(2),atol=1e-12)
        np.testing.assert_allclose(r['snr'],10,atol=1e-10)
        np.testing.assert_allclose(r['times_s'],np.arange(8)+.5)

    def test_revival_prevents_premature_tail_cut_and_no_signal_keeps_full(self):
        times=np.arange(16)*.25+.125
        snr=np.zeros((2,16));snr[0,:3]=10;snr[1,12]=8
        p=propose_intervals(snr,times,.25,4.,.1,5)
        self.assertTrue(all(row['end_s']>=3.5 for row in p))
        p=propose_intervals(np.zeros_like(snr),times,.25,4.,.1,5)
        self.assertTrue(all(row['end_s']==4. for row in p))
        self.assertEqual(p[0]['start_s'],0.)

    def test_complete_windows_and_resolution_are_explicit(self):
        r=band_blocks(np.ones((2,1100)),[1,1],256,[[35,45]],.5)
        self.assertEqual(r['unused_tail_samples'],76)
        self.assertEqual(r['nominal_resolution_hz'],2.)
        with self.assertRaisesRegex(ValueError,'two native'):
            band_blocks(np.ones((2,2048)),[1,1],256,[[40,40.1]],.25)


if __name__=='__main__': unittest.main()
