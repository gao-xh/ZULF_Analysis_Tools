import unittest
import numpy as np
from zulf_tools.crop_diagnostics import band_blocks,propose_intervals,guarded_start


class CropTests(unittest.TestCase):
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
