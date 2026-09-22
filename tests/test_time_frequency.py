import unittest
import numpy as np
from zulf_tools.time_frequency import windowed_fourier,demodulate_band


class TimeFrequencyTests(unittest.TestCase):
    def test_hann_complex_exponential_closed_form_decay_and_phase(self):
        fs=400.; n=2400; t0=.23; t=t0+np.arange(n)/fs
        f=41.7; tau=.72; phase=.63
        y=np.exp(-t/tau+1j*(2*np.pi*f*t+phase))
        r=windowed_fourier(y,fs,[f],.5,.1,t0)
        width=r['window_samples']; starts=np.arange(0,n-width+1,r['hop_samples'])
        # Independent explicit periodic Hann and exact weighted exponential.
        w=(1-np.cos(2*np.pi*np.arange(width)/width))/2
        expected=np.array([np.sum(w*np.exp(-t[s:s+width]/tau))/w.sum()*np.exp(1j*phase) for s in starts])
        np.testing.assert_allclose(r['spectrum'][:,0],expected,atol=2e-14)
        slope=np.polyfit(r['times_s'],np.log(abs(r['spectrum'][:,0])),1)[0]
        self.assertAlmostEqual(-1/slope,tau,places=10)
        self.assertAlmostEqual(r['overlap_fraction'],.8)

    def test_complete_windows_do_not_pad_tail_or_start(self):
        r=windowed_fourier(np.ones(105),100.,[10.],.4,.3,2.)
        np.testing.assert_allclose(r['times_s'],[2.2,2.5,2.8])
        with self.assertRaises(ValueError):
            windowed_fourier(np.ones(20),100.,[10.],.4,.1)

    def test_demodulation_rejects_out_of_band_alias_and_preserves_phase(self):
        fs=1000.; t0=.17;t=t0+np.arange(12000)/fs
        y=np.cos(2*np.pi*100*t+.8)+5*np.cos(2*np.pi*180*t-.4)
        r=demodulate_band(y,fs,[95.,105.],transition_hz=5.,time_origin_s=t0)
        z=r['signal'][r['valid_interior']]
        np.testing.assert_allclose(z,np.full(len(z),np.exp(.8j)),atol=2e-3)
        self.assertGreater(r['decimation'],1)
        self.assertTrue(np.any(~r['valid_interior']))

    def test_demodulated_decay_matches_valid_interior_slope(self):
        fs=1000.;t=np.arange(14000)/fs;tau=2.4
        y=np.exp(-t/tau)*np.cos(2*np.pi*100*t+.8)
        r=demodulate_band(y,fs,[90.,110.],transition_hz=10.)
        keep=r['valid_interior']&(r['times_s']<8)
        slope=np.polyfit(r['times_s'][keep],np.log(abs(r['signal'][keep])),1)[0]
        self.assertAlmostEqual(-1/slope,tau,places=3)


if __name__=='__main__':
    unittest.main()
