import unittest
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes, mode_design


class DecayTests(unittest.TestCase):
    def fixture(self, modes, sg=0, first=64, last=3600):
        fs=512.; n=4096; t=np.arange(n)/fs
        y=sum(a*np.exp(-t/tau)*np.cos(2*np.pi*f*t+phase) for f,tau,a,phase in modes)
        if sg:
            y=y-savgol_filter(y,sg,2,mode='mirror')
        bins=np.flatnonzero((np.fft.rfftfreq(last-first,1/fs)>=35)&(np.fft.rfftfreq(last-first,1/fs)<=45))
        processor=ProcessedSpectrum(fs,n,first,last,bins,sg,2)
        observed=np.fft.rfft(y[first:last])[bins]/(last-first)
        return processor,observed

    def test_exact_forward_matches_independent_real_fid_and_sg_edges(self):
        modes=[(39.7,.42,2.,.9),(41.3,1.8,.7,-1.1)]
        for first,last in [(4,4096),(80,3700)]:
            p,y=self.fixture(modes,sg=41,first=first,last=last)
            c=np.array([[a*np.cos(phase),-a*np.sin(phase)] for f,tau,a,phase in modes]).ravel()
            calculated=mode_design(p,[v[0] for v in modes],[v[1] for v in modes])@c
            np.testing.assert_allclose(calculated,y,atol=2e-12,rtol=2e-9)

    def test_single_frequency_phase_and_decay_recovery(self):
        p,y=self.fixture([(40.17,.74,1.4,1.1)],sg=41)
        result=fit_modes(p,y,[35,45],[.1,3.],starts=3,max_seconds=20)
        self.assertAlmostEqual(result['frequencies_hz'][0],40.17,places=5)
        self.assertAlmostEqual(result['t2star_s'][0],.74,places=5)
        self.assertAlmostEqual(result['phases_rad'][0],1.1,places=5)
        self.assertTrue(result['optimizer_converged'])

    def test_beating_two_frequencies_same_decay(self):
        p,y=self.fixture([(39.6,1.3,1.,.4),(40.5,1.3,.8,-.8)])
        result=fit_modes(p,y,[35,45],[.2,3.],mode_count=2,shared_decay=True,
                         initial_frequencies=[39.5,40.6],starts=3,max_seconds=20)
        np.testing.assert_allclose(result['frequencies_hz'],[39.6,40.5],atol=1e-4)
        np.testing.assert_allclose(result['t2star_s'],[1.3,1.3],atol=1e-4)

    def test_budget_and_out_of_range_decay_are_reported(self):
        p,y=self.fixture([(40.,4.,1.,.2)])
        limited=fit_modes(p,y,[35,45],[.1,2.],max_evaluations=2,starts=2)
        self.assertTrue(limited['budget_exhausted'])
        self.assertFalse(limited['optimizer_converged'])
        result=fit_modes(p,y,[35,45],[.1,2.],starts=2)
        self.assertTrue(any('t2' in key for key in result['boundary_hits']))

    def test_two_distinct_decays_with_independent_time_noise(self):
        fs=512.; n=4096; t=np.arange(n)/fs; first=70; last=3700
        rng=np.random.default_rng(1729)
        y=(1.2*np.exp(-t/.55)*np.cos(2*np.pi*38.8*t+.7)
           +.9*np.exp(-t/1.6)*np.cos(2*np.pi*41.2*t-.9)
           +rng.normal(0,.025,n))
        bins=np.flatnonzero((np.fft.rfftfreq(last-first,1/fs)>=35)&(np.fft.rfftfreq(last-first,1/fs)<=45))
        p=ProcessedSpectrum(fs,n,first,last,bins)
        observed=np.fft.rfft(y[first:last])[bins]/(last-first)
        result=fit_modes(p,observed,[35,45],[.15,3.],mode_count=2,
                         initial_frequencies=[38.7,41.3],starts=4)
        np.testing.assert_allclose(result['frequencies_hz'],[38.8,41.2],atol=.025)
        np.testing.assert_allclose(result['t2star_s'],[.55,1.6],rtol=.07)


if __name__=='__main__':
    unittest.main()
