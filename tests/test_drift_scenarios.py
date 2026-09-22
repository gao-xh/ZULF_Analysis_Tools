import unittest
import numpy as np
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes,mode_design


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.fs=256.;self.n=1536;self.t=np.arange(self.n)/self.fs;self.first=32;self.last=1400
        f=np.fft.rfftfreq(self.last-self.first,1/self.fs)
        self.bins=np.flatnonzero((f>=35)&(f<=45))
        self.p=ProcessedSpectrum(self.fs,self.n,self.first,self.last,self.bins)

    def spectrum(self,y):return np.fft.rfft(y[self.first:self.last])[self.bins]/(self.last-self.first)

    def test_constant_between_repeat_phase_changes_amplitude_not_decay(self):
        env=np.exp(-self.t/1.3);phase=np.pi/3
        a=env*np.cos(2*np.pi*40*self.t-phase);b=env*np.cos(2*np.pi*40*self.t+phase)
        pooled=(a+b)/2
        np.testing.assert_allclose(pooled,.5*env*np.cos(2*np.pi*40*self.t),atol=2e-13)
        r=fit_modes(self.p,self.spectrum(pooled),[35,45],[.1,3.],starts=2)
        self.assertAlmostEqual(r['t2star_s'][0],1.3,places=5)
        self.assertAlmostEqual(r['amplitudes'][0],.5,places=5)

    def test_between_repeat_frequency_drift_mimics_two_modes_in_average(self):
        signals=[np.exp(-self.t/1.3)*np.cos(2*np.pi*f*self.t) for f in [39.7,40.3]]
        observed=self.spectrum(np.mean(signals,axis=0))
        single=fit_modes(self.p,observed,[35,45],[.1,3.],initial_frequencies=[40.],starts=3)
        two=fit_modes(self.p,observed,[35,45],[.1,3.],mode_count=2,shared_decay=True,initial_frequencies=[39.6,40.4],starts=3)
        self.assertLess(single['t2star_s'][0],.8)
        self.assertGreater(single['relative_complex_residual'],.3)
        self.assertLess(two['relative_complex_residual'],1e-6)
        np.testing.assert_allclose(two['t2star_s'],[1.3,1.3],atol=1e-5)
        group_fits=[fit_modes(self.p,self.spectrum(y),[35,45],[.1,3.],starts=2) for y in signals]
        np.testing.assert_allclose([r['frequencies_hz'][0] for r in group_fits],[39.7,40.3],atol=1e-5)
        np.testing.assert_allclose([r['t2star_s'][0] for r in group_fits],[1.3,1.3],atol=1e-5)
        # A pooled two-mode fit does not predict either acquisition separately.
        pred=mode_design(self.p,two['frequencies_hz'],two['t2star_s'])@np.asarray(two['cos_sin_coefficients']).ravel()
        self.assertTrue(all(np.linalg.norm(self.spectrum(y)-pred)/np.linalg.norm(self.spectrum(y))>.4 for y in signals))

    def test_within_fid_frequency_drift_can_converge_but_leaves_structured_error(self):
        y=np.exp(-self.t/1.3)*np.cos(2*np.pi*(40*self.t+.5*.2*self.t**2))
        r=fit_modes(self.p,self.spectrum(y),[35,45],[.1,3.],starts=3)
        self.assertTrue(r['optimizer_converged'])
        self.assertFalse(r['numerical_diagnostics']['requires_review'])
        self.assertGreater(r['relative_complex_residual'],.15)
        self.assertGreater(abs(r['t2star_s'][0]-1.3),.1)


if __name__=='__main__':unittest.main()
