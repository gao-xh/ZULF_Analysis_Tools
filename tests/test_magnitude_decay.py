import unittest
import numpy as np
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.magnitude_decay import fit_magnitude_modes


class MagnitudeTests(unittest.TestCase):
    def fixture(self):
        fs=256.;n=2048;t=np.arange(n)/fs;first=16;last=1800
        y=1.4*np.exp(-t/.74)*np.cos(2*np.pi*40.17*t+1.1)
        f=np.fft.rfftfreq(last-first,1/fs);bins=np.flatnonzero((f>=35)&(f<=45))
        p=ProcessedSpectrum(fs,n,first,last,bins)
        initial=dict(frequencies_hz=[40.1],t2star_s=[.6],cos_sin_coefficients=[[1.,-.4]],shared_decay=False)
        return p,np.fft.rfft(y[first:last])[bins]/(last-first),initial

    def test_magnitude_recovers_decay_but_cannot_determine_global_sign(self):
        p,y,initial=self.fixture();r=fit_magnitude_modes(p,y,[35,45],[.1,2.],initial,starts=2)
        self.assertAlmostEqual(r['frequencies_hz'][0],40.17,places=5)
        self.assertAlmostEqual(r['t2star_s'][0],.74,places=5)
        self.assertLess(r['relative_magnitude_residual'],1e-6)
        flipped=fit_magnitude_modes(p,-y,[35,45],[.1,2.],initial,starts=2)
        np.testing.assert_allclose(r['fitted'],flipped['fitted'],atol=1e-12)
        self.assertTrue(r['phase_sign_ambiguous'])
        np.testing.assert_array_equal(abs(r['fitted']),abs(-r['fitted']))

    def test_budget_preserves_provisional_attempt(self):
        p,y,initial=self.fixture();r=fit_magnitude_modes(p,y,[35,45],[.1,2.],initial,max_evaluations=2)
        self.assertTrue(r['budget_exhausted']);self.assertFalse(r['optimizer_converged'])
        self.assertEqual(r['evaluations'],2)
        with self.assertRaises(InterruptedError):fit_magnitude_modes(p,y,[35,45],[.1,2.],initial,cancel=lambda:True)


if __name__=='__main__':unittest.main()
