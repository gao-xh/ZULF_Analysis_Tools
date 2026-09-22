import unittest
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.transition_decay import fit_transition_decay


class TransitionDecayTests(unittest.TestCase):
    def fixture(self, taus=(.4,1.2)):
        fs=256.;n=2048;t=np.arange(n)/fs
        groups=[dict(frequencies_hz=[38.,39.1],weights=[1.,.6]),
                dict(frequencies_hz=[41.,42.3],weights=[.4,1.])]
        y=np.zeros(n)
        for g,tau,amp,phase in zip(groups,taus,[1.3,.7],[.3,-.8]):
            w=np.asarray(g['weights']);w=w/w.sum()
            for f,weight in zip(g['frequencies_hz'],w):
                y+=amp*weight*np.exp(-t/tau)*np.cos(2*np.pi*f*t+phase)
        y-=savgol_filter(y,41,2,mode='mirror')
        first,last=8,2040;freq=np.fft.rfftfreq(last-first,1/fs)
        bins=np.flatnonzero((freq>=35)&(freq<=45))
        p=ProcessedSpectrum(fs,n,first,last,bins,41,2)
        return p,np.fft.rfft(y[first:last])[bins]/(last-first),groups

    def test_two_weighted_transition_groups_recover_phase_and_distinct_decays(self):
        p,y,g=self.fixture()
        r=fit_transition_decay(p,y,g,[.1,2.],starts=4)
        np.testing.assert_allclose(r['t2star_s'],[.4,1.2],atol=1e-5)
        np.testing.assert_allclose(r['phases_rad'],[.3,-.8],atol=1e-5)
        np.testing.assert_allclose(r['amplitudes'],[1.3,.7],atol=1e-5)
        self.assertTrue(r['optimizer_converged'])
        self.assertFalse(r['numerical_warning_flags'])

    def test_shared_decay_recovers_multiple_transition_beating(self):
        p,y,g=self.fixture((.7,.7))
        r=fit_transition_decay(p,y,g,[.1,2.],shared_decay=True,starts=3)
        np.testing.assert_allclose(r['t2star_s'],[.7,.7],atol=1e-5)

    def test_wrong_transition_structure_remains_bad_even_with_free_decays(self):
        p,y,g=self.fixture()
        g[0]['frequencies_hz']=[38.8,39.9]
        r=fit_transition_decay(p,y,g,[.05,3.],starts=4)
        self.assertGreater(r['relative_complex_residual'],.2)

    def test_budget_bounds_and_invalid_weights(self):
        p,y,g=self.fixture()
        limited=fit_transition_decay(p,y,g,[.1,2.],max_evaluations=2)
        self.assertTrue(limited['budget_exhausted'])
        self.assertIn('search_budget_exhausted',limited['numerical_warning_flags'])
        bounded=fit_transition_decay(p,y,g,[.1,.8],starts=3)
        self.assertIn(1,bounded['boundary_group_indices'])
        g[0]['weights']=[-1.,2.]
        with self.assertRaises(ValueError):
            fit_transition_decay(p,y,g,[.1,2.])


if __name__=='__main__':
    unittest.main()
