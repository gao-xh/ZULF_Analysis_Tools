import unittest
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.demodulated_decay import DemodulatedDecayOperator,fit_demodulated_modes


class DemodulatedTests(unittest.TestCase):
    def test_operator_matches_independent_direct_convolution_including_edges(self):
        fs=256;n=1024;t=np.arange(n)/fs
        y=np.exp(-t/.3)*np.cos(2*np.pi*40.3*t+.7)+.02*t*t
        p=DemodulatedDecayOperator(fs,n,dict(start_s=.125,end_s=3.5,sg_window=31),[35,45],transition_hz=10)
        raw=y-savgol_filter(y,31,2,mode='mirror');raw=raw[32:896];raw-=raw.mean()
        mixed=2*raw*np.exp(-2j*np.pi*40*np.arange(32,896)/fs)
        full=np.convolve(mixed,p.fir,'full');radius=(len(p.fir)-1)//2
        expected=full[radius:radius+len(raw)][::p.decimation]
        np.testing.assert_allclose(p.transform_all(y),expected,atol=1e-12)
        self.assertFalse(p.valid_interior[0]);self.assertTrue(p.selected[0])
        col=p.templates([40.3],[1.],1/.3)@np.array([np.cos(.7),-np.sin(.7)])
        pure=np.exp(-t/.3)*np.cos(2*np.pi*40.3*t+.7)
        np.testing.assert_allclose(col,p.transform(pure),atol=1e-12)

    def test_fast_and_slow_decay_and_phase_recovery_with_matched_edges(self):
        fs=256;n=1024;t=np.arange(n)/fs
        y=np.exp(-t/.18)*np.cos(2*np.pi*39.4*t+.3)+.7*np.exp(-t/.5)*np.cos(2*np.pi*41.8*t-.8)
        op=DemodulatedDecayOperator(fs,n,dict(start_s=.1,end_s=3.,sg_window=31),[35,45],transition_hz=8)
        r=fit_demodulated_modes(op,y,[35,45],[.08,1.],[39.3,41.9],starts=3)
        np.testing.assert_allclose(r['frequencies_hz'],[39.4,41.8],atol=1e-5)
        np.testing.assert_allclose(r['t2star_s'],[.18,.5],atol=1e-5)
        np.testing.assert_allclose(r['phases_rad'],[.3,-.8],atol=1e-5)
        self.assertNotIn('native_fft_spacing_hz',r['numerical_diagnostics']['thresholds'])

    def test_interior_policy_exposes_early_blind_interval(self):
        op=DemodulatedDecayOperator(256,1024,dict(start_s=.1,end_s=3.),[35,45],transition_hz=5,edge_policy='interior')
        self.assertGreater(op.times_s[0],.1+op.metadata['edge_duration_s']-.01)
        np.testing.assert_array_equal(op.selected,op.valid_interior)
        with self.assertRaisesRegex(ValueError,'Fewer than eight'):
            DemodulatedDecayOperator(256,512,dict(end_s=.5),[35,45],transition_hz=5,edge_policy='interior')


if __name__=='__main__': unittest.main()
