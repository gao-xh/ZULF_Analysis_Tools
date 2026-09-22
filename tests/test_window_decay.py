import unittest
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.window_decay import WindowedDecayOperator, fit_windowed_modes


class WindowDecayTests(unittest.TestCase):
    def test_sparse_transform_matches_independent_window_sum_and_sg_edges(self):
        fs=256.;n=768;t=np.arange(n)/fs
        y=np.exp(-t/.4)*np.cos(2*np.pi*40.3*t+.7)+.002*t*t
        p=WindowedDecayOperator(fs,n,{'start_s':.015625,'end_s':2.9,'sg_window':41},[39.,41.],.25,.0625)
        v=(y-savgol_filter(y,41,2,mode='mirror'))[4:int(np.ceil(2.9*fs))]
        v-=v.mean()
        window=.5-.5*np.cos(2*np.pi*np.arange(64)/64)
        expected=[]
        for start in range(0,len(v)-64+1,16):
            for f in [39.,41.]:
                expected.append(2*np.sum(v[start:start+64]*window*np.exp(-2j*np.pi*f*(4+start+np.arange(64))/fs))/window.sum())
        np.testing.assert_allclose(p.transform(y),expected,atol=2e-13)
        basis=p.templates([40.3],[1.],1/.4)
        model=np.exp(-t/.4)*np.cos(2*np.pi*40.3*t+.7)
        np.testing.assert_allclose(basis@np.array([np.cos(.7),-np.sin(.7)]),p.transform(model),atol=1e-13)
        delayed=np.exp(-t/.4)*np.cos(2*np.pi*40.3*(t-.023)+.7)
        basis=p.templates([40.3],[1.],1/.4,phase_delay_s=.023)
        np.testing.assert_allclose(basis@np.array([np.cos(.7),-np.sin(.7)]),p.transform(delayed),atol=1e-13)

    def test_fast_decay_phase_recovery_with_window_longer_than_decay(self):
        fs=256.;n=768;t=np.arange(n)/fs
        p=WindowedDecayOperator(fs,n,{'start_s':.03125,'end_s':1.5,'sg_window':31},[38.,40.,42.],.25,.0625)
        y=1.4*np.exp(-t/.12)*np.cos(2*np.pi*40.2*t+.9)
        result=fit_windowed_modes(p,y,[35,45],[.05,1.],[40.],starts=2,max_seconds=20)
        self.assertAlmostEqual(result['frequencies_hz'][0],40.2,places=4)
        self.assertAlmostEqual(result['t2star_s'][0],.12,places=5)
        self.assertAlmostEqual(result['phases_rad'][0],.9,places=4)
        self.assertNotIn('native_fft_spacing_hz',result['numerical_diagnostics']['thresholds'])

    def test_beating_does_not_require_distinct_decay_times(self):
        fs=256.;n=768;t=np.arange(n)/fs
        p=WindowedDecayOperator(fs,n,{'start_s':.03125},[38.,40.,42.],.25,.0625)
        y=np.exp(-t/.65)*(np.cos(2*np.pi*39.2*t+.2)+.8*np.cos(2*np.pi*41.1*t-.7))
        r=fit_windowed_modes(p,y,[35,45],[.1,2.],[39.,41.],shared_decay=True,starts=2,max_seconds=20)
        np.testing.assert_allclose(r['frequencies_hz'],[39.2,41.1],atol=1e-4)
        np.testing.assert_allclose(r['t2star_s'],[.65,.65],atol=1e-4)

    def test_invalid_windows_and_memory_budget(self):
        for width,hop in [(0.,.1),(.1,0.),(4.,.1)]:
            with self.assertRaises(ValueError):
                WindowedDecayOperator(256.,768,{},[40.],width,hop)
        with self.assertRaisesRegex(ValueError,'two million'):
            WindowedDecayOperator(4000.,65516,{},[120.,130.,140.],1.,.001)


if __name__=='__main__':
    unittest.main()
