import unittest
import numpy as np
from zulf_tools.decay_bounds import noise_aware_t2_proposal


class BoundsTests(unittest.TestCase):
    def fixture(self):
        fs=256;t=np.arange(2048)/fs;rng=np.random.default_rng(947)
        y=np.exp(-t/.6)*np.cos(2*np.pi*40*t)
        return fs,t,y,rng.normal(0,.03,(8,len(t)))

    def test_repeat_noise_changes_horizon_without_claiming_measured_tau(self):
        fs,t,y,noise=self.fixture()
        noisy=noise_aware_t2_proposal(y+noise,np.ones(8),fs,[[35,45]])
        quiet=noise_aware_t2_proposal(y+noise*.001,np.ones(8),fs,[[35,45]])
        self.assertLess(noisy['t2_bounds_s'][1],quiet['t2_bounds_s'][1])
        self.assertLess(noisy['t2_bounds_s'][0],.6);self.assertGreater(noisy['t2_bounds_s'][1],.6)
        self.assertEqual(noisy['full_record_guard_bounds_s'],[.016,16.])

    def test_no_signal_or_single_repeat_preserves_full_record_guard(self):
        fs,t,y,noise=self.fixture()
        # Opposite groups have exactly zero coherent mean, not a missing tail.
        out=noise_aware_t2_proposal(np.array([noise[0],-noise[0]]),[1,1],fs,[[35,45]])
        self.assertEqual(out['t2_bounds_s'],out['full_record_guard_bounds_s'])
        out=noise_aware_t2_proposal(y[None,:],[1],fs,[[35,45]])
        self.assertEqual(out['bands'][0]['status'],'insufficient_repeat_or_window_information')

    def test_late_revival_is_retained_and_time_origin_only_changes_labels(self):
        fs,t,y,noise=self.fixture();y=y+((t>5)&(t<6))*np.cos(2*np.pi*40*t)
        a=noise_aware_t2_proposal(y+noise,np.ones(8),fs,[[35,45]])
        b=noise_aware_t2_proposal(y+noise,np.ones(8),fs,[[35,45]],.2)
        self.assertGreater(a['t2_bounds_s'][1],12.)
        self.assertEqual(a['t2_bounds_s'],b['t2_bounds_s'])
        np.testing.assert_allclose(np.array(b['bands'][0]['times_s'])-a['bands'][0]['times_s'],.2)


if __name__=='__main__':unittest.main()
