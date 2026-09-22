import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes, mode_design, real_projection, fit_diagnostics


class DecayTests(unittest.TestCase):
    def test_best_trial_cannot_borrow_another_starts_convergence(self):
        p,y=self.fixture([(40.,.8,1.,.2)])
        calls=[]
        def endpoint(fun,x,**kwargs):
            start=len(calls);calls.append(start)
            trial=np.array([40.,np.log(.8000001 if start==0 else .8)])
            residual=fun(trial)
            return SimpleNamespace(x=trial,fun=residual,success=start==0,nfev=1,message='Controlled endpoint')
        with patch('zulf_tools.decay.least_squares',side_effect=endpoint):
            r=fit_modes(p,y,[35,45],[.1,2.],starts=2)
        self.assertEqual(r['best_start_index'],1)
        self.assertLess(abs(r['candidates'][0]['score']-r['score']),1e-12)
        self.assertFalse(r['optimizer_converged'])
        self.assertIn('optimizer_not_converged',r['numerical_diagnostics']['numerical_warning_flags'])
        self.assertEqual([c['start_index'] for c in r['candidates']],[0,1])

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
        self.assertEqual(limited['completed_starts'],0)
        self.assertEqual(limited['attempted_starts'],1)
        self.assertEqual(limited['start_attempts'][0]['status'],'budget_exhausted')
        self.assertEqual(limited['start_attempts'][0]['stop_reason'],'max_evaluations')
        self.assertEqual(limited['start_attempts'][0]['evaluations'],2)
        self.assertEqual(limited['best_start_index'],0)
        self.assertLessEqual(limited['best_evaluation'],2)
        result=fit_modes(p,y,[35,45],[.1,2.],starts=2)
        self.assertTrue(any('t2' in key for key in result['boundary_hits']))

    def test_start_ledger_reproduces_automatic_and_explicit_initialization(self):
        p,y=self.fixture([(40.17,.74,1.4,1.1)])
        auto=fit_modes(p,y,[35,45],[.1,3.],starts=4,seed=371)
        explicit=fit_modes(p,y,[35,45],[.1,3.],starts=4,seed=371,
                           initial_frequencies=auto['initialization']['seed_frequencies_hz'])
        self.assertEqual(auto['initialization']['source'],'discovery_spectrum')
        self.assertEqual(explicit['initialization']['source'],'explicit')
        self.assertEqual(auto['start_attempts'],explicit['start_attempts'])
        self.assertEqual(sum(a['evaluations'] for a in auto['start_attempts']),auto['evaluations'])
        self.assertEqual([a['strategy'] for a in auto['start_attempts']],
                         ['seed','local_perturbation','local_perturbation','uniform_band'])
        np.testing.assert_array_equal(auto['fitted'],explicit['fitted'])

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

    def test_duplicate_modes_fail_numerical_screen_despite_exact_prediction(self):
        p,y=self.fixture([(40.,.8,1.,.2)])
        design=mode_design(p,[40.,40.],[.8,.8])
        fitted,_,rank,condition=real_projection(design,y)
        np.testing.assert_allclose(fitted,y,atol=1e-12)
        diagnostic=fit_diagnostics(np.array([40.,40.]),np.array([.8,.8]),[35,45],[.1,3],
            shared_decay=False,native_spacing=p.fs/p.n,rank=rank,columns=4,
            condition=condition,converged=True,budget_exhausted=False)
        self.assertTrue(diagnostic['requires_review'])
        self.assertIn('rank_deficient_amplitude_design',diagnostic['numerical_warning_flags'])
        self.assertEqual(diagnostic['sub_bin_frequency_pairs'][0]['mode_indices'],[0,1])

    def test_same_frequency_distinct_decays_can_recover_without_noise(self):
        p, y = self.fixture([(40.25, .25, .6, .4), (40.25, 1.4, 1., .4)])
        result = fit_modes(p, y, [35, 45], [.1, 3.], mode_count=2,
                           initial_frequencies=[40.2, 40.3], starts=6,
                           max_evaluations=5000, max_seconds=15)
        np.testing.assert_allclose(result['frequencies_hz'], [40.25, 40.25], atol=1e-5)
        np.testing.assert_allclose(sorted(result['t2star_s']), [.25, 1.4], atol=1e-4)
        self.assertLess(result['relative_complex_residual'], 1e-7)
        self.assertTrue(result['numerical_diagnostics']['sub_bin_frequency_pairs'])
        # Frequency matching alone cannot label which decay is which here.

    def test_close_same_frequency_decays_can_hide_below_noise(self):
        p, y = self.fixture([(40.25, .9, .6, .4), (40.25, 1.1, 1., .4)])
        single = fit_modes(p, y, [35, 45], [.1, 3.], starts=3, max_seconds=10)
        self.assertTrue(.9 < single['t2star_s'][0] < 1.1)
        # Independently generated time noise, transformed through the same crop.
        noise = np.random.default_rng(381).normal(0, .025, 4096)[64:3600]
        frequency = np.fft.rfftfreq(len(noise), 1 / 512.)
        band_noise = np.fft.rfft(noise)[(frequency >= 35) & (frequency <= 45)] / len(noise)
        self.assertLess(np.linalg.norm(y-single['fitted']), .3*np.linalg.norm(band_noise))
        # This deterministic counterexample is not a significance/coverage test.

    def test_boundary_indices_follow_sorted_output_not_optimizer_order(self):
        p,y=self.fixture([(38.,.6,1.,.2),(43.,4.,1.,-.5)])
        result=fit_modes(p,y,[35,45],[.1,2.],mode_count=2,
                         initial_frequencies=[43.,38.],starts=3,max_seconds=20)
        self.assertLess(result['frequencies_hz'][0],result['frequencies_hz'][1])
        self.assertGreater(result['t2star_s'][1],1.99)
        self.assertIn('log_t2_1',result['boundary_hits'])
        self.assertNotIn('log_t2_0',result['boundary_hits'])
        self.assertIn('search_boundary',result['numerical_diagnostics']['numerical_warning_flags'])


if __name__=='__main__':
    unittest.main()
