import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.jfit import ProcessedSpectrum


class TemplateMemoryTests(unittest.TestCase):
    def reference(self,fs,n,first,last,bins,window,order,f,w,rate,delay):
        t=np.arange(n)/fs
        wave=np.exp((-rate+2j*np.pi*np.asarray(f)[None,:])*t[:,None])@(np.asarray(w)*np.exp(-2j*np.pi*np.asarray(f)*delay))/sum(w)
        raw=np.column_stack([wave.real,wave.imag])
        if window:raw-=savgol_filter(raw,window,order,axis=0,mode='mirror')
        return np.fft.rfft(raw[first:last],axis=0)[bins]/(last-first)

    def test_large_mirror_map_is_not_allocated(self):
        p=ProcessedSpectrum(4000.,65516,0,65516,np.arange(1,100),60001,2)
        self.assertTrue(p.sampled_backend)
        self.assertGreater(p.estimated_dense_work_bytes,100*1024**3)
        self.assertIsNone(p.edge_fft)
        self.assertEqual(p.edge_inverse.size,0)

    def test_sampled_path_matches_independent_mirror_filter_and_delay(self):
        fs=256.;n=2048;first=20;last=2000;bins=np.arange(1,(last-first)//2,4)
        f=[37.2,69.3];w=[.3,.7];rate=2.;delay=.013
        p=ProcessedSpectrum(fs,n,first,last,bins,1001,2)
        self.assertTrue(p.sampled_backend)
        expected=self.reference(fs,n,first,last,bins,1001,2,f,w,rate,delay)
        np.testing.assert_allclose(p.templates(f,w,rate,delay),expected,atol=2e-12,rtol=2e-9)
        self.assertEqual(p.sampled_template_calls,1)

    def test_fast_decay_avoids_analytic_edge_cancellation(self):
        fs=1000.;n=2048;first=0;last=n;bins=np.arange(1,600)
        f=[123.4];w=[1.];rate=1000.;window=101
        p=ProcessedSpectrum(fs,n,first,last,bins,window,2)
        self.assertFalse(p.sampled_backend)
        actual=p.templates(f,w,rate)
        expected=self.reference(fs,n,first,last,bins,window,2,f,w,rate,0.)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-10)
        self.assertEqual(p.sampled_template_calls,1)

    def test_small_analytic_path_and_forced_sampled_path_agree(self):
        fs=512.;n=4096;bins=np.arange(1,800,3);f=[42.3,83.7];w=[1.,.4]
        p=ProcessedSpectrum(fs,n,16,4000,bins,101,2)
        analytic=p.templates(f,w,1.3,.007)
        self.assertEqual(p.analytic_template_calls,1)
        with patch.object(p,'dense_work_limit_bytes',1):sampled=p.templates(f,w,1.3,.007)
        np.testing.assert_allclose(sampled,analytic,atol=2e-12,rtol=2e-9)


if __name__=='__main__':
    unittest.main()
