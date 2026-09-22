import unittest
from unittest.mock import patch
import numpy as np
from zulf_tools.decay import ModeTemplateCache, fit_modes, mode_design
from zulf_tools.jfit import ProcessedSpectrum


class ModeCacheTests(unittest.TestCase):
    def test_exact_keys_eviction_and_oversized_templates(self):
        class Processor:
            def templates(self,f,w,rate):
                return np.full((8,2),f[0]+1j*rate)
        cache=ModeTemplateCache(Processor())
        cache.byte_limit=512
        a=cache.get(40.,1.)
        self.assertIs(cache.get(40.,1.),a)
        cache.get(np.nextafter(40.,41.),1.)
        self.assertEqual(cache.misses,2)  # no rounding of derivative trials
        cache.get(42.,1.)
        self.assertNotIn((40.,1.),cache.entries)
        self.assertLessEqual(cache.peak_bytes,cache.byte_limit)
        cache.byte_limit=0
        before=cache.misses
        cache.get(50.,1.);cache.get(50.,1.)
        self.assertEqual(cache.misses,before+2)
        self.assertNotIn((50.,1.),cache.entries)

    def test_complete_search_is_identical_without_cache(self):
        fs=256.;n=2048;t=np.arange(n)/fs
        y=np.exp(-t/.6)*np.cos(2*np.pi*39.2*t+.2)+.7*np.exp(-t/1.3)*np.cos(2*np.pi*41.1*t-.5)
        f=np.fft.rfftfreq(n,1/fs);bins=np.flatnonzero((f>=35)&(f<=45))
        p=ProcessedSpectrum(fs,n,0,n,bins)
        observed=np.fft.rfft(y)[bins]/n
        kwargs=dict(mode_count=2,starts=3,max_seconds=30,initial_frequencies=[39.,41.])
        cached=fit_modes(p,observed,[35,45],[.1,3.],**kwargs)
        with patch.object(ModeTemplateCache,'byte_limit',0):
            uncached=fit_modes(p,observed,[35,45],[.1,3.],**kwargs)
        self.assertFalse(cached['budget_exhausted'])
        self.assertFalse(uncached['budget_exhausted'])
        np.testing.assert_array_equal(cached['fitted'],uncached['fitted'])
        self.assertEqual(cached['evaluations'],uncached['evaluations'])
        self.assertGreater(cached['template_cache']['hits'],0)
        self.assertEqual(uncached['template_cache']['hits'],0)
        self.assertEqual(cached['template_cache']['hits']+cached['template_cache']['misses'],2*cached['evaluations'])
        np.testing.assert_allclose(cached['t2star_s'],[.6,1.3],atol=1e-5)


if __name__=='__main__':unittest.main()
