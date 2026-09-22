import unittest
import numpy as np
from zulf_tools.decay_resampling import circular_group_draws, conditional_quantiles


class ResamplingTests(unittest.TestCase):
    def test_reproducible_circular_blocks_preserve_local_pairs(self):
        draws=circular_group_draws(8,200,2,seed=12)
        np.testing.assert_array_equal(draws,circular_group_draws(8,200,2,seed=12))
        self.assertEqual(draws.shape,(200,8))
        np.testing.assert_array_equal((draws[:,1::2]-draws[:,::2])%8,np.ones((200,4)))
        self.assertTrue(np.all(np.bincount(draws.ravel(),minlength=8)>100))
        self.assertTrue(any(len(set(row))<8 for row in draws))
        with self.assertRaises(ValueError): circular_group_draws(8,100,8)

    def test_failed_branches_prevent_deceptively_narrow_summary(self):
        clean=[dict(eligible_for_summary=True,matched_t2star_s=[1.]) for _ in range(40)]
        unstable=[dict(eligible_for_summary=False,matched_t2star_s=[100.]) for _ in range(10)]
        self.assertIsNone(conditional_quantiles(clean+unstable,50)['t2star_percentiles_s'])
        self.assertIsNone(conditional_quantiles(clean,100)['t2star_percentiles_s'])
        self.assertEqual(conditional_quantiles(clean,40)['t2star_percentiles_s'],[[1.],[1.],[1.]])


if __name__=='__main__': unittest.main()
