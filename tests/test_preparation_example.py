import unittest
import numpy as np
from examples.validate_preparation import prepared_response
from zulf_tools import jfit


class PreparationTests(unittest.TestCase):
    def test_unpulsed_full_space_matches_collective_thermal_response(self):
        t=np.linspace(0,.15,47)
        for kind in ['methine','methyl']:
            raw,details=prepared_response(jfit.DEFAULT,kind,t,0.,0.)
            f,w=jfit.transitions(jfit.DEFAULT,kind)
            expected=np.cos(2*np.pi*t[:,None]*f)@w/w.sum()
            np.testing.assert_allclose(raw,expected,atol=1e-8,rtol=1e-8)
            self.assertLess(details['max_direct_propagation_error'],1e-7)

    def test_finite_pulse_complex_weights_match_direct_matrix_propagation(self):
        t=np.linspace(0,.15,47)
        raw,details=prepared_response(jfit.DEFAULT,'methyl',t,10.,.005)
        self.assertTrue(np.isfinite(raw).all())
        self.assertLess(details['max_direct_propagation_error'],1e-7)
        self.assertGreater(details['retained_transition_entries'],0)
        self.assertIn('excluded from both models',details['note'])


if __name__=='__main__':
    unittest.main()
