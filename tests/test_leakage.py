import unittest
import numpy as np
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes


class LeakageTests(unittest.TestCase):
    def fixture(self, outside):
        fs = 256.; n = 2048; first = 16; last = 1600
        t = np.arange(n)/fs
        y = (np.exp(-t/1.2)*np.cos(2*np.pi*40.1*t+.4)
             + 3*np.exp(-t/.25)*np.cos(2*np.pi*outside*t-1.))
        frequency = np.fft.rfftfreq(last-first, 1/fs)
        transformed = np.fft.rfft(y[first:last])/(last-first)
        def fit(band, seeds, background=False):
            bins = np.flatnonzero((frequency >= band[0]) & (frequency <= band[1]))
            processor = ProcessedSpectrum(fs, n, first, last, bins)
            return fit_modes(processor, transformed[bins], band, [.1, 3.],
                             mode_count=len(seeds), initial_frequencies=seeds,
                             background=background, starts=4, max_seconds=10)
        return fit

    def test_nearby_outside_peak_bias_and_explicit_guard_recovery(self):
        fit = self.fixture(42.3)
        omitted = fit([38, 42], [40.])
        self.assertGreater(abs(omitted['t2star_s'][0]-1.2), .2)
        guarded = fit([38, 46], [40., 42.5])
        np.testing.assert_allclose(guarded['frequencies_hz'], [40.1, 42.3], atol=1e-5)
        np.testing.assert_allclose(guarded['t2star_s'], [1.2, .25], atol=1e-5)
        self.assertLess(guarded['relative_complex_residual'], 1e-8)

    def test_constant_background_can_reduce_error_but_worsen_decay_bias(self):
        fit = self.fixture(44.)
        omitted = fit([38, 42], [40.])
        background = fit([38, 42], [40.], True)
        self.assertLess(background['relative_complex_residual'], .5*omitted['relative_complex_residual'])
        self.assertGreater(abs(background['t2star_s'][0]-1.2), 10*abs(omitted['t2star_s'][0]-1.2))
        guarded = fit([38, 46], [40., 44.2])
        np.testing.assert_allclose(guarded['t2star_s'], [1.2, .25], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
