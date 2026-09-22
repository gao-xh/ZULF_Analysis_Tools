"""Reproduce out-of-band leakage and imperfect constant-background correction.

Run as python -m examples.validate_leakage. No experimental input is used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from zulf_tools import storage
from zulf_tools.analysis import plot
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes


def main():
    parameters = dict(fs=256., points=2048, first=16, last=1600,
        target_frequency_hz=40.1, target_t2star_s=1.2, outside_t2star_s=.25,
        outside_amplitude=3., outside_frequencies_hz=[42.3, 44.],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record, directory = storage.begin('synthetic_leakage_validation', parameters)
    fs = parameters['fs']; n = parameters['points']; first = parameters['first']; last = parameters['last']
    t = np.arange(n)/fs; f = np.fft.rfftfreq(last-first, 1/fs)
    arrays = dict(time_s=t, frequency_hz=f); rows = []
    for outside in parameters['outside_frequencies_hz']:
        target = np.exp(-t/1.2)*np.cos(2*np.pi*40.1*t+.4)
        contaminant = 3*np.exp(-t/.25)*np.cos(2*np.pi*outside*t-1.)
        observed = np.fft.rfft((target+contaminant)[first:last])/(last-first)
        target_spectrum = np.fft.rfft(target[first:last])/(last-first)
        leak_spectrum = np.fft.rfft(contaminant[first:last])/(last-first)
        prefix = 'outside_'+str(outside).replace('.', '_')
        arrays[prefix+'_fid'] = target+contaminant
        for label, band, seeds, bg in [('omitted', [38, 42], [40.], False),
                                     ('constant_background', [38, 42], [40.], True),
                                     ('explicit_guard', [38, 46], [40., outside+.2], False)]:
            bins = np.flatnonzero((f >= band[0]) & (f <= band[1]))
            p = ProcessedSpectrum(fs, n, first, last, bins)
            r = fit_modes(p, observed[bins], band, [.1, 3.], mode_count=len(seeds),
                          initial_frequencies=seeds, background=bg, starts=4, max_seconds=10)
            prediction = r.pop('fitted'); r.update(scenario=prefix+'_'+label, range_hz=band,
                outside_frequency_hz=outside, true_target_t2star_s=1.2)
            rows.append(r); arrays[r['scenario']+'_prediction'] = prediction
            plot(directory/(r['scenario']+'_real.png'),
                 [(p.f, observed[bins].real, 'Combined observation'), (p.f, prediction.real, 'Fit'),
                  (p.f, target_spectrum[bins].real, 'True target alone')],
                 'Frequency (Hz)', 'Real FFT / retained length', r['scenario'].replace('_', ' '))
        mask = (f >= 38) & (f <= 42)
        plot(directory/(prefix+'_leakage.png'),
             [(f[mask], abs(target_spectrum[mask]), 'True target'),
              (f[mask], abs(leak_spectrum[mask]), 'Out-of-band oscillator contribution')],
             'Frequency (Hz)', 'Magnitude', 'Finite-record tails enter the selected band')
    np.savez_compressed(directory/'synthetic_arrays.npz', **arrays)
    result = storage.complete(record, scenarios=rows,
        interpretation='Frequency restriction does not remove finite-record contributions of omitted oscillators. A constant complex background can reduce residual and increase decay bias. Explicit guard fitting recovers this known two-mode example, not arbitrary unknown experiments. No automatic band expansion or physical acceptance.')
    print(json.dumps(dict(run_id=result['run_id'], scenarios=[dict(
        name=r['scenario'], t2star_s=r['t2star_s'], residual=r['relative_complex_residual']) for r in rows])))


if __name__ == '__main__':
    main()
