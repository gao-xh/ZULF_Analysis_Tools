"""Same-frequency multi-exponential recovery and finite-noise ambiguity.

Run from the project root: python -m examples.validate_coincident_decays.
Synthetic evidence only; no experimental data read or physical assignments.
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
    parameters = dict(fs_hz=256., points=2048, first=16, last=1600,
                      frequency_hz=40.25, phase_rad=.4, amplitudes=[.6, 1.],
                      t2_bounds_s=[.1, 3.], noise_sd=.025, seed=381,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record, directory = storage.begin('synthetic_coincident_decay_validation', parameters)
    fs = parameters['fs_hz']; n = parameters['points']; t = np.arange(n)/fs
    first, last = parameters['first'], parameters['last']
    f = np.fft.rfftfreq(last-first, 1/fs)
    bins = np.flatnonzero((f >= 35) & (f <= 45))
    op = ProcessedSpectrum(fs, n, first, last, bins)
    def spectrum(y):
        return np.fft.rfft(y[first:last])[bins]/(last-first)
    noise = np.random.default_rng(parameters['seed']).normal(0, parameters['noise_sd'], n)
    band_noise = spectrum(noise)
    arrays = dict(time_s=t, noise=noise, frequency_hz=op.f)
    scenarios = []
    for name, taus in [('separated', [.25, 1.4]), ('close', [.9, 1.1])]:
        envelope = sum(a*np.exp(-t/tau) for a, tau in zip(parameters['amplitudes'], taus))
        y = envelope*np.cos(2*np.pi*parameters['frequency_hz']*t+parameters['phase_rad'])
        clean = spectrum(y)
        arrays[name+'_clean_fid'] = y
        for label, observed, modes in [('clean_single', clean, 1), ('clean_double', clean, 2),
                                      ('noisy_single', clean+band_noise, 1),
                                      ('noisy_double', clean+band_noise, 2)]:
            fit = fit_modes(op, observed, [35, 45], parameters['t2_bounds_s'],
                            mode_count=modes, initial_frequencies=[40.25] if modes == 1 else [40.2, 40.3],
                            starts=6, max_evaluations=5000, max_seconds=10)
            prediction = fit.pop('fitted')
            fit.update(scenario=name+'_'+label, true_t2star_s=taus,
                       clean_model_difference_to_noise_norm=float(np.linalg.norm(clean-prediction)/np.linalg.norm(band_noise)),
                       physically_identified=False)
            scenarios.append(fit)
            arrays[fit['scenario']+'_prediction'] = prediction
            plot(directory/(fit['scenario']+'_real.png'),
                 [(op.f, observed.real, 'Synthetic observation'), (op.f, prediction.real, 'Fit')],
                 'Frequency (Hz)', 'Real spectrum', fit['scenario'].replace('_', ' '))
            plot(directory/(fit['scenario']+'_residual.png'),
                 [(op.f, abs(clean-prediction), 'Difference from clean truth'),
                  (op.f, abs(band_noise), 'Independent time-noise realization')],
                 'Frequency (Hz)', 'Magnitude', 'Descriptive comparison, not a significance test')
    np.savez_compressed(directory/'synthetic_arrays.npz', **arrays)
    result = storage.complete(record, scenarios=scenarios,
        interpretation='Exact noiseless recovery does not establish finite-noise identifiability. Close same-frequency decays may be indistinguishable from one effective decay. Frequency sorting/matching cannot label coincident modes; compare unordered decay sets. No significance or coverage claim.')
    print(json.dumps(dict(run_id=result['run_id'], scenarios=[{
        k: row[k] for k in ['scenario', 'true_t2star_s', 't2star_s',
                           'clean_model_difference_to_noise_norm', 'numerical_diagnostics']}
        for row in scenarios])))


if __name__ == '__main__':
    main()
