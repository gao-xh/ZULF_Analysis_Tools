"""Full-Hilbert-space preparation counterexample, not an experimental pulse claim.

Run as python -m examples.validate_preparation. J and true decay stay fixed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize_scalar
from zulf_tools import storage, simulation, jfit
from zulf_tools.analysis import recipe, plot
from zulf_tools.transition_decay import fit_transition_decay
from zulf_tools.decay import real_projection


def prepared_response(parameters, isotopomer, times, field_ut, duration_s):
    """Return oscillatory response and DC separately, with direct propagation checks."""
    h, mag, _, _ = simulation.build(jfit.model_for(parameters, isotopomer))
    observable = mag[0]/(2*np.pi)
    pulse = expm(-1j*(h+field_ut*mag[2])*duration_s)
    initial = pulse@observable@pulse.conj().T
    energy, vectors = eigh(h/(2*np.pi), driver='evr')
    rho = vectors.conj().T@initial@vectors
    obs = vectors.conj().T@observable@vectors
    i, k = np.triu_indices(len(energy), 1)
    gaps = energy[k]-energy[i]
    weights = 2*rho[i, k]*obs[k, i]/len(energy)
    positive = gaps > 1e-7
    dc = float(np.trace(initial@observable).real/len(energy)-weights[positive].real.sum())
    keep = positive & (abs(weights) > max(abs(weights))*1e-12)
    f, w = gaps[keep], weights[keep]
    reference_scale = float(jfit.transitions(parameters, isotopomer)[1].sum())
    response = np.empty(len(times))
    for start in range(0, len(times), 128):
        response[start:start+128] = (np.exp(2j*np.pi*np.asarray(times[start:start+128])[:,None]*f)@w).real/reference_scale
    checks = []
    for t in [.001, .013, .027]:
        propagator = expm(-1j*h*t)
        direct = float(np.trace(propagator@initial@propagator.conj().T@observable).real/len(energy))
        spectral = float(dc+(np.exp(2j*np.pi*t*f)@w).real)
        checks.append(abs(direct-spectral)/reference_scale)
    return response, dict(dc_before_decay=dc/reference_scale,
        max_direct_propagation_error=max(checks), retained_transition_entries=len(f),
        maximum_frequency_hz=float(f.max()), normalization=reference_scale,
        preparation=dict(initial='gamma-weighted Ix', detection='gamma-weighted Ix',
                         pulse_axis='z', pulse_field_ut=field_ut, pulse_duration_s=duration_s),
        note='Only the nonzero-frequency response is used for decay comparison. The separately recorded DC contribution is deliberately excluded from both models, so this test isolates preparation-dependent transition weights.')


def main():
    parameters = dict(fs_hz=2000., points=4096, first=200, last=3900,
        sg_window=301, true_t2star_s=.8, j_hz=jfit.DEFAULT,
        hypotheses=[dict(name='unpulsed', field_ut=0., duration_s=0.),
                    dict(name='finite_pulse', field_ut=10., duration_s=.005)],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record, directory = storage.begin('synthetic_preparation_validation', parameters)
    fs=parameters['fs_hz'];n=parameters['points'];t=np.arange(n)/fs
    first=parameters['first'];last=parameters['last'];rows=[];arrays=dict(time_s=t)
    preprocessing=dict(start_s=first/fs,end_s=last/fs,sg_window=301,sg_order=2)
    for kind,band in [('methine',[110,150]),('methyl',[230,270])]:
        thermal_f,thermal_w=jfit.transitions(parameters['j_hz'],kind)
        for hypothesis in parameters['hypotheses']:
            name=kind+'_'+hypothesis['name']
            raw,details=prepared_response(parameters['j_hz'],kind,t,hypothesis['field_ut'],hypothesis['duration_s'])
            if details['maximum_frequency_hz'] >= fs/2:
                raise ValueError('Increase synthetic sample rate; retained transitions exceed Nyquist.')
            if details['max_direct_propagation_error'] > 1e-7:
                raise AssertionError('Independent propagation verification failed.')
            def transformed(tau):
                y,_,_=recipe(raw*np.exp(-t/tau),fs,preprocessing)
                return np.fft.rfft(y)/len(y)
            freq=np.fft.rfftfreq(last-first,1/fs);bins=np.flatnonzero((freq>=band[0])&(freq<=band[1]))
            observed=transformed(.8)[bins]
            op=jfit.ProcessedSpectrum(fs,n,first,last,bins,301)
            fit=fit_transition_decay(op,observed,[dict(name=kind,frequencies_hz=thermal_f.tolist(),weights=thermal_w.tolist())],
                [.1,3.],starts=8,max_seconds=20,phase_delay_bounds_s=[-.02,.02])
            prediction=fit.pop('fitted')
            def correct_cost(log_tau):
                predicted,_,_,_=real_projection(transformed(np.exp(log_tau))[bins,None],observed)
                return float(np.linalg.norm(observed-predicted)**2)
            correct=minimize_scalar(correct_cost,bounds=np.log([.1,3.]),method='bounded',options={'xatol':1e-9})
            matched,_,_,_=real_projection(transformed(np.exp(correct.x))[bins,None],observed)
            rows.append(dict(scenario=name,details=details,thermal_fit=fit,
                matched_preparation_t2star_s=float(np.exp(correct.x)),
                matched_preparation_residual=float(np.linalg.norm(observed-matched)/np.linalg.norm(observed))))
            arrays[name+'_undamped_oscillatory_fid']=raw;arrays[name+'_spectrum']=observed;arrays[name+'_thermal_prediction']=prediction
            for part,transform in [('real',np.real),('imaginary',np.imag)]:
                plot(directory/(name+'_'+part+'.png'),[(freq[bins],transform(observed),'Prepared synthetic observation'),
                    (freq[bins],transform(prediction),'Unpulsed weights with fitted decay and delay'),
                    (freq[bins],transform(matched),'Matched preparation model')],
                    'Frequency (Hz)',part.title()+' FFT amplitude',name.replace('_',' ')+': fixed true T2*=0.8 s')
    np.savez_compressed(directory/'synthetic_arrays.npz',**arrays)
    result=storage.complete(record,scenarios=rows,scientifically_validated=False,
        interpretation='A hypothetical finite pulse can change relative complex transition weights. This is not a claim about the experimental preparation. Unknown preparation cannot be repaired by assuming any successful linewidth, phase or J fit is correct. DC was excluded explicitly to isolate transition-weight effects.')
    print(json.dumps(dict(run_id=result['run_id'],scenarios=[dict(name=r['scenario'],thermal_t2star_s=r['thermal_fit']['t2star_s'],
        thermal_residual=r['thermal_fit']['relative_complex_residual'],matched_t2star_s=r['matched_preparation_t2star_s']) for r in rows])))


if __name__=='__main__':
    main()
