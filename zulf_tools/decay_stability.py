"""Bounded group-mean refits, separate from frozen held-out predictions."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import storage
from .repeats import load_group_averages
from .band_relaxation import _indices
from .jfit import ProcessedSpectrum
from .decay import fit_modes, mode_design


def inspect_decay_stability(fit_run_id, candidate_index=0, group_indices=None,
                            settings=None, *, record, directory, cancel, progress):
    from .analysis import recipe, plot
    started=time.perf_counter()
    parent=storage.get_result(fit_run_id)
    if parent['operation']!='fit_frequency_decay':
        raise ValueError('Expected a completed frequency-decay fit.')
    if type(candidate_index) is not int or not 0<=candidate_index<len(parent['candidates']):
        raise ValueError('Invalid candidate index.')
    candidate=parent['candidates'][candidate_index]
    means,counts,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']:
        raise ValueError('Group source does not match parent fit.')
    groups=_indices(parent['validation_groups'] if group_indices is None else group_indices,len(means),'group_indices')
    config=dict(starts=3,max_nfev=150,max_evaluations=3000,max_seconds=20.,total_seconds=180.,seed=20260922)
    if settings and set(settings)-set(config):
        raise ValueError('Unknown stability settings.')
    config.update(settings or {})
    if not np.isfinite(config['total_seconds']) or config['total_seconds']<=0:
        raise ValueError('Total budget must be positive.')
    fs=source['sampling_rate_hz'];spec=parent['parameters'].get('preprocessing') or {}
    rows=[];arrays={};processor=None;halted=False
    reference=np.array(candidate['frequencies_hz'])
    for g in groups:
        if cancel():
            raise InterruptedError('Stability inspection cancelled.')
        remaining=config['total_seconds']-(time.perf_counter()-started)
        if remaining<=0:
            halted=True;break
        y,_,params=recipe(means[g],fs,spec)
        if processor is None:
            f=np.fft.rfftfreq(len(y),1/fs);lo,hi=candidate['range_hz']
            bins=np.flatnonzero((f>=lo)&(f<=hi))
            processor=ProcessedSpectrum(fs,source['points'],params['start_sample'],params['stop_sample'],bins,params['sg_window'],params['sg_order'])
            design=mode_design(processor,reference,candidate['t2star_s'],parent['settings']['background'])
            coef=np.r_[np.asarray(candidate['cos_sin_coefficients']).ravel(),candidate['background_coefficients']]
            frozen=design@coef
            arrays['frequency_hz']=processor.f;arrays['frozen_discovery_prediction']=frozen
        observed=np.fft.rfft(y)[bins]/len(y)
        fit=fit_modes(processor,observed,candidate['range_hz'],parent['t2_bounds_s'],
            mode_count=candidate['mode_count'],shared_decay=candidate['shared_decay'],initial_frequencies=reference,
            background=parent['settings']['background'],starts=config['starts'],max_nfev=config['max_nfev'],
            max_evaluations=config['max_evaluations'],max_seconds=min(config['max_seconds'],remaining),seed=config['seed'],cancel=cancel)
        prediction=fit.pop('fitted')
        _,match=linear_sum_assignment(abs(reference[:,None]-np.array(fit['frequencies_hz'])[None,:]))
        shift=np.array(fit['frequencies_hz'])[match]-reference
        norm=float(np.linalg.norm(observed))
        fit.update(group_index=g,scan_count=int(counts[g]),
            group_role='discovery' if g in parent['discovery_groups'] else 'validation' if g in parent['validation_groups'] else 'other',
            frozen_relative_complex_residual=float(np.linalg.norm(observed-frozen)/norm) if norm else None,
            matched_mode_indices=match.tolist(),frequency_shift_hz=shift.tolist(),
            matched_t2star_s=np.array(fit['t2star_s'])[match].tolist(),
            phase_shift_rad=np.angle(np.exp(1j*(np.array(fit['phases_rad'])[match]-candidate['phases_rad']))).tolist(),
            large_frequency_shift=bool(np.any(abs(shift)>2*fs/len(y))))
        rows.append(fit)
        arrays[f'group_{g}_observed']=observed;arrays[f'group_{g}_refitted']=prediction
        storage.write_json(directory/'group_fits.json',rows)
        progress(len(rows),len(groups))
    if rows:
        x=np.array([r['group_index'] for r in rows])
        for mode in range(len(reference)):
            for key,label in [('matched_t2star_s','Effective T2* candidate (s)'),('frequency_shift_hz','Frequency shift from discovery (Hz)')]:
                plot(directory/f'mode_{mode}_{key}.png',[(x,[r[key][mode] for r in rows],'Group refit')],
                    'Acquisition group index',label,f'{reference[mode]:.3f} Hz: diagnostic group refits; mode identity unverified')
        plot(directory/'group_errors.png',[(x,[r['frozen_relative_complex_residual'] for r in rows],'Frozen discovery prediction'),
            (x,[r['relative_complex_residual'] for r in rows],'Group refit (in-sample)')],
             'Acquisition group index','Relative complex residual','Frozen prediction and diagnostic refit errors')
    np.savez_compressed(directory/'stability_arrays.npz',**arrays)
    return dict(parent_run_id=fit_run_id,candidate_index=candidate_index,source_arrays_sha256=source['arrays_sha256'],
        requested_groups=groups,completed_groups=len(rows),total_budget_exhausted=halted,
        settings=config,group_fits=rows,elapsed_s=time.perf_counter()-started,scientifically_validated=False,
        warnings=['Group refits are in-sample diagnostics; frozen prediction errors remain separate.',
                  'Matching minimizes frequency distance only; it does not establish persistent mode identity.',
                  'Phase differences are raw fitted parameters, not reliable phase corrections or accepted signal phases.',
                  'Numerical flags and shifts exceeding two native FFT bins require review, not automatic rejection.',
                  'Group variation combines noise, model mismatch and drift; it is not a confidence interval.'])
