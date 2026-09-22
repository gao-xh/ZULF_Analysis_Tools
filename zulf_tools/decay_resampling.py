"""Discovery-group resampling with explicit draws and conditional summaries."""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from matplotlib.figure import Figure
from . import storage
from .repeats import load_group_averages
from .jfit import ProcessedSpectrum
from .decay import fit_modes


def circular_group_draws(group_count, draws, block_length=1, seed=0):
    if type(group_count) is not int or group_count<2 or type(draws) is not int or not 1<=draws<=500:
        raise ValueError('Require at least two groups and 1..500 draws.')
    if type(block_length) is not int or not 1<=block_length<=group_count//2:
        raise ValueError('Block length must be 1..floor(group count/2).')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,group_count,size=(draws,int(np.ceil(group_count/block_length))))
    return ((starts[:,:,None]+np.arange(block_length))%group_count).reshape(draws,-1)[:,:group_count]


def conditional_quantiles(rows, requested):
    clean=[r for r in rows if r['eligible_for_summary']]
    ready=len(rows)==requested and len(clean)>=20 and len(clean)/requested>=.9
    return dict(requested_draws=requested,completed_draws=len(rows),clean_draws=len(clean),
        status='conditional_descriptive_percentiles' if ready else 'insufficient_or_unstable_resampling',
        t2star_percentiles_s=np.quantile([r['matched_t2star_s'] for r in clean],[.025,.5,.975],axis=0).tolist() if ready else None,
        percentile_levels=[.025,.5,.975],
        policy='Require all requested draws, at least 20 numerically clean draws and >=90% clean. Heuristic safeguard, not coverage calibration.')


def resample_decay_groups(fit_run_id, candidate_index=0, draws=100, block_length=1,
                           settings=None, *, record, directory, cancel, progress):
    from .analysis import recipe
    started=time.perf_counter();parent=storage.get_result(fit_run_id)
    if parent['operation']!='fit_frequency_decay' or type(candidate_index) is not int or not 0<=candidate_index<len(parent['candidates']):
        raise ValueError('Require a valid frequency-decay candidate.')
    candidate=parent['candidates'][candidate_index]
    means,counts,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']:
        raise ValueError('Group source differs from parent fit.')
    groups=parent['discovery_groups']
    config=dict(starts=2,max_nfev=150,max_evaluations=1500,max_seconds=10.,total_seconds=240.,seed=20260922)
    if settings and set(settings)-set(config): raise ValueError('Unknown resampling settings.')
    config.update(settings or {})
    if not np.isfinite(config['total_seconds']) or config['total_seconds']<=0: raise ValueError('Total budget must be positive.')
    selections=circular_group_draws(len(groups),draws,block_length,config['seed'])
    fs=source['sampling_rate_hz'];spec=parent['parameters'].get('preprocessing') or {}
    spectra=[]
    for g in groups:
        if cancel(): raise InterruptedError('Group resampling cancelled.')
        y,_,params=recipe(means[g],fs,spec);spectra.append(np.fft.rfft(y)/len(y))
    frequency=np.fft.rfftfreq(len(y),1/fs);lo,hi=candidate['range_hz']
    bins=np.flatnonzero((frequency>=lo)&(frequency<=hi));spectra=np.array(spectra)[:,bins]
    p=ProcessedSpectrum(fs,source['points'],params['start_sample'],params['stop_sample'],bins,params['sg_window'],params['sg_order'])
    reference=np.array(candidate['frequencies_hz']);rows=[];halted=False
    selected_counts=counts[groups]
    for i,selection in enumerate(selections):
        if cancel(): raise InterruptedError('Group resampling cancelled.')
        remaining=config['total_seconds']-(time.perf_counter()-started)
        if remaining<=0: halted=True;break
        observed=np.average(spectra[selection],axis=0,weights=selected_counts[selection])
        fit=fit_modes(p,observed,candidate['range_hz'],parent['t2_bounds_s'],mode_count=candidate['mode_count'],
            shared_decay=candidate['shared_decay'],initial_frequencies=reference,background=parent['settings']['background'],
            starts=config['starts'],max_nfev=config['max_nfev'],max_evaluations=config['max_evaluations'],
            max_seconds=min(config['max_seconds'],remaining),seed=config['seed'],cancel=cancel)
        fit.pop('fitted')
        _,match=linear_sum_assignment(abs(reference[:,None]-np.array(fit['frequencies_hz'])[None,:]))
        shift=np.array(fit['frequencies_hz'])[match]-reference
        fit.update(draw_index=i,group_indices=[groups[j] for j in selection],
            matched_t2star_s=np.array(fit['t2star_s'])[match].tolist(),frequency_shift_hz=shift.tolist(),
            eligible_for_summary=bool(not fit['numerical_diagnostics']['requires_review'] and np.all(abs(shift)<=2*fs/len(y))))
        rows.append(fit);storage.write_json(directory/'resampling_fits.json',rows);progress(len(rows),draws)
    summary=conditional_quantiles(rows,draws)
    # Show all retained estimates, marking excluded draws instead of hiding them.
    if rows:
        eligible=np.array([r['eligible_for_summary'] for r in rows]);x=np.arange(len(rows))
        for mode,f in enumerate(reference):
            values=np.array([r['matched_t2star_s'][mode] for r in rows])
            fig=Figure(figsize=(10,4.6),layout='constrained');ax=fig.add_subplot(111)
            ax.scatter(x[eligible],values[eligible],s=12,label='Numerically clean, matched')
            ax.scatter(x[~eligible],values[~eligible],s=24,marker='x',label='Flagged / mode shift')
            ax.axhline(candidate['t2star_s'][mode],color='black',linestyle='--',label='Discovery fit')
            ax.set(xlabel='Resampling draw',ylabel='Effective T2* candidate (s)',title=f'{f:.3f} Hz: discovery-group resampling, block length {block_length}')
            ax.legend(fontsize=8);fig.savefig(directory/f'mode_{mode}_resampling.png',dpi=150)
    storage.write_json(directory/'draw_plan.json',[[groups[j] for j in row] for row in selections])
    return dict(parent_run_id=fit_run_id,candidate_index=candidate_index,discovery_groups=groups,
        untouched_validation_groups=parent['validation_groups'],block_length=block_length,settings=config,
        source_arrays_sha256=source['arrays_sha256'],summary=summary,total_budget_exhausted=halted,
        elapsed_s=time.perf_counter()-started,scientifically_validated=False,
        warnings=['Resamples group means, not individual acquisitions; scan counts weight each sampled copy.',
        'Circular blocks use the parent discovery-list order, which need not equal chronological batches.',
        'Block length 1 assumes exchangeable groups; longer blocks assume local dependence in that list.',
        'Percentiles are conditional on the model, preprocessing, group split and clean mode matching, not calibrated confidence intervals.',
        'Numerical failures and mode shifts are retained; validation data are not resampled.',
        'No coverage claim for few groups, drift, model error or signal/model selection uncertainty.'])
