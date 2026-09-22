"""Provenance-checked evidence summaries; never automatic physical acceptance."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import storage
from .repeats import load_group_averages


def require_compatible(base, other, same_recipe=False):
    for key in ['source_arrays_sha256','discovery_groups','validation_groups']:
        if base.get(key)!=other.get(key) or key not in other:
            raise ValueError('Incompatible evidence: '+key)
    if same_recipe:
        a,b=base['preprocessing'],other['preprocessing']
        for key in ['start_sample','stop_sample','sg_window','sg_order']:
            if a[key]!=b[key]: raise ValueError('Incompatible preprocessing: '+key)
        if a.get('remove_mean',True)!=b.get('remove_mean',True):
            raise ValueError('Incompatible preprocessing: remove_mean')


def evidence_status(signal_supported, flags, unstable, missing, sensitive):
    if not signal_supported: return 'insufficient_signal_evidence'
    if flags: return 'numerically_unreliable'
    if unstable: return 'unstable_decay'
    if missing: return 'incomplete_evidence'
    if sensitive: return 'model_sensitive_candidate'
    return 'exploratory_supported_candidate'


def review_decay_evidence(fit_run_id, candidate_index=0, signal_run_ids=None,
                           stability_run_id=None, resampling_run_ids=None,
                           comparison_refs=None, relative_change_threshold=.1,
                           *, record, directory, cancel, progress):
    parent=storage.get_result(fit_run_id)
    if parent['operation']!='fit_frequency_decay' or type(candidate_index) is not int or not 0<=candidate_index<len(parent['candidates']):
        raise ValueError('Require a frequency-decay candidate.')
    _,_,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']: raise ValueError('Parent source changed.')
    if not np.isfinite(relative_change_threshold) or relative_change_threshold<=0: raise ValueError('Change threshold must be positive.')
    for values in [signal_run_ids or [],resampling_run_ids or []]:
        if not isinstance(values,list) or len(values)>16 or len(set(values))!=len(values):
            raise ValueError('Evidence IDs must be distinct lists of at most 16 runs.')
    candidate=parent['candidates'][candidate_index];freq=np.array(candidate['frequencies_hz'])
    n=parent['preprocessing']['stop_sample']-parent['preprocessing']['start_sample']
    tolerance=2*source['sampling_rate_hz']/n
    supports=[[] for _ in freq]
    for run in signal_run_ids or []:
        if cancel(): raise InterruptedError('Evidence review cancelled.')
        s=storage.get_result(run)
        if s['operation']!='inspect_repeat_signals': raise ValueError('Expected signal inspection evidence.')
        require_compatible(parent,s,True)
        peaks=[c for c in s['candidates'] if candidate['range_hz'][0]<=c['frequency_hz']<=candidate['range_hz'][1]]
        assigned={}
        if peaks:
            cost=abs(freq[:,None]-np.array([c['frequency_hz'] for c in peaks])[None,:])
            a,b=linear_sum_assignment(cost)
            assigned={int(i):dict(peak=peaks[j],distance_hz=float(cost[i,j])) for i,j in zip(a,b) if cost[i,j]<=tolerance}
        for i in range(len(freq)):
            match=assigned.get(i)
            supports[i].append(dict(run_id=run,matched=match is not None,
                classification=match['peak']['classification'] if match else 'unmatched',
                peak_frequency_hz=match['peak']['frequency_hz'] if match else None,
                distance_hz=match['distance_hz'] if match else None))
    stability=None
    if stability_run_id:
        stability=storage.get_result(stability_run_id)
        if stability['operation']!='inspect_decay_stability' or stability['parent_run_id']!=fit_run_id or stability['candidate_index']!=candidate_index:
            raise ValueError('Stability evidence belongs to another candidate.')
    samples=[]
    for run in resampling_run_ids or []:
        s=storage.get_result(run)
        if s['operation']!='resample_decay_groups' or s['parent_run_id']!=fit_run_id or s['candidate_index']!=candidate_index:
            raise ValueError('Resampling evidence belongs to another candidate.')
        samples.append(dict(run_id=run,summary=s['summary'],block_length=s['block_length']))
    comparisons=[[] for _ in freq]
    refs=comparison_refs or []
    if not isinstance(refs,list) or len(refs)>24: raise ValueError('Supply at most 24 explicit comparison references.')
    for ref in refs:
        other=storage.get_result(ref['run_id']);require_compatible(parent,other)
        if other['operation']=='fit_frequency_decay':
            idx=ref.get('candidate_index')
            if type(idx) is not int or not 0<=idx<len(other['candidates']): raise ValueError('Invalid comparison candidate.')
            c=other['candidates'][idx]
        elif other['operation'] in ('fit_window_decay','fit_demodulated_decay'): c=other['fit']
        else: raise ValueError('Comparison must be an FFT, window or demodulated decay fit.')
        if len(c['frequencies_hz'])!=len(freq): raise ValueError('Comparison mode counts must match explicitly.')
        cost=abs(freq[:,None]-np.array(c['frequencies_hz'])[None,:]);a,b=linear_sum_assignment(cost)
        for i,j in zip(a,b):
            comparisons[i].append(dict(run_id=ref['run_id'],operation=other['operation'],distance_hz=float(cost[i,j]),
                matched=bool(cost[i,j]<=tolerance),t2star_s=c['t2star_s'][j],
                relative_t2_change=float(c['t2star_s'][j]/candidate['t2star_s'][i]-1),
                numerically_provisional=bool(c.get('budget_exhausted',False) or not c.get('optimizer_converged',False) or c.get('boundary_hits',[]) or c.get('numerical_diagnostics',{}).get('numerical_warning_flags',[]))))
    flags=candidate.get('numerical_diagnostics',{}).get('numerical_warning_flags')
    if flags is None: flags=['numerical_diagnostics_missing']
    rows=[]
    for i,f in enumerate(freq):
        supported=bool(supports[i]) and all(s['classification']=='reproducible_signal_candidate' for s in supports[i])
        missing=[]
        if not supports[i]: missing.append('signal_reproducibility')
        if not stability: missing.append('group_stability')
        if not samples: missing.append('resampling')
        if not comparisons[i]: missing.append('explicit_sensitivity_comparisons')
        unstable=any(s['summary']['status']!='conditional_descriptive_percentiles' for s in samples)
        group_range=None
        if stability:
            fits=stability['group_fits']
            if len(fits)!=len(stability['requested_groups']): missing.append('incomplete_group_stability')
            if fits:
                values=[g['matched_t2star_s'][i] for g in fits];group_range=[min(values),max(values)]
                unstable=unstable or any(g['numerical_diagnostics']['requires_review'] or abs(g['frequency_shift_hz'][i])>tolerance for g in fits)
        # Sampling variation omits systematic changes of processing and objective.
        # Crossing these descriptive bounds is a sensitivity flag, not a test.
        for c in comparisons[i]:
            c['outside_sampling_ranges']=[s['run_id'] for s in samples
                if s['summary'].get('t2star_percentiles_s') is not None
                and not s['summary']['t2star_percentiles_s'][0][i]<=c['t2star_s']<=s['summary']['t2star_percentiles_s'][2][i]]
        sensitive=any(not c['matched'] or c['numerically_provisional'] or c['outside_sampling_ranges'] or abs(c['relative_t2_change'])>relative_change_threshold for c in comparisons[i])
        status=evidence_status(supported,flags,unstable,missing,sensitive)
        rows.append(dict(mode_index=i,frequency_hz=float(f),candidate_t2star_s=candidate['t2star_s'][i] if supported else None,
            signal_evidence=supports[i],numerical_flags=flags,missing_evidence=missing,stability_range_s=group_range,
            sensitivity_comparisons=comparisons[i],interpretation_status=status,physical_component_accepted=False))
    lines=['# Decay candidate evidence review','','All statuses are operational review labels, not physical acceptance.','',
           '| Frequency (Hz) | Candidate T2* (s) | Status | Missing evidence |','|---|---|---|---|']
    for r in rows: lines.append(f"| {r['frequency_hz']:.4f} | {r['candidate_t2star_s']} | {r['interpretation_status']} | {', '.join(r['missing_evidence'])} |")
    lines+=['','Signal support does not establish decay identifiability. Matched frequencies do not prove mode identity. Sampling percentiles omit model/processing uncertainty. A supported candidate remains exploratory; residual adequacy, coherent interference and mechanism assignment require separate review.']
    (directory/'evidence_review.md').write_text('\n'.join(lines),encoding='utf-8');progress(1,1)
    return dict(parent_run_id=fit_run_id,candidate_index=candidate_index,modes=rows,resampling_evidence=samples,
        signal_run_ids=signal_run_ids or [],stability_run_id=stability_run_id,comparison_refs=refs,
        source_arrays_sha256=source['arrays_sha256'],frequency_match_tolerance_hz=tolerance,
        relative_change_threshold=relative_change_threshold,scientifically_validated=False,
        warnings=['Thresholds and labels are operational heuristics, not statistical acceptance tests.',
        'Missing evidence cannot be interpreted as successful validation.',
        'Changing noise reference bands does not independently change repeat-based classification.',
        'No physical component, intrinsic T2, substance or relaxation mechanism is automatically accepted.'])
