"""Traceable high-band anchor -> component validation -> methine -> joint workflow."""
import numpy as np
from . import storage,jfit


def predict_methyl_low(anchor, comparison_run_id, variant_index, low_range, directory, label):
    """Extrapolate the high-band component without refitting any low-band gain."""
    from .analysis import plot
    parent=storage.get_result(comparison_run_id); v=parent['variants'][variant_index]
    average=storage.get_result(parent['parent_run_id']); p=v['parameters']; fs=parent['sampling_rate_hz']
    with np.load(storage.artifact(comparison_run_id,f'variant_{variant_index}.npz'),allow_pickle=False) as a:
        frequency=a['frequency']; data=a['spectrum']
    bins=np.flatnonzero((frequency>=low_range[0])&(frequency<=low_range[1]))
    if len(bins)<8 or len(bins)>10000: raise ValueError('Invalid low-band prediction size.')
    processor=jfit.ProcessedSpectrum(fs,average['points'],p['start_sample'],p['stop_sample'],bins,p['sg_window'],p['sg_order'])
    freq,w=jfit.transitions(anchor['parameters_hz'],'methyl')
    prediction=processor.templates(freq,w,anchor['decay_rates_per_s']['methyl'])@np.array(anchor['linear_coefficients'][2:4])
    plot(directory/f'{label}_methyl_low_prediction.png',[(frequency[bins],abs(data[bins]),'Experiment'),
         (frequency[bins],abs(prediction),'13CH3: high-band extrapolation, no low-band refit')],
         'Frequency (Hz)','Magnitude (ADC units)','Check methyl prediction before fitting methine')
    np.savez_compressed(directory/f'{label}_low_prediction.npz',frequency_hz=frequency[bins],experiment=data[bins],methyl_prediction=prediction)
    return {'relative_component_norm':float(np.linalg.norm(prediction)/np.linalg.norm(data[bins])),
            'note':'Prediction uses unchanged high-band gains/rate/J. Unknown frequency-dependent detection can invalidate this extrapolation. Magnitudes must not be subtracted.'}


def fit_isopropylamine_staged(comparison_run_id,variant_index,low_range=None,high_range=None,settings=None,
                             *,record,directory,cancel,progress):
    from .analysis import execute
    low_range=low_range or [110.,150.]; high_range=high_range or [230.,275.]
    if len(low_range)!=2 or len(high_range)!=2 or not 0<low_range[0]<low_range[1]<high_range[0]<high_range[1]:
        raise ValueError('Need nonoverlapping low/high bands in increasing order.')
    options=dict(initial=jfit.DEFAULT.copy(),bounds=jfit.BOUNDS.copy(),objective='magnitude',
                 branches=2,anchor_starts=5,anchor_screening=64,methine_starts=3,
                 methine_screening=24,max_nfev=80,bin_stride=3,seed=20260922,
                 high_degradation_tolerance=.15,
                 initial_rates={'methine':4.,'methyl':4.},
                 rate_bounds_by_isotopomer={'methine':[.3,40.],'methyl':[.3,40.]},diff_step=1e-4)
    settings=settings or {}
    if set(settings)-set(options): raise ValueError('Unknown staged-fit settings.')
    options.update(settings)
    if type(options['branches']) is not int or not 1<=options['branches']<=4: raise ValueError('branches must be 1..4.')
    if not np.isfinite(options['high_degradation_tolerance']) or not 0<=options['high_degradation_tolerance']<=1: raise ValueError('Invalid high-band degradation tolerance.')
    children=[]; results=[]
    def run(stage,spec,ranges,branch=None):
        if cancel(): raise InterruptedError('Staged J fit cancelled.')
        storage.write_json(directory/'stage_status.json',{'stage':stage,'branch':branch,'completed_children':children})
        child=execute('fit_isopropylamine_j',{'comparison_run_id':comparison_run_id,'variant_index':variant_index,
                       'ranges':ranges,'settings':spec},cancel=cancel,
                       progress=lambda n,total: storage.write_json(directory/'stage_progress.json',
                           {'stage':stage,'branch':branch,'stage_progress':round(100*n/max(1,total),2)}))
        children.append({'stage':stage,'branch':branch,'run_id':child['run_id']})
        storage.write_json(directory/'children.json',children)
        return child
    for key in ('initial_rates','rate_bounds_by_isotopomer'):
        if not isinstance(options[key],dict) or set(options[key])!={'methine','methyl'}: raise ValueError(key+' must specify methine and methyl.')
    common={k:options[k] for k in ['initial','bounds','objective','max_nfev','bin_stride','seed','diff_step']}
    methyl_free=['J_CH_methyl','J_HH_vicinal','J_Cmethyl_Hmethine','J_Cmethyl_Hother_methyl']
    anchor=run('methyl_high_band',dict(common,isotopomers=['methyl'],free_parameters=methyl_free,
               initial_rates={'methyl':options['initial_rates']['methyl']},
               rate_bounds_by_isotopomer={'methyl':options['rate_bounds_by_isotopomer']['methyl']},
               starts=options['anchor_starts'],screening_samples=options['anchor_screening']),[high_range])
    progress(1,1+2*options['branches'])
    selected=[]
    for candidate in anchor['candidates']:
        if all(max(abs(candidate['parameters_hz'][k]-other['parameters_hz'][k]) for k in methyl_free)>.05 for other in selected):
            selected.append(candidate)
        if len(selected)>=options['branches']: break
    for branch,candidate in enumerate(selected):
        # Reconstruct each selected anchor with unchanged parameters and its own
        # optimized gains before extrapolation; fit max_nfev=2 would move J, so
        # use an exported per-candidate linear solution instead.
        branch_anchor=dict(anchor,parameters_hz=candidate['parameters_hz'],
                           decay_rates_per_s=candidate['decay_rates_per_s'],
                           linear_coefficients=candidate['linear_coefficients'])
        prediction=predict_methyl_low(branch_anchor,comparison_run_id,variant_index,low_range,directory,f'branch_{branch}')
        starting=dict(options['initial'],**candidate['parameters_hz'])
        methine=run('methine_with_methyl_J_and_rate_fixed',dict(common,initial=starting,
            isotopomers=['methine','methyl'],free_parameters=['J_CH_methine','J_Cmethine_Hmethyl'],
            fixed_rates={'methyl':candidate['decay_rates_per_s']['methyl']},
            initial_rates={'methine':options['initial_rates']['methine'],'methyl':candidate['decay_rates_per_s']['methyl']},
            rate_bounds_by_isotopomer=options['rate_bounds_by_isotopomer'],
            starts=options['methine_starts'],screening_samples=options['methine_screening']),[low_range,high_range],branch)
        progress(2+2*branch,1+2*len(selected))
        joint=run('joint_refinement',dict(common,initial=methine['parameters_hz'],
            initial_rates=methine['decay_rates_per_s'],rate_bounds_by_isotopomer=options['rate_bounds_by_isotopomer'],
            isotopomers=['methine','methyl'],free_parameters=jfit.NAMES.copy(),
            starts=1,screening_samples=0),[low_range,high_range],branch)
        high_before=candidate['band_magnitude_relative_residuals'][0]
        high_after=joint['band_magnitude_relative_residuals'][1]
        preserved=high_after<=high_before*(1+options['high_degradation_tolerance'])+1e-8
        results.append({'branch':branch,'anchor_candidate_score':candidate['score'],
            'anchor_parameters_hz':candidate['parameters_hz'],'anchor_optimizer_success':candidate['optimizer_success'],
            'low_band_prediction':prediction,'methine_run_id':methine['run_id'],'joint_run_id':joint['run_id'],
            'parameters_hz':joint['parameters_hz'],'band_magnitude_relative_residuals':joint['band_magnitude_relative_residuals'],
            'methyl_anchor_high_band_residual':high_before,'high_band_preserved':bool(preserved),
            'joint_optimizer_success':joint['optimizer_success'],'joint_weighted_score':joint['weighted_mean_square_residual']})
        storage.write_json(directory/'branches.json',results)
        progress(3+2*branch,1+2*len(selected))
    eligible=[r for r in results if r['high_band_preserved'] and r['joint_optimizer_success'] and r['anchor_optimizer_success']]
    review=min(eligible,key=lambda r:r['joint_weighted_score']) if eligible else None
    return {'parent_run_id':comparison_run_id,'source_variant':variant_index,'low_range_hz':low_range,'high_range_hz':high_range,
            'settings':options,'children':children,'branches':results,
            'candidate_for_review_branch':review['branch'] if review else None,
            'scientifically_validated':False,
            'interpretation':'Staged candidates only. Preserving a high-band fit is necessary for this workflow, not proof of isotope assignment or correct J.',
            'warnings':['High band is treated as methyl-dominated as a testable model assumption, not an experimental assignment.',
                'Low-band prediction is saved for inspection before fitting methine; complex signals are combined, never magnitude-subtracted.',
                'Stage 2 fixes methyl J/shared H-H and methyl rate, but re-estimates gains/phases with both bands; they are not experimentally known.',
                'Anchor uncertainty is represented by retained starts only, not a confidence interval. More branches may exist.',
                'A joint candidate that worsens high-band magnitude residual beyond the stated tolerance is flagged, not silently selected.',
                'Scientific acceptance additionally requires component/residual inspection and independent-data validation.']}
