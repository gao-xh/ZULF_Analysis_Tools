"""Exact finite-record complex demodulation for bounded time-domain decay fits."""
import numpy as np
from scipy.signal import fftconvolve,savgol_filter
from .window_decay import WindowedDecayOperator
from .time_frequency import demodulate_band
from .decay import fit_modes,mode_design


class DemodulatedDecayOperator(WindowedDecayOperator):
    """Cache the filter/mixer; inherited real quadratures retain acquisition time.

    Matched-all fits include zero-extension-affected samples by modelling that
    exact finite-record operator. Interior fits exclude them and can lose most
    of a fast decay. Neither policy creates pre-crop signal observations.
    """
    def __init__(self,fs,full_points,preprocessing,band,transition_hz=None,
                 attenuation_db=80.,edge_policy='matched_all'):
        from .analysis import recipe
        if not np.isfinite(fs) or fs<=0 or type(full_points) is not int or not 32<=full_points<=2_000_000:
            raise ValueError('Invalid sample rate or record size.')
        if edge_policy not in ('matched_all','interior'): raise ValueError('Unknown edge policy.')
        self.fs=fs;self.full_points=full_points
        _,_,self.parameters=recipe(np.zeros(full_points),fs,preprocessing or {})
        self.first=self.parameters['start_sample'];self.last=self.parameters['stop_sample'];self.n=self.last-self.first
        design=demodulate_band(np.zeros(self.n),fs,band,transition_hz,attenuation_db,self.first/fs)
        self.fir=design['fir_coefficients'];self.center=design['center_hz']
        self.decimation=design['decimation'];self.edge_policy=edge_policy
        self.all_times_s=design['times_s'];self.valid_interior=design['valid_interior']
        self.selected=self.valid_interior.copy() if edge_policy=='interior' else np.ones(len(self.all_times_s),bool)
        if self.selected.sum()<8: raise ValueError('Fewer than eight selected demodulated observations; widen filter transition or retain a longer record.')
        self.times_s=self.all_times_s[self.selected]
        self.f=np.full(len(self.times_s),self.center)  # Adapter shape; seeds must be supplied.
        self.time=np.arange(full_points)/fs
        self.mixer=2*np.exp(-2j*np.pi*self.center*(self.first+np.arange(self.n))/fs)
        self.metadata={k:design[k] for k in ['center_hz','output_sampling_rate_hz','decimation','filter_samples','edge_duration_s','transition_hz','attenuation_db']}
        self.metadata.update(edge_policy=edge_policy,selected_samples=int(self.selected.sum()),
            interior_samples=int(self.valid_interior.sum()),first_selected_time_s=float(self.times_s[0]),
            first_interior_time_s=float(self.all_times_s[self.valid_interior][0]) if self.valid_interior.any() else None)

    def transform_all(self,values):
        values=np.asarray(values)
        if values.ndim not in (1,2) or values.shape[0]!=self.full_points or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError('Supply finite real full-record FID columns.')
        vector=values.ndim==1
        if vector: values=values[:,None]
        p=self.parameters
        if p['sg_window']: values=values-savgol_filter(values,p['sg_window'],p['sg_order'],axis=0,mode='mirror')
        retained=values[self.first:self.last].copy()
        if p.get('remove_mean',True): retained-=retained.mean(axis=0)
        mixed=retained*self.mixer[:,None]
        filtered=fftconvolve(mixed,self.fir[:,None],mode='same',axes=0)[::self.decimation]
        return filtered[:,0] if vector else filtered

    def transform(self,values):
        return self.transform_all(values)[self.selected]


def fit_demodulated_modes(operator,full_fid,frequency_bounds,t2_bounds,initial_frequencies,
                          shared_decay=False,**budgets):
    if np.asarray(full_fid).ndim!=1: raise ValueError('Fit one real FID at a time.')
    result=fit_modes(operator,operator.transform(full_fid),frequency_bounds,t2_bounds,
        mode_count=len(initial_frequencies),initial_frequencies=initial_frequencies,
        shared_decay=shared_decay,background=False,**budgets)
    d=result['numerical_diagnostics'];d.pop('sub_bin_frequency_pairs',None)
    d['thresholds'].pop('native_fft_spacing_hz',None)
    result.update(observation_domain='Complex demodulated finite-record time samples',
        interpretation='Effective oscillatory decay under matched filtering; correlated observations, no independent-sample confidence claim.')
    return result


def fit_demodulated_decay(fit_run_id,candidate_index=0,transition_hz=None,
                          attenuation_db=80.,edge_policy='matched_all',settings=None,
                          *,record,directory,cancel,progress):
    from . import storage
    from .analysis import recipe,plot
    from .repeats import load_group_averages
    parent=storage.get_result(fit_run_id)
    if parent['operation']!='fit_frequency_decay' or type(candidate_index) is not int or not 0<=candidate_index<len(parent['candidates']):
        raise ValueError('Require a valid frequency-decay candidate.')
    if parent['settings']['background']: raise ValueError('Constant spectral background has no unique full-FID model.')
    candidate=parent['candidates'][candidate_index]
    means,counts,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']: raise ValueError('Parent source changed.')
    config=dict(starts=3,max_nfev=150,max_evaluations=2000,max_seconds=60.,seed=20260922)
    if settings and set(settings)-set(config): raise ValueError('Unknown demodulated-fit settings.')
    config.update(settings or {})
    spec=parent['parameters'].get('preprocessing') or {};bounds=candidate['range_hz']
    op=DemodulatedDecayOperator(source['sampling_rate_hz'],source['points'],spec,bounds,transition_hz,attenuation_db,edge_policy)
    raw={role:np.average(means[indices],axis=0,weights=counts[indices]) for role,indices in
        [('discovery',parent['discovery_groups']),('validation',parent['validation_groups'])]}
    fit=fit_demodulated_modes(op,raw['discovery'],bounds,parent['t2_bounds_s'],candidate['frequencies_hz'],
        shared_decay=candidate['shared_decay'],cancel=cancel,**config)
    prediction=fit.pop('fitted');progress(1,2)
    def raw_model(c):
        return sum(np.exp(-op.time/tau)*(a*np.cos(2*np.pi*f*op.time)+b*np.sin(2*np.pi*f*op.time))
            for f,tau,(a,b) in zip(c['frequencies_hz'],c['t2star_s'],c['cos_sin_coefficients']))
    values=dict(raw,fitted=raw_model(fit),parent_prediction=raw_model(candidate))
    observations={role:op.transform_all(y) for role,y in values.items()}
    def error(y,p):
        norm=float(np.linalg.norm(y));return float(np.linalg.norm(y-p)/norm) if norm else None
    fit['validation_relative_complex_residual']=error(observations['validation'][op.selected],prediction)
    arrays=dict(times_s=op.all_times_s,valid_interior=op.valid_interior,fit_sample_mask=op.selected,
        fir_coefficients=op.fir,**observations)
    start=op.first/op.fs;end=op.last/op.fs;edge=op.metadata['edge_duration_s']
    edge_ranges=[(start,end)] if 2*edge>=end-start else [(start,start+edge),(end-edge,end)]
    for part,transform in [('magnitude',np.abs),('real',np.real),('imaginary',np.imag)]:
        for view,mask in [('full',np.ones(len(op.all_times_s),bool)),('early',op.all_times_s<=op.all_times_s[0]+1.)]:
            left,right=op.all_times_s[mask][[0,-1]]
            spans=[(max(a,left),min(b,right)) for a,b in edge_ranges if max(a,left)<min(b,right)]
            plot(directory/f'{view}_{part}.png',[(op.all_times_s[mask],transform(y[mask]),role) for role,y in observations.items()],
                'Acquisition time (s)',part.title()+' (ADC units)',f'{bounds[0]:g}-{bounds[1]:g} Hz demodulation; {edge_policy}',
                shaded_ranges=spans)
    residual=observations['validation']-observations['fitted']
    plot(directory/'residual.png',[(op.all_times_s,residual.real,'Real'),(op.all_times_s,residual.imag,'Imaginary')],
        'Acquisition time (s)','Complex residual (ADC units)','Frozen validation residual; all filter samples shown',
        shaded_ranges=edge_ranges)
    plot(directory/'filter_masks.png',[(op.all_times_s,op.valid_interior.astype(int),'Uncontaminated by zero extension'),
        (op.all_times_s,op.selected.astype(int),'Used in fit')],'Acquisition time (s)','Mask (0 or 1)','Filter edge coverage')
    frequency=np.fft.rfftfreq(op.n,1/op.fs);mask=(frequency>=bounds[0])&(frequency<=bounds[1])
    spectra={role:np.fft.rfft(recipe(y,op.fs,spec)[0])[mask]/op.n for role,y in values.items()}
    arrays['fft_frequency_hz']=frequency[mask]
    for role,y in spectra.items(): arrays['fft_'+role]=y
    plot(directory/'fft_crosscheck.png',[(frequency[mask],abs(y),role) for role,y in spectra.items()],
        'Frequency (Hz)','Magnitude (ADC units)','Original frequency-domain cross-check')
    np.savez_compressed(directory/'demodulated_fit_arrays.npz',**arrays);progress(2,2)
    return dict(parent_run_id=fit_run_id,candidate_index=candidate_index,source_arrays_sha256=source['arrays_sha256'],
        discovery_groups=parent['discovery_groups'],validation_groups=parent['validation_groups'],
        preprocessing=spec,range_hz=bounds,t2_bounds_s=parent['t2_bounds_s'],settings=config,filter=op.metadata,fit=fit,
        parent_demodulated_validation_error=error(observations['validation'][op.selected],observations['parent_prediction'][op.selected]),
        fft_validation_error=error(spectra['validation'],spectra['fitted']),
        parent_fft_validation_error=error(spectra['validation'],spectra['parent_prediction']),scientifically_validated=False,
        warnings=['Matched-all uses explicitly modelled zero-extended filter edges; interior policy can exclude rapid early decay.',
        'No imaginary FID channel is invented: real quadratures are mixed and filtered identically for model and data.',
        'Filtered time samples are correlated; no independent-sample uncertainty or likelihood claim.',
        'Transition bands admit outside-band signal absent from this bounded mode model; compare frequency-domain residuals.',
        'A smaller filtered residual or optimizer convergence does not establish a physical relaxation component.'])
