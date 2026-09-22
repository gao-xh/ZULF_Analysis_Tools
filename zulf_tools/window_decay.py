"""Matched complex Hann observations for bounded oscillatory decay fitting."""
import numpy as np
from scipy.signal import get_window, savgol_filter
from scipy.sparse import csr_matrix
from .decay import fit_modes


class WindowedDecayOperator:
    """Cache a sparse real-FID-to-complex-window transform, including phase.

    SG acts on the complete record before cropping. Mean subtraction acts on
    the retained record and is applied to model and data alike. Complete Hann
    windows only; observations are correlated and carry no independent-bin DOF.
    """
    def __init__(self, fs, full_points, preprocessing, frequencies, width_s, hop_s):
        from .analysis import recipe
        if not np.isfinite(fs) or fs<=0 or type(full_points) is not int or not 32<=full_points<=2_000_000:
            raise ValueError('Invalid sample rate or full record size.')
        self.fs=fs;self.full_points=full_points
        _,_,self.parameters=recipe(np.zeros(full_points),fs,preprocessing or {})
        self.first=self.parameters['start_sample'];self.last=self.parameters['stop_sample']
        self.n=self.last-self.first
        self.targets=np.asarray(frequencies,dtype=float)
        if self.targets.ndim!=1 or not 1<=len(self.targets)<=32 or not np.isfinite(self.targets).all() or np.any(self.targets<=0) or np.any(self.targets>=fs/2) or np.any(np.diff(self.targets)<=0):
            raise ValueError('Supply 1..32 sorted distinct positive frequencies inside Nyquist.')
        if not np.isfinite([width_s,hop_s]).all() or min(width_s,hop_s)<=0:
            raise ValueError('Window width and hop must be positive.')
        self.width=int(round(width_s*fs));self.hop=int(round(hop_s*fs))
        if not 8<=self.width<=self.n or self.hop<1:
            raise ValueError('Window must fit the retained record and hop must be at least one sample.')
        starts=np.arange(0,self.n-self.width+1,self.hop)
        rows=len(starts)*len(self.targets)
        if rows*self.width>2_000_000:
            raise ValueError('Sparse window transform exceeds two million coefficients.')
        self.shape=(len(starts),len(self.targets))
        self.times_s=(self.first+starts+self.width/2)/fs
        self.f=np.tile(self.targets,len(starts))
        indices=np.repeat(starts,len(self.targets))[:,None]+np.arange(self.width)[None,:]
        hann=get_window('hann',self.width,fftbins=True)
        values=2*np.exp(-2j*np.pi*self.f[:,None]*(indices+self.first)/fs)*hann/hann.sum()
        self.matrix=csr_matrix((values.ravel(),indices.ravel(),np.arange(rows+1)*self.width),shape=(rows,self.n))
        self.time=np.arange(full_points)/fs

    def transform(self, values):
        values=np.asarray(values)
        if values.ndim not in (1,2) or values.shape[0]!=self.full_points or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError('Supply finite real full-record FID columns.')
        p=self.parameters
        if p['sg_window']:
            values=values-savgol_filter(values,p['sg_window'],p['sg_order'],axis=0,mode='mirror')
        retained=values[self.first:self.last].copy()
        if p.get('remove_mean',True):
            retained-=retained.mean(axis=0)
        return self.matrix@retained

    def templates(self, frequencies, weights, rate, phase_delay_s=0.):
        frequencies=np.asarray(frequencies,dtype=float);weights=np.asarray(weights,dtype=float)
        if frequencies.ndim!=1 or frequencies.shape!=weights.shape or not len(frequencies) or not np.isfinite(frequencies).all() or not np.isfinite(weights).all() or weights.sum()<=0 or not np.isfinite(rate) or rate<=0:
            raise ValueError('Invalid oscillatory template.')
        if not np.isfinite(phase_delay_s):
            raise ValueError('Phase delay must be finite.')
        # Accumulate weighted real quadratures without a full time-by-transition array.
        raw=np.zeros((self.full_points,2))
        decay=np.exp(-rate*self.time)
        for f,w in zip(frequencies,weights/weights.sum()):
            angle=2*np.pi*f*(self.time-phase_delay_s)
            raw[:,0]+=w*decay*np.cos(angle)
            raw[:,1]+=w*decay*np.sin(angle)
        return self.transform(raw)


def fit_windowed_modes(operator, full_fid, frequency_bounds, t2_bounds, initial_frequencies,
                       shared_decay=False, **budgets):
    """Fit correlated complex windows; no independent-window confidence claim."""
    if np.asarray(full_fid).ndim!=1:
        raise ValueError('Fit one real FID at a time.')
    fit=fit_modes(operator,operator.transform(full_fid),frequency_bounds,t2_bounds,
                  mode_count=len(initial_frequencies),initial_frequencies=initial_frequencies,
                  shared_decay=shared_decay,background=False,**budgets)
    # FFT-bin spacing is not the resolution of these window observations.
    diagnostics=fit['numerical_diagnostics']
    diagnostics.pop('sub_bin_frequency_pairs',None)
    diagnostics['thresholds'].pop('native_fft_spacing_hz',None)
    fit.update(observation_domain='Complex complete Hann windows; time-major flattened observations.',
               window_width_s=operator.width/operator.fs,hop_s=operator.hop/operator.fs,
               nominal_window_resolution_hz=operator.fs/operator.width,
               interpretation='Effective oscillatory decay; overlapping windows are correlated. No independent-window confidence interval or physical component assignment.')
    return fit


def fit_window_decay(fit_run_id, candidate_index=0, width_s=.2, hop_s=.04,
                     sample_frequencies_hz=None, settings=None, *, record, directory, cancel, progress):
    """Refit a discovery candidate through matched windows, freezing validation."""
    from . import storage
    from .analysis import recipe, plot
    from .repeats import load_group_averages
    from .decay import mode_design
    parent=storage.get_result(fit_run_id)
    if parent['operation']!='fit_frequency_decay':
        raise ValueError('Expected a completed frequency-decay fit.')
    if type(candidate_index) is not int or not 0<=candidate_index<len(parent['candidates']):
        raise ValueError('Invalid candidate index.')
    if parent['settings']['background']:
        raise ValueError('A constant spectral background has no unique full-FID window model.')
    candidate=parent['candidates'][candidate_index]
    means,counts,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']:
        raise ValueError('Source group arrays do not match parent provenance.')
    config=dict(starts=3,max_nfev=150,max_evaluations=3000,max_seconds=60.,seed=20260922)
    if settings and set(settings)-set(config):
        raise ValueError('Unknown window-fit settings.')
    config.update(settings or {})
    bounds=candidate['range_hz'];spec=parent['parameters'].get('preprocessing') or {}
    targets=np.linspace(*bounds,5) if sample_frequencies_hz is None else sample_frequencies_hz
    op=WindowedDecayOperator(source['sampling_rate_hz'],source['points'],spec,targets,width_s,hop_s)
    raw={name:np.average(means[indices],axis=0,weights=counts[indices])
         for name,indices in [('discovery',parent['discovery_groups']),('validation',parent['validation_groups'])]}
    fit=fit_windowed_modes(op,raw['discovery'],bounds,parent['t2_bounds_s'],candidate['frequencies_hz'],
                          shared_decay=candidate['shared_decay'],cancel=cancel,**config)
    prediction=fit.pop('fitted');progress(1,2)
    observed={name:op.transform(y) for name,y in raw.items()}
    prior=mode_design(op,candidate['frequencies_hz'],candidate['t2star_s'])@np.asarray(candidate['cos_sin_coefficients']).ravel()
    def error(y,p):
        norm=float(np.linalg.norm(y))
        return float(np.linalg.norm(y-p)/norm) if norm else None
    fit['validation_relative_complex_residual']=error(observed['validation'],prediction)
    arrays=dict(times_s=op.times_s,sample_frequencies_hz=op.targets,
        discovery=observed['discovery'].reshape(op.shape),validation=observed['validation'].reshape(op.shape),
        fitted=prediction.reshape(op.shape),parent_prediction=prior.reshape(op.shape))
    for i,f in enumerate(op.targets):
        if cancel():
            raise InterruptedError('Window fit cancelled.')
        for name,transform in [('magnitude',np.abs),('real',np.real),('imaginary',np.imag)]:
            plot(directory/f'frequency_{i}_{name}.png',[(op.times_s,transform(arrays[key][:,i]),label)
                for key,label in [('discovery','Discovery'),('validation','Validation'),('fitted','Window fit'),('parent_prediction','FFT fit')]],
                'Window center: recorded time (s)',name.title()+' (ADC units)',f'{f:.3f} Hz: matched Hann {width_s:g} s')
        residual=arrays['validation'][:,i]-arrays['fitted'][:,i]
        plot(directory/f'frequency_{i}_residual.png',[(op.times_s,residual.real,'Real'),(op.times_s,residual.imag,'Imaginary')],
            'Window center: recorded time (s)','Complex residual (ADC units)',f'{f:.3f} Hz: frozen window prediction residual')
    # Cross-check original native frequency domain; improving windows may worsen it.
    def raw_model(c):
        return sum(np.exp(-op.time/tau)*(a*np.cos(2*np.pi*f*op.time)+b*np.sin(2*np.pi*f*op.time))
                   for f,tau,(a,b) in zip(c['frequencies_hz'],c['t2star_s'],c['cos_sin_coefficients']))
    frequency=np.fft.rfftfreq(op.n,1/op.fs)
    mask=(frequency>=bounds[0])&(frequency<=bounds[1])
    spectra={name:np.fft.rfft(recipe(y,op.fs,spec)[0])[mask]/op.n
             for name,y in dict(raw,fitted=raw_model(fit),parent_prediction=raw_model(candidate)).items()}
    arrays['fft_frequency_hz']=frequency[mask]
    for name,y in spectra.items():
        arrays['fft_'+name]=y
    plot(directory/'fft_crosscheck.png',[(frequency[mask],abs(y),name) for name,y in spectra.items()],
        'Frequency (Hz)','Magnitude (ADC units)','Original frequency-domain cross-check')
    np.savez_compressed(directory/'window_fit_arrays.npz',**arrays)
    progress(2,2)
    return dict(parent_run_id=fit_run_id,candidate_index=candidate_index,source_arrays_sha256=source['arrays_sha256'],
        discovery_groups=parent['discovery_groups'],validation_groups=parent['validation_groups'],settings=config,
        range_hz=bounds,t2_bounds_s=parent['t2_bounds_s'],preprocessing=spec,fit=fit,
        sample_frequencies_hz=op.targets.tolist(),window_count=op.shape[0],
        parent_window_validation_error=error(observed['validation'],prior),
        fft_validation_error=error(spectra['validation'],spectra['fitted']),
        parent_fft_validation_error=error(spectra['validation'],spectra['parent_prediction']),
        scientifically_validated=False,warnings=['Validation coefficients remain frozen; no per-group refitting.',
        'Overlapping windows and sampled frequencies are correlated; no independent-sample uncertainty.',
        'Short windows admit out-of-band signal absent from this bounded model.',
        'A better window residual can worsen native FFT agreement; both are retained.',
        'Window fits are phenomenological candidates, not established relaxation components.'])
