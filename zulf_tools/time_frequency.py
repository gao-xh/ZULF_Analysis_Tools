"""Explicit-window Fourier and anti-aliased complex demodulation diagnostics."""
import numpy as np
from scipy.signal import get_window, firwin, kaiserord, fftconvolve


def windowed_fourier(values, fs, frequencies, width_s, hop_s, time_origin_s=0.):
    """Complete Hann windows only; preserve acquisition-referenced complex phase.

    Frequencies can be off the window FFT grid: this is a direct Fourier sample,
    not extra resolving power. For real input the positive-frequency response is
    doubled; negative-frequency leakage remains part of the exact operator.
    """
    values=np.asarray(values)
    frequencies=np.asarray(frequencies,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite([fs,width_s,hop_s,time_origin_s]).all() or min(fs,width_s,hop_s)<=0:
        raise ValueError('Need a finite one-dimensional FID and positive sampling/window/hop settings.')
    if frequencies.ndim!=1 or not len(frequencies) or not np.isfinite(frequencies).all() or np.any(abs(frequencies)>=fs/2):
        raise ValueError('Frequencies must be finite and inside Nyquist.')
    width=int(round(width_s*fs)); hop=int(round(hop_s*fs))
    if not 8<=width<=len(values) or hop<1:
        raise ValueError('Window must retain 8..N samples and hop at least one sample.')
    starts=np.arange(0,len(values)-width+1,hop)
    if len(starts)*len(frequencies)>2_000_000 or width*len(frequencies)>4_000_000:
        raise ValueError('Windowed Fourier request exceeds diagnostic array budget.')
    window=get_window('hann',width,fftbins=True)
    kernel=np.exp(-2j*np.pi*frequencies[:,None]*np.arange(width)[None,:]/fs)*window/window.sum()
    if not np.iscomplexobj(values):
        kernel*=2
    result=np.empty((len(starts),len(frequencies)),complex)
    for i,start in enumerate(starts):
        result[i]=(kernel@values[start:start+width])*np.exp(-2j*np.pi*frequencies*(time_origin_s+start/fs))
    return {'times_s':time_origin_s+(starts+width/2)/fs,'spectrum':result,
            'frequencies_hz':frequencies,'window_samples':width,'hop_samples':hop,
            'actual_width_s':width/fs,'actual_hop_s':hop/fs,
            'overlap_fraction':max(0.,1-hop/width),
            'phase_convention':'Fourier kernel references acquisition time; complete windows, no padding.',
            'window':'periodic Hann','nominal_resolution_hz':fs/width}


def demodulate_band(values, fs, band, transition_hz=None, attenuation_db=80., time_origin_s=0.):
    """Mix, symmetric FIR low-pass, then decimate with explicit transient mask.

    No Hilbert transform or magnitude operation precedes filtering. The FIR is
    applied with zero extension; samples influenced by extension are marked
    invalid, never silently discarded. Apply the same operator to predictions.
    """
    values=np.asarray(values)
    if values.ndim!=1 or len(values)<32 or not np.isfinite(values).all():
        raise ValueError('Need at least 32 finite samples.')
    if not np.isfinite([fs,time_origin_s,attenuation_db]).all() or fs<=0 or not 40<=attenuation_db<=140:
        raise ValueError('Invalid sample rate, time origin or attenuation.')
    if len(band)!=2 or not np.isfinite(band).all() or not 0<band[0]<band[1]<fs/2:
        raise ValueError('Band must be positive and inside Nyquist.')
    center=sum(band)/2; half=(band[1]-band[0])/2
    transition_hz=max(half*.5,.5) if transition_hz is None else float(transition_hz)
    if not np.isfinite(transition_hz) or transition_hz<=0 or half+transition_hz>=min(center,fs/2):
        raise ValueError('Transition must fit below Nyquist and exclude mixed DC baseline.')
    taps,beta=kaiserord(attenuation_db,transition_hz/(fs/2))
    taps=max(3,taps|1)
    if taps>4*len(values) or taps>262145:
        raise ValueError('Filter is too long for this record; widen transition or band.')
    fir=firwin(taps,half+transition_hz/2,window=('kaiser',beta),fs=fs)
    t=time_origin_s+np.arange(len(values))/fs
    scale=1 if np.iscomplexobj(values) else 2
    mixed=scale*values*np.exp(-2j*np.pi*center*t)
    filtered=fftconvolve(mixed,fir,mode='same')
    # New Nyquist is at least twice the designed stop-band edge.
    decimation=max(1,int(np.floor(fs/(4*(half+transition_hz)))))
    indices=np.arange(0,len(values),decimation)
    radius=(taps-1)//2
    valid=(indices>=radius)&(indices<len(values)-radius)
    return {'times_s':t[indices],'signal':filtered[indices],'valid_interior':valid,
            'center_hz':center,'output_sampling_rate_hz':fs/decimation,
            'decimation':decimation,'fir_coefficients':fir,'filter_samples':taps,
            'edge_duration_s':radius/fs,'transition_hz':transition_hz,
            'attenuation_db':attenuation_db,
            'note':'Zero-extended FIR edge samples are marked invalid. Match filtering for model comparisons.'}


def inspect_decay_time_frequency(fit_run_id, candidate_index=0, widths_s=None,
                                 hop_fraction=.1, *, record, directory, cancel, progress):
    """Compare a frozen frequency-decay candidate through matched time operators."""
    from . import storage
    from .analysis import recipe, plot
    from .repeats import load_group_averages
    result=storage.get_result(fit_run_id)
    if result['operation']!='fit_frequency_decay':
        raise ValueError('Expected a completed fit_frequency_decay run.')
    if type(candidate_index) is not int or not 0<=candidate_index<len(result['candidates']):
        raise ValueError('Invalid candidate_index.')
    candidate=result['candidates'][candidate_index]
    widths_s=[.25,.5,1.] if widths_s is None else widths_s
    if not isinstance(widths_s,list) or not 1<=len(widths_s)<=6 or not np.isfinite(widths_s).all() or min(widths_s)<=0:
        raise ValueError('Supply 1..6 positive window widths in seconds.')
    if not np.isfinite(hop_fraction) or not 0<hop_fraction<=1:
        raise ValueError('hop_fraction must be in (0,1].')
    means,counts,parent=load_group_averages(result['parent_run_id'])
    if parent['arrays_sha256']!=result['source_arrays_sha256']:
        raise ValueError('Source group arrays disagree with fitting provenance.')
    fs=parent['sampling_rate_hz']; full_t=np.arange(parent['points'])/fs
    model=np.zeros(len(full_t))
    for f,tau,(cosine,sine) in zip(candidate['frequencies_hz'],candidate['t2star_s'],candidate['cos_sin_coefficients']):
        model+=np.exp(-full_t/tau)*(cosine*np.cos(2*np.pi*f*full_t)+sine*np.sin(2*np.pi*f*full_t))
    specification=result['parameters'].get('preprocessing') or {}
    predicted,_,params=recipe(model,fs,specification)
    observed={}
    for label,indices in [('discovery',result['discovery_groups']),('validation',result['validation_groups'])]:
        average=np.average(means[indices],axis=0,weights=counts[indices])
        observed[label]=recipe(average,fs,specification)[0]
    origin=params['actual_start_s']; frequencies=candidate['frequencies_hz']
    arrays={}; comparisons=[]
    for wi,width in enumerate(widths_s):
        if cancel():
            raise InterruptedError('Time-frequency diagnostics cancelled.')
        hop=width*hop_fraction
        transforms={label:windowed_fourier(values,fs,frequencies,width,hop,origin)
                    for label,values in dict(observed,prediction=predicted).items()}
        tf=transforms['prediction']; times=tf['times_s']
        arrays[f'window_{wi}_times_s']=times
        for label,tr in transforms.items():
            arrays[f'window_{wi}_{label}']=tr['spectrum']
        for fi,f in enumerate(frequencies):
            plot(directory/f'window_{wi}_frequency_{fi}_magnitude.png',
                 [(times,abs(tr['spectrum'][:,fi]),label.title()) for label,tr in transforms.items()],
                 'Window center: recorded time (s)','Windowed amplitude (ADC units)',
                 f'{f:.4f} Hz: Hann {tf["actual_width_s"]:g} s, hop {tf["actual_hop_s"]:g} s')
            early=times<=min(times[-1],origin+max(2.,4*max(candidate['t2star_s'])))
            plot(directory/f'window_{wi}_frequency_{fi}_early.png',
                 [(times[early],abs(tr['spectrum'][early,fi]),label.title()) for label,tr in transforms.items()],
                 'Window center: recorded time (s)','Windowed amplitude (ADC units)',
                 f'{f:.4f} Hz: early windowed response (Hann {tf["actual_width_s"]:g} s)')
        y=transforms['validation']['spectrum']; yhat=tf['spectrum']
        norm=float(np.linalg.norm(y))
        comparisons.append({'width_s':tf['actual_width_s'],'hop_s':tf['actual_hop_s'],
                            'overlap_fraction':tf['overlap_fraction'],
                            'nominal_resolution_hz':tf['nominal_resolution_hz'],
                            'validation_complex_relative_residual':float(np.linalg.norm(y-yhat)/norm) if norm else None})
        progress(wi+1,len(widths_s)+1)
    band=candidate['range_hz']
    demod={label:demodulate_band(values,fs,band,time_origin_s=origin)
           for label,values in dict(observed,prediction=predicted).items()}
    ref=demod['prediction']; t=ref['times_s']; interior=ref['valid_interior']
    arrays['demod_times_s']=t;arrays['demod_valid_interior']=interior
    arrays['demod_fir_coefficients']=ref['fir_coefficients']
    for label,tr in demod.items():
        arrays[f'demod_{label}']=tr['signal']
    for part,transform in [('real',np.real),('imaginary',np.imag),('magnitude',np.abs)]:
        plot(directory/f'demodulated_{part}.png',[(t,transform(tr['signal']),label.title()) for label,tr in demod.items()],
             'Recorded time (s)',part.title()+' (ADC units)',
             f'Demodulated {band[0]:g}-{band[1]:g} Hz; edge duration {ref["edge_duration_s"]:.3g} s')
    norm=float(np.linalg.norm(demod['validation']['signal'][interior]))
    err=float(np.linalg.norm((demod['validation']['signal']-ref['signal'])[interior])/norm) if norm else None
    np.savez_compressed(directory/'time_frequency_arrays.npz',**arrays)
    progress(len(widths_s)+1,len(widths_s)+1)
    return {'parent_run_id':fit_run_id,'candidate_index':candidate_index,'range_hz':band,
            'frequencies_hz':frequencies,'window_comparisons':comparisons,
            'demodulation':{k:ref[k] for k in ['center_hz','output_sampling_rate_hz','decimation','filter_samples','edge_duration_s','transition_hz','attenuation_db']},
            'demod_valid_interior_samples':int(interior.sum()),'demod_validation_relative_residual':err,
            'demodulation_status':'interior_available' if interior.any() else 'insufficient_record_for_filter_interior',
            'warnings':['No refitting: candidate prediction is frozen and transformed identically to data.',
                        'Selected-frequency STFT samples are not independently resolved transitions.',
                        'Overlapping windows are correlated; no independent-window confidence interval.',
                        'Oscillatory magnitude can reflect beating, not multiple physical decays.',
                        'Demodulation plots include boundary-affected samples; numerical interior metric excludes them.',
                        'Any fitted complex spectral background is omitted from time prediction; it has no unique broadband time model.',
                        'A band-only model omits other experimental frequencies, which can leak through short windows.']}
