"""Discovery/validation peak checks and empirical repeat-noise diagnostics."""
import numpy as np
from scipy.signal import find_peaks
from . import storage
from .repeats import load_group_averages
from .repeat_statistics import weighted_repeat_statistics,accumulation_diagnostic,classify_reproducibility
from .band_relaxation import _indices


def inspect_repeat_signals(group_run_id, ranges, noise_ranges, discovery_groups,
                           validation_groups, preprocessing=None, max_candidates=8,
                           snr_threshold=5., frequency_tolerance_hz=None,
                           interference_ranges=None, *, record,directory,cancel,progress):
    from .analysis import recipe,plot
    means,counts,parent=load_group_averages(group_run_id)
    train=_indices(discovery_groups,len(means),'discovery_groups')
    valid=_indices(validation_groups,len(means),'validation_groups')
    if len(train)<2 or len(valid)<2 or set(train)&set(valid):
        raise ValueError('Need at least two disjoint discovery and validation groups each.')
    if type(max_candidates) is not int or not 1<=max_candidates<=20 or not np.isfinite(snr_threshold) or snr_threshold<3:
        raise ValueError('Invalid candidate budget or operational SNR threshold.')
    fs=parent['sampling_rate_hz']; interference_ranges=interference_ranges or []
    for label,bands in [('ranges',ranges),('noise_ranges',noise_ranges),('interference_ranges',interference_ranges)]:
        if not isinstance(bands,list) or len(bands)>8 or (label!='interference_ranges' and not bands):
            raise ValueError('Invalid '+label)
        for pair in bands:
            if len(pair)!=2 or not np.isfinite(pair).all() or not 0<pair[0]<pair[1]<fs/2:
                raise ValueError('Bands must lie strictly inside Nyquist.')
    if any(max(a,c)<min(b,d) for a,b in ranges for c,d in noise_ranges):
        raise ValueError('Reference noise bands must not overlap target bands.')
    spectra=[]
    for row in means:
        y,_,params=recipe(row,fs,preprocessing or {})
        spectra.append(np.fft.rfft(y)/len(y))
    spectra=np.asarray(spectra); f=np.fft.rfftfreq(len(y),1/fs);df=fs/len(y)
    tolerance=2*df if frequency_tolerance_hz is None else float(frequency_tolerance_hz)
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Frequency tolerance must be finite and nonnegative.')
    noise_mask=np.zeros(len(f),bool)
    for lo,hi in noise_ranges:
        noise_mask|=(f>=lo)&(f<=hi)
    if noise_mask.sum()<8:
        raise ValueError('Reference noise bands need at least eight native bins.')
    discovery=weighted_repeat_statistics(spectra[train],counts[train])
    validation=weighted_repeat_statistics(spectra[valid],counts[valid])
    records=[]; arrays={'frequency_hz':f,'discovery_mean':discovery['mean'],
                        'validation_mean':validation['mean'],'noise_reference_mask':noise_mask}
    for label,stat in [('discovery',discovery),('validation',validation)]:
        for key in ['standard_error','repeat_snr','coherence']:
            arrays[label+'_'+key]=stat[key]
    phase_mask=(discovery['repeat_snr']>=snr_threshold)&(validation['repeat_snr']>=snr_threshold)
    arrays['phase_reliability_mask']=phase_mask
    for bi,(lo,hi) in enumerate(ranges):
        if cancel():raise InterruptedError('Repeat-signal inspection cancelled.')
        bins=np.flatnonzero((f>=lo)&(f<=hi))
        if len(bins)<8:raise ValueError('Each target range needs at least eight native bins.')
        peaks,_=find_peaks(abs(discovery['mean'][bins]))
        selected=peaks[np.argsort(abs(discovery['mean'][bins[peaks]]))[::-1][:max_candidates]]
        val_peaks,_=find_peaks(abs(validation['mean'][bins]))
        val_f=f[bins[val_peaks]]
        plot(directory/f'band_{bi}_spectrum.png',[(f[bins],abs(stat['mean'][bins]),label) for label,stat in [('Discovery',discovery),('Validation',validation)]],
             'Frequency (Hz)','Magnitude (ADC units)','Disjoint group means')
        traces=[]
        for label,stat in [('Discovery',discovery),('Validation',validation)]:
            phase=np.angle(stat['mean'][bins]*np.exp(-2j*np.pi*f[bins]*params['actual_start_s']))
            traces.append((f[bins],np.where(phase_mask[bins],phase,np.nan),label))
        plot(directory/f'band_{bi}_phase.png',traces,'Frequency (Hz)','Wrapped phase (rad)',
             'Acquisition-referenced phase: repeat-SNR-masked bins only')
        for pi,local in enumerate(selected):
            index=int(bins[local]);frequency=float(f[index])
            distance=float(np.min(abs(val_f-frequency))) if len(val_f) else None
            ds=float(discovery['repeat_snr'][index]);vs=float(validation['repeat_snr'][index])
            known=any(a<=frequency<=b for a,b in interference_ranges)
            label=classify_reproducibility(ds,vs,distance is not None and distance<=tolerance,known,snr_threshold)
            # Accumulation diagnostics use discovery only; validation is not
            # reused to choose signal frequencies or reference noise bands.
            small=np.column_stack([spectra[train,index],spectra[train][:,noise_mask]])
            small_mask=np.ones(small.shape[1],bool);small_mask[0]=False
            accumulation=accumulation_diagnostic(small,counts[train],small_mask,0)
            levels=accumulation['levels'];sizes=[r['median_total_scans'] for r in levels]
            errors=[r['median_difference_noise'] for r in levels]
            plot(directory/f'band_{bi}_peak_{pi}_accumulation.png',
                 [(sizes,errors,'Measured pool-difference noise'),
                  (sizes,errors[0]*np.sqrt(sizes[0]/np.asarray(sizes)),'Reference 1/sqrt(N)')],
                 'Total scans in two disjoint pools','Noise estimate (ADC units)','Measured accumulation trend (correlated subset summaries)')
            group_phase=np.angle(spectra[:,index]*discovery['mean'][index].conjugate())
            single_noise=np.sqrt(discovery['single_scan_variance'][index]/counts)
            reliable=abs(spectra[:,index])>=snr_threshold*single_noise
            plot(directory/f'band_{bi}_peak_{pi}_group_phase.png',
                 [(np.arange(len(means)),np.where(reliable,group_phase,np.nan),'Relative phase')],
                 'Acquisition subset index','Wrapped relative phase (rad)',f'{frequency:.4f} Hz: no alignment applied')
            records.append({'band_index':bi,'frequency_hz':frequency,'classification':label,
                            'discovery_repeat_snr':ds if np.isfinite(ds) else None,
                            'validation_repeat_snr':vs if np.isfinite(vs) else None,
                            'zero_scatter':not np.isfinite(ds) or not np.isfinite(vs),
                            'validation_nearest_peak_distance_hz':distance,
                            'discovery_coherence':float(discovery['coherence'][index]),
                            'validation_coherence':float(validation['coherence'][index]),
                            'known_interference_band':known,'accumulation':accumulation,
                            'group_phase_rad':[float(p) if good else None for p,good in zip(group_phase,reliable)]})
        progress(bi+1,len(ranges))
    np.savez_compressed(directory/'repeat_signal_arrays.npz',**arrays)
    return {'parent_run_id':group_run_id,'source_arrays_sha256':parent['arrays_sha256'],
            'discovery_groups':train,'validation_groups':valid,'preprocessing':params,
            'ranges_hz':ranges,'reference_noise_ranges_hz':noise_ranges,'candidates':records,
            'operational_snr_threshold':snr_threshold,'frequency_tolerance_hz':tolerance,
            'warnings':['Repeat SNR includes group variation and drift, not a calibrated detection p value.',
                        'Peaks are proposed only from discovery data; validation checks reuse no fitted model.',
                        'Coherent interference can be reproducible and obey sqrt(N) SNR growth.',
                        'Reference noise bands are caller assumptions; signal leakage or drift can contaminate them.',
                        'Subset accumulation levels and permutations are correlated; no slope significance is claimed.',
                        'Phase is hidden below the operational repeat-SNR mask; no phase correction was applied.',
                        'No multiple-testing correction or automatic molecular assignment is performed.']}
