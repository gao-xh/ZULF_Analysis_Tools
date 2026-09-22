"""Discovery-only crop proposals with held-out diagnostics, never automatic trimming."""
import numpy as np
from scipy.signal import get_window
from . import storage
from .repeats import load_group_averages
from .repeat_statistics import weighted_repeat_statistics
from .band_relaxation import _indices


def band_blocks(values, counts, fs, ranges, width_s):
    """Complete nonoverlapping Hann blocks; band RMS is a diagnostic, not a decay fit."""
    values=np.asarray(values,float)
    if values.ndim!=2 or not np.isfinite(values).all(): raise ValueError('Require finite group FIDs.')
    if not np.isfinite([fs,width_s]).all() or fs<=0 or width_s<=0: raise ValueError('Invalid window settings.')
    width=int(round(width_s*fs));blocks=values.shape[1]//width if width else 0
    if width<16 or blocks<4: raise ValueError('Require at least four complete windows of 16 samples.')
    f=np.fft.rfftfreq(width,1/fs);window=get_window('hann',width,fftbins=True)
    data=values[:,:blocks*width].reshape(len(values),blocks,width)
    spectrum=np.fft.rfft(data*window,axis=-1)*2/window.sum()
    stats=weighted_repeat_statistics(spectrum.reshape(len(values),-1),counts)
    mean=stats['mean'].reshape(blocks,-1);sem=stats['standard_error'].reshape(blocks,-1)
    amplitude=[];error=[]
    for lo,hi in ranges:
        mask=(f>=lo)&(f<=hi)
        if mask.sum()<2: raise ValueError('Each band needs at least two native window FFT bins; increase width.')
        amplitude.append(np.sqrt(np.mean(abs(mean[:,mask])**2,axis=1)))
        error.append(np.sqrt(np.mean(sem[:,mask]**2,axis=1)))
    amplitude=np.array(amplitude);error=np.array(error)
    # Finite floor avoids infinite manifest values for synthetic identical groups.
    snr=amplitude/np.maximum(error,np.finfo(float).eps*np.maximum(1,amplitude))
    return dict(times_s=(np.arange(blocks)+.5)*width/fs,amplitude=amplitude,
        standard_error=error,snr=snr,actual_width_s=width/fs,
        unused_tail_samples=values.shape[1]-blocks*width,nominal_resolution_hz=fs/width)


def propose_intervals(discovery_snr, times_s, width_s, duration_s, start_s, threshold):
    """Keep full record plus guarded alternatives; last activity includes revivals."""
    active=np.any(np.asarray(discovery_snr)>=threshold,axis=0)
    last=np.flatnonzero(active)
    # No detected signal does not justify discarding all data.
    end=duration_s if not len(last) else min(duration_s,float(times_s[last[-1]]+1.5*width_s))
    end=max(end,start_s+4*width_s)
    end=min(end,duration_s)
    proposals=[dict(start_s=0.,end_s=duration_s,reason='Untrimmed baseline')]
    for start,stop,reason in [(start_s,duration_s,'Discovery baseline/SG guard; preserve full tail'),
        (start_s,end,'After last discovery-supported window plus one window guard'),
        (max(0,start_s-width_s),end,'Earlier start sensitivity'),
        (min(start_s+width_s,duration_s-4*width_s),end,'Later start sensitivity')]:
        if stop-start>=4*width_s and not any(abs(p['start_s']-start)<1e-10 and abs(p['end_s']-stop)<1e-10 for p in proposals):
            proposals.append(dict(start_s=float(start),end_s=float(stop),reason=reason))
    return proposals


def guarded_start(recovery_s, guard_s, duration_s):
    cap=min(1.,duration_s*.1)
    rejected=recovery_s is None or recovery_s>cap
    return min(cap,max(0. if rejected else recovery_s,guard_s)),rejected,guard_s>cap


def inspect_fid_crops(group_run_id, ranges, discovery_groups, validation_groups,
                      sg_window=0, sg_order=2, width_s=.25, snr_threshold=5.,
                      *,record,directory,cancel,progress):
    from .analysis import recipe,plot
    means,counts,source=load_group_averages(group_run_id)
    train=_indices(discovery_groups,len(means),'discovery_groups');valid=_indices(validation_groups,len(means),'validation_groups')
    if min(len(train),len(valid))<2 or set(train)&set(valid): raise ValueError('Need two disjoint groups per role.')
    fs=source['sampling_rate_hz'];duration=source['points']/fs
    if not isinstance(ranges,list) or not 1<=len(ranges)<=6: raise ValueError('Specify 1..6 frequency ranges.')
    for band in ranges:
        if len(band)!=2 or not np.isfinite(band).all() or not 0<band[0]<band[1]<fs/2: raise ValueError('Invalid frequency range.')
    if not np.isfinite(snr_threshold) or snr_threshold<3: raise ValueError('Use an operational SNR threshold of at least 3.')
    processed=[]
    for row in means:
        if cancel(): raise InterruptedError('Crop inspection cancelled.')
        y,_,params=recipe(row,fs,dict(sg_window=sg_window,sg_order=sg_order,remove_mean=False))
        processed.append(y)
    processed=np.array(processed);diagnostics={};arrays={}
    width=int(round(width_s*fs))
    for label,indices in [('discovery',train),('validation',valid)]:
        d=band_blocks(processed[indices],counts[indices],fs,ranges,width_s)
        raw=np.average(means[indices],axis=0,weights=counts[indices])
        coherent=np.average(processed[indices],axis=0,weights=counts[indices])
        local=means[indices,:len(d['times_s'])*width].reshape(len(indices),-1,width).mean(axis=-1)
        stat=weighted_repeat_statistics(local,counts[indices])
        d['local_raw_mean']=stat['mean'].real;d['local_mean_sem']=stat['standard_error']
        diagnostics[label]=d
        arrays[label+'_raw_fid']=raw;arrays[label+'_processed_fid']=coherent
        for key in ['times_s','amplitude','standard_error','snr','local_raw_mean','local_mean_sem']: arrays[label+'_'+key]=d[key]
    d=diagnostics['discovery'];times=d['times_s'];actual=d['actual_width_s']
    tail=d['local_raw_mean'][-max(3,len(times)//4):]
    center=float(np.median(tail));spread=float(1.4826*np.median(abs(tail-center)))
    noise=max(spread,float(np.median(d['local_mean_sem'])),np.finfo(float).eps)
    settled=abs(d['local_raw_mean']-center)<=10*noise
    # Three consecutive blocks are a heuristic baseline-recovery proposal only.
    first=next((i for i in range(len(times)-2) if settled[i:i+3].all()),None)
    raw_start=0. if first is None else max(0.,float(times[first]-actual/2))
    # A long raw baseline can coexist with a short-lived signal after SG.
    # Do not turn the cap into a presumed clean start and discard that signal.
    guard=(sg_window//2)/fs if sg_window else 0.
    start,fallback,capped=guarded_start(raw_start if first is not None else None,guard,duration)
    proposals=propose_intervals(d['snr'],times,actual,duration,start,snr_threshold)
    for p in proposals:
        omitted=times-actual/2>=p['end_s']
        p['validation_supported_windows_after_end']=int(np.sum(np.any(diagnostics['validation']['snr'][:,omitted]>=snr_threshold,axis=0)))
        p['requires_review']=bool(p['validation_supported_windows_after_end'])
    t=np.arange(source['points'])/fs
    for label,mask in [('full',np.ones(len(t),bool)),('early',t<min(1.,duration/4)),('tail',t>=duration-min(2.,duration/4))]:
        indices=np.flatnonzero(mask);indices=indices[::max(1,len(indices)//15000)]
        for kind in ['raw','processed']:
            plot(directory/f'{label}_{kind}_fid.png',[(t[indices],arrays[role+'_'+kind+'_fid'][indices],role.title()) for role in diagnostics],
                'Acquisition time (s)','ADC amplitude',f'{label.title()} {kind} FID; display sampled, data unchanged')
    plot(directory/'local_baseline.png',[(times,diagnostics[role]['local_raw_mean'],role.title()) for role in diagnostics],
        'Acquisition time (s)','Local raw mean (ADC units)','Baseline recovery diagnostic; complete blocks')
    for i,band in enumerate(ranges):
        for key,ylabel in [('amplitude','Window band RMS (ADC units)'),('snr','Band RMS / repeat-scatter SEM')]:
            plot(directory/f'band_{i}_{key}.png',[(times,diagnostics[role][key][i],role.title()) for role in diagnostics],
                'Acquisition time (s)',ylabel,f'{band[0]:g}-{band[1]:g} Hz: Hann-window diagnostic')
    np.savez_compressed(directory/'crop_diagnostics.npz',**arrays)
    storage.write_json(directory/'crop_candidates.json',proposals);progress(1,1)
    return dict(parent_run_id=group_run_id,source_arrays_sha256=source['arrays_sha256'],
        discovery_groups=train,validation_groups=valid,ranges_hz=ranges,preprocessing=params,
        actual_width_s=actual,nominal_resolution_hz=d['nominal_resolution_hz'],unused_tail_samples=d['unused_tail_samples'],
        snr_threshold=snr_threshold,baseline_tail_reference=center,baseline_tolerance=10*noise,
        baseline_recovery_candidate_s=raw_start if first is not None else None,
        baseline_start_rule_rejected=fallback,start_proposal_capped=capped,
        candidate_intervals=proposals,scientifically_validated=False,
        warnings=['Proposals use discovery only; held-out diagnostics do not select or change them.',
        'Tail baseline reference may contain a slow signal or drift; no baseline recovery is proven.',
        'Hann band RMS mixes peaks, beating and leakage; it is not a relaxation envelope or a noise significance test.',
        'Nonoverlapping windows remain correlated under filtering and acquisition drift.',
        'SG is full-record mirror baseline subtraction; no data are trimmed by this tool.',
        'Raw first point and acquisition time are retained. Compare candidate fits with identical simulation processing.',
        'Undetected tail signal may remain. Preserve the untrimmed comparison and inspect all independent figures.'])
