"""Noise-informed numerical exploration bounds, never measured physical limits."""
import numpy as np


def noise_aware_t2_proposal(processed_groups,counts,fs,ranges,start_s=0.):
    from .crop_diagnostics import band_blocks
    y=np.asarray(processed_groups,float);counts=np.asarray(counts,float)
    if y.ndim!=2 or y.shape[1]<32 or not np.isfinite(y).all() or not np.isfinite([fs,start_s]).all() or fs<=0 or start_s<0:
        raise ValueError('Require finite processed discovery FIDs and sampling settings.')
    if counts.shape!=(len(y),) or not np.isfinite(counts).all() or np.any(counts<=0): raise ValueError('Invalid group counts.')
    duration=y.shape[1]/fs;guard=[max(4/fs,duration/500),duration*2]
    rows=[];horizons=[]
    for band in ranges:
        if len(band)!=2 or not np.isfinite(band).all() or not 0<band[0]<band[1]<fs/2: raise ValueError('Invalid proposal frequency range.')
        width=max(16/fs,3/(band[1]-band[0]),duration/64)
        if len(y)<2 or round(width*fs)*4>y.shape[1]:
            rows.append(dict(range_hz=band,status='insufficient_repeat_or_window_information',informative_horizon_s=duration));horizons.append(duration);continue
        diagnostic=band_blocks(y,counts,fs,[band],width)
        active=diagnostic['snr'][0]>=5.
        supported=np.flatnonzero(active)
        # Isolated support is insufficient to justify shortening the search span.
        ready=len(supported)>=2
        horizon=min(duration,float(diagnostic['times_s'][supported[-1]]+1.5*diagnostic['actual_width_s'])) if ready else duration
        horizons.append(horizon)
        rows.append(dict(range_hz=band,status='exploratory_supported_horizon' if ready else 'insufficient_signal_support',
            informative_horizon_s=horizon,actual_width_s=diagnostic['actual_width_s'],
            nominal_resolution_hz=diagnostic['nominal_resolution_hz'],unused_tail_samples=diagnostic['unused_tail_samples'],
            times_s=(start_s+diagnostic['times_s']).tolist(),band_rms=diagnostic['amplitude'][0].tolist(),
            repeat_scatter_sem=diagnostic['standard_error'][0].tolist(),rms_ratio=diagnostic['snr'][0].tolist(),
            supported_windows=int(active.sum())))
    if not horizons: raise ValueError('Require at least one frequency band.')
    bounds=[max(4/fs,min(horizons)/500),2*max(horizons)]
    return dict(t2_bounds_s=bounds,full_record_guard_bounds_s=guard,retained_duration_s=duration,
        actual_start_s=start_s,bands=rows,operational_rms_ratio_threshold=5.,
        selection='Discovery groups only; union of band-specific supported horizons.',
        rule='Lower: max(4 sample periods, shortest supported horizon/500). Upper: twice longest supported horizon. Inadequate evidence retains full-record horizon.',
        warnings=['Numerical exploration limits, not a T2* estimate, confidence interval or measured physical prior.',
        'Repeat scatter includes drift; Hann band RMS mixes peaks, leakage and beating. Threshold 5 is an operational heuristic.',
        'Weak slow components may fall below support threshold; retain the full-record guard range for explicit sensitivity checks.',
        'A boundary hit requires limited expansion, not acceptance. Signal classification and crop diagnostics remain separate.'])
