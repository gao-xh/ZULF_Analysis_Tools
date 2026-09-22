"""Empirical repeat diagnostics; no automatic molecular assignment or p values."""
import numpy as np


def weighted_repeat_statistics(spectra, counts):
    """Mean and repeat-scatter standard error under independent equal-variance scans.

    Counts weight group means. Drift inflates scatter; stable coherent interference
    does not, so high repeat SNR is not evidence of molecular origin.
    """
    spectra=np.asarray(spectra,complex); counts=np.asarray(counts,float)
    if spectra.ndim!=2 or len(spectra)<2 or counts.shape!=(len(spectra),) or not np.isfinite(spectra).all() or not np.isfinite(counts).all() or np.any(counts<=0):
        raise ValueError('Need at least two finite group spectra with positive counts.')
    total=counts.sum(); mean=np.average(spectra,axis=0,weights=counts)
    single_scan_variance=np.sum(counts[:,None]*abs(spectra-mean)**2,axis=0)/(len(spectra)-1)
    standard_error=np.sqrt(single_scan_variance/total)
    snr=np.divide(abs(mean),standard_error,out=np.full(len(mean),np.inf),where=standard_error>0)
    coherence=np.divide(abs(mean),np.average(abs(spectra),axis=0,weights=counts),
                        out=np.zeros(len(mean)),where=np.average(abs(spectra),axis=0,weights=counts)>0)
    return {'mean':mean,'standard_error':standard_error,'repeat_snr':snr,
            'coherence':coherence,'single_scan_variance':single_scan_variance}


def accumulation_diagnostic(spectra, counts, noise_mask, target_index, permutations=32, seed=0):
    """Measure differences between disjoint pools at increasing acquisition counts.

    Noise is measured from each pair, not imposed as 1/sqrt(N). Subsets overlap
    between permutations and sizes: slopes are descriptive, not independent-test
    significance. Caller must supply and justify the reference noise bins.
    """
    spectra=np.asarray(spectra,complex); counts=np.asarray(counts,float)
    weighted_repeat_statistics(spectra,counts)
    noise_mask=np.asarray(noise_mask,bool)
    if noise_mask.shape!=(spectra.shape[1],) or noise_mask.sum()<8:
        raise ValueError('Provide at least eight reference noise bins.')
    if type(target_index) is not int or not 0<=target_index<spectra.shape[1]:
        raise ValueError('Invalid target bin.')
    if type(permutations) is not int or not 4<=permutations<=256:
        raise ValueError('Use 4..256 permutations.')
    rng=np.random.default_rng(seed); sizes=[]; k=1
    while k<=len(spectra)//2:
        sizes.append(k);k*=2
    records=[]
    for k in sizes:
        noise=[]; signal=[]; totals=[]
        for _ in range(permutations):
            order=rng.permutation(len(spectra));a=order[:k];b=order[k:2*k]
            na=counts[a].sum();nb=counts[b].sum()
            ma=np.average(spectra[a],axis=0,weights=counts[a]);mb=np.average(spectra[b],axis=0,weights=counts[b])
            pooled=(na*ma+nb*mb)/(na+nb)
            difference_rms=np.sqrt(np.mean(abs((ma-mb)[noise_mask])**2))
            pooled_noise=difference_rms*np.sqrt(na*nb)/(na+nb)
            noise.append(pooled_noise);signal.append(abs(pooled[target_index]));totals.append(na+nb)
        n=float(np.median(totals));s=float(np.median(signal));e=float(np.median(noise))
        records.append({'groups_per_pool':k,'median_total_scans':n,'median_signal_amplitude':s,
                        'median_difference_noise':e,'snr':s/e if e>0 else None,
                        'noise_interquartile_range':np.percentile(noise,[25,75]).tolist()})
    good=[r for r in records if r['median_difference_noise']>0]
    slope=float(np.polyfit(np.log([r['median_total_scans'] for r in good]),np.log([r['median_difference_noise'] for r in good]),1)[0]) if len(good)>=3 else None
    return {'levels':records,'noise_log_log_slope':slope,'independent_noise_expected_slope':-.5,
            'note':'Measured disjoint-pool differences; drift included. Correlated subset statistics, no significance claim.'}


def classify_reproducibility(discovery_snr, validation_snr, frequency_agreement,
                             known_interference=False, threshold=5.):
    """Operational labels; caller records thresholds, never calls them p values."""
    if known_interference:
        return 'suspected_interference'
    if min(discovery_snr,validation_snr)>=threshold and frequency_agreement:
        return 'reproducible_signal_candidate'
    if max(discovery_snr,validation_snr)<3:
        return 'noise_compatible'
    return 'insufficient_evidence'


def spectral_coherence(spectra,counts,reference,single_scan_variance,threshold=5.):
    """Descriptive norm cancellation and masked common phase, no correction.

    The noise proxy is a band RMS, with no independent-bin aggregation gain.
    A common complex gain is a shape diagnostic, not a frequency-drift estimate.
    """
    spectra=np.asarray(spectra,complex);counts=np.asarray(counts,float)
    reference=np.asarray(reference,complex);variance=np.asarray(single_scan_variance,float)
    weighted_repeat_statistics(spectra,counts)
    if reference.shape!=(spectra.shape[1],) or variance.shape!=reference.shape or not np.isfinite(reference).all() or not np.isfinite(variance).all() or np.any(variance<0) or not np.isfinite(threshold) or threshold<3:
        raise ValueError('Invalid coherence reference, variance or threshold.')
    norms=np.linalg.norm(spectra,axis=1);pooled=np.average(spectra,axis=0,weights=counts)
    denominator=float(np.average(norms,weights=counts));refnorm=float(np.linalg.norm(reference))
    rows=[]
    for i,y in enumerate(spectra):
        proxy=float(np.sqrt(variance.sum()/counts[i]));ratio=float(norms[i]/max(proxy,1e-15))
        overlap=np.vdot(reference,y)
        similarity=float(np.clip(abs(overlap)/(refnorm*norms[i]),0,1)) if refnorm*norms[i]>0 else 0.
        reliable=bool(refnorm>0 and norms[i]>0 and ratio>=threshold and similarity>=.8)
        rows.append(dict(relative_phase_rad=float(np.angle(overlap)) if reliable else None,
            gain_magnitude=float(abs(overlap)/refnorm**2) if refnorm>0 else None,
            normalized_shape_overlap=similarity,group_rms_signal_to_scatter=ratio,
            phase_reliable=reliable,zero_scatter_proxy=proxy==0))
    return dict(coherent_to_mean_group_norm=float(np.clip(np.linalg.norm(pooled)/denominator,0,1)) if denominator>0 else None,
        group_diagnostics=rows,shape_overlap_threshold=.8,rms_signal_to_scatter_threshold=threshold,
        note='Descriptive coherence includes noise, phase and spectral-shape differences. No alignment or independent-bin significance claim.')


def directional_residual_diagnostic(discovery_residual, validation_residuals, counts, threshold=3.):
    """Freeze discovery's direction, then estimate scatter of validation projections.

    Projection occurs before estimating scatter, retaining correlations among
    frequency bins. This is descriptive, not a calibrated hypothesis test.
    """
    direction = np.asarray(discovery_residual, complex)
    validation = np.asarray(validation_residuals, complex)
    stats = weighted_repeat_statistics(validation, counts)
    if direction.shape != stats['mean'].shape or not np.isfinite(direction).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Invalid discovery residual or threshold.')
    norm = float(np.linalg.norm(direction))
    floor = np.finfo(float).eps*max(norm, float(np.linalg.norm(stats['mean'])), 1.)*100
    if norm <= floor:
        return dict(available=False, requires_review=False, status='no_discovery_direction')
    projection = (validation @ (direction/norm).conj()).real
    projected = weighted_repeat_statistics(projection[:, None], counts)
    mean = float(projected['mean'][0].real)
    sem = float(projected['standard_error'][0])
    return dict(available=bool(sem > floor), requires_review=bool(mean > threshold*max(sem, floor)),
        status='measured' if sem > floor else 'zero_projection_scatter',
        validation_projection_mean=mean, validation_projection_sem=sem,
        signed_projection_to_sem=mean/sem if sem > floor else None,
        validation_group_projections=projection.tolist(), ratio_threshold=threshold,
        direction_selected_on='discovery_residual_only',
        note='Signed projection onto a frozen discovery direction. SEM comes from projected validation groups, retaining bin correlations. No p value, coverage claim or protection against repeated model selection or correlated acquisition groups.')


def residual_repeat_diagnostic(discovery,discovery_counts,validation,validation_counts,prediction,threshold=3.):
    """Band residual norms versus empirical mean uncertainty; not a chi-square test."""
    if len(discovery)<2 or len(validation)<2:
        return dict(available=False,requires_review=False,status='insufficient_repeat_groups')
    d=weighted_repeat_statistics(discovery,discovery_counts);v=weighted_repeat_statistics(validation,validation_counts)
    prediction=np.asarray(prediction,complex)
    if prediction.shape!=d['mean'].shape or v['mean'].shape!=prediction.shape or not np.isfinite(prediction).all() or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Invalid residual prediction or threshold.')
    rd=d['mean']-prediction;rv=v['mean']-prediction
    nd=float(np.linalg.norm(rd));nv=float(np.linalg.norm(rv))
    sd=float(np.linalg.norm(d['standard_error']));sv=float(np.linalg.norm(v['standard_error']))
    floor=np.finfo(float).eps*max(float(np.linalg.norm(d['mean'])),float(np.linalg.norm(v['mean'])),1.)*100
    ratios=[nd/sd if sd>floor else None,nv/sv if sv>floor else None]
    norm_mismatch=nv>threshold*max(sv,floor)
    directional=directional_residual_diagnostic(rd,np.asarray(validation)-prediction,validation_counts,threshold)
    mismatch=norm_mismatch or directional['requires_review']
    alignment=float(np.clip(np.vdot(rd,rv).real/(nd*nv),-1,1)) if min(nd,nv)>floor else None
    repeated=bool((norm_mismatch and nd>threshold*max(sd,floor) and alignment is not None and alignment>.5) or directional['requires_review'])
    return dict(available=bool(sd>floor and sv>floor),requires_review=bool(mismatch),
        status='reproducible_unmodelled_structure' if repeated else 'validation_residual_exceeds_repeat_scatter' if mismatch else 'no_large_residual_flag',
        discovery_residual_to_sem=ratios[0],validation_residual_to_sem=ratios[1],
        discovery_residual_norm=nd,validation_residual_norm=nv,discovery_sem_norm=sd,validation_sem_norm=sv,
        residual_alignment=alignment,reproducible_residual=repeated,ratio_threshold=threshold,
        directional_validation=directional,band_norm_requires_review=bool(norm_mismatch),
        note='Band norm ratios are operational diagnostics, not p values or independent-bin chi-square statistics. Discovery residual is in-sample; validation prediction is frozen.')
