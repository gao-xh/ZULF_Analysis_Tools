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
