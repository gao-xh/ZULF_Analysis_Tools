"""Range-restricted effective decay analysis with disjoint-group evaluation."""
import time
import hashlib
import numpy as np
from . import storage
from .repeats import load_group_averages
from .jfit import ProcessedSpectrum
from .decay import fit_modes, mode_design, real_projection
from .repeat_statistics import weighted_repeat_statistics


def _indices(value, size, label):
    if not isinstance(value,list) or not value or any(type(i) is not int or not 0<=i<size for i in value):
        raise ValueError(label+' must contain valid group indices.')
    if len(set(value))!=len(value):
        raise ValueError(label+' must not contain duplicates.')
    return value


def fit_frequency_decay(group_run_id, ranges, discovery_groups, validation_groups,
                        t2_bounds=None, components=None, preprocessing=None,
                        settings=None, *, record, directory, cancel, progress):
    """Fit discovery mean; predict validation without changing any coefficient.

    Validation is a model-comparison set, not an untouched final test set after
    model selection. No physical component assignment or automatic acceptance.
    """
    from .analysis import recipe, plot
    started=time.perf_counter()
    means,counts,parent=load_group_averages(group_run_id)
    train=_indices(discovery_groups,len(means),'discovery_groups')
    valid=_indices(validation_groups,len(means),'validation_groups')
    if set(train)&set(valid):
        raise ValueError('Discovery and validation groups must be disjoint.')
    fs=parent['sampling_rate_hz']; points=parent['points']
    if not isinstance(ranges,list) or not 1<=len(ranges)<=6:
        raise ValueError('Supply one to six disjoint frequency ranges.')
    for i,pair in enumerate(ranges):
        if len(pair)!=2 or not np.isfinite(pair).all() or not 0<pair[0]<pair[1]<fs/2:
            raise ValueError('Frequency ranges must be finite and inside Nyquist.')
        if i and pair[0]<=ranges[i-1][1]:
            raise ValueError('Frequency ranges must be sorted and disjoint.')
    components=[1,2] if components is None else components
    if not isinstance(components,list) or not components or any(type(k) is not int or not 1<=k<=8 for k in components) or len(set(components))!=len(components):
        raise ValueError('components must be distinct integers from 1 to 8.')
    s=dict(starts=4,max_nfev=150,max_evaluations=3000,max_seconds=60.,
           total_seconds=600.,seed=20260922,compare_shared_decay=True,background=False,
           initial_frequencies_hz=None)
    if settings and set(settings)-set(s):
        raise ValueError('Unknown frequency-decay settings.')
    s.update(settings or {})
    initial=s['initial_frequencies_hz']
    if initial is not None:
        if len(components)!=1 or not isinstance(initial,list) or len(initial)!=len(ranges):
            raise ValueError('Explicit initial frequencies require one component count and one list per band.')
        for seeds,(lo,hi) in zip(initial,ranges):
            values=np.asarray(seeds,dtype=float)
            if values.shape!=(components[0],) or not np.isfinite(values).all() or np.any((values<lo)|(values>hi)):
                raise ValueError('Initial frequencies must match the mode count and lie inside their band.')
    if not np.isfinite(s['total_seconds']) or s['total_seconds']<=0 or type(s['compare_shared_decay']) is not bool:
        raise ValueError('Invalid total budget or compare_shared_decay flag.')
    spec=preprocessing or {}
    processed=[]
    for group in means:
        y,_,params=recipe(group,fs,spec)
        processed.append(y)
    processed=np.asarray(processed)
    n=processed.shape[1]; duration=n/fs
    bounds_proposal=None
    if t2_bounds is None:
        from .decay_bounds import noise_aware_t2_proposal
        bounds_proposal=noise_aware_t2_proposal(processed[train],counts[train],fs,ranges,params['actual_start_s'])
        t2_bounds=bounds_proposal['t2_bounds_s']
        bounds_origin='Exploratory discovery repeat-scatter/window horizon; not measured physical limits.'
        storage.write_json(directory/'t2_search_proposal.json',bounds_proposal)
        for i,row in enumerate(bounds_proposal['bands']):
            if 'times_s' in row:
                plot(directory/f'band_{i}_bounds_support.png',[(row['times_s'],row['rms_ratio'],'Discovery band RMS / repeat SEM')],
                    'Acquisition time (s)','Operational RMS ratio','Exploratory bound support; weak slow signal may remain')
    else:
        bounds_origin='Explicit caller-specified interval in seconds.'
    if len(t2_bounds)!=2 or not np.isfinite(t2_bounds).all() or not 0<t2_bounds[0]<t2_bounds[1]:
        raise ValueError('T2* bounds must be finite, positive and increasing.')
    spectra=np.fft.rfft(processed,axis=1)/n
    frequency=np.fft.rfftfreq(n,1/fs)
    train_y=np.average(spectra[train],axis=0,weights=counts[train])
    validation_y=np.average(spectra[valid],axis=0,weights=counts[valid])
    t=np.arange(n)/fs+params['actual_start_s']
    stride=max(1,n//12000)
    for label,indices in [('discovery',train),('validation',valid)]:
        average=np.average(processed[indices],axis=0,weights=counts[indices])
        plot(directory/f'{label}_fid.png',[(t[::stride],average[::stride],label)],
             'Recorded time (s)','ADC amplitude',label.title()+' mean after explicit preprocessing')
    records=[]; arrays={}; work=0; halted=False
    configurations=[(k,shared) for k in sorted(components) for shared in ([True,False] if k>1 and s['compare_shared_decay'] else [False])]
    total=len(ranges)*len(configurations)
    band_inputs=[]
    schedule=[dict(band_index=band,mode_count=k,shared_decay=shared)
              for k,shared in configurations for band in range(len(ranges))]
    for band,(lo,hi) in enumerate(ranges):
        bins=np.flatnonzero((frequency>=lo)&(frequency<=hi))
        if len(bins)<8 or len(bins)>5000:
            raise ValueError('Each requested band must contain 8..5000 native FFT bins.')
        if 2*len(bins)<=4*max(components)+2:
            raise ValueError(f'Band {band} has too few native observations for the requested mode counts.')
        p=ProcessedSpectrum(fs,points,params['start_sample'],params['stop_sample'],bins,params['sg_window'],params['sg_order'])
        observed=train_y[bins]; held=validation_y[bins]
        # Empirical repeat scatter, including drift. Not stationary thermal noise.
        if len(train)>1:
            scatter=weighted_repeat_statistics(spectra[train][:,bins],counts[train])['standard_error']
        else:
            scatter=np.full(len(bins),np.nan)
        arrays[f'band_{band}_frequency_hz']=p.f
        arrays[f'band_{band}_discovery']=observed
        arrays[f'band_{band}_validation']=held
        arrays[f'band_{band}_group_spectra']=spectra[:,bins]
        arrays[f'band_{band}_discovery_scatter']=scatter
        band_inputs.append((band,(lo,hi),bins,p,observed,held))
    # Give every band its simplest requested baseline before spending the
    # remaining budget on more complex configurations of any one band.
    for modes,shared in configurations:
        for band,(lo,hi),bins,p,observed,held in band_inputs:
            if cancel():
                raise InterruptedError('Frequency-decay analysis cancelled.')
            remaining=s['total_seconds']-(time.perf_counter()-started)
            if remaining<=0:
                halted=True
                break
            fit=fit_modes(p,observed,[lo,hi],t2_bounds,mode_count=modes,shared_decay=shared,
                          initial_frequencies=None if initial is None else initial[band],
                          starts=s['starts'],max_nfev=s['max_nfev'],max_evaluations=s['max_evaluations'],
                          max_seconds=min(s['max_seconds'],remaining),seed=s['seed']+band*100+modes,
                          background=s['background'],cancel=cancel)
            prediction=fit.pop('fitted')
            key=f'band_{band}_modes_{modes}_{"shared" if shared else "independent"}'
            arrays[key+'_prediction']=prediction
            # Frozen prediction is the primary held-out metric. Conditional gain
            # refits are a separately named drift/model-shape diagnostic only.
            denominator=float(np.linalg.norm(held))
            validation_error=float(np.linalg.norm(held-prediction)/denominator) if denominator else None
            group_errors=[]; group_coefficients=[]
            design=mode_design(p,fit['frequencies_hz'],fit['t2star_s'],s['background'])
            for g in valid:
                target=spectra[g,bins]
                conditional,coef,_,_=real_projection(design,target)
                norm=float(np.linalg.norm(target))
                group_errors.append({'group_index':g,
                    'frozen_relative_complex_residual':float(np.linalg.norm(target-prediction)/norm) if norm else None,
                    'conditional_gain_relative_complex_residual':float(np.linalg.norm(target-conditional)/norm) if norm else None})
                group_coefficients.append(coef.tolist())
            fit.update(band_index=band,range_hz=[lo,hi],mode_count=modes,
                       validation_relative_complex_residual=validation_error,
                       validation_group_errors=group_errors,
                       validation_conditional_coefficients=group_coefficients,
                       conditional_refit_note='Fixed discovery frequencies/T2*, amplitudes and phases re-estimated for diagnostics only.',
                       artifact_prefix=key,scientifically_validated=False)
            checkpoint=directory/(key+'_checkpoint.npz')
            np.savez_compressed(checkpoint,frequency_hz=p.f,discovery=observed,
                validation=held,prediction=prediction,group_spectra=spectra[:,bins],
                discovery_scatter=arrays[f'band_{band}_discovery_scatter'])
            fit['checkpoint_artifact']=checkpoint.name
            fit['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            records.append(fit)
            # Publish only after the uniquely named numeric checkpoint closes.
            # A subsequent cancellation or plotting failure retains this trial.
            storage.write_json(directory/'candidates.json',records)
            for part,transform in [('magnitude',np.abs),('real',np.real),('imaginary',np.imag)]:
                plot(directory/f'{key}_{part}.png',[(p.f,transform(observed),'Discovery mean'),
                     (p.f,transform(held),'Validation mean'),(p.f,transform(prediction),'Frozen prediction')],
                     'Frequency (Hz)',part.title()+' (ADC units)',f'{lo:g}-{hi:g} Hz: {modes} mode(s), '+('shared T2*' if shared else 'independent T2*'))
            plot(directory/f'{key}_residual.png',[(p.f,(held-prediction).real,'Validation real residual'),
                 (p.f,(held-prediction).imag,'Validation imaginary residual')],
                 'Frequency (Hz)','Complex residual (ADC units)','Held-out prediction residual')
            work+=1; progress(work,total)
        if halted:
            break
    # Preserve the established completed-result candidate ordering. Progress
    # files follow execution order until this final canonical rewrite.
    records.sort(key=lambda r:(r['band_index'],r['mode_count'],not r['shared_decay']))
    storage.write_json(directory/'candidates.json',records)
    np.savez_compressed(directory/'band_arrays.npz',**arrays)
    return {'parent_run_id':group_run_id,'source_arrays_sha256':parent['arrays_sha256'],
            'discovery_groups':train,'validation_groups':valid,'preprocessing':params,
            'ranges_hz':ranges,'t2_bounds_s':list(t2_bounds),'bounds_origin':bounds_origin,
            'bounds_proposal':bounds_proposal,
            'components':components,'settings':s,'candidates':records,
            'template_diagnostics':[dict(band_index=b,**p.template_diagnostics()) for b,_,_,p,_,_ in band_inputs],
            'configuration_schedule':schedule,
            'candidate_order':'Band index, mode count, shared before independent. Execution prioritizes simpler configurations across bands.',
            'completed_configurations':work,'requested_configurations':total,
            'total_budget_exhausted':halted,'elapsed_s':time.perf_counter()-started,
            'scientifically_validated':False,
            'warnings':['Modes are effective FID oscillators, not substances or intrinsic T2.',
                'No alignment or experimental phase correction was applied.',
                'Validation predictions freeze all discovery parameters, including gains and phases.',
                'Reusing validation for model choice makes it a comparison set, not a final untouched test set.',
                'Repeat scatter includes drift and signal variation; it is not a calibrated noise floor.',
                'Finite-record leakage from outside the selected band remains possible.',
                'This operation does not yet perform signal classification, confidence intervals or crop sensitivity.',
                'Candidate convergence or smaller residual is not scientific acceptance.']}
