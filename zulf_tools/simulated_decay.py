"""Conditional effective decay fits using a fixed isopropylamine J model."""
import numpy as np
from matplotlib.figure import Figure
from . import storage
from .repeats import load_group_averages
from .jfit import transitions, ProcessedSpectrum
from .transition_decay import fit_transition_decay, transition_design


def fit_simulated_decay(fit_run_id, model_run_id, shared_decay=False, isotopomers=None,
                         source_j_fit_run_id=None, settings=None, *, record, directory, cancel, progress):
    from .analysis import recipe, plot
    parent=storage.get_result(fit_run_id);model=storage.get_result(model_run_id)
    if parent['operation']!='fit_frequency_decay' or model['operation']!='build_isopropylamine_model':
        raise ValueError('Require completed frequency-decay and isopropylamine model runs.')
    if source_j_fit_run_id is not None:
        source_fit=storage.get_result(source_j_fit_run_id)
        if source_fit.get('parameters_hz')!=model['parameters_hz']:
            raise ValueError('J source fit parameters differ from the supplied model.')
    names=['methine','methyl'] if isotopomers is None else isotopomers
    if not isinstance(names,list) or not names or len(set(names))!=len(names) or any(n not in ('methine','methyl') for n in names):
        raise ValueError('Invalid isotopomers.')
    config=dict(starts=6,max_nfev=150,max_evaluations=2000,max_seconds=60.,seed=20260922,equal_band_weight=True)
    if settings and set(settings)-set(config):
        raise ValueError('Unknown simulated decay settings.')
    config.update(settings or {})
    if type(config['equal_band_weight']) is not bool:
        raise ValueError('equal_band_weight must be boolean.')
    means,counts,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']:
        raise ValueError('Group source does not match fitting provenance.')
    fs=source['sampling_rate_hz'];spec=parent['parameters'].get('preprocessing') or {}
    spectra={}
    for name,indices in [('discovery',parent['discovery_groups']),('validation',parent['validation_groups'])]:
        raw=np.average(means[indices],axis=0,weights=counts[indices])
        y,_,params=recipe(raw,fs,spec)
        spectra[name]=np.fft.rfft(y)/len(y)
    frequency=np.fft.rfftfreq(len(y),1/fs);bins=[];membership=[]
    for i,(lo,hi) in enumerate(parent['ranges_hz']):
        selected=np.flatnonzero((frequency>=lo)&(frequency<=hi));bins.extend(selected);membership.extend([i]*len(selected))
    bins=np.asarray(bins,dtype=int);membership=np.asarray(membership)
    if len(bins)>5000:
        raise ValueError('Selected ranges exceed 5000 native bins.')
    p=ProcessedSpectrum(fs,source['points'],params['start_sample'],params['stop_sample'],bins,params['sg_window'],params['sg_order'])
    observed=spectra['discovery'][bins];held=spectra['validation'][bins]
    scale=np.ones(len(bins))
    if config['equal_band_weight']:
        for i in range(len(parent['ranges_hz'])):
            mask=membership==i
            scale[mask]=max(float(np.sqrt(np.mean(abs(observed[mask])**2))),1e-15)
    groups=[]
    for name in names:
        if cancel(): raise InterruptedError('Simulated decay cancelled.')
        f,w=transitions(model['parameters_hz'],name)
        groups.append(dict(name=name,frequencies_hz=f.tolist(),weights=w.tolist()))
        fig=Figure(figsize=(10,4.6),layout='constrained');ax=fig.add_subplot(111)
        ax.vlines(f,0,w/w.sum(),linewidth=.8)
        ax.set(xlabel='Transition frequency (Hz)',ylabel='Normalized model weight',title=name.title()+': all retained simulated transitions')
        fig.savefig(directory/f'{name}_transition_sticks.png',dpi=150)
    storage.write_json(directory/'transition_groups.json',groups)
    budgets={k:v for k,v in config.items() if k!='equal_band_weight'}
    fit=fit_transition_decay(p,observed,groups,parent['t2_bounds_s'],shared_decay=shared_decay,
                             observation_scale=scale,cancel=cancel,**budgets)
    prediction=fit.pop('fitted');progress(1,2)
    def error(a,b):
        norm=float(np.linalg.norm(a))
        return float(np.linalg.norm(a-b)/norm) if norm else None
    design=transition_design(p,groups,fit['t2star_s'])
    components=[design[:,2*i:2*i+2]@np.asarray(fit['cos_sin_coefficients'][i]) for i in range(len(groups))]
    arrays=dict(frequency_hz=p.f,discovery=observed,validation=held,prediction=prediction,observation_scale=scale)
    for name,c in zip(names,components): arrays['component_'+name]=c
    bands=[]
    for i,bounds in enumerate(parent['ranges_hz']):
        if cancel(): raise InterruptedError('Simulated decay cancelled.')
        mask=membership==i
        bands.append(dict(range_hz=bounds,discovery_error=error(observed[mask],prediction[mask]),validation_error=error(held[mask],prediction[mask])))
        for part,transform in [('magnitude',np.abs),('real',np.real),('imaginary',np.imag)]:
            traces=[(p.f[mask],transform(a[mask]),name) for name,a in [('Discovery',observed),('Validation',held),('Frozen simulation',prediction)]]
            traces += [(p.f[mask],transform(c[mask]),name) for name,c in zip(names,components)]
            plot(directory/f'band_{i}_{part}.png',traces,'Frequency (Hz)',part.title()+' (ADC units)',f'Fixed J, {bounds[0]:g}-{bounds[1]:g} Hz: conditional decay fit')
        residual=held[mask]-prediction[mask]
        plot(directory/f'band_{i}_residual.png',[(p.f[mask],residual.real,'Real'),(p.f[mask],residual.imag,'Imaginary')],
            'Frequency (Hz)','Complex residual (ADC units)','Frozen conditional prediction residual')
    np.savez_compressed(directory/'simulated_decay_arrays.npz',**arrays);progress(2,2)
    return dict(parent_run_id=fit_run_id,model_run_id=model_run_id,source_j_fit_run_id=source_j_fit_run_id,
        parameters_hz=model['parameters_hz'],model_parameter_digest=storage.digest(model['parameters_hz']),
        source_arrays_sha256=source['arrays_sha256'],isotopomers=names,shared_decay=shared_decay,
        discovery_groups=parent['discovery_groups'],validation_groups=parent['validation_groups'],
        t2_bounds_s=parent['t2_bounds_s'],preprocessing=spec,settings=config,fit=fit,bands=bands,
        validation_relative_complex_residual=error(held,prediction),scientifically_validated=False,
        warnings=['J and relative transition weights are fixed assumptions, not independently established facts.',
        'If J was estimated from all scans, validation is conditional on that selection, not untouched validation of J.',
        'One gain/phase pair per isotopomer; all its supplied transitions share a decay time.',
        'Unpulsed isotropic thermal response weights approximate unknown experimental preparation.',
        'NH2/14N and inter-methyl coupling are omitted as specified by the source model.',
        'Component magnitudes are shown separately but are never summed to construct total magnitude.',
        'Effective FID T2* and a smaller residual do not establish intrinsic T2 or unique J.'])
