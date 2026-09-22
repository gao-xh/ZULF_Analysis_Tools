"""Bounded magnitude-only comparison initialized from a complex decay candidate."""
import time
import numpy as np
from scipy.optimize import least_squares
from .decay import mode_design,fit_diagnostics,_BudgetReached,real_projection


def fit_magnitude_modes(processor,observed,frequency_bounds,t2_bounds,initial,
                        starts=3,max_nfev=150,max_evaluations=3000,max_seconds=60.,
                        seed=0,cancel=lambda:False):
    """Optimize |sum complex modes|; real cos/sin gains are nonlinear here.

    A complex-fit warm start is explicit, not an independent phase-free discovery.
    Simultaneously reversing all real gains has exactly identical magnitudes.
    """
    observed=np.asarray(observed,complex);f0=np.asarray(initial['frequencies_hz'],float)
    taus=np.asarray(initial['t2star_s'],float);coef=np.asarray(initial['cos_sin_coefficients'],float)
    k=len(f0);shared=initial['shared_decay'];nt=1 if shared else k
    if type(shared) is not bool or not 1<=k<=8 or taus.shape!=(k,) or coef.shape!=(k,2): raise ValueError('Invalid initial mode arrays.')
    if observed.shape!=np.asarray(processor.f).shape or len(observed)<max(8,3*k+nt+1) or not np.isfinite(observed).all() or np.linalg.norm(observed)==0:
        raise ValueError('Insufficient finite nonzero magnitude observations.')
    if len(frequency_bounds)!=2 or len(t2_bounds)!=2 or not np.isfinite([*frequency_bounds,*t2_bounds]).all() or not 0<frequency_bounds[0]<frequency_bounds[1]<processor.fs/2 or not 0<t2_bounds[0]<t2_bounds[1]:
        raise ValueError('Invalid frequency or T2* bounds.')
    if not np.isfinite(f0).all() or not np.isfinite(taus).all() or not np.isfinite(coef).all() or np.any(taus<=0): raise ValueError('Invalid initial parameters.')
    for value,lo,hi in [(starts,1,16),(max_nfev,2,2000),(max_evaluations,2,100000)]:
        if type(value) is not int or not lo<=value<=hi: raise ValueError('Invalid magnitude fitting budget.')
    if not np.isfinite(max_seconds) or max_seconds<=0: raise ValueError('Invalid time budget.')
    scales=np.repeat(np.maximum(np.linalg.norm(coef,axis=1),1e-12),2)
    lower=np.r_[np.full(k,frequency_bounds[0]),np.full(nt,np.log(t2_bounds[0])),np.full(2*k,-np.inf)]
    upper=np.r_[np.full(k,frequency_bounds[1]),np.full(nt,np.log(t2_bounds[1])),np.full(2*k,np.inf)]
    x0=np.r_[f0,np.log(taus[:1] if shared else taus),coef.ravel()/scales]
    scale=float(np.sqrt(np.mean(abs(observed)**2)));began=time.perf_counter();evaluations=0;best=None;history=[]
    def decode(x):
        t=np.exp(x[k:k+nt]);return x[:k],np.repeat(t,k) if shared else t,x[k+nt:]*scales
    def evaluate(x):
        nonlocal evaluations,best
        if cancel(): raise InterruptedError('Magnitude fit cancelled.')
        if evaluations>=max_evaluations or (evaluations and time.perf_counter()-began>=max_seconds): raise _BudgetReached()
        evaluations+=1;freq,t,c=decode(x);prediction=mode_design(processor,freq,t)@c
        residual=(abs(prediction)-abs(observed))/scale;score=float(np.mean(residual**2))
        if best is None or score<best['score']:best=dict(x=x.copy(),fitted=prediction,score=score)
        return residual
    rng=np.random.default_rng(seed);stopped=False
    for start in range(starts):
        x=x0.copy()
        if start:
            x[:k]+=rng.normal(0,(frequency_bounds[1]-frequency_bounds[0])/30,k)
            x[k:k+nt]+=rng.normal(0,.25,nt)
            angles=rng.normal(0,.5,k);c=x[k+nt:].reshape(k,2).copy()
            x[k+nt:]=np.column_stack([c[:,0]*np.cos(angles)-c[:,1]*np.sin(angles),c[:,0]*np.sin(angles)+c[:,1]*np.cos(angles)]).ravel()
        x=np.minimum(np.maximum(x,lower+1e-10),upper-1e-10)
        try: solution=least_squares(evaluate,x,bounds=(lower,upper),max_nfev=max_nfev,x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)
        except _BudgetReached:stopped=True;break
        ff,tt,cc=decode(solution.x)
        history.append(dict(score=float(np.mean(solution.fun**2)),success=bool(solution.success),frequencies_hz=ff.tolist(),t2star_s=tt.tolist(),cos_sin_coefficients=cc.reshape(k,2).tolist(),nfev=int(solution.nfev)))
    freq,t,c=decode(best['x']);order=np.argsort(freq);c=c.reshape(k,2)[order]
    converged=any(h['success'] and abs(h['score']-best['score'])<=max(1e-12,best['score']*1e-5) for h in history)
    _,_,rank,condition=real_projection(mode_design(processor,freq,t),observed)
    diagnostic=fit_diagnostics(freq[order],t[order],frequency_bounds,t2_bounds,shared_decay=shared,native_spacing=processor.fs/processor.n,rank=rank,columns=2*k,condition=condition,converged=converged,budget_exhausted=stopped)
    return dict(frequencies_hz=freq[order].tolist(),t2star_s=t[order].tolist(),cos_sin_coefficients=c.tolist(),
        amplitudes=np.linalg.norm(c,axis=1).tolist(),phases_rad=np.arctan2(-c[:,1],c[:,0]).tolist(),shared_decay=shared,
        fitted=best['fitted'],score=best['score'],relative_magnitude_residual=float(np.linalg.norm(abs(best['fitted'])-abs(observed))/np.linalg.norm(observed)),
        relative_complex_residual=float(np.linalg.norm(best['fitted']-observed)/np.linalg.norm(observed)),
        optimizer_converged=converged,budget_exhausted=stopped,boundary_hits=diagnostic['boundary_hits'],numerical_diagnostics=diagnostic,
        candidates=history,evaluations=evaluations,elapsed_s=time.perf_counter()-began,
        phase_sign_ambiguous=True,initialization='Complex-fit frequencies, decays and gains; phase is not independently identified by magnitude.')


def compare_decay_objectives(fit_run_id,candidate_index=0,settings=None,*,record,directory,cancel,progress):
    from . import storage
    from .analysis import recipe,plot
    from .repeats import load_group_averages
    from .jfit import ProcessedSpectrum
    parent=storage.get_result(fit_run_id)
    if parent['operation']!='fit_frequency_decay' or type(candidate_index) is not int or not 0<=candidate_index<len(parent['candidates']): raise ValueError('Require a frequency-decay candidate.')
    if parent['settings']['background']:raise ValueError('Compare candidates without spectral background.')
    initial=parent['candidates'][candidate_index];means,counts,source=load_group_averages(parent['parent_run_id'])
    if source['arrays_sha256']!=parent['source_arrays_sha256']:raise ValueError('Parent source changed.')
    config=dict(starts=3,max_nfev=150,max_evaluations=3000,max_seconds=60.,seed=20260922)
    if settings and set(settings)-set(config):raise ValueError('Unknown objective-comparison settings.')
    config.update(settings or {});spec=parent['parameters'].get('preprocessing') or {};fs=source['sampling_rate_hz'];bounds=initial['range_hz']
    raw={role:np.average(means[indices],axis=0,weights=counts[indices]) for role,indices in [('discovery',parent['discovery_groups']),('validation',parent['validation_groups'])]}
    processed={}
    for role,y in raw.items():processed[role],_,params=recipe(y,fs,spec)
    n=len(processed['discovery']);freq=np.fft.rfftfreq(n,1/fs);bins=np.flatnonzero((freq>=bounds[0])&(freq<=bounds[1]))
    p=ProcessedSpectrum(fs,source['points'],params['start_sample'],params['stop_sample'],bins,params['sg_window'],params['sg_order'])
    observed={role:np.fft.rfft(y)[bins]/n for role,y in processed.items()}
    fit=fit_magnitude_modes(p,observed['discovery'],bounds,parent['t2_bounds_s'],initial,cancel=cancel,**config)
    prediction=fit.pop('fitted');prior=mode_design(p,initial['frequencies_hz'],initial['t2star_s'])@np.asarray(initial['cos_sin_coefficients']).ravel();progress(1,2)
    def metrics(y,model):
        return dict(complex=float(np.linalg.norm(y-model)/np.linalg.norm(y)),magnitude=float(np.linalg.norm(abs(y)-abs(model))/np.linalg.norm(y))) if np.linalg.norm(y)>0 else dict(complex=None,magnitude=None)
    comparison={role:{'complex_fit':metrics(y,prior),'magnitude_fit':metrics(y,prediction),'sign_reversed_magnitude_fit':metrics(y,-prediction)} for role,y in observed.items()}
    for part,transform in [('magnitude',np.abs),('real',np.real),('imaginary',np.imag)]:
        plot(directory/f'{part}.png',[(p.f,transform(y),role) for role,y in dict(observed,complex_fit=prior,magnitude_fit=prediction).items()],
            'Frequency (Hz)',part.title()+' (ADC units)','Same modes and bounds: complex vs magnitude objective')
    residual=observed['validation']-prediction
    plot(directory/'residual.png',[(p.f,residual.real,'Real'),(p.f,residual.imag,'Imaginary')],'Frequency (Hz)','ADC units','Frozen magnitude-fit complex residual')
    time_axis=np.arange(source['points'])/fs
    def model_fid(c):
        return sum(np.exp(-time_axis/tau)*(a*np.cos(2*np.pi*f*time_axis)+b*np.sin(2*np.pi*f*time_axis)) for f,tau,(a,b) in zip(c['frequencies_hz'],c['t2star_s'],c['cos_sin_coefficients']))
    # Selected-band model is not a reconstruction of all measured FID components.
    model_traces={name:recipe(model_fid(c),fs,spec)[0] for name,c in [('complex_fit',initial),('magnitude_fit',fit)]}
    tt=time_axis[params['start_sample']:params['stop_sample']];keep=tt<=tt[0]+.5
    plot(directory/'signed_model_fid.png',[(tt[keep],y[keep],name) for name,y in model_traces.items()]+[(tt[keep],-model_traces['magnitude_fit'][keep],'Magnitude-equivalent sign reversal')],
        'Acquisition time (s)','Selected-band model (ADC units)','Signed model FIDs: magnitude does not determine global sign')
    np.savez_compressed(directory/'objective_arrays.npz',frequency_hz=p.f,**observed,complex_prediction=prior,magnitude_prediction=prediction)
    progress(2,2)
    return dict(parent_run_id=fit_run_id,candidate_index=candidate_index,source_arrays_sha256=source['arrays_sha256'],
        discovery_groups=parent['discovery_groups'],validation_groups=parent['validation_groups'],preprocessing=spec,
        range_hz=bounds,t2_bounds_s=parent['t2_bounds_s'],settings=config,fit=fit,comparison=comparison,scientifically_validated=False,
        warnings=['Warm-started from complex fit; not an independent magnitude-only discovery or exhaustive global search.',
        'Magnitude uses abs(sum of complex modes), never sum of individual magnitudes.',
        'Global sign reversal has exactly identical magnitude; reported phases are not uniquely established.',
        'Validation coefficients remain frozen. Magnitude noise bias and parameter ambiguity remain.',
        'Signed FID plots are selected-band model predictions, not a fit to the complete experimental FID.'])
