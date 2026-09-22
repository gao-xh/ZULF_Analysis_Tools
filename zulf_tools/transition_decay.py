"""Decay-only variable projection for complete fixed transition groups."""
import time
import numpy as np
from scipy.optimize import least_squares
from .decay import real_projection, _BudgetReached


def transition_design(processor, groups, taus, phase_delay_s=0.):
    return np.column_stack([processor.templates(g['frequencies_hz'],g['weights'],1/tau,phase_delay_s=phase_delay_s)
                            for g,tau in zip(groups,taus)])


def fit_transition_decay(processor, observed, groups, t2_bounds, *, shared_decay=False,
                          observation_scale=None, starts=6, max_nfev=150,
                          max_evaluations=2000, max_seconds=60., seed=0,
                          phase_delay_bounds_s=None, cancel=lambda:False):
    """Keep all frequencies/relative transition weights fixed; fit group T2*.

    One real cosine/sine coefficient pair per group. No per-transition phase,
    extra spectral background or hidden transition trimming. Group order remains
    fixed. The caller must document preparation and transition-weight assumptions.
    """
    y=np.asarray(observed,dtype=complex)
    if y.shape!=np.asarray(processor.f).shape or len(y)<8 or not np.isfinite(y).all() or np.linalg.norm(y)==0:
        raise ValueError('Need finite nonzero complex observations matching the processor.')
    if not isinstance(groups,list) or not 1<=len(groups)<=4:
        raise ValueError('Supply one to four fixed transition groups.')
    for g in groups:
        f=np.asarray(g['frequencies_hz'],dtype=float);w=np.asarray(g['weights'],dtype=float)
        if f.ndim!=1 or not 1<=len(f)<=5000 or f.shape!=w.shape or not np.isfinite(f).all() or not np.isfinite(w).all() or np.any(f<=0) or np.any(f>=processor.fs/2) or np.any(w<0) or w.sum()<=0:
            raise ValueError('Invalid transition frequencies or nonnegative weights.')
    if len(t2_bounds)!=2 or not np.isfinite(t2_bounds).all() or not 0<t2_bounds[0]<t2_bounds[1]:
        raise ValueError('T2* bounds must be positive and increasing.')
    if type(shared_decay) is not bool:
        raise ValueError('shared_decay must be boolean.')
    for value,lo,hi in [(starts,1,32),(max_nfev,2,2000),(max_evaluations,2,100000)]:
        if type(value) is not int or not lo<=value<=hi:
            raise ValueError('Invalid optimizer budget.')
    if not np.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('Time budget must be positive.')
    scale=np.ones(len(y)) if observation_scale is None else np.asarray(observation_scale,dtype=float)
    if scale.shape!=y.shape or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Observation scale must be finite, positive and match the observations.')
    count=1 if shared_decay else len(groups)
    lo,hi=np.log(t2_bounds);rng=np.random.default_rng(seed)
    with_delay=phase_delay_bounds_s is not None
    lower=np.full(count,lo);upper=np.full(count,hi)
    if with_delay:
        if len(phase_delay_bounds_s)!=2 or not np.isfinite(phase_delay_bounds_s).all() or not phase_delay_bounds_s[0]<phase_delay_bounds_s[1]:
            raise ValueError('Phase-delay bounds must be finite and increasing.')
        lower=np.r_[lower,phase_delay_bounds_s[0]];upper=np.r_[upper,phase_delay_bounds_s[1]]
    clock=time.perf_counter();evaluations=0;best=None;history=[];stopped=False
    amplitude=max(float(np.sqrt(np.mean(abs(y/scale)**2))),np.finfo(float).tiny)
    def evaluate(x):
        nonlocal best,evaluations
        if cancel():
            raise InterruptedError('Transition-decay fit cancelled.')
        if evaluations>=max_evaluations or (evaluations and time.perf_counter()-clock>=max_seconds):
            raise _BudgetReached()
        evaluations+=1
        taus=np.repeat(np.exp(x[:count]),len(groups)) if shared_decay else np.exp(x[:count])
        delay=float(x[-1]) if with_delay else 0.
        design=transition_design(processor,groups,taus,delay)
        _,coeff,rank,condition=real_projection(design/scale[:,None],y/scale)
        prediction=design@coeff
        r=(prediction-y)/scale/amplitude
        residual=np.r_[r.real,r.imag];score=float(np.mean(residual**2))
        if best is None or score<best['score']:
            best=dict(score=score,taus=taus,delay=delay,coeff=coeff,prediction=prediction,rank=rank,condition=condition)
        return residual
    for k in range(starts):
        fraction=(k+.5)/starts
        x=np.full(count,lo+(hi-lo)*fraction) if k%2==0 else rng.uniform(lo,hi,count)
        if with_delay:
            x=np.r_[x,np.clip(0.,lower[-1],upper[-1]) if k==0 else rng.uniform(lower[-1],upper[-1])]
        try:
            solution=least_squares(evaluate,x,bounds=(lower,upper),x_scale=upper-lower,
                max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        except _BudgetReached:
            stopped=True;break
        history.append(dict(log_t2_parameters=solution.x[:count].tolist(),phase_delay_s=float(solution.x[-1]) if with_delay else 0.,score=float(np.mean(solution.fun**2)),
                            success=bool(solution.success),nfev=int(solution.nfev),message=str(solution.message)))
    converged=any(h['success'] and abs(h['score']-best['score'])<=max(1e-12,best['score']*1e-5) for h in history)
    coefficients=best['coeff'].reshape(-1,2)
    hits=[i for i,tau in enumerate(best['taus']) if min(np.log(tau)-lo,hi-np.log(tau))<.01*(hi-lo)]
    flags=[]
    delay_boundary=bool(with_delay and min(best['delay']-lower[-1],upper[-1]-best['delay'])<.01*(upper[-1]-lower[-1]))
    if delay_boundary: flags.append('phase_delay_boundary')
    if with_delay and all(np.ptp(np.asarray(g['frequencies_hz'])[np.asarray(g['weights'])>0])<1e-10 for g in groups):
        flags.append('phase_delay_unidentifiable_single_frequency_groups')
    if hits: flags.append('search_boundary')
    if best['rank']<2*len(groups): flags.append('rank_deficient_amplitude_design')
    if best['condition'] is None or not np.isfinite(best['condition']) or best['condition']>1e8: flags.append('ill_conditioned_amplitude_design')
    if not converged: flags.append('optimizer_not_converged')
    if stopped: flags.append('search_budget_exhausted')
    return dict(t2star_s=best['taus'].tolist(),shared_decay=shared_decay,
        phase_delay_s=best['delay'],phase_delay_bounds_s=phase_delay_bounds_s,
        phase_delay_note='Model-only phase -2*pi*f*delay; recorded time and decay envelope are unchanged. Delay may be ambiguous or confounded with preparation.',
        cos_sin_coefficients=coefficients.tolist(),amplitudes=np.linalg.norm(coefficients,axis=1).tolist(),
        phases_rad=np.arctan2(-coefficients[:,1],coefficients[:,0]).tolist(),
        phase_convention='A exp(-t/T2*) sum(normalized weights * cos(2 pi f (t-delay) + phase)); acquisition time.',
        score=best['score'],relative_complex_residual=float(np.linalg.norm(y-best['prediction'])/np.linalg.norm(y)),
        linear_rank=best['rank'],linear_condition_number=best['condition'],boundary_group_indices=hits,
        numerical_warning_flags=flags,optimizer_converged=converged,budget_exhausted=stopped,
        evaluations=evaluations,elapsed_s=time.perf_counter()-clock,candidates=history,fitted=best['prediction'],
        interpretation='Conditional effective decays with fixed transition structure and relative weights; no automatic physical acceptance.')
