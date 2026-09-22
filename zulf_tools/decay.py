"""Bounded local damped-mode fits on exact complex native FFT bins.

Independent oscillators are phenomenological modes, not assigned substances.
Real cos/sin coefficients express amplitude and acquisition-referenced phase.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


class _BudgetReached(Exception):
    pass


def real_projection(design, observed):
    """Real coefficients fitting complex observations; preserve phase information."""
    matrix = np.vstack([design.real, design.imag])
    target = np.r_[observed.real, observed.imag]
    scales = np.maximum(np.linalg.norm(matrix, axis=0), 1e-30)
    normalized = matrix/scales
    solution, _, rank, singular = np.linalg.lstsq(normalized, target, rcond=1e-10)
    coefficients = solution/scales
    condition = float(singular[0]/singular[-1]) if len(singular) and singular[-1] > 0 else None
    return design@coefficients, coefficients, int(rank), condition


def mode_design(processor, frequencies, taus, background=False):
    columns = [processor.templates([f], [1.], 1/tau) for f, tau in zip(frequencies, taus)]
    design = np.column_stack(columns)
    if background:
        design = np.column_stack([design, np.ones(len(processor.f)), 1j*np.ones(len(processor.f))])
    return design


def frequency_seeds(f, observed, count, bounds):
    """Discovery-only initial guesses; no significance or substance classification."""
    peak_indices, _ = find_peaks(abs(observed))
    candidates = list(peak_indices[np.argsort(abs(observed[peak_indices]))[::-1]])
    candidates += list(np.argsort(abs(observed))[::-1])
    chosen = []
    spacing = float(np.median(np.diff(f)))
    for index in candidates:
        value = float(f[index])
        if all(abs(value-other) >= 2*spacing for other in chosen):
            chosen.append(value)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        chosen = np.linspace(bounds[0], bounds[1], count+2)[1:-1].tolist()
    return np.asarray(sorted(chosen))


def fit_modes(processor, observed, frequency_bounds, t2_bounds, mode_count=1,
              shared_decay=False, initial_frequencies=None, starts=4,
              max_nfev=150, max_evaluations=4000, max_seconds=60., seed=0,
              background=False, cancel=lambda: False):
    """Variable projection with hard per-fit evaluation and wall-clock budgets.

    The budget includes finite-difference and initial evaluations. Best evaluated
    point is retained if stopped. A budget-limited result is never convergence.
    All observations must already correspond to processor's native FFT bins.
    """
    observed = np.asarray(observed, dtype=complex)
    f = np.asarray(processor.f)
    if observed.shape != f.shape or len(f) < 8 or not np.isfinite(observed).all():
        raise ValueError('Supply finite complex observations for at least eight native bins.')
    if not np.isfinite([*frequency_bounds,*t2_bounds]).all() or not 0 < t2_bounds[0] < t2_bounds[1]:
        raise ValueError('Require finite positive increasing T2* bounds in seconds.')
    lo, hi = frequency_bounds
    if not 0 < lo < hi < processor.fs/2 or f[0] < lo or f[-1] > hi:
        raise ValueError('Frequency bounds must contain selected bins and lie inside Nyquist.')
    for value, low, high in [(mode_count,1,8),(starts,1,32),(max_nfev,2,2000),(max_evaluations,2,100000)]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Invalid integer fit budget or mode count.')
    if type(shared_decay) is not bool or type(background) is not bool:
        raise ValueError('shared_decay and background must be boolean.')
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be finite and positive.')
    if len(f)*2 <= 4*mode_count+2:
        raise ValueError('Too few observations for requested mode count.')
    amplitude_scale = float(np.sqrt(np.mean(abs(observed)**2)))
    if amplitude_scale <= np.finfo(float).tiny:
        raise ValueError('Cannot fit zero signal.')
    seeds = (frequency_seeds(f, observed, mode_count, frequency_bounds)
             if initial_frequencies is None else np.asarray(initial_frequencies, dtype=float))
    if seeds.shape != (mode_count,) or not np.isfinite(seeds).all() or np.any((seeds<lo)|(seeds>hi)):
        raise ValueError('Initial frequencies must match count and lie inside the band.')
    tau_count = 1 if shared_decay else mode_count
    lower = np.r_[np.full(mode_count,lo), np.full(tau_count,np.log(t2_bounds[0]))]
    upper = np.r_[np.full(mode_count,hi), np.full(tau_count,np.log(t2_bounds[1]))]
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter(); evaluations = 0; best = None; history = []

    def decode(x):
        taus = np.exp(x[mode_count:])
        return x[:mode_count], np.repeat(taus,mode_count) if shared_decay else taus

    def evaluate(x):
        nonlocal evaluations, best
        if cancel():
            raise InterruptedError('Decay fit cancelled.')
        if evaluations >= max_evaluations or (evaluations and time.perf_counter()-start_time >= max_seconds):
            raise _BudgetReached()
        evaluations += 1
        frequencies, taus = decode(x)
        design = mode_design(processor,frequencies,taus,background)
        fitted, coefficients, rank, condition = real_projection(design,observed)
        residual = np.r_[(fitted-observed).real,(fitted-observed).imag]/amplitude_scale
        score = float(np.mean(residual**2))
        if best is None or score < best['score']:
            best = dict(score=score,x=x.copy(),fitted=fitted,coefficients=coefficients,
                        rank=rank,condition=condition)
        return residual

    stopped = False
    for start in range(starts):
        fraction = (start+.5)/starts
        trial_f = seeds.copy()
        if start:
            # Local starts favor the observed cluster; every third start explores
            # the full allowed band. These are not exhaustive global searches.
            trial_f = (rng.uniform(lo,hi,mode_count) if start%3 == 0 else
                       seeds+rng.normal(0,max((hi-lo)/20,1/(processor.n/processor.fs)),mode_count))
        x = np.r_[trial_f,np.full(tau_count,np.log(t2_bounds[0])*(1-fraction)+np.log(t2_bounds[1])*fraction)]
        x = np.clip(x,lower+1e-10,upper-1e-10)
        try:
            solution = least_squares(evaluate,x,bounds=(lower,upper),x_scale='jac',
                                     max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        except _BudgetReached:
            stopped = True
            break
        frequencies, taus = decode(solution.x)
        history.append({'score':float(np.mean(solution.fun**2)),
                        'frequencies_hz':frequencies.tolist(),'t2star_s':taus.tolist(),
                        'success':bool(solution.success),'nfev':int(solution.nfev),
                        'message':str(solution.message)})
    frequencies, taus = decode(best['x'])
    order = np.argsort(frequencies,kind='stable')
    coefficients = best['coefficients'][:2*mode_count].reshape(-1,2)[order]
    hits = []
    for i in range(len(best['x'])):
        if min(best['x'][i]-lower[i],upper[i]-best['x'][i]) < .01*(upper[i]-lower[i]):
            hits.append(('frequency_' if i < mode_count else 'log_t2_')+str(i if i<mode_count else i-mode_count))
    # best may be a finite-difference trial rather than a converged endpoint;
    # require close numerical agreement with a successful recorded endpoint.
    converged = any(h['success'] and abs(h['score']-best['score']) <= max(1e-12,best['score']*1e-5) for h in history)
    return {'frequencies_hz':frequencies[order].tolist(),'t2star_s':taus[order].tolist(),
            'cos_sin_coefficients':coefficients.tolist(),
            'amplitudes':np.linalg.norm(coefficients,axis=1).tolist(),
            'phases_rad':np.arctan2(-coefficients[:,1],coefficients[:,0]).tolist(),
            'phase_convention':'A exp(-t/T2*) cos(2 pi f t + phase); t is acquisition time.',
            'background_coefficients':best['coefficients'][2*mode_count:].tolist(),
            'relative_complex_residual':float(np.linalg.norm(observed-best['fitted'])/np.linalg.norm(observed)),
            'relative_magnitude_residual':float(np.linalg.norm(abs(observed)-abs(best['fitted']))/np.linalg.norm(observed)),
            'score':best['score'],'boundary_hits':hits,'linear_rank':best['rank'],
            'linear_condition_number':best['condition'],'optimizer_converged':converged,
            'budget_exhausted':stopped,'evaluations':evaluations,'elapsed_s':time.perf_counter()-start_time,
            'completed_starts':len(history),'candidates':history,'shared_decay':shared_decay,
            'fitted':best['fitted'],
            'interpretation':'Phenomenological effective FID modes, not intrinsic T2 or substance assignments.'}
