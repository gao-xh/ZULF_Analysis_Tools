"""Bounded local damped-mode fits on exact complex native FFT bins.

Independent oscillators are phenomenological modes, not assigned substances.
Real cos/sin coefficients express amplitude and acquisition-referenced phase.
"""
import time
from collections import OrderedDict
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


class ModeTemplateCache:
    """Per-fit exact-key LRU; never round nonlinear search parameters."""
    byte_limit = 8*1024**2
    entry_limit = 128

    def __init__(self, processor):
        self.processor=processor
        self.entries=OrderedDict()
        self.bytes=0
        self.peak_bytes=0
        self.hits=0
        self.misses=0

    def get(self, frequency, tau):
        key=(float(frequency),float(tau))
        if key in self.entries:
            self.hits+=1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses+=1
        value=self.processor.templates([frequency],[1.],1/tau)
        if value.nbytes<=self.byte_limit and self.entry_limit>0:
            while self.entries and (self.bytes+value.nbytes>self.byte_limit or len(self.entries)>=self.entry_limit):
                _,old=self.entries.popitem(last=False)
                self.bytes-=old.nbytes
            self.entries[key]=value
            self.bytes+=value.nbytes
            self.peak_bytes=max(self.peak_bytes,self.bytes)
        return value

    def diagnostics(self):
        return dict(hits=self.hits,misses=self.misses,entries=len(self.entries),
                    retained_array_bytes=self.bytes,peak_retained_array_bytes=self.peak_bytes,
                    array_byte_limit=self.byte_limit,entry_limit=self.entry_limit,
                    note='Per-fit exact frequency/T2* keys; no rounding. Limits apply to retained template arrays, not total process memory or temporary designs.')


def mode_design(processor, frequencies, taus, background=False, template_cache=None):
    columns = [(processor.templates([f], [1.], 1/tau) if template_cache is None else template_cache.get(f,tau))
               for f, tau in zip(frequencies, taus)]
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


def fit_diagnostics(frequencies, taus, frequency_bounds, t2_bounds, *, shared_decay,
                    native_spacing, rank, columns, condition, converged, budget_exhausted):
    """Numerical warning screen, not a statistical identifiability test.

    Indices refer to the returned frequency-sorted modes. A sub-bin separation
    is descriptive: parametric recovery below FFT-bin spacing can be possible.
    """
    hits = []
    for label, values, bounds in [('frequency', frequencies, frequency_bounds),
                                  ('log_t2', np.log(taus[:1] if shared_decay else taus),
                                   np.log(t2_bounds))]:
        lo, hi = bounds
        for i, value in enumerate(values):
            if min(value-lo, hi-value) < .01*(hi-lo):
                hits.append(f'{label}_{i}')
    close_pairs = [{'mode_indices': [i,j], 'separation_hz': float(abs(frequencies[j]-frequencies[i]))}
                   for i in range(len(frequencies)) for j in range(i+1,len(frequencies))
                   if abs(frequencies[j]-frequencies[i]) < native_spacing]
    flags = []
    if hits:
        flags.append('search_boundary')
    if rank < columns:
        flags.append('rank_deficient_amplitude_design')
    if condition is None or not np.isfinite(condition) or condition > 1e8:
        flags.append('ill_conditioned_amplitude_design')
    if not converged:
        flags.append('optimizer_not_converged')
    if budget_exhausted:
        flags.append('search_budget_exhausted')
    return {'boundary_hits': hits, 'numerical_warning_flags': flags,
            'requires_review': bool(flags), 'sub_bin_frequency_pairs': close_pairs,
            'thresholds': {'boundary_fraction': .01, 'normalized_design_condition': 1e8,
                           'native_fft_spacing_hz': float(native_spacing)},
            'index_convention': 'Frequency-sorted, zero-based mode indices; shared log_t2_0 applies to all modes.',
            'interpretation': 'A clean numerical screen does not establish parameter identifiability, signal status or physical validity. Thresholds are numerical heuristics, not calibrated confidence limits.'}


def fit_modes(processor, observed, frequency_bounds, t2_bounds, mode_count=1,
              shared_decay=False, initial_frequencies=None, starts=4,
              max_nfev=150, max_evaluations=4000, max_seconds=60., seed=0,
              background=False, cancel=lambda: False):
    """Variable projection with hard per-fit evaluation and wall-clock budgets.

    The budget includes finite-difference and initial evaluations. Best evaluated
    point is retained if stopped. A budget-limited result is never convergence.
    Observations must correspond to the processor's exact complex transform.
    Windowed adapters must supply explicit initial frequencies and must not
    interpret repeated window observations as independent FFT bins.
    """
    observed = np.asarray(observed, dtype=complex)
    f = np.asarray(processor.f)
    if observed.shape != f.shape or len(f) < 8 or not np.isfinite(observed).all():
        raise ValueError('Supply finite complex observations for at least eight transform samples.')
    if not np.isfinite([*frequency_bounds,*t2_bounds]).all() or not 0 < t2_bounds[0] < t2_bounds[1]:
        raise ValueError('Require finite positive increasing T2* bounds in seconds.')
    lo, hi = frequency_bounds
    if not 0 < lo < hi < processor.fs/2 or np.min(f) < lo or np.max(f) > hi:
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
    template_cache=ModeTemplateCache(processor)

    def decode(x):
        taus = np.exp(x[mode_count:])
        return x[:mode_count], np.repeat(taus,mode_count) if shared_decay else taus

    def evaluate(x):
        nonlocal evaluations, best
        if cancel():
            raise InterruptedError('Decay fit cancelled.')
        if evaluations >= max_evaluations or (evaluations and time.perf_counter()-start_time >= max_seconds):
            raise _BudgetReached('max_evaluations' if evaluations >= max_evaluations else 'max_seconds')
        evaluations += 1
        frequencies, taus = decode(x)
        design = mode_design(processor,frequencies,taus,background,template_cache)
        fitted, coefficients, rank, condition = real_projection(design,observed)
        residual = np.r_[(fitted-observed).real,(fitted-observed).imag]/amplitude_scale
        score = float(np.mean(residual**2))
        if best is None or score < best['score']:
            best = dict(score=score,x=x.copy(),fitted=fitted,coefficients=coefficients,
                        rank=rank,condition=condition,start_index=start,evaluation=evaluations)
        return residual

    stopped = False
    attempts = []
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
        initial_f, initial_t = decode(x)
        attempt = dict(start_index=start, initial_frequencies_hz=initial_f.tolist(),
                       initial_t2star_s=initial_t.tolist(),
                       strategy='seed' if start == 0 else 'uniform_band' if start%3 == 0 else 'local_perturbation')
        before = evaluations
        try:
            solution = least_squares(evaluate,x,bounds=(lower,upper),x_scale='jac',
                                     max_nfev=max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        except _BudgetReached as exc:
            attempt.update(status='budget_exhausted', stop_reason=str(exc),
                           evaluations=evaluations-before)
            attempts.append(attempt)
            stopped = True
            break
        attempt.update(status='converged' if solution.success else 'optimizer_stopped',
                       stop_reason=str(solution.message), evaluations=evaluations-before)
        attempts.append(attempt)
        frequencies, taus = decode(solution.x)
        history.append({'score':float(np.mean(solution.fun**2)),
                        'start_index':start,
                        'frequencies_hz':frequencies.tolist(),'t2star_s':taus.tolist(),
                        'success':bool(solution.success),'nfev':int(solution.nfev),
                        'message':str(solution.message)})
    frequencies, taus = decode(best['x'])
    order = np.argsort(frequencies,kind='stable')
    coefficients = best['coefficients'][:2*mode_count].reshape(-1,2)[order]
    # best may be a finite-difference trial rather than a converged endpoint;
    # require close numerical agreement with a successful recorded endpoint.
    converged = any(h['start_index']==best['start_index'] and h['success']
                    and abs(h['score']-best['score']) <= max(1e-12,best['score']*1e-5)
                    and np.allclose(h['frequencies_hz'],frequencies,rtol=1e-5,atol=1e-8)
                    and np.allclose(h['t2star_s'],taus,rtol=1e-5,atol=1e-8)
                    for h in history)
    diagnostics = fit_diagnostics(frequencies[order], taus[order], frequency_bounds, t2_bounds,
                                  shared_decay=shared_decay, native_spacing=processor.fs/processor.n,
                                  rank=best['rank'], columns=2*mode_count+2*int(background),
                                  condition=best['condition'], converged=converged, budget_exhausted=stopped)
    return {'frequencies_hz':frequencies[order].tolist(),'t2star_s':taus[order].tolist(),
            'cos_sin_coefficients':coefficients.tolist(),
            'amplitudes':np.linalg.norm(coefficients,axis=1).tolist(),
            'phases_rad':np.arctan2(-coefficients[:,1],coefficients[:,0]).tolist(),
            'phase_convention':'A exp(-t/T2*) cos(2 pi f t + phase); t is acquisition time.',
            'background_coefficients':best['coefficients'][2*mode_count:].tolist(),
            'relative_complex_residual':float(np.linalg.norm(observed-best['fitted'])/np.linalg.norm(observed)),
            'relative_magnitude_residual':float(np.linalg.norm(abs(observed)-abs(best['fitted']))/np.linalg.norm(observed)),
            'score':best['score'],'boundary_hits':diagnostics['boundary_hits'],'linear_rank':best['rank'],
            'numerical_diagnostics':diagnostics,
            'linear_condition_number':best['condition'],'optimizer_converged':converged,
            'budget_exhausted':stopped,'evaluations':evaluations,'elapsed_s':time.perf_counter()-start_time,
            'completed_starts':len(history),'candidates':history,'shared_decay':shared_decay,
            'initialization':{'source':'discovery_spectrum' if initial_frequencies is None else 'explicit',
                              'seed_frequencies_hz':seeds.tolist(),'random_seed':seed},
            'start_attempts':attempts,'attempted_starts':len(attempts),
            'best_start_index':best['start_index'],'best_evaluation':best['evaluation'],
            'template_cache':template_cache.diagnostics(),
            'start_ledger_note':'Zero-based start indices; evaluations include finite-difference trials. A budget-stopped attempt may have zero evaluations. Completed starts include optimizer endpoints without convergence. Unattempted starts are not recorded.',
            'fitted':best['fitted'],
            'interpretation':'Phenomenological effective FID modes, not intrinsic T2 or substance assignments.'}
