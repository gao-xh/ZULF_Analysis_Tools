"""Exploratory structure-constrained natural-abundance isopropylamine J fitting.

Exact A3B3MX collective-spin Hamiltonians; finite-record complex spectra with
the same SG/crop as observations. Unknown preparation is approximated by one
cosine/sine coefficient pair per isotopomer, not independent fitted peak positions.
"""
from functools import lru_cache
import hashlib
import itertools
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.signal import savgol_coeffs
from scipy.stats import qmc
from . import storage, simulation

NAMES = ['J_CH_methine', 'J_CH_methyl', 'J_HH_vicinal',
         'J_Cmethine_Hmethyl', 'J_Cmethyl_Hmethine', 'J_Cmethyl_Hother_methyl']
DEFAULT = dict(zip(NAMES, [135., 125., 6.5, -3., -3., 2.]))
BOUNDS = dict(zip(NAMES, [[100.,175.], [100.,160.], [0.,12.], [-12.,12.], [-12.,12.], [-8.,8.]]))


def model_for(parameters, isotopomer):
    """Export a conventional eight-spin model consumable by simulate_spin_dynamics."""
    p = dict(DEFAULT, **parameters)
    j = np.zeros((8,8))
    for i in range(6): j[i,6] = j[6,i] = p['J_HH_vicinal']
    for i in range(7):
        key = ('J_CH_methine' if i == 6 else 'J_Cmethine_Hmethyl') if isotopomer == 'methine' else (
            'J_CH_methyl' if i < 3 else 'J_Cmethyl_Hmethine' if i == 6 else 'J_Cmethyl_Hother_methyl')
        j[i,7] = j[7,i] = p[key]
    return {'isotopes':['1H']*7+['13C'], 'couplings_hz':j.tolist(), 'symmetry_groups':[[0,1,2],[3,4,5]]}


def angular_momentum(spin):
    m = np.arange(spin, -spin-1, -1)
    plus = np.zeros((len(m),len(m)), complex)
    for column in range(1,len(m)):
        plus[column-1,column] = np.sqrt(spin*(spin+1)-m[column]*(m[column]+1))
    return [(plus+plus.T.conj())/2, (plus-plus.T.conj())/(2j), np.diag(m).astype(complex)]


@lru_cache(maxsize=1)
def collective_blocks():
    blocks = []
    for sa,sb in itertools.product([.5,1.5], repeat=2):
        spins = [sa,sb,.5,.5]; dims = [int(2*s+1) for s in spins]
        ops = []
        for site,s in enumerate(spins):
            vector = []
            for a in angular_momentum(s):
                op = np.ones((1,1),complex)
                for k,dim in enumerate(dims): op = np.kron(op, a if k == site else np.eye(dim))
                vector.append(op)
            ops.append(vector)
        pair = {(i,k):sum(a@b for a,b in zip(ops[i],ops[k])).real for i in range(4) for k in range(i+1,4)}
        # Rotational invariance permits z instead of x for the unpulsed thermal
        # response; this yields real eigensystems and the same zero-field spectrum.
        obs = sum(simulation.GAMMA['1H']*ops[i][2] for i in range(3))+simulation.GAMMA['13C']*ops[3][2]
        mult = (2 if sa == .5 else 1)*(2 if sb == .5 else 1)
        blocks.append((pair,obs.real,mult))
    return blocks


def transitions(parameters, isotopomer):
    p = dict(DEFAULT, **parameters)
    frequencies, weights = [], []
    for pair,obs,mult in collective_blocks():
        h = p['J_HH_vicinal']*(pair[0,2]+pair[1,2])
        if isotopomer == 'methine':
            h = h+p['J_CH_methine']*pair[2,3]+p['J_Cmethine_Hmethyl']*(pair[0,3]+pair[1,3])
        else:
            h = h+p['J_CH_methyl']*pair[0,3]+p['J_Cmethyl_Hmethine']*pair[2,3]+p['J_Cmethyl_Hother_methyl']*pair[1,3]
        e,v = eigh(h,driver='evr')  # eigenvalues in Hz, not rad/s
        o = v.T@obs@v
        i,k = np.triu_indices(len(e),1)
        f,w = e[k]-e[i], 2*mult*o[i,k]**2/256
        frequencies.extend(f.tolist()); weights.extend(w.tolist())
    f,w = np.asarray(frequencies),np.asarray(weights)
    # Numerical-zero transitions only, not a user significance/peak threshold.
    keep = (f > 1e-7)&(w > max(w.max()*1e-13,1e-20))
    f,w = f[keep],w[keep]
    # Merge degenerate frequencies; do not round their actual weighted centers.
    keys = np.round(f,7); _,inverse = np.unique(keys,return_inverse=True)
    totals = np.bincount(inverse,weights=w)
    centers = np.bincount(inverse,weights=w*f)/totals
    return centers,totals


class ProcessedSpectrum:
    """Exact FFT bins of damped modes after full-record mirror SG and time crop."""
    def __init__(self, fs, full_points, first, last, bins, sg_window=0, sg_order=2):
        self.fs,self.full_points,self.first,self.last = fs,full_points,first,last
        self.n = last-first
        self.bins = np.asarray(bins,int)
        self.f = self.bins*fs/self.n
        self.window = sg_window
        self.coeff = savgol_coeffs(sg_window,sg_order) if sg_window else np.array([1.])
        self.offsets = np.arange(-(sg_window//2),sg_window//2+1) if sg_window else np.array([0])
        m = sg_window//2
        self.edge = np.concatenate([np.arange(first,min(last,m)), np.arange(max(first,full_points-m),last)]) if m else np.array([],int)
        self.edge_fft = np.exp(-2j*np.pi*self.f[:,None]*(self.edge-first)[None,:]/fs)/self.n
        idx = self.edge[:,None]+self.offsets if len(self.edge) else np.zeros((0,len(self.offsets)),int)
        idx = np.where(idx < 0,-idx,idx); idx = np.where(idx >= full_points,2*full_points-2-idx,idx)
        self.edge_indices,self.edge_inverse = np.unique(idx,return_inverse=True)

    def templates(self, frequencies, weights, rate):
        f,w = np.asarray(frequencies),np.asarray(weights)
        w = w/max(w.sum(),1e-30)
        lam = -rate+2j*np.pi*np.r_[f,-f]
        coeff = np.column_stack([np.r_[w,w]/2, np.r_[w,-w]/(2j)])
        gain = 1-(np.exp(lam[:,None]*self.offsets/self.fs)@self.coeff) if self.window else np.ones(len(lam))
        numerator = np.exp(lam*self.first/self.fs)*np.expm1(lam*self.n/self.fs)/self.n
        denominator = np.expm1(lam[None,:]/self.fs-2j*np.pi*self.f[:,None]/self.fs)
        spectrum = (numerator[None,:]/denominator)@(gain[:,None]*coeff)
        if len(self.edge):
            # Correct only retained mirror-edge samples; interior transfer is exact.
            raw_unique = np.exp(self.edge_indices[:,None]*lam[None,:]/self.fs)@coeff
            smooth = np.sum(raw_unique[self.edge_inverse].reshape(len(self.edge),len(self.offsets),2)*self.coeff[None,:,None],axis=1)
            modes = np.exp(self.edge[:,None]*lam[None,:]/self.fs)
            difference = modes@coeff-smooth-modes@(gain[:,None]*coeff)
            spectrum += self.edge_fft@difference
        # All selected bins are strictly positive native bins: mean removal is zero there.
        return spectrum


def build_isopropylamine_model(parameters=None, *,record,directory,cancel,progress):
    p = dict(DEFAULT,**(parameters or {}))
    if set(p) != set(NAMES) or not np.isfinite(list(p.values())).all(): raise ValueError('Unknown or nonfinite J parameters.')
    models = {name:model_for(p,name) for name in ('methine','methyl')}
    for name,model in models.items(): storage.write_json(directory/f'{name}_model.json',model)
    return {'parameters_hz':p,'suggested_search_bounds_hz':BOUNDS,'models':models,
            'atom_order':['H_A1','H_A2','H_A3','H_B1','H_B2','H_B3','H_methine','C13'],
            'relative_molecule_counts':{'methine':1,'methyl':2},
            'assumptions':['Carbon-bound proton skeleton only: omit 14N and NH2.',
                'Within-methyl H-H couplings are unobservable in this symmetric response; fixed to zero.',
                'Inter-methyl H-H coupling fixed to zero as an approximation, not measured.',
                'Initial values and bounds are search settings, not literature or fitted values.']}


def fit_isopropylamine_j(comparison_run_id, variant_index, ranges, settings=None, *,record,directory,cancel,progress):
    from .analysis import plot
    start_time = time.perf_counter()
    s = dict(initial=DEFAULT.copy(),bounds=BOUNDS.copy(),free_parameters=NAMES.copy(),
             starts=8,max_nfev=80,screening_samples=128,seed=20260922,bin_stride=2,
             rate_bounds=[.3,40.],initial_rate=4.,objective='complex')
    settings = settings or {}
    if set(settings)-set(s): raise ValueError('Unknown fitting settings.')
    s.update(settings)
    if s['objective'] not in ('complex','magnitude'): raise ValueError('objective must be complex or magnitude.')
    initial = dict(DEFAULT,**s['initial']); bounds = dict(BOUNDS,**s['bounds'])
    free = s['free_parameters']
    if not free or len(set(free)) != len(free) or any(k not in NAMES for k in free): raise ValueError('Invalid free J parameter list.')
    if set(initial)!=set(NAMES) or set(bounds)!=set(NAMES) or not np.isfinite(list(initial.values())).all(): raise ValueError('Invalid J settings.')
    for key in NAMES:
        b = bounds[key]
        if len(b)!=2 or not np.isfinite(b).all() or not b[0]<b[1] or not b[0]<=initial[key]<=b[1]: raise ValueError('Invalid J bounds/initial: '+key)
    for key,lo,hi in [('starts',1,32),('max_nfev',2,1000),('screening_samples',0,4096),('bin_stride',1,32)]:
        if type(s[key]) is not int or not lo<=s[key]<=hi: raise ValueError('Invalid '+key)
    rb = s['rate_bounds']
    if len(rb)!=2 or not np.isfinite(rb).all() or not 0<rb[0]<rb[1] or not rb[0]<=s['initial_rate']<=rb[1]: raise ValueError('Invalid decay-rate bounds.')
    parent = storage.get_result(comparison_run_id)
    if parent['operation']!='compare_preprocessing' or type(variant_index) is not int or not 0<=variant_index<len(parent['variants']): raise ValueError('Invalid comparison/variant.')
    variant = parent['variants'][variant_index]; params=variant['parameters']; fs=parent['sampling_rate_hz']
    path=storage.artifact(comparison_run_id,f'variant_{variant_index}.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=variant['sha256']: raise ValueError('Preprocessing artifact changed.')
    average=storage.get_result(parent['parent_run_id'])
    if not isinstance(ranges,list) or not 1<=len(ranges)<=6: raise ValueError('Provide 1..6 frequency ranges.')
    for i,b in enumerate(ranges):
        if len(b)!=2 or not np.isfinite(b).all() or not 0<b[0]<b[1]<fs/2 or (i and b[0]<=ranges[i-1][1]): raise ValueError('Ranges must be sorted, disjoint, positive and below Nyquist.')
    with np.load(path,allow_pickle=False) as arrays: full_f=arrays['frequency']; full_y=arrays['spectrum']
    indices=[]; memberships=[]
    for i,(lo,hi) in enumerate(ranges):
        bins=np.flatnonzero((full_f>=lo)&(full_f<=hi))[::s['bin_stride']]
        if len(bins)<8: raise ValueError('Each range must retain at least eight bins.')
        indices.extend(bins.tolist()); memberships.extend([i]*len(bins))
    indices=np.array(indices); membership=np.array(memberships); f=full_f[indices]; y=full_y[indices]
    if len(indices)>5000 or not np.isfinite(y).all() or np.linalg.norm(y)==0: raise ValueError('Invalid data or more than 5000 fit bins; narrow ranges/increase bin_stride.')
    processor=ProcessedSpectrum(fs,average['points'],params['start_sample'],params['stop_sample'],indices,params['sg_window'],params['sg_order'])
    # Equalize each band's energy so a weak second multiplet remains informative.
    band_scale=np.array([max(np.sqrt(np.mean(abs(y[membership==i])**2)),1e-15) for i in range(len(ranges))])
    scale=band_scale[membership]
    lower=np.array([bounds[k][0] for k in free]+[np.log(rb[0])]*2)
    upper=np.array([bounds[k][1] for k in free]+[np.log(rb[1])]*2)
    x0=np.array([initial[k] for k in free]+[np.log(s['initial_rate'])]*2)
    evaluations=0; best_score=float('inf'); history=[]
    def predict(x,detail=False):
        nonlocal evaluations,best_score
        if cancel(): raise InterruptedError('J search cancelled.')
        p=dict(initial,**dict(zip(free,x[:len(free)])))
        cols=[]
        for i,name in enumerate(('methine','methyl')):
            freq,weight=transitions(p,name)
            cols.extend(processor.templates(freq,weight,np.exp(x[len(free)+i])).T)
        # Independent real cosine/sine gains, summed before magnitude. Isotopic
        # number weights are absorbed by unknown response amplitudes, not inferred.
        for i in range(len(ranges)):
            mask=(membership==i).astype(float)
            cols.extend([mask,1j*mask])
        design=np.column_stack(cols); weighted=design/scale[:,None]; target=y/scale
        a=np.vstack([weighted.real,weighted.imag]); b=np.r_[target.real,target.imag]
        norms=np.maximum(np.linalg.norm(a,axis=0),1e-30)
        coef=np.linalg.lstsq(a/norms,b,rcond=1e-10)[0]/norms
        if s['objective']=='magnitude':
            # Fit |sum complex signals|, never sum their individual magnitudes.
            # Analytic inner derivatives avoid extra Hamiltonian evaluations.
            normalized=weighted/norms
            def inner_residual(c): return abs(normalized@c)-abs(target)
            def inner_jac(c):
                z=normalized@c
                return (z.conj()[:,None]*normalized).real/np.maximum(abs(z)[:,None],1e-15)
            inner=least_squares(inner_residual,coef*norms,jac=inner_jac,max_nfev=60,
                                ftol=1e-9,xtol=1e-9,gtol=1e-9)
            coef=inner.x/norms
        fitted=design@coef
        residual=((abs(fitted)-abs(y))/scale) if s['objective']=='magnitude' else np.r_[((fitted-y)/scale).real,((fitted-y)/scale).imag]
        if not np.isfinite(residual).all(): raise FloatingPointError('Nonfinite fit residual.')
        evaluations+=1; score=float(np.mean(residual**2))
        if score<best_score:
            best_score=score
            storage.write_json(directory/'best_so_far.json',{'status':'provisional','evaluations':evaluations,
                'score':score,'parameters_hz':p,'decay_rates_per_s':np.exp(x[len(free):]).tolist()})
        if evaluations%25==0:
            storage.write_json(directory/'search_progress.json',{'evaluations':evaluations,'best_score':best_score})
        return (residual,fitted,coef,design,p) if detail else residual
    # A seeded Latin-hypercube screen supplies varied starts; it is not exhaustive.
    candidates=[(float(np.mean(predict(x0)**2)),x0)]
    if s['screening_samples']:
        sample=qmc.LatinHypercube(len(x0),seed=s['seed']).random(s['screening_samples'])
        for i,x in enumerate(lower+sample*(upper-lower)):
            candidates.append((float(np.mean(predict(x)**2)),x))
            progress(round(2000*(i+1)/s['screening_samples']),10000)
    candidates.sort(key=lambda pair:pair[0])
    starts=[x0]  # Always refine the explicit user/default initial point as well.
    for score,x in candidates:
        if len(starts)>=s['starts']: break
        if all(np.linalg.norm((x-other)/(upper-lower))>.08 for other in starts): starts.append(x)
        if len(starts)>=s['starts']: break
    solutions=[]
    for i,x in enumerate(starts):
        fit=least_squares(predict,np.clip(x,lower+1e-9,upper-1e-9),bounds=(lower,upper),
            x_scale='jac',diff_step=1e-4,max_nfev=s['max_nfev'],ftol=1e-7,xtol=1e-7,gtol=1e-7)
        score=float(np.mean(fit.fun**2)); solutions.append((score,fit))
        history.append({'score':score,'parameters_hz':dict(initial,**dict(zip(free,fit.x[:len(free)]))),
                        'decay_rates_per_s':np.exp(fit.x[len(free):]).tolist(),
                        'optimizer_success':bool(fit.success),'message':fit.message,'nfev':fit.nfev})
        storage.write_json(directory/'candidates.json',sorted(history,key=lambda r:r['score']))
        progress(2000+round(8000*(i+1)/len(starts)),10000)
    score,best=min(solutions,key=lambda row:row[0])
    residual,fitted,coef,design,p=predict(best.x,True)
    # Sensitivity is descriptive: no confidence interval from correlated processed bins.
    jac=best.jac; norms=np.linalg.norm(jac,axis=0); gram=jac.T@jac
    correlation=gram/np.maximum(norms[:,None]*norms[None,:],1e-30)
    sv=np.linalg.svd(jac*(upper-lower)[None,:],compute_uv=False)
    boundary=[free[i] for i in range(len(free)) if min(best.x[i]-lower[i],upper[i]-best.x[i])<.01*(upper[i]-lower[i])]
    near=[row for row in history if row['score']<=score*1.1+1e-12]
    spread={key:[min(row['parameters_hz'][key] for row in near),
                 max(row['parameters_hz'][key] for row in near)] for key in free}
    for i,(lo,hi) in enumerate(ranges):
        mask=membership==i
        plot(directory/f'fit_range_{i}.png',[(f[mask],abs(y[mask]),'Experiment'),(f[mask],abs(fitted[mask]),'Candidate model')],
             'Frequency (Hz)','Magnitude (ADC units)',f'Candidate J fit: {lo:g}-{hi:g} Hz')
        plot(directory/f'residual_range_{i}.png',[(f[mask],(y-fitted)[mask].real,'Real'),(f[mask],(y-fitted)[mask].imag,'Imaginary')],
             'Frequency (Hz)','Complex residual (ADC units)',f'Residual: {lo:g}-{hi:g} Hz')
    np.savez_compressed(directory/'fit_arrays.npz',frequency_hz=f,experiment=y,fitted=fitted,residual=y-fitted,
                        methine=design[:,:2]@coef[:2],methyl=design[:,2:4]@coef[2:4],membership=membership)
    for name in ('methine','methyl'): storage.write_json(directory/f'{name}_model.json',model_for(p,name))
    storage.write_json(directory/'parameter_sensitivity.json',{'parameters':free+['log_rate_methine','log_rate_methyl'],
        'jacobian_norms':norms.tolist(),'jacobian_column_cosines':correlation.tolist(),'scaled_singular_values':sv.tolist()})
    report={'parent_run_id':comparison_run_id,'source_variant':variant_index,'source_sha256':variant['sha256'],
            'preprocessing':params,'settings':s,'ranges_hz':ranges,'free_parameters':free,
            'parameters_hz':p,'fixed_parameters_hz':{k:p[k] for k in NAMES if k not in free},
            'decay_rates_per_s':dict(zip(['methine','methyl'],np.exp(best.x[len(free):]).tolist())),
            'linear_coefficients':coef.tolist(),'weighted_mean_square_residual':score,
            'relative_complex_residual':float(np.linalg.norm(y-fitted)/np.linalg.norm(y)),
            'relative_magnitude_residual':float(np.linalg.norm(abs(y)-abs(fitted))/np.linalg.norm(abs(y))),
            'parameters_near_bounds':boundary,'optimizer_success':bool(best.success),
            'optimizer_message':best.message,'candidates':sorted(history,key=lambda r:r['score']),
            'actual_starts':len(starts),'within_10_percent_score_parameter_ranges_hz':spread,
            'candidate_spread_note':'Descriptive multi-start range only, not a confidence interval; even a single found solution need not be unique.',
            'evaluations':evaluations,'elapsed_s':time.perf_counter()-start_time,
            'interpretation':'Candidate J values conditional on a simplified skeleton and response model; not a unique/full molecular J determination.',
            'warnings':['14N, NH2, exchange and residual fields omitted.',
                'Relative transition weights assume unpulsed gamma-weighted thermal preparation; only one global phase/amplitude per isotopomer is free.',
                'Positive direct C-H and vicinal H-H signs are search assumptions. Other sign-equivalent solutions are not excluded.',
                'No confidence intervals: processed Fourier bins are correlated; repeat-group validation not performed by this tool.',
                'Oscillatory transitions only: zero-frequency modes removed explicitly; a complex constant per band absorbs background.',
                'Inter-methyl H-H set to zero as an approximation; within-methyl equivalent H-H is unobservable here.',
                'Natural-abundance 1:2 molecule counts are absorbed into unknown response gains; amplitudes do not measure abundance.',
                'Only requested bands are fitted; no claim of validation outside them.']}
    if boundary: report['warnings'].append('Some J parameters reached search boundaries; widen/review the model before interpretation.')
    if not best.success: report['warnings'].append('Best local optimization exhausted its budget or did not converge.')
    if s['objective']=='magnitude': report['warnings'].append('Magnitude fitting discards observed phase and has noise bias; the complex residual is diagnostic, not its optimized objective.')
    if len(near)>1: report['warnings'].append('Multiple candidate starts are within 10% of the best objective; inspect their parameter differences before selecting a J set.')
    return report
