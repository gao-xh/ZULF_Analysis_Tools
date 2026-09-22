"""Explicit 15N hypotheses, with absolute thermal weights and natural populations.

The coherent ten-spin skeleton is not an exchange/relaxation mechanism model.
Unknown couplings must be supplied explicitly; none are fitted as literature facts.
"""
from functools import lru_cache
from math import comb
import hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares
from matplotlib.figure import Figure
from . import storage
from .jfit import angular_momentum, ProcessedSpectrum, transitions
from .simulation import GAMMA


def spin_sectors(n):
    return [(n/2-k,comb(n,k)-(comb(n,k-1) if k else 0)) for k in range(n//2+1)]


@lru_cache(maxsize=4)
def nitrogen_blocks(include_nh2=True, methyl_count=6):
    blocks=[]
    for sm,mult in spin_sectors(methyl_count):
        for sn,mn in (spin_sectors(2) if include_nh2 else [(0.,1)]):
            spins=[sm,.5,sn,.5];dims=[int(2*s+1) for s in spins];ops=[]
            for i,s in enumerate(spins):
                vec=[]
                for a in angular_momentum(s):
                    op=np.ones((1,1),complex)
                    for k,dim in enumerate(dims):op=np.kron(op,a if k==i else np.eye(dim))
                    vec.append(op)
                ops.append(vec)
            pair={(i,k):sum(a@b for a,b in zip(ops[i],ops[k])).real for i in range(4) for k in range(i+1,4)}
            obs=(GAMMA['1H']*sum(ops[i][2] for i in range(3))+GAMMA['15N']*ops[3][2]).real
            blocks.append((pair,obs,mult*mn))
    return blocks


def nitrogen_transitions(p, include_nh2=True, methyl_count=6):
    frequencies=[];weights=[];dim=2**(methyl_count+2+(2 if include_nh2 else 0))
    for pair,obs,mult in nitrogen_blocks(include_nh2,methyl_count):
        h=p['J_HH_vicinal']*pair[0,1]+p['J_N_methyl']*pair[0,3]+p['J_N_methine']*pair[1,3]
        if include_nh2:h=h+p['J_NH']*pair[2,3]+p['J_NH_Hmethine']*pair[1,2]
        e,v=eigh(h);o=v.T@obs@v;i,k=np.triu_indices(len(e),1)
        frequencies.extend((e[k]-e[i]).tolist());weights.extend((2*mult*o[i,k]**2/dim).tolist())
    f,w=np.array(frequencies),np.array(weights)
    keep=(f>1e-7)&(w>max(w.max()*1e-13,1e-20));f,w=f[keep],w[keep]
    if not len(f):raise ValueError('Hypothesis has no observable oscillatory transitions.')
    _,inv=np.unique(np.round(f,7),return_inverse=True);tot=np.bincount(inv,weights=w)
    return np.bincount(inv,weights=w*f)/tot,tot


def inspect_n15_hypotheses(j_fit_run_id, hypotheses, rates_per_s, abundance_n15=.003663,
                          abundance_c13=.0107, *,record,directory,cancel,progress):
    """Screen fixed 15N spectra against a frozen carbon model; no identification."""
    required={'J_HH_vicinal','J_N_methyl','J_N_methine','J_NH','J_NH_Hmethine'}
    if not isinstance(hypotheses,list) or not 1<=len(hypotheses)<=64:raise ValueError('Supply 1..64 explicit hypotheses.')
    for h in hypotheses:
        if set(h)!={'label','couplings_hz','include_nh2'} or not isinstance(h['label'],str) or type(h['include_nh2']) is not bool:raise ValueError('Invalid hypothesis fields.')
        if set(h['couplings_hz'])!=required or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or abs(v)>200 for v in h['couplings_hz'].values()):raise ValueError('Invalid explicit couplings.')
    if not isinstance(rates_per_s,list) or not 1<=len(rates_per_s)<=8 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not .1<=v<=100 for v in rates_per_s):raise ValueError('Supply 1..8 rates in 0.1..100 /s.')
    if not 0<abundance_n15<.1 or not 0<abundance_c13<.1:raise ValueError('Invalid natural abundance scenario.')
    r=storage.get_result(j_fit_run_id)
    if r['operation']!='fit_isopropylamine_j' or 'methyl' not in r['active_isotopomers']:raise ValueError('Require a carbon J fit with methyl.')
    parent=storage.get_result(r['parent_run_id']);avg=storage.get_result(parent['parent_run_id']);p=r['preprocessing'];fs=parent['sampling_rate_hz']
    with np.load(storage.artifact(j_fit_run_id,'fit_arrays.npz')) as z:a={k:z[k].copy() for k in z.files}
    f=a['frequency_hz'];y=a['experiment'];base=a['fitted'];membership=a['membership']
    bins=np.rint(f*(p['stop_sample']-p['start_sample'])/fs).astype(int)
    op=ProcessedSpectrum(fs,avg['points'],p['start_sample'],p['stop_sample'],bins,p['sg_window'],p['sg_order'])
    # Exact, disjoint isotope probabilities. Doubly labeled species are omitted.
    pn=abundance_n15*(1-abundance_c13)**3
    pc=2*abundance_c13*(1-abundance_c13)**2*(1-abundance_n15)
    _,cw=transitions(r['parameters_hz'],'methyl')
    cc=np.asarray(r['linear_coefficients'][2:4]);calibration=cc/(pc*cw.sum())
    scale=np.array([max(np.sqrt(np.mean(abs(y[membership==i])**2)),1e-15) for i in range(len(r['ranges_hz']))])[membership]
    score=lambda z:float(np.mean(((abs(z)-abs(y))/scale)**2))
    rows=[];arrays=dict(frequency_hz=f,experiment=y,carbon_prediction=base,membership=membership)
    spectra=[];best=None
    # Wide prediction shows peaks excluded from fitting; sampled frequencies are native bins.
    allf=np.fft.rfftfreq(p['stop_sample']-p['start_sample'],1/fs)
    widebins=np.flatnonzero((allf>=1)&(allf<=280))[::2]
    wideop=ProcessedSpectrum(fs,avg['points'],p['start_sample'],p['stop_sample'],widebins,p['sg_window'],p['sg_order'])
    arrays['wide_frequency_hz']=allf[widebins]
    for hi,h in enumerate(hypotheses):
        if cancel():raise InterruptedError('15N hypotheses cancelled.')
        tf,tw=nitrogen_transitions(h['couplings_hz'],h['include_nh2'])
        spectra.append(dict(label=h['label'],frequency_hz=tf.tolist(),thermal_weights=tw.tolist(),population=pn))
        for rate in rates_per_s:
            if cancel():raise InterruptedError('15N hypotheses cancelled.')
            # Undo internal template normalization before multiplying population.
            design=op.templates(tf,tw,rate)*(tw.sum()*pn)
            natural=design@calibration
            normalized=design*np.linalg.norm(calibration)
            def residual(c):return (abs(base+normalized@c)-abs(y))/scale
            # Zero is an explicit nested null; common N15 phase/gain only.
            starts=[np.zeros(2),calibration/max(np.linalg.norm(calibration),1e-30)]
            candidates=[least_squares(residual,s,max_nfev=80) for s in starts]
            fit=min(candidates,key=lambda q:np.mean(q.fun**2))
            fitted=base+normalized@fit.x
            if score(fitted)>score(base):fitted=base.copy();coef=np.zeros(2)
            else:coef=fit.x
            key=f'h{hi}_r{rate:g}'
            arrays[key+'_n15_natural']=natural;arrays[key+'_n15_fitted']=normalized@coef
            wide=wideop.templates(tf,tw,rate)*(tw.sum()*pn)
            arrays[key+'_wide_natural']=wide@calibration;arrays[key+'_wide_fitted']=wide@(np.linalg.norm(calibration)*coef)
            row=dict(key=key,hypothesis_index=hi,label=h['label'],rate_per_s=rate,
                score=score(fitted),natural_score=score(base+natural),response_factor=float(np.linalg.norm(coef)),
                phase_relative_to_methyl_rad=float(np.angle(complex(*coef))-np.angle(complex(*calibration))),
                optimizer_success=bool(fit.success),coefficients_in_calibration_norm_units=coef.tolist(),
                band_residuals=[float(np.linalg.norm((abs(fitted)-abs(y))[membership==i])/np.linalg.norm(y[membership==i])) for i in range(len(r['ranges_hz']))],
                weight_fraction_above_200_hz=float(tw[tf>200].sum()/tw.sum()))
            rows.append(row)
            if best is None or row['score']<best['score']:best=row
        storage.write_json(directory/'candidates.json',rows);progress(hi+1,len(hypotheses))
    key=best['key'];added=arrays[key+'_n15_fitted'];nat=arrays[key+'_n15_natural']
    h=hypotheses[best['hypothesis_index']];tr=spectra[best['hypothesis_index']]
    for i,(lo,hi) in enumerate(r['ranges_hz']):
        mask=membership==i;fig=Figure(figsize=(11,6),layout='constrained');ax=fig.add_subplot(111)
        for curve,label,color in [(y,'Experiment','.55'),(base,'Frozen carbon model','#0072B2'),(base+added,'Carbon + fitted 15N hypothesis','#D55E00'),(added,'15N component','#009E73')]:
            ax.plot(f[mask],abs(curve[mask]),lw=1.2,color=color,label=label)
        tf=np.array(tr['frequency_hz']);tw=np.array(tr['thermal_weights']);keep=(tf>=lo)&(tf<=hi)
        if np.any(keep):ax.vlines(tf[keep],.025,.025+.075*tw[keep]/tw[keep].max(),transform=ax.get_xaxis_transform(),color='#009E73',lw=.7,label='15N unbroadened positions')
        ax.set(xlabel='Frequency (Hz)',ylabel='Magnitude (ADC units)',title=f'Exploratory 15N addition: {lo:g}-{hi:g} Hz');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.supxlabel('Frozen carbon parameters. Same-data amplitude/phase screen, not isotope identification.\nComponent magnitudes do not add; N15 couplings and exchange regime are hypotheses.',fontsize=8)
        fig.savefig(directory/f'n15_comparison_band_{i}.png',dpi=150)
    fig=Figure(figsize=(12,5.5),layout='constrained');ax=fig.add_subplot(111)
    ax.plot(arrays['wide_frequency_hz'],abs(arrays[key+'_wide_natural']),label='Natural population + methyl-calibrated thermal response')
    ax.plot(arrays['wide_frequency_hz'],abs(arrays[key+'_wide_fitted']),label='Fitted N15 response hypothesis')
    ax.axvspan(110,150,alpha=.08,color='gray');ax.axvspan(230,275,alpha=.08,color='gray')
    ax.set(xlabel='Frequency (Hz)',ylabel='Magnitude (ADC units)',title='15N predicted spectrum, including frequencies outside the fit');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.supxlabel('Natural abundance does not establish identical preparation/transfer response. Unfitted frequencies are predictions.',fontsize=8)
    fig.savefig(directory/'n15_wide_prediction.png',dpi=150)
    np.savez_compressed(directory/'n15_arrays.npz',**arrays)
    storage.write_json(directory/'transitions.json',spectra)
    return dict(parent_run_id=j_fit_run_id,source_arrays_sha256=hashlib.sha256(storage.artifact(j_fit_run_id,'fit_arrays.npz').read_bytes()).hexdigest(),
        preprocessing=p,baseline_score=score(base),best=best,candidates=rows,n15_only_population=pn,methyl_c13_only_population=pc,
        calibration_note='Methyl carbon fitted coefficients divided by its population and absolute thermal weight. Unknown isotope-specific preparation/detection makes this conditional.',
        warnings=['Unknown couplings are explicit hypotheses, not a measured full N15 matrix.',
            'No physical exchange superoperator: NH2 omitted is a decoupled limit, not a finite exchange-rate simulation.',
            'Carbon parameters frozen; N15 gain/phase selection uses the same observations and is not independent validation.',
            'Free response amplitude may absorb carbon-model errors. Improvement alone does not identify N15.',
            'Double 13C/15N labels, 14N effects, residual fields and NH-to-methyl H-H couplings omitted.',
            'Six methyl protons are assumed equivalent to external spins; the two NH protons are equivalent.',
            'All component spectra sum in complex form before taking magnitude.'])
