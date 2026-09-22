"""J-constrained, explicit transition-cluster decay fits on repeated FIDs."""
import numpy as np
from . import storage
from .jfit import transitions, model_for, ProcessedSpectrum
from .nitrogen import nitrogen_transitions
from .repeats import load_group_averages
from .band_relaxation import _indices
from .transition_decay import fit_transition_decay, transition_design


def partition_transitions(frequencies, weights, edges, family):
    """Partition ALL transitions; edge lists split, never crop, the model."""
    f=np.asarray(frequencies,dtype=float);w=np.asarray(weights,dtype=float)
    cuts=np.asarray(edges,dtype=float)
    if cuts.ndim!=1 or not np.isfinite(cuts).all() or np.any(cuts<=0) or np.any(np.diff(cuts)<=0):
        raise ValueError('Cluster edges must be finite positive and strictly increasing.')
    if f.ndim!=1 or f.shape!=w.shape or not len(f) or not np.isfinite(f).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid transition table.')
    result=[];which=np.searchsorted(cuts,f,side='right')
    for i in range(len(cuts)+1):
        m=which==i
        if not m.any():continue
        result.append(dict(name=f'{family}_{i}',family=family,transition_indices=np.flatnonzero(m).tolist(),
            frequencies_hz=f[m].tolist(),weights=w[m].tolist(),
            fraction_of_family_weight=float(w[m].sum()/w.sum()),
            lower_edge_hz=None if i==0 else float(cuts[i-1]),
            upper_edge_hz=None if i==len(cuts) else float(cuts[i]),
            normalized_weight_frequency_hz=float(np.average(f[m],weights=w[m]))))
    return result


def fit_j_cluster_decay(group_run_id, carbon_run_id, ranges, discovery_groups, validation_groups,
                        nitrogen_run_id=None, cluster_edges=None, t2_bounds=None,
                        preprocessing=None, settings=None, *, record, directory, cancel, progress):
    """Fixed J; free gain/phase and decay per explicit cluster, with all tails.

    An empty family edge list gives a shared family response/decay baseline.
    N15 is optional and provisional. No automatic assignment or acceptance.
    """
    from .analysis import recipe, plot
    carbon=storage.get_result(carbon_run_id)
    if carbon.get('operation')!='fit_isopropylamine_j':raise ValueError('Require a carbon J-fit run.')
    nit=storage.get_result(nitrogen_run_id) if nitrogen_run_id is not None else None
    if nit is not None and 'couplings_hz' not in nit:raise ValueError('N15 source must contain explicit couplings_hz.')
    families=['methine','methyl']+(['N15'] if nit is not None else [])
    edges={k:[] for k in families};edges['methyl']=[190.]
    if cluster_edges is not None:
        if not isinstance(cluster_edges,dict) or set(cluster_edges)-set(families):raise ValueError('Unknown cluster family.')
        edges.update(cluster_edges)
    if not isinstance(ranges,list) or not 1<=len(ranges)<=6:raise ValueError('Supply one to six ranges.')
    for i,b in enumerate(ranges):
        if len(b)!=2 or not np.isfinite(b).all() or not 0<b[0]<b[1] or (i and b[0]<=ranges[i-1][1]):raise ValueError('Ranges must be positive, ordered and disjoint.')
    bounds=[.05,3.] if t2_bounds is None else t2_bounds
    if len(bounds)!=2 or not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1]:raise ValueError('Invalid T2* bounds.')
    config=dict(starts=4,max_nfev=150,max_evaluations=6000,max_seconds=120.,seed=20260922,phase_delay_bounds_s=None)
    if settings and set(settings)-set(config):raise ValueError('Unknown solver setting.')
    config.update(settings or {})
    groups=[];tables={}
    for name in families:
        f,w=nitrogen_transitions(nit['couplings_hz']) if name=='N15' else transitions(carbon['parameters_hz'],name)
        tables[name]=dict(frequencies_hz=f.tolist(),weights=w.tolist())
        groups.extend(partition_transitions(f,w,edges[name],name))
    if len(groups)>32:raise ValueError('At most 32 nonempty clusters are supported.')
    storage.write_json(directory/'transition_tables.json',tables)
    storage.write_json(directory/'transition_groups.json',groups)
    storage.write_json(directory/'J_models.json',dict(carbon_parameters_hz=carbon['parameters_hz'],
        carbon_models={k:model_for(carbon['parameters_hz'],k) for k in ['methine','methyl']},
        carbon_atom_order=['H_A1','H_A2','H_A3','H_B1','H_B2','H_B3','H_CH','C13'],
        nitrogen_parameters_hz=None if nit is None else nit['couplings_hz']))
    means,counts,source=load_group_averages(group_run_id);fs=source['sampling_rate_hz']
    train=_indices(discovery_groups,len(means),'discovery_groups');valid=_indices(validation_groups,len(means),'validation_groups')
    if set(train)&set(valid):raise ValueError('Discovery and validation groups must be disjoint.')
    if ranges[-1][1]>=fs/2:raise ValueError('Ranges exceed Nyquist.')
    spec=preprocessing or {};spectra={}
    for idx in sorted(set(train+valid)):
        processed,_,params=recipe(means[idx],fs,spec)
        spectra[idx]=np.fft.rfft(processed)/len(processed)
    ff=np.fft.rfftfreq(len(processed),1/fs);bins=[];membership=[]
    for i,(lo,hi) in enumerate(ranges):
        selected=np.flatnonzero((ff>=lo)&(ff<=hi))
        if len(selected)<8:raise ValueError('Each range needs at least eight native bins.')
        bins.extend(selected);membership.extend([i]*len(selected))
    bins=np.asarray(bins,dtype=int);membership=np.asarray(membership)
    if len(bins)>5000:raise ValueError('More than 5000 observed bins; select narrower ranges explicitly.')
    observed=np.average([spectra[i][bins] for i in train],axis=0,weights=counts[train])
    held=np.average([spectra[i][bins] for i in valid],axis=0,weights=counts[valid])
    p=ProcessedSpectrum(fs,source['points'],params['start_sample'],params['stop_sample'],bins,params['sg_window'],params['sg_order'])
    # Equal total band weight, including differing native-bin counts.
    scale=np.ones(len(bins))
    for i in range(len(ranges)):
        m=membership==i;scale[m]=max(float(np.linalg.norm(observed[m])),1e-15)
    fit=fit_transition_decay(p,observed,groups,bounds,observation_scale=scale,cancel=cancel,**config)
    pred=fit.pop('fitted');progress(1,3)
    design=transition_design(p,groups,fit['t2star_s'],fit['phase_delay_s'])
    terms=np.array([design[:,2*i:2*i+2]@fit['cos_sin_coefficients'][i] for i in range(len(groups))])
    np.testing.assert_allclose(terms.sum(axis=0),pred,atol=1e-10)
    error=lambda y:float(np.linalg.norm(y-pred)/np.linalg.norm(y))
    bands=[];rows=[]
    for i,g in enumerate(groups):
        rows.append(dict(g,t2star_s=fit['t2star_s'][i],amplitude=fit['amplitudes'][i],phase_rad=fit['phases_rad'][i],
            boundary=i in fit['boundary_group_indices'],observed_transition_count=sum(any(lo<=f<=hi for lo,hi in ranges) for f in g['frequencies_hz']),
            physical_assignment_accepted=False,interpretation='Conditional cluster decay; inspect residual and stability before interpretation.'))
    for b,(lo,hi) in enumerate(ranges):
        m=membership==b
        bands.append(dict(range_hz=[lo,hi],validation_complex_error=float(np.linalg.norm(held[m]-pred[m])/np.linalg.norm(held[m]))))
        for part,fn in [('magnitude',np.abs),('real',np.real),('imaginary',np.imag)]:
            traces=[(p.f[m],fn(y[m]),name) for y,name in [(observed,'Discovery'),(held,'Validation'),(pred,'Frozen prediction')]]
            traces += [(p.f[m],fn(y[m]),g['name']) for y,g in zip(terms,groups)]
            plot(directory/f'band_{b}_{part}.png',traces,'Frequency (Hz)',part.title()+' (ADC units)',f'Fixed-J cluster decay: {lo:g}-{hi:g} Hz')
        plot(directory/f'band_{b}_residual.png',[(p.f[m],(held-pred)[m].real,'Real'),(p.f[m],(held-pred)[m].imag,'Imaginary')],
             'Frequency (Hz)','Residual (ADC units)','Frozen validation residual')
    progress(2,3)
    np.savez_compressed(directory/'cluster_arrays.npz',frequency_hz=p.f,discovery=observed,validation=held,prediction=pred,components=terms,
                        membership=membership,observation_scale=scale,group_indices=sorted(set(train+valid)),
                        group_spectra=np.array([spectra[i][bins] for i in sorted(set(train+valid))]))
    storage.write_json(directory/'cluster_results.json',rows);progress(3,3)
    return dict(group_run_id=group_run_id,carbon_run_id=carbon_run_id,nitrogen_run_id=nitrogen_run_id,
        source_arrays_sha256=source['arrays_sha256'],carbon_parameter_digest=storage.digest(carbon['parameters_hz']),
        nitrogen_parameter_digest=None if nit is None else storage.digest(nit['couplings_hz']),
        preprocessing=params,ranges_hz=ranges,t2_bounds_s=bounds,settings=config,cluster_edges=edges,
        discovery_groups=train,validation_groups=valid,clusters=rows,fit=fit,bands=bands,
        validation_relative_complex_residual=error(held),validation_group_errors={str(i):error(spectra[i][bins]) for i in valid},
        template_diagnostics=p.template_diagnostics(),scientifically_validated=False,
        warnings=['J fits used all-scan data: validation is conditional, not untouched validation of J.',
        'One independent response/phase and decay per explicit cluster; thermal weights fixed only within clusters.',
        'All simulated oscillatory transitions retained, including out-of-range tails. No observed phase correction.',
        'Spectral model mismatch and nonidentifiability can bias effective FID T2*. No intrinsic T2 assignment.',
        'N15 is a hypothesis; source J optimizer may not have converged.',
        'Methyl frequency partitions are empirical, not established symmetry relaxation sectors.'])
