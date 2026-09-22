"""Reproduce independent synthetic drift counterexamples; writes only .analysis.

Run from the project root: python -m examples.validate_drift
No experimental files or metadata are used.
"""
import json
import numpy as np
from zulf_tools import storage
from zulf_tools.analysis import plot
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes,mode_design


def main():
    parameters=dict(fs_hz=256.,points=1536,true_t2star_s=1.3,first=32,last=1400,
        frequency_offsets_hz=[-.3,.3],phase_offsets_rad=[-np.pi/3,np.pi/3],chirp_hz_per_s=.2)
    record,directory=storage.begin('synthetic_drift_validation',parameters)
    fs=parameters['fs_hz'];n=parameters['points'];t=np.arange(n)/fs;first=parameters['first'];last=parameters['last']
    f=np.fft.rfftfreq(last-first,1/fs);bins=np.flatnonzero((f>=35)&(f<=45));op=ProcessedSpectrum(fs,n,first,last,bins)
    def spectrum(y):return np.fft.rfft(y[first:last])[bins]/(last-first)
    env=np.exp(-t/1.3)
    groups=np.array([env*np.cos(2*np.pi*(40+df)*t) for df in [-.3,.3]])
    pooled=groups.mean(axis=0)
    phase_mean=np.mean([env*np.cos(2*np.pi*40*t+phase) for phase in [-np.pi/3,np.pi/3]],axis=0)
    chirp=env*np.cos(2*np.pi*(40*t+.5*.2*t*t))
    cases=[('frequency_drift_single',pooled,1,False,[40.]),('frequency_drift_two',pooled,2,True,[39.6,40.4]),
        ('phase_offset_average',phase_mean,1,False,[40.]),('within_fid_chirp',chirp,1,False,[40.])]
    results=[];arrays=dict(time_s=t,frequency_drift_groups=groups,frequency_drift_average=pooled,phase_offset_average=phase_mean,chirp=chirp)
    for name,y,k,shared,seeds in cases:
        observed=spectrum(y);r=fit_modes(op,observed,[35,45],[.1,3.],mode_count=k,shared_decay=shared,initial_frequencies=seeds,starts=3)
        prediction=r.pop('fitted');r['scenario']=name
        r['individual_group_frozen_errors']=[float(np.linalg.norm(spectrum(g)-prediction)/np.linalg.norm(spectrum(g))) for g in groups] if name.startswith('frequency_drift') else None
        results.append(r);arrays[name+'_spectrum']=observed;arrays[name+'_prediction']=prediction
        plot(directory/(name+'_real.png'),[(op.f,observed.real,'Independent synthetic observation'),(op.f,prediction.real,'Stationary mode model')],
            'Frequency (Hz)','Real spectrum','Known T2*=1.3 s: '+name.replace('_',' '))
        residual=observed-prediction
        plot(directory/(name+'_residual.png'),[(op.f,residual.real,'Real'),(op.f,residual.imag,'Imaginary')],
            'Frequency (Hz)','Complex residual',name.replace('_',' '))
    plot(directory/'frequency_drift_fids.png',[(t[t<=2],y[t<=2],label) for y,label in [(groups[0],'Acquisition at 39.7 Hz'),(groups[1],'Acquisition at 40.3 Hz'),(pooled,'Coherent mean')]],
        'Acquisition time (s)','Amplitude','Each acquisition has one mode and T2*=1.3 s')
    np.savez_compressed(directory/'synthetic_arrays.npz',**arrays)
    result=storage.complete(record,scenarios=results,scientifically_validated=False,
        warnings=['Synthetic counterexamples, not evidence of actual experimental drift.',
        'A perfect two-mode average fit cannot distinguish within-acquisition components from between-acquisition frequency heterogeneity.',
        'Constant between-repeat phase offsets can reduce amplitude without changing a common decay.',
        'Within-FID chirp can pass numerical convergence checks while biasing stationary-model decay.'])
    print(json.dumps(dict(run_id=result['run_id'],scenarios=[dict(name=r['scenario'],t2star_s=r['t2star_s'],relative_complex_residual=r['relative_complex_residual']) for r in results])))


if __name__=='__main__':main()
