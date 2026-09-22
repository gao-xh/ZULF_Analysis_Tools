"""Conditional decay-grid inversion; known synthetic frequency and phase only.

Run python -m examples.validate_regularization. This is not a physical
distribution estimator for unknown experimental peaks or phases.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from matplotlib.figure import Figure
from zulf_tools import storage
from zulf_tools.analysis import plot
from zulf_tools.jfit import ProcessedSpectrum


def main():
    parameters=dict(fs_hz=256.,points=2048,first=16,last=1600,
        frequency_hz=40.25,phase_rad=.4,t2_bounds_s=[.1,3.],noise_sd=.025,
        grid_counts=[32,64],alphas=[0.,1e-8,1e-5,.01,1.],seed=381,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record,directory=storage.begin('synthetic_decay_regularization',parameters)
    started=time.perf_counter();rows=[];arrays={}
    fs=parameters['fs_hz'];n=parameters['points'];t=np.arange(n)/fs
    first,last=parameters['first'],parameters['last']
    f=np.fft.rfftfreq(last-first,1/fs);bins=np.flatnonzero((f>=35)&(f<=45))
    op=ProcessedSpectrum(fs,n,first,last,bins)
    noise=np.random.default_rng(parameters['seed']).normal(0,parameters['noise_sd'],n)
    arrays.update(time_s=t,noise=noise,frequency_hz=op.f)
    for name,taus,amplitudes in [('single',[.8],[1.]),('close_pair',[.9,1.1],[.6,1.])]:
        raw=sum(a*np.exp(-t/tau) for a,tau in zip(amplitudes,taus))*np.cos(2*np.pi*parameters['frequency_hz']*t+parameters['phase_rad'])
        observed=np.fft.rfft((raw+noise)[first:last])[bins]/(last-first)
        arrays[name+'_clean_fid']=raw;arrays[name+'_observed']=observed
        for count in parameters['grid_counts']:
            loglo,loghi=np.log(parameters['t2_bounds_s']);h=(loghi-loglo)/count
            grid=np.exp(loglo+(np.arange(count)+.5)*h)
            phase=np.array([np.cos(parameters['phase_rad']),-np.sin(parameters['phase_rad'])])
            design=np.column_stack([op.templates([parameters['frequency_hz']],[1.],1/tau)@phase for tau in grid])
            scale=np.linalg.norm(observed)
            matrix=np.vstack([design.real,design.imag])/scale
            target=np.r_[observed.real,observed.imag]/scale
            # Weights are masses in uniform log(T) cells. Smooth the density
            # w/h: integral of squared curvature is ||D2 w / h**2.5||**2.
            penalty=np.diff(np.eye(count),n=2,axis=0)/h**2.5
            curves=[]
            for alpha in parameters['alphas']:
                augmented=np.vstack([matrix,np.sqrt(alpha)*penalty])
                rhs=np.r_[target,np.zeros(len(penalty))]
                fit=lsq_linear(augmented,rhs,bounds=(0.,np.inf),method='bvls',tol=1e-10,max_iter=300)
                weights=fit.x;prediction=design@weights
                key=f'{name}_grid_{count}_alpha_{alpha:g}'
                arrays[key+'_weights']=weights;arrays[key+'_prediction']=prediction
                rows.append(dict(scenario=name,grid_count=count,alpha=alpha,
                    optimizer_success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),
                    relative_complex_residual=float(np.linalg.norm(observed-prediction)/scale),
                    density_curvature_norm=float(np.linalg.norm(penalty@weights)),
                    integrated_amplitude=float(weights.sum()),physically_identified=False))
                curves.append((grid,weights/h,f'alpha={alpha:g}'+(' (provisional)' if not fit.success else '')))
                plot(directory/(key+'_real.png'),[(op.f,observed.real,'Synthetic observation'),(op.f,prediction.real,'Conditional prediction')],
                     'Frequency (Hz)','Real spectrum','Known frequency/phase; numerical prediction only')
            arrays[f'{name}_grid_{count}_t2star_s']=grid
            fig=Figure(figsize=(10,5),layout='constrained');ax=fig.add_subplot(111)
            for x,y,label in curves:ax.semilogx(x,y,label=label)
            for tau in taus:ax.axvline(tau,color='black',ls='--',alpha=.5)
            ax.set(xlabel='T2* grid (s)',ylabel='Conditional amplitude density per log(T2*)',
                   title=f'{name}: {count} log cells; regularization sensitivity')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.supxlabel('Dashed lines: synthetic truth. Frequency and phase known; no physical distribution inferred.',fontsize=9)
            fig.savefig(directory/f'{name}_grid_{count}_density.png',dpi=150)
            storage.write_json(directory/'candidates.json',rows)
    np.savez_compressed(directory/'regularization_arrays.npz',**arrays)
    result=storage.complete(record,candidates=rows,elapsed_s=time.perf_counter()-started,
        interpretation='Exploratory nonnegative conditional grid with fixed true frequency/phase. No automatic alpha selection, significance, uncertainty interval or experimental physical distribution claim. Grid spacing is included in the density-curvature penalty.')
    storage.write_json(storage.ROOT/'regularization_validation_current.json',{'run_id':result['run_id']})
    print(json.dumps(dict(run_id=result['run_id'],candidates=rows)),flush=True)


if __name__=='__main__':main()
