"""Finite Monte Carlo check, not calibration of experimental confidence intervals.

Run: python -m examples.validate_resampling_coverage
Paired synthetic datasets contrast independent and correlated group noise.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from zulf_tools import storage
from zulf_tools.jfit import ProcessedSpectrum
from zulf_tools.decay import fit_modes
from zulf_tools.decay_resampling import circular_group_draws, conditional_quantiles


def main():
    parameters=dict(trials=40,draws=60,groups=8,points=2048,fs_hz=256.,
                    first=16,true_frequency_hz=40.25,true_t2star_s=.8,
                    phase_rad=.4,group_noise_sd=.2,correlations=[0.,.8],seed=713911,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record,directory=storage.begin('synthetic_resampling_coverage',parameters)
    storage.write_json(storage.ROOT/'resampling_coverage_current.json',{'run_id':record['run_id']})
    began=time.perf_counter();rng=np.random.default_rng(parameters['seed'])
    n=parameters['points'];fs=parameters['fs_hz'];first=parameters['first']
    t=np.arange(n)/fs
    clean=np.exp(-t/parameters['true_t2star_s'])*np.cos(2*np.pi*parameters['true_frequency_hz']*t+parameters['phase_rad'])
    freq=np.fft.rfftfreq(n-first,1/fs);bins=np.flatnonzero((freq>=35)&(freq<=45))
    op=ProcessedSpectrum(fs,n,first,n,bins)
    summaries=[]
    def fit(observed,initial=None):
        result=fit_modes(op,observed,[35,45],[.1,3.],initial_frequencies=initial,
                         starts=1,max_nfev=100,max_evaluations=500,max_seconds=2.,seed=17)
        result.pop('fitted')
        return result
    try:
        for trial in range(parameters['trials']):
            shared=rng.normal(size=n)
            independent=rng.normal(size=(parameters['groups'],n))
            plan=circular_group_draws(parameters['groups'],parameters['draws'],1,parameters['seed']+trial)
            for rho in parameters['correlations']:
                means=clean+parameters['group_noise_sd']*(np.sqrt(rho)*shared+np.sqrt(1-rho)*independent)
                spectra=np.fft.rfft(means[:,first:],axis=1)[:,bins]/(n-first)
                parent=fit(spectra.mean(axis=0));rows=[]
                for i,selection in enumerate(plan):
                    result=fit(spectra[selection].mean(axis=0),parent['frequencies_hz'])
                    shift=abs(result['frequencies_hz'][0]-parent['frequencies_hz'][0])
                    result.update(draw_index=i,group_indices=selection.tolist(),
                                  matched_t2star_s=result['t2star_s'],
                                  eligible_for_summary=not result['numerical_diagnostics']['requires_review'] and shift<=2*fs/(n-first))
                    rows.append(result)
                summary=conditional_quantiles(rows,parameters['draws'])
                quantiles=summary['t2star_percentiles_s']
                interval=[quantiles[0][0],quantiles[2][0]] if quantiles else None
                entry=dict(trial=trial,group_noise_correlation=rho,parent=parent,
                           summary=summary,interval_s=interval,
                           covers_truth=interval[0]<=parameters['true_t2star_s']<=interval[1] if interval else None)
                summaries.append(entry)
                name=f'trial_{trial:03d}_rho_{rho:g}'
                storage.write_json(directory/(name+'.json'),dict(entry,draw_fits=rows))
                np.savez_compressed(directory/(name+'.npz'),group_means=means,draw_plan=plan)
            storage.write_json(directory/'trial_summaries.json',summaries)
            print(json.dumps(dict(run_id=record['run_id'],completed_trials=trial+1)),flush=True)
        aggregate=[]
        for rho in parameters['correlations']:
            selected=[r for r in summaries if r['group_noise_correlation']==rho]
            available=[r for r in selected if r['interval_s'] is not None]
            covered=sum(r['covers_truth'] for r in available)
            aggregate.append(dict(group_noise_correlation=rho,requested_trials=len(selected),
                available_intervals=len(available),covered_trials=covered,
                coverage_among_available=covered/len(available) if available else None,
                covered_fraction_all_trials=covered/len(selected),
                median_interval_width_s=float(np.median([r['interval_s'][1]-r['interval_s'][0] for r in available])) if available else None))
            fig=Figure(figsize=(10,5),layout='constrained');ax=fig.add_subplot(111)
            ax.axhline(parameters['true_t2star_s'],color='black',ls='--',label='Known synthetic T2*')
            for row in available:
                color='#246ba6' if row['covers_truth'] else '#b33a3a'
                ax.vlines(row['trial'],*row['interval_s'],color=color)
                ax.scatter(row['trial'],row['parent']['t2star_s'][0],color=color,s=14)
            ax.set(xlabel='Independent synthetic dataset index',ylabel='T2* (s)',
                   title=f'Conditional percentile spans; group-noise correlation {rho:g}')
            ax.legend();fig.supxlabel('Red: truth outside span. Fixed correct single-mode model; not experimental confidence calibration.',fontsize=9)
            fig.savefig(directory/f'coverage_rho_{rho:g}.png',dpi=150)
        result=storage.complete(record,aggregate=aggregate,elapsed_s=time.perf_counter()-began,
            interpretation='Finite Monte Carlo at one correct-model operating point. Paired correlation scenarios share random draws. Independent-group bootstrap cannot recover unobserved common group noise; no actual experimental coverage guarantee or inferred correlation.')
        print(json.dumps(dict(run_id=result['run_id'],aggregate=aggregate)),flush=True)
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_s=time.perf_counter()-began)
        storage.write_json(directory/'result.json',record)
        raise


if __name__=='__main__':main()
