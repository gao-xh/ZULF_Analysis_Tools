"""Read-only compact result summaries; does not fit, import backend, or alter data."""
import argparse,json,re
from pathlib import Path

FIELDS=('run_id','operation','status','created_utc','parent_run_id','group_run_id',
        'comparison_run_id','carbon_run_id','nitrogen_run_id','preprocessing',
        'ranges_hz','parameters_hz','couplings_hz','decay_rates_per_s',
        'validation_relative_complex_residual','scientifically_validated','warnings')
FIT_FIELDS=('optimizer_converged','budget_exhausted','relative_complex_residual',
            'validation_complex_error','validation_band_errors','training_complex_error',
            'linear_rank','linear_condition_number','boundary_group_indices',
            'numerical_warning_flags','beta','message')
def brief(r):
    out={k:r[k] for k in FIELDS if k in r}
    out['manifest_keys']=list(r)
    if 'fit' in r:out['fit']={k:r['fit'][k] for k in FIT_FIELDS if k in r['fit']}
    if 'bands' in r:
        out['bands']=[{k:v for k,v in b.items() if k=='range_hz' or (('error' in k or 'residual' in k) and not isinstance(v,(dict,list)))} for b in r['bands']]
    if 'main_fits' in r:out['main_fits']=[{k:f[k] for k in FIT_FIELDS if k in f} for f in r['main_fits']]
    if 'group_refits' in r:out['group_refits']={'count':len(r['group_refits']),'converged':sum(bool(g.get('optimizer_converged')) for g in r['group_refits'])}
    for key in ('clusters','transitions','candidates','artifacts'):
        if key in r:out[key+'_count']=len(r[key])
    return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',required=True,type=Path);p.add_argument('run_ids',nargs='+')
    args=p.parse_args();result=[]
    for rid in args.run_ids:
        if not re.fullmatch('[0-9a-f]{32}',rid):p.error('Run IDs must be 32 lowercase hexadecimal characters')
        path=args.project.resolve()/'.analysis'/'runs'/rid/'result.json'
        try:
            r=json.loads(path.read_text(encoding='utf8'));row=brief(r);row['manifest_path']=str(path)
        except (OSError,ValueError,TypeError) as e:row={'run_id':rid,'error':str(e)}
        result.append(row)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(any('error' in r for r in result))
if __name__=='__main__':raise SystemExit(main())
