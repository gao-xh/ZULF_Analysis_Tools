import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.linalg import eigh
from zulf_tools import jfit,simulation,storage
from zulf_tools.analysis import recipe,execute


class JFitTests(unittest.TestCase):
    def test_exact_processed_modes_against_sampled_fft(self):
        fs=1000.; n=1800; freq=np.array([124.17,132.86]); w=np.array([.3,.7]); rate=1.6
        t=np.arange(n)/fs
        for first,last,window in [(0,n,101),(20,n-20,101),(100,1500,101),(0,n,0)]:
            bins=np.arange(1,(last-first)//2,11)
            processor=jfit.ProcessedSpectrum(fs,n,first,last,bins,window,2)
            templates=processor.templates(freq,w,rate)
            for k,func in enumerate((np.cos,np.sin)):
                fid=(func(2*np.pi*t[:,None]*freq)@w)*np.exp(-rate*t)
                processed,_,_=recipe(fid,fs,{'start_s':first/fs,'end_s':last/fs,'sg_window':window})
                ref=np.fft.rfft(processed)[bins]/len(processed)
                np.testing.assert_allclose(templates[:,k],ref,rtol=1e-9,atol=1e-12)

    def test_collective_multiplicities_match_full_eight_spin_space(self):
        p=dict(jfit.DEFAULT)
        for kind in ('methine','methyl'):
            model=jfit.model_for(p,kind)
            h,m,_,_=simulation.build(model)
            e,v=eigh(h/(2*np.pi),driver='evr'); o=v.conj().T@m[2]@v/(2*np.pi)
            i,k=np.triu_indices(len(e),1); gaps=e[k]-e[i]; weights=2*abs(o[i,k])**2/256
            mask=gaps>1e-7
            f,w=jfit.transitions(p,kind)
            t=np.linspace(0,.073,17)
            full=np.cos(2*np.pi*t[:,None]*gaps[mask])@weights[mask]
            reduced=np.cos(2*np.pi*t[:,None]*f)@w
            np.testing.assert_allclose(reduced,full,rtol=1e-9,atol=1e-7)

    def test_synthetic_recovery_and_cancel(self):
        fs=1000.; n=2400; truth=dict(jfit.DEFAULT,J_CH_methine=134.2,J_CH_methyl=124.3)
        t=np.arange(n)/fs; values=np.zeros(n)
        for kind,scale in [('methine',1.),('methyl',1.7)]:
            f,w=jfit.transitions(truth,kind); w=w/w.sum()
            values+=scale*(np.cos(2*np.pi*t[:,None]*f)@w)*np.exp(-3*t)
        with tempfile.TemporaryDirectory() as tmp, patch.object(storage,'ROOT',Path(tmp)):
            record,d=storage.begin('compute_average',{})
            np.save(d/'average.npy',values)
            import hashlib
            storage.complete(record,points=n,sampling_rate_hz=fs,average_sha256=hashlib.sha256((d/'average.npy').read_bytes()).hexdigest())
            comparison=execute('compare_preprocessing',{'average_run_id':record['run_id'],'variants':[{'start_s':.05,'sg_window':101}]})
            args={'comparison_run_id':comparison['run_id'],'variant_index':0,'ranges':[[110,150],[230,270]],
                  'settings':{'initial':dict(truth,J_CH_methine=134.5,J_CH_methyl=124.6),
                              'free_parameters':['J_CH_methine','J_CH_methyl'],'starts':1,
                              'screening_samples':0,'max_nfev':35,'initial_rate':3.,'bin_stride':2}}
            r=execute('fit_isopropylamine_j',args)
            for key in args['settings']['free_parameters']: self.assertAlmostEqual(r['parameters_hz'][key],truth[key],places=3)
            self.assertLess(r['relative_complex_residual'],1e-4)
            args['settings']['objective']='magnitude'
            magnitude=execute('fit_isopropylamine_j',args)
            self.assertLess(magnitude['relative_magnitude_residual'],.002)
            for key in args['settings']['free_parameters']:
                self.assertAlmostEqual(magnitude['parameters_hz'][key],truth[key],places=2)
            self.assertEqual(len(magnitude['candidates']),1)
            # Methyl-only anchor must not contain a hidden methine component.
            anchor_args=dict(args,settings=dict(args['settings'],objective='complex',
                 isotopomers=['methyl'],free_parameters=['J_CH_methyl'],fixed_rates={'methyl':3.},max_nfev=8),ranges=[[230,270]])
            anchor=execute('fit_isopropylamine_j',anchor_args)
            self.assertEqual(anchor['decay_rates_per_s'],{'methyl':3.})
            with np.load(storage.artifact(anchor['run_id'],'fit_arrays.npz')) as a:
                np.testing.assert_array_equal(a['methine'],np.zeros_like(a['methine']))
            # Full orchestration, child provenance, fixed methyl parameters, and
            # no automatic claim of scientific validation.
            staged=execute('fit_isopropylamine_staged',{'comparison_run_id':comparison['run_id'],
                'variant_index':0,'low_range':[110,150],'high_range':[230,270],
                'settings':{'initial':truth,'objective':'complex','branches':1,'anchor_starts':1,
                    'anchor_screening':0,'methine_starts':1,'methine_screening':0,'max_nfev':5,'bin_stride':4}})
            self.assertEqual(len(staged['children']),3)
            self.assertFalse(staged['scientifically_validated'])
            middle=storage.get_result(staged['branches'][0]['methine_run_id'])
            for key in ['J_CH_methyl','J_HH_vicinal','J_Cmethyl_Hmethine','J_Cmethyl_Hother_methyl']:
                self.assertEqual(middle['parameters_hz'][key],staged['branches'][0]['anchor_parameters_hz'][key])
            with self.assertRaises(InterruptedError): execute('fit_isopropylamine_j',args,cancel=lambda:True)
            artifact=storage.artifact(comparison['run_id'],'variant_0.npz')
            with artifact.open('ab') as out: out.write(b'changed')
            with self.assertRaises(ValueError): execute('fit_isopropylamine_j',args)


if __name__=='__main__': unittest.main()
