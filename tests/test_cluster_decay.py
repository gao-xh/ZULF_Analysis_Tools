import unittest
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.cluster_decay import partition_transitions
from zulf_tools.transition_decay import fit_transition_decay
from zulf_tools.jfit import ProcessedSpectrum


class ClusterDecayTests(unittest.TestCase):
    def test_partition_retains_every_transition_and_boundary(self):
        f=np.array([1.,30.,35.,40.,100.]);w=np.arange(1.,6.)
        g=partition_transitions(f,w,[30.,40.],'test')
        self.assertEqual([x['transition_indices'] for x in g],[[0],[1,2],[3,4]])
        np.testing.assert_array_equal(np.concatenate([x['frequencies_hz'] for x in g]),f)
        self.assertAlmostEqual(sum(x['fraction_of_family_weight'] for x in g),1.)
        for cuts in [[40,30],[30,30],[float('nan')]]:
            with self.assertRaises(ValueError):partition_transitions(f,w,cuts,'test')

    def test_six_clusters_recover_independently_sampled_fid(self):
        fs=256.;points=1024;t=np.arange(points)/fs;y=np.zeros(points);groups=[]
        taus=np.array([.2,.3,.45,.6,.8,1.1]);phases=np.array([.1,-.2,.3,-.4,.5,-.6])
        for i,(tau,phase) in enumerate(zip(taus,phases)):
            f=np.array([25+8*i,25.6+8*i]);w=np.array([.7,.3]);groups.append(dict(frequencies_hz=f.tolist(),weights=w.tolist()))
            for ff,ww in zip(f,w):y+=(1+i*.1)*ww*np.exp(-t/tau)*np.cos(2*np.pi*ff*t+phase)
        y-=savgol_filter(y,31,2,mode='mirror');a,b=20,1000
        freq=np.fft.rfftfreq(b-a,1/fs);bins=np.flatnonzero((freq>20)&(freq<75))
        op=ProcessedSpectrum(fs,points,a,b,bins,31,2)
        fit=fit_transition_decay(op,np.fft.rfft(y[a:b])[bins]/(b-a),groups,[.1,2.],starts=2,max_seconds=30)
        np.testing.assert_allclose(fit['t2star_s'],taus,atol=1e-5)
        np.testing.assert_allclose(fit['phases_rad'],phases,atol=1e-5)
        self.assertTrue(fit['optimizer_converged'])


if __name__=='__main__':unittest.main()
