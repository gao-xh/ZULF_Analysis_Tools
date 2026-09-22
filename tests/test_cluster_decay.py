import unittest
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools.cluster_decay import partition_transitions
from zulf_tools.transition_decay import fit_transition_decay, response_projection
from zulf_tools.jfit import ProcessedSpectrum


class ClusterDecayTests(unittest.TestCase):
    def test_response_penalty_exposes_bias_for_cancelling_true_components(self):
        rng=np.random.default_rng(11)
        first=rng.normal(size=(40,2))+1j*rng.normal(size=(40,2))
        second=first+.001*(rng.normal(size=(40,2))+1j*rng.normal(size=(40,2)))
        d=np.column_stack([first,second]);truth=np.array([100.,30.,-100.,-30.]);y=d@truth
        free,*_=response_projection(d,y,0.)
        bounded,*_=response_projection(d,y,.01)
        def energy(c):return sum(np.linalg.norm(d[:,i:i+2]@c[i:i+2])**2 for i in [0,2])
        self.assertLess(energy(bounded),energy(free)/100)
        self.assertGreater(np.linalg.norm(d@bounded-y),np.linalg.norm(d@free-y)*1000)
        # Lower cancelling response is not automatically a physically truer fit.

    def test_response_penalty_matches_independent_block_system(self):
        rng=np.random.default_rng(4)
        d=rng.normal(size=(30,6))+1j*rng.normal(size=(30,6));y=rng.normal(size=30)+1j*rng.normal(size=30)
        alpha=.07
        coeff,_,_,roots=response_projection(d,y,alpha)
        real=np.vstack([d.real,d.imag]);penalty=np.zeros((180,6))
        for i in range(3):penalty[i*60:(i+1)*60,i*2:i*2+2]=real[:,i*2:i*2+2]
        expected=np.linalg.lstsq(np.vstack([real,np.sqrt(alpha)*penalty]),np.r_[y.real,y.imag,np.zeros(180)],rcond=None)[0]
        np.testing.assert_allclose(coeff,expected,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(roots@coeff)**2,alpha*np.linalg.norm(penalty@coeff)**2,atol=1e-12)
        # A phase rotation within each group cannot change the predicted spectrum.
        angle=.8;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotation=np.kron(np.eye(3),rot)
        other,*_=response_projection(d@rotation,y,alpha)
        np.testing.assert_allclose(d@coeff,d@rotation@other,atol=1e-12)

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
