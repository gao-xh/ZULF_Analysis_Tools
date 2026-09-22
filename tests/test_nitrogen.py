import unittest
import numpy as np
from scipy.linalg import eigh
from zulf_tools.nitrogen import nitrogen_transitions,nitrogen_blocks
from zulf_tools.simulation import build


class NitrogenTests(unittest.TestCase):
    def test_collective_against_full_signed_gamma_response(self):
        p=dict(J_HH_vicinal=6.2,J_N_methyl=2.7,J_N_methine=-1.,J_NH=-65.,J_NH_Hmethine=4.)
        # Three equivalent carbon-bound H, one CH, two NH, one N: full 128 states.
        j=np.zeros((7,7))
        for i in range(3):j[i,3]=j[3,i]=p['J_HH_vicinal'];j[i,6]=j[6,i]=p['J_N_methyl']
        j[3,6]=j[6,3]=p['J_N_methine']
        for i in [4,5]:j[i,6]=j[6,i]=p['J_NH'];j[i,3]=j[3,i]=p['J_NH_Hmethine']
        h,m,_,_=build(dict(isotopes=['1H']*6+['15N'],couplings_hz=j.tolist()))
        e,v=eigh(h/(2*np.pi));obs=v.conj().T@m[2]@v/(2*np.pi)
        ii,kk=np.triu_indices(len(e),1);f=e[kk]-e[ii];w=2*abs(obs[ii,kk])**2/len(e);keep=f>1e-7
        tf,tw=nitrogen_transitions(p,True,3);t=np.linspace(0,.31,41)
        np.testing.assert_allclose(np.cos(2*np.pi*t[:,None]*tf)@tw,np.cos(2*np.pi*t[:,None]*f[keep])@w[keep],atol=1e-8,rtol=1e-9)

    def test_ten_spin_dimension_and_isolated_ax2_frequency(self):
        self.assertEqual(sum(len(o)*mult for _,o,mult in nitrogen_blocks()),1024)
        p=dict(J_HH_vicinal=0.,J_N_methyl=0.,J_N_methine=0.,J_NH=-65.,J_NH_Hmethine=0.)
        f,w=nitrogen_transitions(p)
        np.testing.assert_allclose(f,[97.5],atol=1e-8)
        self.assertGreater(w.sum(),0)


if __name__=='__main__':unittest.main()
