"""Matched complex Hann observations for bounded oscillatory decay fitting."""
import numpy as np
from scipy.signal import get_window, savgol_filter
from scipy.sparse import csr_matrix
from .decay import fit_modes


class WindowedDecayOperator:
    """Cache a sparse real-FID-to-complex-window transform, including phase.

    SG acts on the complete record before cropping. Mean subtraction acts on
    the retained record and is applied to model and data alike. Complete Hann
    windows only; observations are correlated and carry no independent-bin DOF.
    """
    def __init__(self, fs, full_points, preprocessing, frequencies, width_s, hop_s):
        from .analysis import recipe
        if not np.isfinite(fs) or fs<=0 or type(full_points) is not int or not 32<=full_points<=2_000_000:
            raise ValueError('Invalid sample rate or full record size.')
        self.fs=fs;self.full_points=full_points
        _,_,self.parameters=recipe(np.zeros(full_points),fs,preprocessing or {})
        self.first=self.parameters['start_sample'];self.last=self.parameters['stop_sample']
        self.n=self.last-self.first
        self.targets=np.asarray(frequencies,dtype=float)
        if self.targets.ndim!=1 or not 1<=len(self.targets)<=32 or not np.isfinite(self.targets).all() or np.any(self.targets<=0) or np.any(self.targets>=fs/2) or np.any(np.diff(self.targets)<=0):
            raise ValueError('Supply 1..32 sorted distinct positive frequencies inside Nyquist.')
        if not np.isfinite([width_s,hop_s]).all() or min(width_s,hop_s)<=0:
            raise ValueError('Window width and hop must be positive.')
        self.width=int(round(width_s*fs));self.hop=int(round(hop_s*fs))
        if not 8<=self.width<=self.n or self.hop<1:
            raise ValueError('Window must fit the retained record and hop must be at least one sample.')
        starts=np.arange(0,self.n-self.width+1,self.hop)
        rows=len(starts)*len(self.targets)
        if rows*self.width>2_000_000:
            raise ValueError('Sparse window transform exceeds two million coefficients.')
        self.shape=(len(starts),len(self.targets))
        self.times_s=(self.first+starts+self.width/2)/fs
        self.f=np.tile(self.targets,len(starts))
        indices=np.repeat(starts,len(self.targets))[:,None]+np.arange(self.width)[None,:]
        hann=get_window('hann',self.width,fftbins=True)
        values=2*np.exp(-2j*np.pi*self.f[:,None]*(indices+self.first)/fs)*hann/hann.sum()
        self.matrix=csr_matrix((values.ravel(),indices.ravel(),np.arange(rows+1)*self.width),shape=(rows,self.n))
        self.time=np.arange(full_points)/fs

    def transform(self, values):
        values=np.asarray(values)
        if values.ndim not in (1,2) or values.shape[0]!=self.full_points or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError('Supply finite real full-record FID columns.')
        p=self.parameters
        if p['sg_window']:
            values=values-savgol_filter(values,p['sg_window'],p['sg_order'],axis=0,mode='mirror')
        retained=values[self.first:self.last].copy()
        if p.get('remove_mean',True):
            retained-=retained.mean(axis=0)
        return self.matrix@retained

    def templates(self, frequencies, weights, rate):
        frequencies=np.asarray(frequencies,dtype=float);weights=np.asarray(weights,dtype=float)
        if frequencies.ndim!=1 or frequencies.shape!=weights.shape or not len(frequencies) or not np.isfinite(frequencies).all() or not np.isfinite(weights).all() or weights.sum()<=0 or not np.isfinite(rate) or rate<=0:
            raise ValueError('Invalid oscillatory template.')
        # Accumulate weighted real quadratures without a full time-by-transition array.
        raw=np.zeros((self.full_points,2))
        decay=np.exp(-rate*self.time)
        for f,w in zip(frequencies,weights/weights.sum()):
            angle=2*np.pi*f*self.time
            raw[:,0]+=w*decay*np.cos(angle)
            raw[:,1]+=w*decay*np.sin(angle)
        return self.transform(raw)


def fit_windowed_modes(operator, full_fid, frequency_bounds, t2_bounds, initial_frequencies,
                       shared_decay=False, **budgets):
    """Fit correlated complex windows; no independent-window confidence claim."""
    if np.asarray(full_fid).ndim!=1:
        raise ValueError('Fit one real FID at a time.')
    fit=fit_modes(operator,operator.transform(full_fid),frequency_bounds,t2_bounds,
                  mode_count=len(initial_frequencies),initial_frequencies=initial_frequencies,
                  shared_decay=shared_decay,background=False,**budgets)
    # FFT-bin spacing is not the resolution of these window observations.
    diagnostics=fit['numerical_diagnostics']
    diagnostics.pop('sub_bin_frequency_pairs',None)
    diagnostics['thresholds'].pop('native_fft_spacing_hz',None)
    fit.update(observation_domain='Complex complete Hann windows; time-major flattened observations.',
               window_width_s=operator.width/operator.fs,hop_s=operator.hop/operator.fs,
               nominal_window_resolution_hz=operator.fs/operator.width,
               interpretation='Effective oscillatory decay; overlapping windows are correlated. No independent-window confidence interval or physical component assignment.')
    return fit
