# Method design notes and references

## Existing code reviewed

The legacy `ZULF_Relaxation/zulf_relaxation/fitting.py` uses variable projection:
linear amplitudes/phases/offsets conditional on nonlinear frequencies and decay
times. Its individual-scan formulation can be expensive and requires frequency
seeds. The new workflow will use explicit group means and band-only observations;
it must preserve raw-time phase conventions and must not infer multiple decays
from a beat envelope alone.

The user's `signal_selection/src/ui_main.py` computes Hann-window STFT with
90% overlap, then takes magnitude. `analyze_stft_t2` follows one nearest
frequency bin from its maximum onward and fits `A exp(-t/T2*) + C`.
`processing.py::fit_exponential_decay` also offers a baseline-subtracted
log-linear route. This is useful as an exploratory comparator. Its overlapping
windows are correlated; magnitude has noise bias, a single bin can mix nearby
frequencies, and choosing the maximum as a fit start is data-dependent. The new
implementation must compare the explicitly windowed forward prediction and
multiple widths/hops rather than assume every magnitude slice is an exponential.

`zulf_tools/jfit.py::ProcessedSpectrum` provides exact finite-record Fourier
templates with full-record SG baseline subtraction and mirror-edge corrections.
The decay core reuses this forward operator. Independent real time-domain tests
check the result instead of duplicating the analytic implementation.

## DOSY and Laplace inversion references

- Cherni, Chouzenoux and Delsuc, *PALMA, an improved algorithm for DOSY signal
  processing*, revised 2016, [arXiv:1608.07055](https://arxiv.org/abs/1608.07055),
  [DOI:10.1039/C6AN01902A](https://doi.org/10.1039/C6AN01902A).
  The abstract describes a proximity-operator splitting scheme with combined
  maximum-entropy and L1 regularization. It supports examining regularization
  sensitivity for distribution reconstructions, not interpreting every recovered
  distribution peak as a physical component. Full algorithm review is still
  required before claiming a PALMA implementation.
- Song et al., *T(1)--T(2) correlation spectra obtained using a fast
  two-dimensional Laplace inversion*, JMR 154 (2002), 261-268,
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/11846583/),
  [DOI:10.1006/jmre.2001.2474](https://doi.org/10.1006/jmre.2001.2474).
  This is a reference for noise-aware regularized inversion and two-dimensional
  relaxation visualization; it does not imply the current FID experiment measures
  a T1-T2 correlation spectrum.

## Scope of analogy

A frequency versus effective T2* display can borrow DOSY's presentation ideas.
The present acquisition variable is FID time, not a diffusion-gradient b value.
Its kernel is oscillatory and complex; applying a positive Laplace inversion
directly to a multi-frequency magnitude envelope can misinterpret beating.
Bounded parametric complex fits and independent-repeat checks are the first
implementation; any continuous-distribution extension remains exploratory.

## Numerical core conventions

Each mode is `A exp(-t/T2*) cos(2 pi f t + phase)`, with `t` measured from
acquisition start. Cosine/sine coefficients are real and unrestricted; phase is
`atan2(-sine, cosine)`. A spectral band does not eliminate finite-record leakage
from modes outside it. Sensitivity to band edges and explicit nuisance models
must therefore be checked before experimental interpretation.

The native-bin complex residual is minimized; no confidence interval is inferred
from an independent-bin assumption after filtering. Function budgets count
finite-difference calls. Budget interruption retains a provisional best fit and
is not itself convergence. Shared-decay two-frequency fits explicitly represent
beating without introducing an additional decay component.

## Independent-repeat noise diagnostics

For group means z_g with scan counts n_g, use the count-weighted mean m and
estimate single-scan complex variance as sum(n_g |z_g-m|^2)/(G-1). Its pooled
mean standard error is the square root of this variance divided by total scan
count. This assumes independent equal-variance raw scans; drift inflates this
empirical scatter and is not silently removed. These diagnostics do not infer
molecular origin or calibrated multiple-testing significance.

For accumulation checks, form disjoint pools A/B at each size and measure their
spectral differences in explicitly chosen reference bins. Multiply difference
RMS by sqrt(N_A N_B)/(N_A+N_B) to estimate the noise of their pooled mean.
The noise-vs-count slope is measured rather than imposed. Repeated permutations
and sizes share observations, so slope statistics are descriptive. Coherent
interference can survive averaging and still obey the expected SNR trend.
# Numerical screening of decay candidates

Each bounded mode fit now includes `numerical_diagnostics`. This reports search
boundaries, rank deficiency or severe conditioning of the column-normalized
real amplitude design, incomplete optimization, and exhausted search budgets.
Boundary indices refer to the frequency-sorted returned modes; a shared
`log_t2_0` applies to every mode. Boundary proximity uses 1% of the frequency
interval or logarithmic decay interval. A normalized design condition number
above 1e8 triggers review. These are explicit numerical heuristics, not
confidence limits or a statistical identifiability test.

Sub-bin frequency pairs are also recorded, without automatically rejecting
them: finite FFT spacing is not a universal limit on parametric estimation.
Conversely, passing the screen cannot establish a signal, a relaxation
component, or a physical mechanism. Repeat validation, preprocessing stability,
and uncertainty checks remain necessary. A synthetic exact-duplicate example
verifies that an essentially perfect prediction can still have rank-deficient
amplitudes. Reversed optimizer mode order verifies sorted boundary indexing.
## Matched complex-window fitting core

`window_decay.WindowedDecayOperator` caches a sparse complete-Hann transform
with acquisition-referenced phase. Real cosine/sine templates undergo the same
full-record mirror SG subtraction, time crop, and retained-mean subtraction as
the experimental FID. The latter matters for off-grid short-window samples,
even though a constant vanishes at positive native full-record FFT bins.

`fit_windowed_modes` uses bounded variable projection on complex observations,
starting from explicit frequencies. It retains negative-frequency contributions
of real FIDs and window averaging of rapidly decaying signals. The optimizer's
nominal sample count is not a statistical count of independent observations:
overlapping windows and nearby Fourier samples are correlated. No confidence
interval is computed from that count. Finite window width can admit signals
outside a selected nominal band; matched processing alone cannot fix an
incomplete oscillatory model.

Four independent tests cover direct Hann sums including mirror SG edges,
0.12-second decay/phase recovery through a 0.25-second window, same-decay
two-frequency beating, and invalid/resource-limited requests. Sparse storage
is capped at two million coefficients. `fit_window_decay` exposes this core as
a CLI/MCP background operation. It preserves the parent discovery/validation
split and checks each window-domain candidate in the original FFT band as well.
Phase-reversed held-out data verify that validation is not silently refitted.
Method stability and physical acceptance require further experimental checks.
