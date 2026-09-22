# Frequency-range relaxation goal and evidence ledger

Primary objective: reproducible, range-restricted effective FID decay analysis,
with explicit T2* bounds and independent acquisition-group validation. J fitting
is auxiliary, not a prerequisite. The full user goal remains active until the
implementation and experimental validation below have been inspected.

## Requirements and planned evidence

- [x] Streaming, disjoint acquisition-group averages with source hashes, group
  membership and recorded time preserved; verify against independent fixtures.
- [ ] One/multiple frequency bands, T2* bounds, component counts, crop and work
  budgets through JSON CLI and typed MCP tools; rejection tests and transport test.
- [ ] Complex demodulation with anti-aliasing, sliding-window diagnostics and
  bounded damped-oscillator fits; compare with independently constructed signals.
  Explicitly review the user's legacy short-time FT decay implementation and
  compare window width, hop, overlap and matched-window model predictions.
- [ ] Single vs multiple components using held-out groups, not training error;
  export parameter stability, boundary hits and identifiability diagnostics.
- [ ] Signal discovery separated from validation, local noise and accumulation
  scaling, cautious interference/noise/insufficient-evidence classification.
- [ ] Full/early/tail FID figures, candidate crop boundaries and sensitivity;
  preserve fast/slow components and acquisition-relative model time.
- [ ] Raw vs SG baseline subtraction; matched experimental/model processing;
  sensitivity to frequency-band edges, filter and sliding-window parameters.
- [ ] Complex/real/imaginary/magnitude/residual and trustworthy-phase plots;
  assess group drift and cancellation before applying any alignment.
- [ ] Multiple candidate J models as optional constraints, report effect on T2*;
  do not silently validate current broad-envelope methyl fit.
- [ ] Synthetic validation of noise, close frequencies, beating, multiple decays,
  drift, crop/filter effects, bound hits and underidentified cases.
- [ ] Actual isopropylamine band analysis and disjoint-group prediction checks,
  using the existing read-only raw dataset, with inspected independent figures.
- [ ] Reusable tools, examples and English documentation; local reproducible
  reports and compute costs; before/after GitHub code backups, no experiment uploads.

## Constraints

- Raw experiments are read-only; output arrays/reports stay in `.analysis`.
- No default individual-scan fits or full-spectrum fitting. Begin with coherent
  averages; retain an unaligned reference. Acquisition numbering is not proven
  physical batch metadata and must never be presented as such.
- Report effective T2*, not unmeasured T2/T1 or automatically identified species.
- A narrower peak, more components, smaller training residual or solver success
  does not establish a physical model. Negative results are valid deliverables.
- Unknown excitation and nitrogen/exchange effects remain explicit assumptions.
- Routine choices are autonomous; keep progress and failed routes on disk.

## Starting evidence

- Before-change remote branch: `backup/before-band-relaxation-20260922`.
- Existing total average: `100548557cf94d6cb6dc2bb02bca885a` (5677 acquisitions).
- Existing preprocessing: `f9c8dc86f77048e4844756b9c45dbcce`, SG baseline 301/2,
  100 ms crop. This is a comparison baseline, not a mandated final recipe.
- Existing joint J candidate: `71c86d71a33445b9942fc6ac46bbbdf7`.
- Methyl-only trials `8acc557dc48b47bcab4e43ec138c8956` and
  `e97f7c635de04d1398dcedfd807ee5bc` do not establish resolved methyl J values.

## Progress

The expanded user objective requires at least seven hours of actual autonomous
iteration, including literature and legacy-method review (DOSY-inspired inverse
methods and visualization where appropriate). Start: 2026-09-22 09:27:40 UTC;
earliest completion: 2026-09-22 16:27:40 UTC. Elapsed time alone is insufficient:
all deliverables and evidence gates above still apply. DOSY analogy does not
turn FID decay into a measured diffusion coefficient.

Implementation begins with provenance-preserving group averages and FID diagnostics.
Subsequent completion claims require linked numerical tests and experimental runs.

Milestone 1: `tests/test_tools.py` passes 10 tests including group-weighted
pooling, overlapping-group rejection, raw-file preservation, tamper detection and
real MCP STDIO discovery. Actual group run `c2d483bf4de84a969ca65317cf6de044`
contains 16 disjoint groups of 354/355 acquisitions (5677 total), completed in
22.6 seconds. Recombined pooled mean differs from the previous full mean by at
most 1.42e-14 ADC units (floating-point summation). Prespecified alternating
group indices separate discovery and validation; split stored locally in
`.analysis/relaxation_goal_progress.json`. Early FID plot shows a very large
transient/baseline relative to the spectral signal; explicit crop/filter
sensitivity is essential before decay interpretation.

Milestone 2 (numerical core, not yet the complete user-facing operation):
`zulf_tools/decay.py` evaluates only the requested native complex FFT bins,
using exact finite-record real oscillators with matched SG/crop response from
`ProcessedSpectrum`. It provides bounded frequencies/T2*, shared or independent
decays, variable-projection amplitude/phase, multistart search, actual objective
evaluation/time budgets and explicit boundary/convergence diagnostics.
`tests/test_decay.py` passes five independent time-signal tests: SG mirror-edge
response, phase/single-decay recovery, equal-decay beating, noisy two-decay
recovery, and budget/bound reporting. API integration, discovery/validation,
demodulation, sliding windows, uncertainty and real-data evaluation remain open.

Milestone 3: `fit_frequency_decay` is registered in CLI and typed MCP, with
independent magnitude/real/imaginary/residual plots. It enforces disjoint groups
and reports fully frozen validation predictions separately from conditional
gain diagnostics. Eleven tool/transport tests pass, including validation-only
phase inversion that leaves discovery parameters unchanged but worsens frozen
prediction error. Pilot experimental evaluation completed (local progress file
records its run ID); this is not yet a validated relaxation conclusion. Legacy
short-time FT magnitude-slice fitting was reviewed in `RELAXATION_METHODS.md`.

Milestone 4: matched-window Fourier and demodulation primitives pass four
independent tests (closed-form damped complex exponential, unpadded windows,
out-of-band rejection/phase preservation, decay slope after anti-alias filtering).
The `inspect_decay_time_frequency` MCP/CLI tool transforms frozen predictions
and observations identically; the 11 tool/transport tests cover its short-record
no-interior state as well. Both pilot frequency bands were inspected with
0.25/0.5/1-second windows and 90% overlap. Local pilot run IDs are stored in
`.analysis/time_frequency_pilot_runs.json`. Window dependence and beating are
visible; STFT fitting and statistically justified decay acceptance remain open.

Milestone 5: addressed the observed Windows atomic JSON replacement failure
with bounded retries limited to permission/sharing failures. Tests verify the
old file remains intact until success, permanent failure is surfaced after a
bounded retry, and no invalid JSON or temporary files are left behind.
Repeat-statistics primitives pass three tests: unequal group sizes recover
known single-scan noise variance, measured disjoint-pool noise decreases without
imposing sqrt(N), and phase cancellation/known-interference labels remain
separate from molecular assignment. The frequency-decay scatter calculation now
uses the exact count-weighted variance estimate. Classification tool integration
and experimental accumulation evaluation remain open.

Milestone 6: `inspect_repeat_signals` is exposed through JSON CLI and typed MCP.
Twelve tool/transport tests pass including an end-to-end discovery-only peak
proposal. Pilot experimental run recorded locally in the progress file checks
two bands with explicit reference-noise intervals and a prespecified split.
Independent phase and accumulation figures were inspected. These are operational
reproducibility labels, not calibrated significance or molecular assignment;
alternate reference bands, drift diagnosis and downstream acceptance remain open.
Milestone 7: six local crop/SG sensitivity cases completed with 24 candidate
models. Reports and independently inspected figures remain in local analysis
storage. The next method checks must separate optimizer branch changes from
preprocessing sensitivity; no new physical decay assignment was accepted.

Milestone 8: decay candidates expose an explicit numerical review screen for
boundary solutions, deficient/ill-conditioned amplitude designs, incomplete
optimization and exhausted budgets. Sorted output indices now identify the
correct boundary parameters. Seven decay tests pass, including exact prediction
with duplicate-mode rank deficiency and reversed optimizer mode ordering.
This screen does not replace nonlinear uncertainty or physical validation.
Milestone 9: the band tool accepts explicit per-band frequency initializations
for one mode count, preserving bounded multistart optimization. Twelve transport
and integration tests pass, including invalid/out-of-band/nonfinite seeds and
frozen validation behavior. Common discovery-derived seeds are being evaluated
across preprocessing variants; this does not assert a globally optimal fit.
Milestone 10: `inspect_decay_stability` is available through JSON CLI and MCP.
It refits explicit group means with discovery-derived frequency starts, retaining
frozen predictions separately from conditional parameter estimates. Twelve
integration/transport tests pass, including phase-reversed validation data: a
good group refit must not erase the bad frozen prediction. Group spread is
descriptive and does not establish confidence intervals or stable mode identity.
Milestone 11: matched complex-window fitting core passes four new tests and
eleven existing decay/time-frequency tests. It caches a bounded sparse Hann
operator, preserves real-FID quadratures and acquisition phase, and applies SG,
crop and mean removal identically. A real-record-size synthetic benchmark is
saved locally. Tool integration and actual-data fitting remain unfinished;
overlapping windows must not be counted as independent uncertainty samples.
Milestone 12: `fit_window_decay` is exposed through CLI and typed MCP. Twelve
integration/transport tests pass, including independently phase-reversed
validation that must retain a large frozen window prediction error. Every run
saves complex window arrays, independent figures and native FFT cross-checks.
Exploratory low/high-band runs use finite per-fit budgets; numerical completion
does not by itself establish a valid relaxation model.
Milestone 13: fixed transition-group decay core passes four independent numerical
tests. The model shares one gain/phase pair across each supplied weighted group,
fits bounded shared/separate T2*, preserves all transition support and exposes
budget/boundary/conditioning diagnostics. Experimental fixed-J tool integration
remains open, including conditional-validation provenance for historical J fits.
Milestone 14: fixed-J decay tool is available through CLI/MCP with source-model
parameter verification, all-transition sticks and conditional held-out errors.
Twelve integration/transport tests pass, including an explicit controlled
transition fixture for phase-reversed validation and mismatched J provenance.
The separate spin-physics and decay-core tests support the numerical components;
historical all-scan J fits must not be called untouched validation priors.
Milestone 15: optional bounded global phase delay is implemented in native FFT,
window templates and fixed-transition fitting. Zero-delay behavior remains the
default. Six transition-core tests include independent delay recovery and a
single-frequency nonidentifiability counterexample. Three historical J models
are being compared with the same exploratory delay bounds and preserved baselines.
