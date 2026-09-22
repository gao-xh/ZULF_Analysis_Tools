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
