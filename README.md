# ZULF Analysis Tools

A Codex-centered local analysis project. Tell Codex the scientific question;
Codex calls typed analysis tools, inspects independent figures, and discusses
evidence with you. Python performs the computation. No OpenAI API key or separate
model API is required by this backend.

## Version 0.1 workflow

`inspect_dataset → compute_average → compare_preprocessing → inspect_frequency_ranges`

The first release establishes data fidelity and reproducible processing before
automated decay interpretation. It is independent of the previous desktop apps.
Tools return JSON manifests, opaque run IDs and absolute local artifact paths.
Every graph is its own PNG, and numerical arrays are stored alongside it.

| Tool | Behavior |
| --- | --- |
| `inspect_dataset` | Check numbered DAT/INI files, sample rate, lengths, sample statistics and compiled-reference metadata. |
| `compute_average` | Stream selected raw FIDs into a coherent mean; hash inputs and compare any compiled reference. |
| `compute_group_averages` | Stream 2..32 explicit disjoint scan groups into means, preserving scan membership and hashes for independent validation. |
| `fit_frequency_decay` | Fit bounded damped modes on selected complex FFT bands; predict disjoint validation groups with frozen parameters. |
| `inspect_decay_time_frequency` | Compare a frozen decay candidate through matched Hann windows and anti-aliased complex demodulation. |
| `inspect_repeat_signals` | Propose peaks on discovery groups, check validation reproducibility, measured accumulation and masked phase. |
| `inspect_decay_stability` | Refit bounded group means from discovery initializations; separate frozen prediction errors from diagnostic refits. |
| `fit_demodulated_decay` | Fit bounded oscillatory modes through matched complex FIR demodulation, with explicit edge policy and frozen validation. |
| `fit_window_decay` | Refit complex Hann observations with matched processing, frozen validation and native FFT cross-checks. |
| `fit_simulated_decay` | Fit bounded decays of complete fixed-J transition groups with explicit model provenance and conditional validation. |
| `resample_decay_groups` | Resample discovery-group means with saved circular-block draws, bounded refits and conditional percentile diagnostics. |
| `inspect_fid_crops` | Propose discovery-only time crops from baseline and complete-window band diagnostics; retain full-record comparison and inspect held-out tails. |
| `review_decay_evidence` | Link a decay candidate to compatible signal, group, resampling and sensitivity evidence; expose missing or conflicting evidence without physical acceptance. |
| `compare_preprocessing` | Compare explicit time crops and SG baseline subtraction recipes on an existing average. |
| `inspect_frequency_ranges` | Plot each recipe in selected bands, with local vertical scaling, and rank local maxima. |
| `start_analysis` | Start any operation as a persistent background job. |
| `get_job` | Query progress, failure/cancellation, or completed result. |
| `cancel_job` | Request cancellation at the next file/recipe checkpoint. |
| `get_result` | Retrieve a completed analysis manifest without recomputation. |

MCP averaging, preprocessing and frequency inspection return a job ID immediately.
Direct MCP inspection is synchronous; use `start_analysis` for large inventories.
The CLI provides both direct execution and the same background-job API. Job
requests/status/logs survive client reconnection. A machine restart or externally
killed worker can leave its last status as running; inspect `worker.log` and the
recorded PID rather than treating that as ongoing progress. Automatic recovery
and durable job resumption are not implemented yet.

## Installation and Codex connection

Use Python 3.10+ in a project virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
```

Copy `local.example.json` to `local.json` and set only the experimental input
directories to be analyzed. Those directories are read-only to the tool code.
All outputs go to `.analysis/`, excluded from Git. There is no arbitrary shell
execution or source-file write operation among the exposed tools.

Register the local STDIO server using absolute paths:

```powershell
codex mcp add zulf-analysis-tools -- C:/path/to/project/.venv/Scripts/python.exe C:/path/to/project/run_server.py
```

The adapter uses the official MCP Python SDK v1 maintenance line (`mcp<2`),
tested with 1.30.0. The currently configured environment reuses the existing
scientific packages through a local virtual environment; SDK dependencies are
installed only into that environment. Other machines should use a normal clean
environment as shown above.

Follow the [official Codex MCP configuration documentation](https://developers.openai.com/codex/mcp)
to manage STDIO servers. A newly registered server may require a new session or
client reload before its tools appear; registration alone does not prove the
current conversation's active tool catalog has refreshed. The CLI remains usable:

```powershell
.venv/Scripts/python.exe -m zulf_tools.cli --request examples/inspect.json
```

CLI request shape is `{"tool": "get_job", "arguments": {"job_id": "..."}}`.
Omit `--request` to send JSON through stdin. Successful calls emit JSON to stdout;
errors return a JSON error and a nonzero exit code. Run IDs refer to completed
artifacts; job IDs refer to asynchronous execution. Do not interchange them.

## Suggested first conversation

Ask Codex to inspect a folder, average the chosen scan IDs, compare raw versus
explicit baseline/crop recipes, and show the 115–140 Hz band. Review the original
mean and early-time transient before selecting a processing recipe.

Example preprocessing recipes:

```json
[
  {"label": "Raw mean", "start_s": 0, "sg_window": 0},
  {"label": "SG baseline 301", "start_s": 0, "sg_window": 301, "sg_order": 2},
  {"label": "Crop 50 ms", "start_s": 0.05, "sg_window": 0},
  {"label": "SG then crop 50 ms", "start_s": 0.05, "sg_window": 301, "sg_order": 2}
]
```

These are comparison examples, not validated scientific defaults. Operation order
is SG baseline subtraction on the complete averaged FID using mirror edges,
then crop, then optional constant-mean removal (default true). SG means
`FID - savgol_filter(FID)`, not replacing the FID with its smoothed baseline.
No apodization, interpolation, zero filling, automatic alignment, resampling or
normalization of individual acquisitions is performed.

## Data fidelity and limits

- Decoder: reverse bytes, little-endian int16, retain `[20:-2]`, reverse samples.
  This reproduces `signal_selection`'s convention; it is not a verified hardware
  format specification. Configuration-vs-decoded length differences are reported.
  The first decoded point is retained explicitly.
- Original scan IDs mean the names in the selected directory; a previously
  exported selection may have renamed files. Do not assume original acquisition
  chronology without checking the selection report.
- Sampling rate comes from `0.ini [NMRduino] SampleRate`. Available per-scan INI
  files must agree. The schema does not assert all other experimental conditions
  were identical. Missing per-scan INI files are reported.
- Inspection fingerprints paths, size, modification time and INI content. Averaging
  additionally hashes every DAT payload and rechecks the inventory at completion.
  It never silently trims mismatched scans. Local artifacts are verified before
  reuse. Reusing a completed average means analyzing that recorded snapshot;
  it does not claim the source folder has remained unchanged since the run.
- Configured input folders remain untouched. No experiments, means or reports
  are sent to GitHub. MCP responses contain summaries and local artifact paths.
  Data shared with the Codex conversation is subject to the user's Codex settings.
- Mean FIDs and recipes are persisted for reuse; automatic content-addressed
  cache lookup is not implemented. Pass the existing run ID to avoid recomputing.
- Plots can be display-sampled; numerical FFTs use all retained samples. Each band
  plot contains only that band, so outside peaks cannot inflate its y-axis.
- Range maxima are exploratory, including noise. No significance threshold, SNR
  scaling validation, substance identification or decay fitting is included yet.
  Those will be added after reproducing the user's trusted processing workflow.

Each run saves parameters, source/parent provenance, software versions and an
implementation hash. Failed/cancelled runs retain their status and partial files
but cannot be consumed as completed results. Numerical work uses float64.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Tests cover independent DAT fixtures, exact streaming means, input preservation,
sample-rate/length rejection, recorded time origins, SG baseline subtraction,
artifact integrity, cancellation, range peak recovery and real MCP STDIO
initialization/list/call behavior. Synthetic tests do not establish physical
interpretations of experimental spectra.

## Simulation extension

Four additional tools expose the Python TwoD simulation workflow, optional CUDA
and validated equivalent-spin sectors. See [SIMULATION.md](SIMULATION.md) for
units, sequence definitions, limitations, installation and examples. Legacy
desktop applications are unchanged. The MCP server now exposes 12 tools; reload
the Codex connection after updating an already running server.

Two further tools now build natural-abundance isopropylamine skeleton models and
fit candidate J values to saved experimental complex spectra. The server exposes
26 tools including `review_decay_evidence`, `resample_decay_groups`, `fit_simulated_decay`, `fit_window_decay`, `inspect_decay_stability`, `inspect_repeat_signals`, `inspect_decay_time_frequency`, `fit_frequency_decay`, `compute_group_averages` and `fit_isopropylamine_staged`, which anchors the methyl high band
before fitting the overlapping methine component. See [J_FITTING.md](J_FITTING.md) for the explicit
model assumptions, parameter mapping, optimizer and interpretation limits.

## Next analysis milestones

1. Reproduce the trusted isopropylamine spectrum, including the first-point offset.
2. Add reviewed transient/baseline diagnostics and batch/scan accumulation checks.
3. Add candidate discovery with defensible noise estimation and explicit peak review.
4. Expose decay models with consistent processing of both observations and model.

Keep legacy applications available for comparison throughout this transition.

## Range-restricted relaxation development

`resample_decay_groups` takes a frequency `fit_run_id`, `candidate_index`,
`draws` (default 100), `block_length` (default 1) and bounded optimizer settings.
Only parent discovery groups are sampled. Each sampled copy retains its original
scan-count weight. Circular blocks follow the parent list order, not an inferred
chronological order. Length 1 assumes group exchangeability; larger blocks
preserve local dependence in that list. All draws and flagged estimates are
saved, with independent mode plots. Descriptive 2.5/50/97.5 percentiles require
all requested draws, at least 20 clean fits and at least 90% clean fits; otherwise
the summary is suppressed. These thresholds are heuristics, not coverage
guarantees. Percentiles exclude flagged/mismatched modes and remain conditional
on model, processing, grouping and that selection; they are not calibrated CIs.

`fit_simulated_decay` takes an existing band `fit_run_id` and a
`build_isopropylamine_model` `model_run_id`. Optional `source_j_fit_run_id`
must contain identical J parameters. It compares `shared_decay: true/false`
in separate calls, using `isotopomers: ["methine", "methyl"]` by default.
All simulated transitions are retained, including those outside observed bands;
one phase/gain pair applies to each isotopomer. Bounds, preprocessing and group
splits come from the parent. Equal band RMS weighting is explicit and enabled
by default. Settings also expose start/evaluation/time limits. Result artifacts
include transition sticks, component plots, real/imaginary residuals and model
provenance. Historical J fits using all scans do not provide an untouched
validation prior. Unknown preparation weights remain a model assumption.

An optional `settings.phase_delay_bounds_s: [-0.1, 0.1]` enables one global
phase-equivalent delay in a fixed-J decay fit. Bounds are exploratory settings,
not measured hardware timing. The model uses phase `-2*pi*f*delay` for each
transition while retaining the original acquisition time and decay envelope;
group gains absorb any common envelope reference factor. Omit the setting for
the zero-delay baseline. Saved starts retain alternative delay solutions and
boundary warnings. Single-frequency groups cannot distinguish this delay from
their free group phases and are explicitly flagged. The parameter does not
authorize raw-data alignment or establish a physical instrument delay.

`fit_window_decay` takes a parent FFT `fit_run_id` and `candidate_index`, with
`width_s`, `hop_s`, optional `sample_frequencies_hz`, and optimizer `settings`.
It inherits the parent's group split, preprocessing, bounds and mode count.
Five evenly spaced observation frequencies span the candidate band by default;
these samples are not five independently resolved peaks. Fit parameters are
estimated only on discovery means and frozen for validation. Independent
magnitude/real/imaginary/residual figures and a native FFT cross-check are saved.
A constant spectral-background parent is rejected because that term has no
unique full-FID window counterpart. Sparse transform memory and optimization
work are bounded; budget-limited results remain explicitly provisional.

`inspect_decay_stability` takes a completed `fit_run_id`, `candidate_index`,
optional `group_indices` (default: the parent's validation groups), and bounded
optimizer `settings`. It refits group averages with the parent's preprocessing,
frequency and T2* bounds, and mode count. Frozen discovery prediction errors
remain separate from group refit errors. Per-mode T2* and frequency-shift figures
are independent; raw fitted phase differences are stored without recommending
phase correction. Frequency matching is only a bookkeeping convention, and
group spread is not a confidence interval. All refits and numerical warning
flags are retained, including provisional budget-limited candidates.

For reproducible comparisons with a single requested mode count, pass
`settings.initial_frequencies_hz` as one list per frequency band, for example
`[[133.4, 134.8, 136.5], [246.1, 250.1, 254.0]]` with `components: [3]`.
These are starting points, not fixed frequencies or a physical assignment.
Bounds and multistart exploration remain active. Record the discovery-only
source of starting values when comparing preprocessing variants; do not use
validation data to choose them. Omit this setting for automatic initialization
or when comparing several mode counts.

The active requirements/evidence ledger is [RELAXATION_PLAN.md](RELAXATION_PLAN.md).
`compute_group_averages` takes `folder` and `groups`, for example
`[[0, 2, 4], [1, 3, 5]]` when those explicit scan IDs exist. It returns a
background job through MCP, or a completed manifest through direct JSON CLI.
All selected groups must be nonempty and disjoint. It stores `group_averages.npz`
with `means`, `counts`, and the scan-count-weighted `pooled` mean, plus file
hashes and membership. No filtering, phase/frequency alignment or trimming is
performed. Explicit subsets are not asserted to be physical acquisition batches.
Keep discovery and validation groups disjoint; the pooled mean is not held-out data.

`fit_frequency_decay` uses such a completed group run. See
[examples/band_decay.json](examples/band_decay.json). Frequency ranges are in Hz;
T2* bounds are in seconds; group indices are zero-based. Example bounds and
preprocessing are exploratory, not established physical priors. `components`
means oscillator counts, not substance counts. A two-frequency shared-decay
model can describe beating without two distinct decay times. All coefficients,
including phases/gains, are frozen for the main validation prediction. Separate
conditional gain refits are explicitly diagnostic, not the held-out score.
Selecting a model on validation makes it a comparison set rather than a final
untouched test set. The tool reports all candidates without scientific acceptance.
Plots of magnitude, real part, imaginary part and validation residual are separate.
`max_seconds` and `max_evaluations` limit each fit, while `total_seconds` limits
starting/continuing fits across the operation; output serialization/plotting may
extend wall time beyond the optimization budget. Other goal requirements such
as drift correction, confidence intervals and automated crop
sensitivity remain in development and are not claimed by this operation.

`inspect_decay_time_frequency` consumes a completed decay fit and zero-based
candidate index. `widths_s` defaults to `[0.25, 0.5, 1]`, and `hop_fraction=0.1`
matches the legacy 90% overlap. Complete periodic-Hann windows retain complex
phase referenced to acquisition time; off-grid Fourier evaluation does not
increase spectral resolution. Data and frozen model undergo identical processing.
The demodulator mixes to the band center, applies a Kaiser FIR before decimation,
and exports an explicit mask for zero-extension edge effects. A short record
may have no unaffected interior; this is reported with a null interior error,
not silently treated as a successful comparison. Full and early-window plots
are independent. Overlap correlation and magnitude-noise bias remain relevant;
this diagnostic is not an independent confidence interval or substance assignment.

`inspect_repeat_signals` requires explicit `ranges`, nonoverlapping reference
`noise_ranges`, and at least two disjoint groups in each discovery/validation
split. It ranks discovery local maxima and checks validation peaks within an
explicit tolerance (default two native bins). `snr_threshold` is operational,
not a p value; default 5. Optional `interference_ranges` flags known/suspected
external bands supplied by the caller. Labels distinguish reproducible signal
candidates, suspected interference, noise-compatible and insufficient evidence.
Frequency agreement and repeat SNR do not establish molecular origin. The same
reference-band accumulation curve appears for each candidate; only the signal
amplitude/SNR differs. Reference bands may contain leakage or drift, so compare
alternative reference bands before interpretation. Phase plots hide bins below
the repeat-SNR threshold and never apply an alignment or unwrap across gaps.


## Reviewing decay evidence

`review_decay_evidence` takes `fit_run_id`, `candidate_index`, optional
`signal_run_ids`, `stability_run_id`, `resampling_run_ids` and `comparison_refs`.
Each comparison is an explicit `{"run_id": "...", "candidate_index": 1}`
for FFT fits, or `{"run_id": "..."}` for window fits. Mode counts must match;
frequency matching is one-to-one within two native FFT bins and does not prove
physical identity. Source hashes and discovery/validation membership must match.
Signal evidence additionally requires identical processing. Stability and
resampling must reference this exact parent candidate.

The tool writes `evidence_review.md` and per-mode evidence in its manifest.
Missing signal support suppresses the reported candidate decay value. Numerical
warnings, unstable group/resampling results, and missing evidence prevent a
supported label. Sensitivity is flagged if a matched comparison changes T2*
beyond `relative_change_threshold` (default 0.1), falls outside a supplied
conditional sampling percentile range, or is numerically provisional. These are
operational diagnostics, not statistical acceptance tests. Even an
`exploratory_supported_candidate` has `physical_component_accepted: false`:
residual adequacy, interference, model error and mechanism remain separate
scientific questions. Repeating signal checks with different reference noise
bands does not provide independent repeat-classification evidence.


For fixed-J decay boundary checks, `fit_simulated_decay` accepts optional
`t2_bounds: [lower_seconds, upper_seconds]`. It inherits bounds when omitted.
The result records both `parent_t2_bounds_s` and actual `t2_bounds_s`, plus
`t2_bounds_overridden`. Keep the original run and change bounds explicitly;
a boundary-free solution is still not evidence of model adequacy. Band ranges,
source groups, preprocessing and validation remain inherited from the parent.


## FID crop diagnostics

`inspect_fid_crops` takes a `group_run_id`, frequency `ranges`, disjoint
`discovery_groups` and `validation_groups` (at least two each), optional
`sg_window`/`sg_order`, `width_s` (default 0.25) and `snr_threshold` (default 5).
It retains the first sample and acquisition time; SG means full-record mirror
baseline subtraction. No trimming, alignment or mean removal is performed.
Each band needs two native bins per complete nonoverlapping Hann window.

Outputs include independent full/early/tail raw and processed FID figures,
local baseline and window band-amplitude/SEM-ratio figures, diagnostic arrays,
and up to six candidate intervals. Starts use a heuristic raw-baseline
recovery check against the final quarter of blocks and an SG half-window guard,
capped at min(1 second, 10% of record). A raw baseline that recovers later than
that cap rejects the baseline-based start rule and falls back to the earlier
SG guard (or zero), rather than discarding early signal up to the cap.
Tail references may contain real signal
or drift. End proposals retain the last discovery-supported window plus one
window margin; a later revival prevents an earlier cutoff. No detected signal
keeps the full endpoint. Full data are always included as a baseline candidate.

Held-out groups do not determine proposals; their supported windows beyond each
suggested endpoint are reported separately. These ratios are operational
repeat-scatter diagnostics, not significance levels. Short windows mix nearby
peaks and leakage, and band RMS is not an exponential envelope. Compare the
proposed fits under identical processing before adopting any crop; preserve
failed and sensitive results. Plots use display subsampling only.


Early-start refinement uses `early_block_s` (default 0.005 s),
`early_duration_s` (default 1 s) and `early_amplitude_ratios` (default [5,10,20]).
Processed-FID block RMS is compared with its median in the latter half of this
early interval. A threshold alternative requires excess amplitude near the
acquisition start and its disappearance before the reference interval; the
candidate retains a one-fine-block margin after the last excess. Isolated later
bursts and unresolved excess do not automatically remove preceding data.
Distinct discovery alternatives replace coarse start offsets when available;
the full record and a full-tail alternative remain. Independent validation
large-amplitude blocks retained by each crop are recorded against discovery
thresholds. `early_amplitude_thresholds.png` and `candidate_start_zoom.png`
show this decision. Ratios and reference intervals are explicit heuristics:
large early amplitude can itself include fast molecular signal. They cannot
prove an artifact-free crop, so alternative fits must still be compared.


## Matched demodulated-time fitting

`fit_demodulated_decay` inherits an FFT `fit_run_id` and `candidate_index`,
frequency and T2* bounds, mode count, shared/independent decay, acquisition groups
and processing. It accepts `transition_hz`, `attenuation_db` (default 80),
`edge_policy` and bounded `settings` (starts, max_nfev, max_evaluations,
max_seconds, seed). Explicit discovery FFT initial frequencies are used.

The model creates real full-record damped quadratures at acquisition time,
subtracts the same full-record SG baseline, crops and removes the retained mean
as configured, then mixes, convolves with the cached Kaiser FIR and decimates.
Anti-alias settings match the diagnostic demodulator. Filter transitions admit
outside-band contributions, so target-band-only models can be incomplete.

`matched_all` (default) includes zero-extension-affected samples and models the
same finite-record operation exactly. `interior` uses only samples unaffected
by zero extension, potentially losing most of a fast decay. This distinction
is explicit in the `filter` manifest and `filter_masks.png`; neither policy
invents pre-crop observations. All-sample predictions are plotted even for an
interior-only fit. Full/early magnitude, real and imaginary figures, residuals,
arrays and an original FFT cross-check are independent artifacts. The fitted
coefficients are frozen for validation. Filtered samples are correlated and
are not treated as independent observations for uncertainty estimates.

`review_decay_evidence` accepts these runs as explicit sensitivity comparisons
alongside window and FFT fits. Smaller demodulated residuals are not sufficient
to establish decay identifiability or a physical component.


`inspect_repeat_signals` additionally reports `band_coherence` on native bins
selected only by discovery repeat-SNR, excluding declared interference ranges.
For each role it compares the norm of the scan-weighted coherent mean with the
weighted mean of group spectral norms. A low ratio can reflect phase, noise or
spectral-shape changes; it is not a pure phase-loss estimate. A common complex
overlap with the discovery reference gives group phase diagnostics, masked
unless the group band-RMS/scatter ratio reaches the operational threshold and
normalized shape overlap is at least 0.8. Repeating bins does not inflate this
RMS ratio. Discovery phases share their reference and are not independent
validation measurements. The independent group-phase and overlap figures apply
no alignment or normalization and do not estimate frequency drift. Frequency
changes remain separate diagnostics from `inspect_decay_stability`; unstable
mode assignments must not be interpreted as physical drift.
