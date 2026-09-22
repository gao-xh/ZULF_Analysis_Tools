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
19 tools including `inspect_repeat_signals`, `inspect_decay_time_frequency`, `fit_frequency_decay`, `compute_group_averages` and `fit_isopropylamine_staged`, which anchors the methyl high band
before fitting the overlapping methine component. See [J_FITTING.md](J_FITTING.md) for the explicit
model assumptions, parameter mapping, optimizer and interpretation limits.

## Next analysis milestones

1. Reproduce the trusted isopropylamine spectrum, including the first-point offset.
2. Add reviewed transient/baseline diagnostics and batch/scan accumulation checks.
3. Add candidate discovery with defensible noise estimation and explicit peak review.
4. Expose decay models with consistent processing of both observations and model.

Keep legacy applications available for comparison throughout this transition.

## Range-restricted relaxation development

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
