# Bounded frequency-range decay analysis

Start with a frequency cluster and plausible time bounds, not exact peak positions.
The output is a reviewable effective FID T2* model. A completed job or converged
optimizer does not identify an intrinsic T2, substance or relaxation mechanism.

## 1. Establish the data and diagnostic baseline

Use `inspect_dataset` on a new folder, then `compute_group_averages` with explicit,
disjoint scan groups. Keep discovery and validation membership fixed. Group means
reduce computation; fitting every raw acquisition is not the default workflow.
Numeric file order is not automatically a verified acquisition-time sequence.

Use `inspect_fid_crops` to inspect independent full, early and tail FID figures,
local baseline and band-window diagnostics. Proposals are hypotheses: keep the
untrimmed reference, compare a few justified alternatives, and retain real
acquisition time. A visually flat full-scale trace can still contain weak signal.
Display extrema are not new fitting data.

Use `inspect_repeat_signals` with explicit target and nonoverlapping reference
noise bands. Candidate frequencies come from discovery; validation checks them.
Inspect magnitude, masked phase, accumulation and group coherence. A square-root
SNR trend supports averaging behavior, but cannot exclude coherent interference
or common noise. Change the reference band as a sensitivity check, not a way to
force a favorable classification.

SG means **full-record baseline subtraction**, `FID - SG(FID)`, before cropping.
It does not mean smoothing the fitted FID. Compare with no SG and a justified
alternative window. Data and model must undergo the same operation.

## 2. Start a small bounded fit

Copy [bounded_decay_request.json](examples/bounded_decay_request.json) to a local
request file. Replace the group run ID and group indices; adapt ranges, crop,
SG and time bounds to the inspected data. The supplied numbers are illustrative,
not measured physical limits. No exact frequency seeds are supplied.

```powershell
.venv/Scripts/python.exe -m zulf_tools.cli --request .analysis/my_decay_request.json
```

Save the returned job ID. Poll `get_job` with `{"job_id":"..."}` through the same
CLI or MCP interface. Use `cancel_job` to request cancellation. An observation
timeout is not a failed job and is not a reason to submit a duplicate.

`ranges` are sorted, disjoint positive-frequency intervals in Hz. `t2_bounds`
are positive seconds. Omit the latter only when an explicitly documented
discovery-based exploratory proposal is acceptable. `components: [1, 2]` tests
simple oscillator counts first; two-mode shared and independent T2* candidates
are compared by default. These are mathematical terms, not physical populations.

Each band must have sufficient native FFT bins for the requested complexity.
Zero padding or denser off-grid plotting does not create independent information.
The same T2* interval applies to every band within one request; use separate
requests if bands need different justified intervals.

Budgets are cooperative: a single evaluation, preprocessing or figure write can
extend past a time limit. Check both run-level completeness and candidate flags.
Inspect returned candidate metadata rather than guessing a candidate index.

Frequency-decay requests reject invalid fit controls before group preprocessing:
`starts` is an integer from 1 to 32, `max_nfev` from 2 to 2000, and
`max_evaluations` from 2 to 100000. Time budgets must be finite positive numbers;
an infinite per-fit limit is not silently replaced by the total budget. `seed`
must be a nonnegative integer, and model flags must be actual JSON booleans.
Explicit T2* bounds are also checked before preprocessing.

## 3. Decide what to do next

| Evidence | Next action |
| --- | --- |
| Missing/failed signal support | Withhold decay interpretation; inspect inputs, noise reference and interference evidence. |
| Budget exhausted or optimizer not converged | Keep the provisional trial; correct numerical limits or make one justified bounded extension. |
| Parameter near a bound | Make a limited boundary sensitivity check; do not quote the bound as a measurement. |
| Repeatable residual structure | Inspect matched time-domain/window responses and omitted signal before adding terms. |
| Strong dependence on crop, SG or objective | Report model/processing dependence; a narrow resampling interval does not remove it. |
| Evidence agrees across groups and justified comparisons | Report an exploratory effective T2* candidate with the remaining assumptions. |

Do not repeatedly increase mode count because the residual decreases. A complex
model can fit noise, unresolved neighbors or outside-band leakage. Stop an
unproductive route and preserve its result and reasoning.

## 4. Validate the chosen numerical candidate

Use `inspect_decay_time_frequency` first to transform its frozen prediction
through the same Hann windows and anti-aliased demodulator as the data. Width
sets the observation interval; hop sets spacing between overlapping windows.
Overlapping points are correlated. Real/imaginary views can reveal disagreement
hidden by magnitude; phase near zero amplitude is not meaningful.

Use `fit_window_decay` or `fit_demodulated_decay` for a justified alternative
objective. Check the original FFT cross-check as well as the fitted-domain loss.
Filter-edge shading marks finite-filter influence, not a physical transient or
automatic crop. Individual exponential envelopes can decline while their
complex sum revives through beating.

Use `inspect_decay_stability` for independent validation-group refits and
`resample_decay_groups` for conditional discovery-group variability. Their
parent run and candidate index must match the candidate being reviewed. Similar
parameters from another run do not transfer parent-specific evidence. Historical
J parameters selected on all scans also do not provide untouched validation.

Use `review_decay_evidence` to link compatible signal, stability, resampling and
explicit sensitivity comparisons. Inspect its independent evidence map and
residual figures. Model selection reuses validation: it is a comparison set,
not a fresh final confirmatory experiment. Unsupported T2* values are withheld;
other candidates retain numerical, instability or unresolved-residual labels.

## 5. Keep the analysis reproducible

Record requests, source hashes, run IDs, preprocessing, bounds, optimizer flags,
failed attempts and independent figures. Completed frequency-decay candidates
have numeric checkpoint filenames and hashes. A later cancelled/failed run can
retain these for review but cannot act as a completed parent run; checkpoints
do not resume an optimizer.

The persistent [workflow example](examples/relaxation_workflow.json) demonstrates
job orchestration. Its diagnostic stages explicitly review **candidate 0**;
they do not select the best candidate or incorporate crop proposals automatically.
Treat it as an execution example, then make a new explicit plan after reviewing
scientific evidence. Reusing its name with a changed plan is rejected.

Keep raw experiments in configured read-only input roots and generated reports
in local artifact storage. Back up code separately. For optional J constraints,
follow [J_FITTING.md](J_FITTING.md)
and preserve alternative assignments and preparation assumptions.
