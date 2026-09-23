---
name: zulf-j-fit
description: Fit and compare ZULF experimental FIDs against spin simulations to estimate candidate scalar J matrices, using the local ZULF_Analysis_Tools backend. Use for simulation-guided J refinement, isotope-component comparisons, linewidth checks, or continuing an existing J fit; optionally hand the fixed transitions to T2* analysis.
---

# ZULF simulation-guided J fitting

Use existing numerical tools rather than reconstructing this workflow in chat. Produce conditional J candidates with reproducible experimental comparisons, not an automatically accepted molecular assignment. This skill does not make the isopropylamine-specific fitter a general molecular inverse solver.

## Locate and reuse

Resolve `ZULF_Analysis_Tools` from the user's workspace or explicit project path. Read its `AGENTS.md`. Experimental inputs are read-only; analysis outputs belong in `.analysis`. Use English figures/reports and the user's language for conversation.

Before recomputing, identify the requested dataset, completed run and preprocessing variant. Run `scripts/summarize_run.py --project PROJECT RUN_ID [RUN_ID ...]` with the project Python to read compact local summaries. Fetch selected arrays/plots only when needed. Never dump whole candidate logs or raw arrays into context.

For tool names, CLI requests and staged fitting, read [references/workflow.md](references/workflow.md). For display-only work or T2* continuation, read [references/decay-and-display.md](references/decay-and-display.md). Historical run pointers belong in the local project's `.analysis`, not in this portable skill; request or discover the appropriate manifest when continuing an experiment.

## Scientific decisions

1. Verify sample, spin model and explicit scan membership. Inspect a new experimental directory with `inspect_dataset` before averaging. Keep coherent means and disjoint group means reproducible. Do not silently discard scans, reset acquisition time, smooth FIDs, or change crop/FFT grids.
2. Define the comparison before fitting: ranges, crop, SG baseline subtraction, target (magnitude, complex, or real), J bounds/sign assumptions, damping hypotheses and search budget. Apply the same acquisition/preprocessing operator to simulation. SG means `FID - SG(FID)` on the full record, then crop.
3. For natural-abundance isopropylamine, use the staged methyl high-band anchor, unchanged low-band prediction, methine competition, then joint refinement. Read the project's `J_FITTING.md` for this branch. Keep alternative candidates when identifiable. High-band methyl dominance is a hypothesis, not guaranteed truth.
4. Refine on native bins (`bin_stride=1`). Inspect the component spectra, unbroadened transitions, per-band residuals and boundary hits. Broadening can conceal incorrect J or preparation weights. Compare common methyl damping with separate low/high effective rates when relevant; label the latter empirical, not established symmetry channels.
5. Evaluate frozen predictions on separate acquisition groups where possible. Once groups are used to select a model they are not untouched tests. If J used all scans, subsequent held-group decay validation is conditional on J. Stop bounded searches at the recorded budget and expose nonconvergence; do not restart indefinitely until a preferred appearance emerges.
6. Export atom ordering, J matrix in Hz, fixed/free parameters, source IDs/hashes, actual time/frequency grids, objective and unpenalized residuals, alternative branches and limitations. Inspect separate figures before delivery. Numerical convergence and lower loss do not establish a unique J matrix.

## Essential distinctions

- Scalar isotropic J matrices are not spatial anisotropic coupling tensors.
- Sum complex component spectra before taking magnitude. Never subtract magnitude spectra to isolate a molecule.
- Real-display phase, empirical phase slope and AsLS are separate from changing J or decay. More positive real spectra do not prove correct absorption phases or assignments.
- Distinguish fitted effective FID T2* from intrinsic T2; boundary hits and regularization sensitivity are not precise measurements. Min–max spreads are not confidence intervals.
- N15 is optional and hypothetical until supported; include isotope probability, signed gamma, preparation and relative signal weighting. Low abundance alone neither rules it out nor establishes it.
- Reuse jobs/results by ID. Check the current backend schema before adding an option. Existing `.analysis` scripts may hard-code old runs: inspect/adapt before executing on new inputs.

## Efficient execution

Use `start_analysis` / `get_job` for production fits. Save request JSON and job IDs. Read summaries first, then only relevant plots/arrays. Prefer existing deterministic export code to regenerating plotting logic. Do not claim GPU acceleration without verifying the selected computation path.

This skill itself does not authorize experiment uploads, GitHub publication, data deletion or changes to unrelated configuration. Follow existing project backup requirements for substantive production edits; ordinary analysis stays under `.analysis`.
