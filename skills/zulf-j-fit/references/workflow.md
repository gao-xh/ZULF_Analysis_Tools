# Tools and request protocol

Project CLI (PowerShell, working directory = project):

```powershell
.venv/Scripts/python.exe -m zulf_tools.cli --request .analysis/requests/task.json
```

Request shape: `{"tool":"start_analysis","arguments":{"operation":"fit_isopropylamine_staged","parameters":{...}}}`. `get_job` takes `job_id`; `get_result` takes `run_id`. These identifiers are different. The CLI and MCP share the implementation. Use active MCP capabilities when available; CLI is a functional fallback, not evidence the MCP connection is live.

Production operations verified in `zulf_tools/analysis.py`:

| Need | Operation / source |
|---|---|
| Inventory and coherent mean | `inspect_dataset`, `compute_average` |
| Repeated-data groups | `compute_group_averages` |
| Explicit crop and baseline recipes | `compare_preprocessing` |
| Simulate a spin model | `simulate_spin_dynamics` |
| Inspect spin symmetry | `analyze_spin_symmetry` |
| Isopropylamine model | `build_isopropylamine_model` |
| Staged J fit | `fit_isopropylamine_staged` |
| Fine or restricted J fit | `fit_isopropylamine_j` |
| Provisional nitrogen hypothesis | `inspect_n15_hypotheses` |
| Fixed-J cluster decay | `fit_j_cluster_decay` |

Use project `examples/j_fit_staged.json` as the staged request template. Set a real completed `comparison_run_id` and correct `variant_index`. Its 110–150 / 230–275 Hz windows are isopropylamine examples, not universal defaults. It anchors methyl in the high band, preserves its low-band extrapolation, fits the competing methine contribution and jointly refines candidate branches. The 15% high-band deterioration guard is operational, not statistical significance.

Use `examples/j_fit.json` or `examples/j_fit_fine.json` for ordinary fitting. The fine example imposes rate limits and therefore represents a narrow-width hypothesis, not an automatically appropriate default. Preserve an unrestricted-width control. Inspect `zulf_tools/jfit.py` and `J_FITTING.md` for the exact settings accepted by the installed version.

For methyl independent effective damping, ordinary-fit `settings.methyl_split_hz` can be placed in the unfitted gap (190 Hz in this experiment). Rate keys become `methyl_low`, `methyl_high`, plus `methine`; the staged operation retains common methyl damping. Keep a common-rate baseline, model all tails in both bands, and report bound hits. Never splice spectra at the split.

Fine comparison should record `bin_stride=1`, explicit local J bounds and starts, `diff_step` and optimizer budgets. Local refinement cannot exclude remote alternative J solutions. Fixed experimental record length defines frequency spacing; zero padding or plotting interpolation does not add independent information.

Long operations should save the returned job ID immediately. Query status rather than launching duplicates. A stale running state requires checking worker PID/logs; it does not prove work is continuing. A budget-stopped best candidate is provisional even if it looks good.

For another molecule, inspect supported generic spin simulation inputs and construct a matching inverse model; do not run the isopropylamine fitter under a new sample label.
