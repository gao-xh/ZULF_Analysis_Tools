# Codex skills

## ZULF J Fit

[`zulf-j-fit/SKILL.md`](zulf-j-fit/SKILL.md) packages simulation-guided J fitting decisions and connects them to this repository's existing numerical tools. It includes staged isotope comparisons, fine fitting, linewidth controls, frozen validation, and optional decay/display guidance.

The skill is an orchestration layer, not a replacement simulation engine or a standalone tool server. Use this repository's configured Python environment and CLI/MCP backend. Experiment data, local run identifiers, credentials and generated results are not included.

To install as a personal Codex skill, copy the complete `zulf-j-fit` folder to your Codex skills directory (`$CODEX_HOME/skills`, or `~/.codex/skills` when unset). Do not overwrite an existing modified copy without reviewing it. Reload skills or start a new Codex session if it is not discovered immediately.

Example request:

> Use $zulf-j-fit to refine candidate J matrices for my experimental data, compare the isotope components and report frozen validation errors.

The bundled summary helper is read-only and can be run before inspecting large manifests:

```text
python skills/zulf-j-fit/scripts/summarize_run.py --project /path/to/ZULF_Analysis_Tools RUN_ID
```

It prints compact metrics while numerical arrays remain on disk. Detailed methods are loaded only when relevant. Token savings depend on the task; this package does not claim a measured reduction or automatically validate fitted physics.

The advanced individual-transition decay pilot mentioned in the reference is a local analysis script, not included in the public skill or exposed as a production tool. The supported production fixed-J decay operation is `fit_j_cluster_decay`.
