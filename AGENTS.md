# Codex analysis workflow

This project is a local scientific tool backend. Codex orchestrates tool calls,
inspects artifacts, explains evidence and asks the user about scientific choices.
Do not build a new desktop GUI as the primary workflow.

- Keep tools, reports, plots, code and documentation in English. Discuss with the
  user in their preferred language. Preserve source file names.
- Experimental roots are read-only. Write only under the project's `.analysis`.
- Read `README.md` and call `inspect_dataset` before analyzing a new folder.
- Use the JSON CLI or MCP tools. Do not invent computed values in chat.
- Prefer `start_analysis` for expensive work; poll `get_job`, cancel via
  `cancel_job`. Report failures and incomplete jobs explicitly.
- Every completed analysis writes parameters, source provenance, versions and
  independent figures. Inspect figures before drawing scientific conclusions.
- Keep the legacy decoder explicit. Do not silently drop a first point, align,
  normalize or average scans. Average only via the named averaging tool.
- SG baseline subtraction means `FID - SG(FID)`, not SG-smoothed FID.
- Processing comparisons are exploratory. Do not interpret changes in apparent
  peaks as physical modes without stability checks and an appropriate fit model.
- Preserve raw FIDs and acquisition-relative time origins. Never automatically
  identify substances or declare a relaxation mechanism from peak detection.
- Test numerical operations against independent synthetic signals and verify
  the MCP transport. Do not upload experiments or generated reports to GitHub.
- Maintain before/after GitHub backups for subsequent substantive modifications.

