"""Typed local STDIO MCP tools. Experiments never leave the local machine."""
from mcp.server.fastmcp import FastMCP
from .analysis import execute
from . import jobs, storage

mcp = FastMCP('ZULF Analysis Tools')


@mcp.tool()
def inspect_dataset(folder: str, scan_ids: list[int] | None = None) -> dict:
    """Inspect numbered NMRduino DAT/INI files and sample checks. Source is read-only.
    Reports declared vs decoded points; saves a full manifest locally. For large
    folders prefer start_analysis to avoid tool timeouts. null scan_ids means all.
    """
    return execute('inspect_dataset', {'folder': folder, 'scan_ids': scan_ids})


@mcp.tool()
def compute_average(folder: str, scan_ids: list[int] | None = None) -> dict:
    """Start a background coherent average of explicitly selected raw FIDs.
    Streams scans without filtering, trimming or alignment. Returns job_id; use
    get_job until terminal state. Completed run_id is input to preprocessing.
    """
    return jobs.start_analysis('compute_average', {'folder': folder, 'scan_ids': scan_ids})


@mcp.tool()
def compare_preprocessing(average_run_id: str, variants: list[dict]) -> dict:
    """Start comparison of explicit recipes on an existing average. Recipe keys:
    label, start_s, end_s (exclusive), sg_window (0=off; otherwise odd), sg_order
    (default 2), remove_mean (default true). SG baseline subtraction is applied
    before crop with mirror edges. Returns job_id, independent figures and
    provenance on completion. This is exploratory; no fitting or auto selection.
    """
    return jobs.start_analysis('compare_preprocessing', {'average_run_id': average_run_id, 'variants': variants})


@mcp.tool()
def inspect_frequency_ranges(comparison_run_id: str, ranges: list[list[float]], max_candidates: int = 10) -> dict:
    """Start native-bin range plots for all recipes in a completed comparison.
    Ranges are [low_hz, high_hz]. Local maxima are ranked by prominence, NOT
    declared significant or assigned to substances. Figures use local y limits.
    Returns job_id. No SNR threshold or decay fitting is performed in v0.1.
    """
    return jobs.start_analysis('inspect_frequency_ranges', {'comparison_run_id': comparison_run_id,
                                                           'ranges': ranges, 'max_candidates': max_candidates})


@mcp.tool()
def start_analysis(operation: str, parameters: dict) -> dict:
    """Start inspect_dataset, compute_average, compare_preprocessing or
    inspect_frequency_ranges in a background process. Returns job_id immediately.
    Query get_job for progress/results, or cancel_job for cooperative cancellation.
    """
    return jobs.start_analysis(operation, parameters)


@mcp.tool()
def get_job(job_id: str) -> dict:
    """Read persistent job progress. Terminal states: complete, failed, cancelled.
    Complete jobs include run_id, JSON report and absolute local artifact paths.
    """
    return jobs.get_job(job_id)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Request cooperative cancellation. Poll get_job until cancellation finishes.
    Existing completed artifacts are never deleted. Checkpoints occur between
    files/variants, not inside a single FFT or plotting operation.
    """
    return jobs.cancel_job(job_id)


@mcp.tool()
def get_result(run_id: str) -> dict:
    """Read a completed run manifest and artifact paths without recomputation."""
    return storage.get_result(run_id)


def main():
    mcp.run(transport='stdio')

if __name__ == '__main__':
    main()

