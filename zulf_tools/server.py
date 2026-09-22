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
def inspect_simulation_backend() -> dict:
    """Test CPU/GPU availability including actual CUDA complex128 linear algebra."""
    return execute('inspect_simulation_backend', {})


@mcp.tool()
def import_spin_model(folder: str) -> dict:
    """Read legacy structure.csv and optional symmetry.csv under local model_roots.
    Returns model with explicit Hz couplings and zero-based symmetry indices.
    Only 1H, 13C, 15N spin-1/2 nuclei; input files remain unchanged.
    """
    return execute('import_spin_model', {'folder': folder})


@mcp.tool()
def analyze_spin_symmetry(model: dict) -> dict:
    """Start equivalent-spin validation and exact collective-spin sector analysis.
    Model keys: isotopes, couplings_hz, optional gamma_hz_per_ut isotope mapping,
    optional symmetry_groups (disjoint zero-based groups; omitted=detect, []=none).
    Returns job_id. Reports commutators and sector dimensions, not geometric symmetry.
    """
    return jobs.start_analysis('analyze_spin_symmetry', {'model': model})


@mcp.tool()
def simulate_spin_dynamics(model: dict, settings: dict | None = None) -> dict:
    """Start 1D/2D/MQ spin-1/2 J dynamics using eigenbasis transition sums.
    Model: isotopes, symmetric zero-diagonal couplings_hz, optional symmetry_groups
    (zero-based), gamma_hz_per_ut. Settings: sequence fid/2d/mq; backend cpu/gpu;
    symmetry boolean; npoints, sampling_rate_hz, t1_points, t1_step_s, tm_s,
    t2star_s, field1_ut, field2_ut, pulse_field_ut [Bx,By,Bz], pulse_duration_s.
    Defaults: CPU, symmetry enabled, zero pulse/field, 1024 points at 1000 Hz.
    Positive Zeeman sign; preparation/detection gamma-weighted Ix. Returns job_id.
    All times seconds, fields microtesla. GPU failure is explicit, never silent fallback.
    """
    return jobs.start_analysis('simulate_spin_dynamics', {'model': model, 'settings': settings})


@mcp.tool()
def build_isopropylamine_model(parameters: dict | None = None) -> dict:
    """Build the two single-13C eight-spin carbon/proton skeletons and J mapping.
    Parameters (Hz): J_CH_methine, J_CH_methyl, J_HH_vicinal,
    J_Cmethine_Hmethyl, J_Cmethyl_Hmethine, J_Cmethyl_Hother_methyl.
    Defaults are search guesses, not measurements. NH2/14N omitted explicitly.
    """
    return execute('build_isopropylamine_model', {'parameters': parameters})


@mcp.tool()
def fit_isopropylamine_j(comparison_run_id: str, variant_index: int,
                       ranges: list[list[float]], settings: dict | None = None) -> dict:
    """Start exploratory multi-start J fitting to a saved complex experimental FFT.
    Uses both natural-abundance single-13C skeletons, exact SG/crop response,
    independent isotopomer amplitude/phase and decay rates. ranges: sorted disjoint
    positive [low,high] Hz bands. Settings: initial (J dict), bounds (J-> [lo,hi]),
    free_parameters (names), starts, max_nfev, screening_samples, seed, bin_stride,
    rate_bounds [lo,hi] in 1/s, initial_rate, objective complex/magnitude (default
    complex). Magnitude compares abs of coherently summed complex predictions,
    fitting nuisance gains/phases internally. Also accepts isotopomers (nonempty
    subset of methine/methyl) and fixed_rates (isotopomer->1/s). Returns job_id. Saves candidates,
    J matrices, sensitivity, overlays and residuals. Does NOT uniquely identify
    the full molecule's J values, perform repeat validation or infer confidence intervals.
    """
    return jobs.start_analysis('fit_isopropylamine_j', {'comparison_run_id': comparison_run_id,
        'variant_index': variant_index, 'ranges': ranges, 'settings': settings})


@mcp.tool()
def fit_isopropylamine_staged(comparison_run_id: str, variant_index: int,
                             low_range: list[float] | None = None,
                             high_range: list[float] | None = None, settings: dict | None = None) -> dict:
    """Start methyl-high-band anchor, unchanged low-band extrapolation, methine
    fit with methyl J/rate fixed, then joint refinement of retained branches.
    Default low/high bands: [110,150]/[230,275] Hz. Settings: initial, bounds,
    objective complex/magnitude, branches (1..4), anchor_starts, anchor_screening,
    methine_starts, methine_screening, max_nfev, bin_stride, seed,
    high_degradation_tolerance (default .15). All child run IDs are preserved.
    High-band assignment is an assumption; no candidate is scientifically validated
    automatically. Component magnitudes are never subtracted to form a residual.
    """
    return jobs.start_analysis('fit_isopropylamine_staged',{'comparison_run_id':comparison_run_id,
        'variant_index':variant_index,'low_range':low_range,'high_range':high_range,'settings':settings})


@mcp.tool()
def start_analysis(operation: str, parameters: dict) -> dict:
    """Start any registered experimental or simulation operation in a background
    process. Returns job_id immediately.
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
