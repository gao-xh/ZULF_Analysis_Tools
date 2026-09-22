"""Typed local STDIO MCP tools. Experiments never leave the local machine."""
from mcp.server.fastmcp import FastMCP
from .analysis import execute
from . import jobs, storage

mcp = FastMCP('ZULF Analysis Tools')


@mcp.tool()
def review_decay_evidence(fit_run_id: str, candidate_index: int = 0,
                           signal_run_ids: list[str] | None = None, stability_run_id: str | None = None,
                           resampling_run_ids: list[str] | None = None, comparison_refs: list[dict] | None = None,
                           relative_change_threshold: float = .1) -> dict:
    """Review matched signal, group, resampling and explicit sensitivity evidence.
    Checks source/split compatibility and exact signal preprocessing. Comparison
    refs contain run_id and candidate_index for FFT fits, or run_id for window
    fits. Missing/conflicting evidence is explicit. All statuses remain exploratory;
    no automatic physical acceptance or intrinsic T2 assignment.
    """
    return jobs.start_analysis('review_decay_evidence',dict(fit_run_id=fit_run_id,candidate_index=candidate_index,
        signal_run_ids=signal_run_ids,stability_run_id=stability_run_id,resampling_run_ids=resampling_run_ids,
        comparison_refs=comparison_refs,relative_change_threshold=relative_change_threshold))


@mcp.tool()
def resample_decay_groups(fit_run_id: str, candidate_index: int = 0, draws: int = 100,
                           block_length: int = 1, settings: dict | None = None) -> dict:
    """Start bounded discovery-group resampling; validation groups are untouched.
    Circular blocks follow parent discovery-list order; block length 1 assumes
    exchangeability. Saves every draw and flagged candidate. Conditional
    percentile summaries require enough clean completed draws and are not
    calibrated confidence intervals. Settings: starts, max_nfev, max_evaluations,
    max_seconds per draw, total_seconds, seed.
    """
    return jobs.start_analysis('resample_decay_groups',dict(fit_run_id=fit_run_id,candidate_index=candidate_index,
        draws=draws,block_length=block_length,settings=settings))


@mcp.tool()
def fit_simulated_decay(fit_run_id: str, model_run_id: str, shared_decay: bool = False,
                         isotopomers: list[str] | None = None, source_j_fit_run_id: str | None = None,
                         settings: dict | None = None, t2_bounds: list[float] | None = None) -> dict:
    """Start conditional decay-only fitting of complete fixed-J transition groups.
    Inherits parent band ranges, preprocessing and group split. Optional t2_bounds
    explicitly overrides inherited bounds for bounded sensitivity checks. Model
    comes from build_isopropylamine_model; optional source J fit must match it.
    Defaults to methine+methyl, one phase/gain pair per group, equal band weighting.
    Settings: starts, max_nfev, max_evaluations, max_seconds, seed, equal_band_weight,
    phase_delay_bounds_s (optional bounded global frequency-dependent phase).
    All-scan J estimation makes validation conditional, not an untouched J test.
    """
    return jobs.start_analysis('fit_simulated_decay',dict(fit_run_id=fit_run_id,model_run_id=model_run_id,
        shared_decay=shared_decay,isotopomers=isotopomers,source_j_fit_run_id=source_j_fit_run_id,settings=settings,t2_bounds=t2_bounds))


@mcp.tool()
def fit_window_decay(fit_run_id: str, candidate_index: int = 0, width_s: float = .2,
                     hop_s: float = .04, sample_frequencies_hz: list[float] | None = None,
                     settings: dict | None = None) -> dict:
    """Start complex Hann-window refitting of a discovery FFT decay candidate.
    Inherits original preprocessing, frequency/T2* bounds and mode count. Uses
    five uniformly spaced observation frequencies by default. Freezes all fitted
    coefficients for validation and cross-checks original native FFT agreement.
    Settings: starts, max_nfev, max_evaluations, max_seconds, seed. Overlapping
    windows are correlated; no independent-window confidence interval is claimed.
    """
    return jobs.start_analysis('fit_window_decay',dict(fit_run_id=fit_run_id,candidate_index=candidate_index,
        width_s=width_s,hop_s=hop_s,sample_frequencies_hz=sample_frequencies_hz,settings=settings))


@mcp.tool()
def inspect_decay_stability(fit_run_id: str, candidate_index: int = 0,
                            group_indices: list[int] | None = None,
                            settings: dict | None = None) -> dict:
    """Start bounded per-group mean decay refits, defaulting to validation groups.
    Frequencies initialize from discovery, with original bands/T2* bounds and
    preprocessing. Returns frozen errors separately from in-sample refits,
    frequency-matched parameters, raw phase shifts and independent figures.
    Settings: starts, max_nfev, max_evaluations, max_seconds, total_seconds, seed.
    Group variation is diagnostic, not a confidence interval or physical assignment.
    """
    return jobs.start_analysis('inspect_decay_stability',dict(fit_run_id=fit_run_id,
        candidate_index=candidate_index,group_indices=group_indices,settings=settings))


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
def compute_group_averages(folder: str, groups: list[list[int]]) -> dict:
    """Start streaming means of 2..32 explicit disjoint acquisition groups.
    groups contains scan-ID lists, not index ranges. No alignment, filtering or
    trimming. Returns job_id; completed run stores hashes, membership, group
    means and scan-weighted pooled mean. Groups are not proven physical batches.
    Use separate groups for discovery and validation, never pooled held-out data.
    """
    return jobs.start_analysis('compute_group_averages', {'folder': folder, 'groups': groups})


@mcp.tool()
def fit_frequency_decay(group_run_id: str, ranges: list[list[float]],
                        discovery_groups: list[int], validation_groups: list[int],
                        t2_bounds: list[float] | None = None,
                        components: list[int] | None = None,
                        preprocessing: dict | None = None, settings: dict | None = None) -> dict:
    """Start band-only complex FFT effective T2* fitting on disjoint group means.
    ranges: sorted disjoint Hz intervals. t2_bounds: [min,max] seconds; omitted
    uses an explicitly reported exploratory interval. components: mode counts,
    default [1,2]. Group indices are zero-based and disjoint. preprocessing uses
    the explicit SG-baseline/crop recipe. Settings: starts, max_nfev,
    max_evaluations, max_seconds per fit, total_seconds, seed,
    compare_shared_decay, background, initial_frequencies_hz. Explicit initial
    frequencies require one component count and one frequency list per band;
    they initialize the search without fixing frequencies. Validation freezes discovery predictions;
    conditional gain refits are diagnostics only. No automatic physical assignment.
    """
    return jobs.start_analysis('fit_frequency_decay',dict(group_run_id=group_run_id,
        ranges=ranges,discovery_groups=discovery_groups,validation_groups=validation_groups,
        t2_bounds=t2_bounds,components=components,preprocessing=preprocessing,settings=settings))


@mcp.tool()
def inspect_decay_time_frequency(fit_run_id: str, candidate_index: int = 0,
                                 widths_s: list[float] | None = None,
                                 hop_fraction: float = .1) -> dict:
    """Start matched-window STFT and anti-aliased demodulation comparisons.
    Uses a frozen candidate from fit_frequency_decay; no refitting. Defaults to
    Hann widths 0.25/0.5/1 seconds and hop 10% of width (90% overlap). Retains
    complex acquisition-referenced phase, complete windows only. Demodulated
    arrays include FIR boundary masks. Independent plots and raw arrays saved.
    """
    return jobs.start_analysis('inspect_decay_time_frequency',dict(fit_run_id=fit_run_id,
        candidate_index=candidate_index,widths_s=widths_s,hop_fraction=hop_fraction))


@mcp.tool()
def inspect_repeat_signals(group_run_id: str, ranges: list[list[float]],
                           noise_ranges: list[list[float]], discovery_groups: list[int],
                           validation_groups: list[int], preprocessing: dict | None = None,
                           max_candidates: int = 8, snr_threshold: float = 5.,
                           frequency_tolerance_hz: float | None = None,
                           interference_ranges: list[list[float]] | None = None) -> dict:
    """Start discovery/validation peak reproducibility and measured noise checks.
    Need two or more disjoint groups per split. Explicit nonoverlapping noise
    reference bands are assumptions, not automatically signal-free. Includes
    measured disjoint-pool accumulation and masked phase plots. Labels are
    operational candidates, not significance tests or molecular identification.
    """
    return jobs.start_analysis('inspect_repeat_signals',dict(group_run_id=group_run_id,
        ranges=ranges,noise_ranges=noise_ranges,discovery_groups=discovery_groups,
        validation_groups=validation_groups,preprocessing=preprocessing,max_candidates=max_candidates,
        snr_threshold=snr_threshold,frequency_tolerance_hz=frequency_tolerance_hz,
        interference_ranges=interference_ranges))


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
    subset of methine/methyl), fixed_rates and initial_rates (isotopomer->1/s),
    rate_bounds_by_isotopomer (isotopomer->[lo,hi] in 1/s), diff_step (relative
    nonlinear finite-difference step). bin_stride=1 fits all native FFT bins.
    Returns job_id. Saves candidates,
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
    high_degradation_tolerance (default .15), initial_rates and
    rate_bounds_by_isotopomer (both isotope keys required), diff_step.
    Joint refinement warm-starts decay rates as well as J. All child run IDs are preserved.
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
