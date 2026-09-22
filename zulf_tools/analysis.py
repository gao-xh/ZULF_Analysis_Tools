"""Deterministic operations returning manifests and independent figure artifacts."""
import hashlib
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from scipy.signal import savgol_filter, find_peaks
from . import data, storage


def plot(path, traces, xlabel, ylabel, title):
    fig = Figure(figsize=(10, 4.6), layout='constrained')
    ax = fig.add_subplot(111)
    for x, y, label in traces:
        ax.plot(x, y, linewidth=.8, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
    ax.grid(alpha=.2)
    if len(traces) > 1:
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)


def inspect_dataset(folder, scan_ids=None, *, record, directory, cancel, progress):
    manifest = data.inventory(folder, scan_ids, cancel, progress)
    storage.write_json(directory/'sources.json', manifest)
    warnings = []
    if manifest['points'] != manifest['declared_points']:
        warnings.append('Configured and decoded sample counts differ. No first-point deletion or correction was applied.')
    if manifest['missing_per_scan_ini']:
        warnings.append('Some per-scan INI files are absent; these scans use 0.ini sampling rate.')
    sample = []
    for i in sorted(set([0, len(manifest['scans'])//2, len(manifest['scans'])-1])):
        values, sha = data.read_scan(manifest['scans'][i])
        sample.append({'scan_id': manifest['scans'][i]['scan_id'], 'sha256': sha,
                       'minimum': float(values.min()), 'maximum': float(values.max()),
                       'first_16_samples': values[:16].tolist()})
    auxiliary = []
    for name in ('halp_compiled.npy', 'sampling_rate_compiled.npy', 'acq_time_compiled.npy'):
        path = Path(manifest['folder'])/name
        if path.exists():
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            auxiliary.append({'name': name, 'shape': list(array.shape), 'dtype': str(array.dtype),
                              'scalar': float(array) if array.ndim == 0 else None})
    return {'scan_count': len(manifest['scans']), 'sampling_rate_hz': manifest['sampling_rate_hz'],
            'decoded_points': manifest['points'], 'declared_points': manifest['declared_points'],
            'duration_s': manifest['points']/manifest['sampling_rate_hz'],
            'float64_array_gib': len(manifest['scans'])*manifest['points']*8/1024**3,
            'dataset_fingerprint': manifest['fingerprint'], 'decoder': data.DECODER,
            'sample_checks': sample, 'auxiliary_arrays': auxiliary, 'warnings': warnings}


def compute_average(folder, scan_ids=None, *, record, directory, cancel, progress):
    manifest = data.inventory(folder, scan_ids, cancel, progress)
    total = np.zeros(manifest['points'], dtype=np.float64)
    for n, row in enumerate(manifest['scans'], 1):
        if cancel():
            raise InterruptedError('Average cancelled.')
        values, sha = data.read_scan(row)
        row['sha256'] = sha
        total += values
        if n % 50 == 0:
            progress(n, len(manifest['scans']))
    # Reject metadata changes during computation instead of mixing epochs.
    current = data.inventory(folder, scan_ids, cancel)
    if current['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Source metadata changed during averaging; discard this run and retry.')
    average = total / len(manifest['scans'])
    np.save(directory/'average.npy', average, allow_pickle=False)
    storage.write_json(directory/'sources.json', manifest)
    fs = manifest['sampling_rate_hz']
    time = np.arange(len(average))/fs
    stride = max(1, len(average)//12000)
    plot(directory/'average_fid.png', [(time[::stride], average[::stride], 'Mean FID')],
         'Recorded time (s)', 'ADC amplitude', 'Coherent average · full record (display sampled)')
    early = min(len(average), max(32, math.ceil(.2*fs)))
    plot(directory/'early_fid.png', [(time[:early], average[:early], 'Mean FID')],
         'Recorded time (s)', 'ADC amplitude', 'Coherent average · first 0.2 s')
    comparison = {'status': 'No compiled reference present'}
    path = Path(manifest['folder'])/'halp_compiled.npy'
    if path.exists() and path.stat().st_size < 64*1024**2:
        other = np.load(path, allow_pickle=False)
        comparison = {'status': 'Reference comparison only; origin is not assumed', 'points': int(other.size),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if other.ndim == 1 and np.isfinite(other).all():
            if len(other) == len(average)-1:
                comparison.update(rmse_after_dropping_first=float(np.sqrt(np.mean((average[1:]-other)**2))),
                                  rmse_after_dropping_last=float(np.sqrt(np.mean((average[:-1]-other)**2))))
            elif len(other) == len(average):
                comparison['rmse_same_length'] = float(np.sqrt(np.mean((average-other)**2)))
    return {'scan_count': len(manifest['scans']), 'sampling_rate_hz': fs,
            'points': len(average), 'time_origin_s': 0., 'decoder': data.DECODER,
            'dataset_fingerprint': manifest['fingerprint'], 'compiled_reference': comparison,
            'average_sha256': hashlib.sha256((directory/'average.npy').read_bytes()).hexdigest(),
            'warnings': ['Coherent averaging assumes sufficiently stable phase and frequency.',
                         'Raw first point retained. No alignment, trimming, normalization or filtering applied.']}


def recipe(values, fs, specification):
    allowed = {'label', 'start_s', 'end_s', 'sg_window', 'sg_order', 'remove_mean'}
    if set(specification)-allowed:
        raise ValueError(f'Unknown preprocessing options: {set(specification)-allowed}')
    start = float(specification.get('start_s', 0))
    end = float(specification.get('end_s', len(values)/fs))
    if not np.isfinite([start,end]).all() or not 0 <= start < end <= len(values)/fs:
        raise ValueError('Invalid recorded-time interval.')
    first, last = math.ceil(start*fs), min(len(values), math.ceil(end*fs))
    if last-first < 32:
        raise ValueError('Retain at least 32 samples.')
    window, order = specification.get('sg_window', 0), specification.get('sg_order', 2)
    if type(window) is not int or type(order) is not int or order < 0 or window < 0:
        raise ValueError('SG window/order must be nonnegative integers.')
    if window and (window % 2 == 0 or window <= order or window > len(values)):
        raise ValueError('SG window must be odd, greater than order and no longer than the original FID.')
    if type(specification.get('remove_mean', True)) is not bool:
        raise ValueError('remove_mean must be a boolean.')
    baseline = savgol_filter(values, window, order, mode='mirror') if window else np.zeros_like(values)
    processed = (values-baseline)[first:last].copy()
    offset = float(processed.mean()) if specification.get('remove_mean', True) else 0.
    processed -= offset
    parameters = dict(specification, start_sample=first, stop_sample=last,
                      actual_start_s=first/fs, actual_stop_exclusive_s=last/fs,
                      sg_window=window, sg_order=order, removed_constant=offset,
                      order_of_operations='SG baseline subtraction on full FID, crop, optional mean removal')
    return processed, baseline, parameters


def compare_preprocessing(average_run_id, variants, *, record, directory, cancel, progress):
    parent = storage.get_result(average_run_id)
    if parent['operation'] != 'compute_average':
        raise ValueError('average_run_id must refer to compute_average.')
    if not isinstance(variants, list) or not 1 <= len(variants) <= 12:
        raise ValueError('Supply 1 to 12 explicit preprocessing variants.')
    source = storage.artifact(average_run_id, 'average.npy')
    if hashlib.sha256(source.read_bytes()).hexdigest() != parent['average_sha256']:
        raise ValueError('Average artifact changed after creation.')
    values = np.load(source, allow_pickle=False)
    fs = parent['sampling_rate_hz']
    records = []
    for i, specification in enumerate(variants):
        if cancel():
            raise InterruptedError('Comparison cancelled.')
        processed, baseline, parameters = recipe(values, fs, specification)
        label = str(specification.get('label', f'Variant {i+1}'))
        frequency = np.fft.rfftfreq(len(processed), 1/fs)
        spectrum = np.fft.rfft(processed)/len(processed)
        np.savez(directory/f'variant_{i}.npz', fid=processed, frequency=frequency, spectrum=spectrum)
        time = np.arange(len(processed))/fs+parameters['actual_start_s']
        stride = max(1, len(time)//12000)
        plot(directory/f'variant_{i}_fid.png', [(time[::stride], processed[::stride], label)],
             'Recorded time (s)', 'ADC amplitude', label)
        plot(directory/f'variant_{i}_spectrum.png', [(frequency, abs(spectrum), label)],
             'Frequency (Hz)', '|FFT| / retained sample count', label)
        if parameters['sg_window']:
            x = np.arange(len(values))/fs
            plot(directory/f'variant_{i}_baseline.png', [(x[::stride], values[::stride], 'Original mean'),
                                                       (x[::stride], baseline[::stride], 'SG baseline')],
                 'Recorded time (s)', 'ADC amplitude', label+' · baseline estimate')
        records.append({'index': i, 'label': label, 'parameters': parameters,
                        'points': len(processed), 'frequency_bin_hz': fs/len(processed),
                        'sha256': hashlib.sha256((directory/f'variant_{i}.npz').read_bytes()).hexdigest()})
        progress(i+1, len(variants))
    return {'parent_run_id': average_run_id, 'sampling_rate_hz': fs, 'variants': records,
            'warnings': ['Exploratory preprocessing comparison; no variant is automatically selected as physically correct.',
                         'SG removes a fitted baseline; it is not a direct smoothed-FID replacement.',
                         'Different crop lengths change Fourier bin spacing and amplitude normalization.']}


def inspect_frequency_ranges(comparison_run_id, ranges, max_candidates=10, *, record, directory, cancel, progress):
    parent = storage.get_result(comparison_run_id)
    if parent['operation'] != 'compare_preprocessing':
        raise ValueError('comparison_run_id must refer to compare_preprocessing.')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 50:
        raise ValueError('max_candidates must be 1 to 50.')
    if not isinstance(ranges, list) or not 1 <= len(ranges) <= 10:
        raise ValueError('Supply 1 to 10 [low_hz, high_hz] ranges.')
    for pair in ranges:
        if len(pair) != 2 or not np.isfinite(pair).all() or not 0 <= pair[0] < pair[1] <= parent['sampling_rate_hz']/2:
            raise ValueError('Each range must lie within 0 to Nyquist and increase.')
    report = []
    for v in parent['variants']:
        if cancel():
            raise InterruptedError('Range inspection cancelled.')
        path = storage.artifact(comparison_run_id, f"variant_{v['index']}.npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != v['sha256']:
            raise ValueError('Preprocessing artifact changed after creation.')
        with np.load(path, allow_pickle=False) as arrays:
            frequency, spectrum = arrays['frequency'], abs(arrays['spectrum'])
        for r, (lo, hi) in enumerate(ranges):
            mask = (frequency >= lo) & (frequency <= hi)
            x, y = frequency[mask], spectrum[mask]
            if len(x) < 3:
                raise ValueError('Range contains fewer than three native Fourier bins.')
            peaks, props = find_peaks(y, prominence=0)
            selected = np.argsort(props['prominences'])[::-1][:max_candidates]
            candidates = [{'frequency_hz': float(x[peaks[j]]), 'amplitude': float(y[peaks[j]]),
                           'prominence': float(props['prominences'][j])} for j in selected]
            plot(directory/f"variant_{v['index']}_range_{r}.png", [(x, y, v['label'])],
                 'Frequency (Hz)', '|FFT| / retained sample count', f"{v['label']} · {lo:g}–{hi:g} Hz")
            report.append({'variant': v['index'], 'label': v['label'], 'range_hz': [lo,hi],
                           'bins': len(x), 'minimum': float(y.min()), 'maximum': float(y.max()),
                           'local_maxima_count': len(peaks), 'candidates': candidates})
    return {'parent_run_id': comparison_run_id, 'ranges': report,
            'warnings': ['Local maxima are ranked by prominence with no significance threshold; noise peaks are included.',
                         'Range endpoints cannot be identified as peaks. No substance or decay assignments are made.']}


from .simulation import (inspect_simulation_backend, import_spin_model, analyze_spin_symmetry,
                         simulate_spin_dynamics)

OPERATIONS = {f.__name__: f for f in (inspect_dataset, compute_average, compare_preprocessing,
              inspect_frequency_ranges, inspect_simulation_backend, import_spin_model,
              analyze_spin_symmetry, simulate_spin_dynamics)}


def execute(operation, parameters, cancel=lambda: False, progress=lambda n, total: None):
    if operation not in OPERATIONS:
        raise ValueError('Unknown analysis operation.')
    record, directory = storage.begin(operation, parameters)
    try:
        result = OPERATIONS[operation](**parameters, record=record, directory=directory, cancel=cancel, progress=progress)
        if cancel():
            raise InterruptedError('Cancelled before result publication.')
        return storage.complete(record, **result)
    except Exception as exc:
        record.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed', error=str(exc))
        storage.write_json(directory/'result.json', record)
        raise
