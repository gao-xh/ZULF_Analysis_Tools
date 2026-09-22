"""Disjoint acquisition means; never silently align, trim or reweight raw FIDs."""
import hashlib
import time
import numpy as np
from . import data, storage


def compute_group_averages(folder, groups, *, record, directory, cancel, progress):
    """Stream explicitly specified disjoint scan groups in a single read pass."""
    from .analysis import plot
    started = time.perf_counter()
    if not isinstance(groups, list) or not 2 <= len(groups) <= 32:
        raise ValueError('Supply 2..32 explicitly selected disjoint scan groups.')
    selected = []
    for group in groups:
        if not isinstance(group, list) or not group or any(type(i) is not int or i < 0 for i in group):
            raise ValueError('Each group needs nonnegative integer scan IDs.')
        selected.extend(group)
    if len(set(selected)) != len(selected):
        raise ValueError('Scan groups must be disjoint, with no duplicate scan IDs.')
    manifest = data.inventory(folder, selected, cancel, progress)
    points = manifest['points']; fs = manifest['sampling_rate_hz']
    memory_bytes = len(groups)*points*8
    if memory_bytes > 512*1024**2:
        raise ValueError('Group accumulators exceed 512 MiB; use fewer groups.')
    assignment = {scan: index for index, group in enumerate(groups) for scan in group}
    totals = np.zeros((len(groups), points), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=int)
    for n, row in enumerate(manifest['scans'], 1):
        if cancel():
            raise InterruptedError('Group averaging cancelled.')
        values, sha = data.read_scan(row)
        row['sha256'] = sha
        g = assignment[row['scan_id']]
        row['group_index'] = g
        totals[g] += values
        counts[g] += 1
        if n % 50 == 0:
            progress(n, len(selected))
    current = data.inventory(folder, selected, cancel)
    if current['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Source metadata changed during group averaging.')
    means = totals/counts[:, None]
    pooled = totals.sum(axis=0)/counts.sum()
    destination = directory/'group_averages.npz'
    np.savez_compressed(destination, means=means, counts=counts, pooled=pooled)
    storage.write_json(directory/'sources.json', manifest)
    storage.write_json(directory/'groups.json', groups)
    t = np.arange(points)/fs
    stride = max(1, points//12000)
    plot(directory/'group_fids.png', [(t[::stride], x[::stride], f'Group {i+1} (n={counts[i]})')
         for i, x in enumerate(means)], 'Recorded time (s)', 'ADC amplitude',
         'Unaligned acquisition-group means (display sampled)')
    plot(directory/'pooled_fid.png', [(t[::stride], pooled[::stride], 'Scan-weighted mean')],
         'Recorded time (s)', 'ADC amplitude', 'Pooled FID (display sampled)')
    for name, mask in [('early', t < .2), ('tail', t >= max(0, t[-1]-2))]:
        plot(directory/f'{name}_fid.png', [(t[mask], pooled[mask], 'Scan-weighted mean')],
             'Recorded time (s)', 'ADC amplitude', f'Pooled FID: {name}')
    return {'groups': groups, 'group_counts': counts.tolist(), 'scan_count': len(selected),
            'sampling_rate_hz': fs, 'points': points, 'time_origin_s': 0.,
            'decoder': data.DECODER, 'dataset_fingerprint': manifest['fingerprint'],
            'arrays_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'sources_sha256': hashlib.sha256((directory/'sources.json').read_bytes()).hexdigest(),
            'accumulator_bytes': memory_bytes, 'elapsed_s': time.perf_counter()-started,
            'warnings': ['No filtering, alignment, first-point deletion or individual-scan fitting.',
                         'Groups are explicit acquisition subsets, not verified physical batches.',
                         'Disjoint membership does not prove independence of instrument drift.',
                         'Pooled mean is scan-count weighted; never use pooled data as held-out validation.']}


def load_group_averages(run_id):
    """Verify the saved arrays and group manifest before downstream analysis."""
    result = storage.get_result(run_id)
    if result['operation'] != 'compute_group_averages':
        raise ValueError('Expected a completed compute_group_averages run.')
    source = storage.artifact(run_id, 'group_averages.npz')
    if hashlib.sha256(source.read_bytes()).hexdigest() != result['arrays_sha256']:
        raise ValueError('Group-average artifact changed after creation.')
    manifest = storage.artifact(run_id, 'sources.json')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != result['sources_sha256']:
        raise ValueError('Group source manifest changed after creation.')
    with np.load(source, allow_pickle=False) as arrays:
        means, counts = arrays['means'].copy(), arrays['counts'].copy()
    if means.shape != (len(result['groups']), result['points']) or not np.isfinite(means).all():
        raise ValueError('Invalid stored group arrays.')
    if counts.tolist() != result['group_counts']:
        raise ValueError('Group counts disagree with manifest.')
    return means, counts, result
