"""Spin-1/2 isotropic J simulations, exact collective-spin sectors and optional CUDA.

Hamiltonians are in rad/s. Fields are microtesla, times seconds, J values Hz.
Positive Zeeman sign, gamma-weighted x preparation/detection follow TwoD_simulation.
"""
import csv
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh
from . import storage

GAMMA = {'1H': 42.58, '13C': 10.71, '15N': -4.316}  # gamma/(2 pi), Hz/uT
_DLL_HANDLES = []


def configure_cuda():
    """Discover CUDA wheel DLLs in this interpreter; never alter system settings."""
    import sysconfig
    cache = storage.ROOT/'cupy_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR', str(cache))
    root = Path(sysconfig.get_paths()['purelib'])/'nvidia'
    if os.name == 'nt' and root.exists() and not _DLL_HANDLES:
        bins = sorted(root.glob('*/bin'))
        for folder in bins:
            _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
        os.environ['PATH'] = os.pathsep.join(str(p) for p in bins)+os.pathsep+os.environ.get('PATH', '')
        if (root/'cuda_runtime').exists():
            os.environ.setdefault('CUDA_PATH', str(root/'cuda_runtime'))


def validate(model):
    allowed = {'isotopes', 'couplings_hz', 'gamma_hz_per_ut', 'symmetry_groups'}
    if set(model) - allowed:
        raise ValueError('Unknown model keys: ' + str(set(model) - allowed))
    isotopes = model['isotopes']
    n = len(isotopes)
    if not 1 <= n <= 8 or any(s not in GAMMA for s in isotopes):
        raise ValueError('Supports 1..8 spin-1/2 nuclei: 1H, 13C, 15N. 14N is not spin-1/2.')
    j = np.asarray(model['couplings_hz'], dtype=float)
    gamma = np.array([model.get('gamma_hz_per_ut', {}).get(s, GAMMA[s]) for s in isotopes])
    if j.shape != (n, n) or not np.isfinite(j).all() or not np.isfinite(gamma).all():
        raise ValueError('Finite square J matrix and gamma values required.')
    if not np.allclose(j, j.T, atol=1e-10, rtol=0) or np.any(np.diag(j) != 0):
        raise ValueError('J matrix must be symmetric with zero diagonal; no implicit repair.')
    groups = model.get('symmetry_groups')
    if groups is None:
        groups, unused = [], set(range(n))
        while unused:
            i = min(unused)
            group = [i]
            for k in sorted(unused - {i}):
                perm = list(range(n)); perm[i], perm[k] = k, i
                if isotopes[i] == isotopes[k] and np.allclose(j, j[np.ix_(perm, perm)], atol=1e-10, rtol=0):
                    group.append(k)
            unused.difference_update(group)
            if len(group) > 1:
                groups.append(group)
    seen = set()
    for group in groups:
        if len(group) < 2 or any(type(i) is not int or not 0 <= i < n or i in seen for i in group) or len(set(group)) != len(group):
            raise ValueError('Symmetry groups require disjoint, zero-based spin indices, at least two per group.')
        seen.update(group)
        for k in group[1:]:
            i = group[0]; perm = list(range(n)); perm[i], perm[k] = k, i
            if isotopes[i] != isotopes[k] or not np.allclose(j, j[np.ix_(perm, perm)], atol=1e-10, rtol=0):
                raise ValueError('Requested symmetry is broken by isotope or coupling differences.')
    return isotopes, j, gamma, groups


def spin_operators(n):
    pauli = [np.array([[0, 1], [1, 0]])/2,
             np.array([[0, -1j], [1j, 0]])/2, np.diag([.5, -.5])]
    result = []
    for axis in pauli:
        ops = []
        for i in range(n):
            op = np.ones((1, 1), complex)
            for k in range(n):
                op = np.kron(op, axis if k == i else np.eye(2))
            ops.append(op)
        result.append(ops)
    return result


def build(model):
    isotopes, j, gamma, groups = validate(model)
    ops = spin_operators(len(isotopes)); d = 2**len(isotopes)
    h = np.zeros((d, d), complex)
    for i in range(len(isotopes)):
        for k in range(i+1, len(isotopes)):
            h += 2*np.pi*j[i, k]*sum(axis[i] @ axis[k] for axis in ops)
    magnetization = [2*np.pi*sum(g*op for g, op in zip(gamma, axis)) for axis in ops]
    return h, magnetization, ops, groups


def sectors(ops, groups, enabled):
    """Simultaneous collective S_g^2 sectors; retain every multiplicity state."""
    d = ops[0][0].shape[0]
    blocks = [(np.eye(d, dtype=complex), [])]
    if not enabled:
        return blocks
    for group in groups:
        collective = [sum(axis[i] for i in group) for axis in ops]
        casimir = sum(s @ s for s in collective)
        new = []
        for q, labels in blocks:
            a = q.conj().T @ casimir @ q
            e, v = eigh((a+a.conj().T)/2, driver='evr')
            labels_e = np.round(e, 7)
            for value in np.unique(labels_e):
                new.append((q @ v[:, labels_e == value], labels + [float((np.sqrt(1+4*value)-1)/2)]))
        blocks = new
    return blocks


def gpu_probe():
    try:
        configure_cuda()
        import cupy as cp
        x = cp.asarray([[2., 1.], [1., 2.]], dtype=cp.complex128)
        eig = cp.linalg.eigh(x)[0]
        check = cp.asnumpy(cp.exp(x) @ x)
        cp.cuda.Stream.null.synchronize()
        if not np.allclose(cp.asnumpy(eig), [1, 3]) or not np.isfinite(check).all():
            raise RuntimeError('GPU numerical smoke test failed.')
        props = cp.cuda.runtime.getDeviceProperties(0)
        return {'available': True, 'cupy': cp.__version__, 'device': props['name'].decode(),
                'free_bytes': int(cp.cuda.runtime.memGetInfo()[0]), 'precision': 'complex128'}
    except Exception as exc:
        return {'available': False, 'reason': str(exc)}


def inspect_simulation_backend(*, record, directory, cancel, progress):
    return {'cpu': 'NumPy complex128', 'gpu': gpu_probe(),
            'note': 'GPU accelerates block eigensolvers, pulse propagation and transition sums. Setup/symmetry is CPU; small blocks may be faster on CPU.'}


def import_spin_model(folder, *, record, directory, cancel, progress):
    path = Path(folder).resolve(strict=True)
    config = storage.read_json(storage.PROJECT/'local.json')
    if not any(path.is_relative_to(Path(p).resolve()) for p in config.get('model_roots', [])):
        raise ValueError('Model folder is outside configured read-only model_roots.')
    source = path/'structure.csv'
    raw = source.read_bytes()
    rows = list(csv.reader(raw.decode('utf-8-sig').splitlines()))
    model = {'isotopes': [s.strip() for s in rows[0]], 'couplings_hz': [[float(v) for v in row] for row in rows[1:] if row]}
    hashes = {'structure.csv': hashlib.sha256(raw).hexdigest()}
    if (path/'symmetry.csv').exists():
        sym = (path/'symmetry.csv').read_bytes()
        groups = []
        for row in csv.reader(sym.decode('utf-8-sig').splitlines()):
            if not row: continue
            group = [int(s)-1 for s in ' '.join(row[1:]).replace(',', ' ').split()]
            if row[0].strip() != f'S{len(group)}':
                raise ValueError('Only legacy S_n permutation groups are supported.')
            groups.append(group)
        model['symmetry_groups'] = groups
        hashes['symmetry.csv'] = hashlib.sha256(sym).hexdigest()
    validate(model)
    storage.write_json(directory/'model.json', model)
    return {'model': model, 'source_folder': str(path), 'source_sha256': hashes,
            'index_conversion': 'Legacy CSV one-based indices converted to zero-based JSON.'}


def analyze_spin_symmetry(model, *, record, directory, cancel, progress):
    h, m, ops, groups = build(model)
    blocks = sectors(ops, groups, True)
    commutators = []
    for group in groups:
        c = sum(sum(axis[i] for i in group) @ sum(axis[i] for i in group) for axis in ops)
        commutators.append(max(float(np.linalg.norm(c @ a-a @ c)/max(1., np.linalg.norm(a))) for a in [h]+m))
    return {'groups_zero_based': groups, 'dimension': len(h),
            'sectors': [{'dimension': q.shape[1], 'group_total_spins': label} for q,label in blocks],
            'commutator_relative_norms': commutators,
            'matrix_cube_ratio': float(len(h)**3/sum(q.shape[1]**3 for q,_ in blocks)),
            'note': 'Exact collective-spin sectors, not full irreducible permutation decomposition. All multiplicities retained; cube ratio is not a measured speedup.'}


def compute(model, settings, cancel=lambda: False, progress=lambda n,total: None):
    allowed = {'sequence', 'npoints', 'sampling_rate_hz', 't1_points', 't1_step_s', 'tm_s',
               't2star_s', 'field1_ut', 'field2_ut', 'pulse_field_ut', 'pulse_duration_s', 'backend', 'symmetry'}
    if set(settings)-allowed:
        raise ValueError('Unknown simulation settings: ' + str(set(settings)-allowed))
    s = dict(sequence='fid', npoints=1024, sampling_rate_hz=1000., t1_points=64,
             t1_step_s=.0017, tm_s=.0017, t2star_s=1., field1_ut=0., field2_ut=0.,
             pulse_field_ut=[0., 0., 0.], pulse_duration_s=0., backend='cpu', symmetry=True)
    s.update(settings)
    if s['sequence'] not in ('fid', '2d', 'mq') or s['backend'] not in ('cpu', 'gpu') or type(s['symmetry']) is not bool:
        raise ValueError('sequence: fid/2d/mq; backend: cpu/gpu; symmetry: boolean.')
    for key in ('npoints', 't1_points'):
        if type(s[key]) is not int or not 2 <= s[key] <= 131072:
            raise ValueError('Integer point counts must be between 2 and 131072.')
    for key in ('sampling_rate_hz', 't1_step_s', 't2star_s', 'tm_s', 'pulse_duration_s', 'field1_ut', 'field2_ut'):
        if not np.isfinite(s[key]) or (key not in ('field1_ut', 'field2_ut') and s[key] < 0):
            raise ValueError('Invalid finite time, field or sample rate: '+key)
    if min(s['sampling_rate_hz'], s['t1_step_s'], s['t2star_s']) <= 0:
        raise ValueError('Sample rate, t1 step and T2* must be positive.')
    pulse = np.asarray(s['pulse_field_ut'], float)
    if pulse.shape != (3,) or not np.isfinite(pulse).all():
        raise ValueError('pulse_field_ut must be finite [Bx, By, Bz].')
    rows = 1 if s['sequence'] == 'fid' else s['t1_points']
    if rows*s['npoints'] > 4_000_000:
        raise ValueError('Output limited to 4 million complex samples; choose an explicit smaller grid.')
    start = time.perf_counter()
    h, mag, ops, groups = build(model)
    blocklist = sectors(ops, groups, s['symmetry'])
    if rows*max(q.shape[1]**2 for q,_ in blocklist) > 16_000_000:
        raise ValueError('Transition weights exceed 256 MB; reduce t1_points or use validated symmetry.')
    xp, device = np, {'backend': 'cpu'}
    if s['backend'] == 'gpu':
        device = gpu_probe()
        if not device['available']:
            raise RuntimeError('GPU explicitly requested but unavailable: '+device['reason'])
        import cupy as xp
        device['backend'] = 'gpu'
    to_cpu = (lambda x: x) if xp is np else xp.asnumpy
    # Use the robust MRRR driver on CPU; divide-and-conquer failed for a
    # finite, highly degenerate 24-dimensional ethanol pulse sector on this LAPACK build.
    def diagonalize(a):
        a = (a+a.conj().T)/2
        if xp is np:
            return eigh(a, driver='evr')
        from cupyx import errstate
        with errstate(linalg='raise'):
            return xp.linalg.eigh(a)
    t2 = np.arange(s['npoints'])/s['sampling_rate_hz']
    t1 = np.arange(rows)*s['t1_step_s']
    result = np.zeros((rows, len(t2)), complex)
    h1, h2 = h+s['field1_ut']*mag[2], h+s['field2_ut']*mag[2]
    hp = h1+sum(b*m for b,m in zip(pulse, mag))
    # Verify that ALL sequence operators preserve the chosen sectors before discarding cross blocks.
    qfull = np.concatenate([q for q,_ in blocklist], axis=1)
    sizes = [q.shape[1] for q,_ in blocklist]
    for a in (h1, h2, hp, mag[0]):
        transformed = qfull.conj().T @ a @ qfull
        off = transformed.copy(); offset = 0
        for size in sizes:
            off[offset:offset+size, offset:offset+size] = 0; offset += size
        if np.linalg.norm(off) > 1e-9*max(1., np.linalg.norm(a)):
            raise ValueError('Sequence breaks requested symmetry; use symmetry=false.')
    max_f1 = max_f2 = 0.
    for bi, (q, labels) in enumerate(blocklist):
        if cancel(): raise InterruptedError('Simulation cancelled.')
        block = lambda a: xp.asarray(q.conj().T @ a @ q)
        e1, v1 = diagonalize(block(h1)); e2, v2 = diagonalize(block(h2))
        ep, vp = diagonalize(block(hp))
        up = (vp*xp.exp(-1j*ep*s['pulse_duration_s'])) @ vp.conj().T
        rho0, obs = block(mag[0]), block(mag[0])
        if s['sequence'] == 'mq':
            um = (v1*xp.exp(-1j*e1*s['tm_s'])) @ v1.conj().T
            exc = up @ um @ up
        else:
            exc = up
        rho = exc @ rho0 @ exc.conj().T
        r1 = v1.conj().T @ rho @ v1
        transfer = v2.conj().T @ up @ v1
        obs2 = v2.conj().T @ obs @ v2
        delta1, delta2 = e1[:, None]-e1[None, :], e2[:, None]-e2[None, :]
        max_f1 = max(max_f1, float(to_cpu(xp.max(xp.abs(delta1))))/(2*np.pi))
        max_f2 = max(max_f2, float(to_cpu(xp.max(xp.abs(delta2))))/(2*np.pi))
        weights = xp.empty((rows, len(e2)**2), dtype=xp.complex128)
        for i, ti in enumerate(t1):
            if cancel(): raise InterruptedError('Simulation cancelled.')
            if s['sequence'] == 'fid':
                r2 = v2.conj().T @ rho @ v2
            else:
                evolved = r1*xp.exp(-1j*delta1*ti)
                r2 = transfer @ evolved @ transfer.conj().T
            weights[i] = (r2*obs2.T).ravel()
        # Bound phase and weight workspaces instead of allocating a time x D x D cube.
        chunk = max(1, min(256, 1_000_000//(len(e2)**2)))
        for k in range(0, len(t2), chunk):
            if cancel(): raise InterruptedError('Simulation cancelled.')
            phase = xp.exp(-1j*delta2.reshape(-1, 1)*xp.asarray(t2[k:k+chunk]))
            result[:, k:k+chunk] += to_cpu(weights @ phase)
            progress(bi*len(t2)+min(k+chunk, len(t2)), len(blocklist)*len(t2))
    result *= np.exp(-t2/s['t2star_s'])[None, :]
    if not np.isfinite(result).all():
        raise FloatingPointError('Simulation produced nonfinite values; no completed result published.')
    warnings = ['Phenomenological T2* decay is applied only along t2, as in legacy fid2d; no t1 relaxation.',
                'Initial state and detection are unnormalized gamma-weighted Ix; amplitudes are arbitrary units.']
    if max_f2 >= s['sampling_rate_hz']/2 or (rows > 1 and max_f1 >= .5/s['t1_step_s']):
        warnings.append('Some Hamiltonian transition gaps exceed Nyquist; inspect sampling before interpretation (bound includes dark transitions).')
    return result, t1, t2, {'settings': s, 'device': device, 'groups_zero_based': groups,
             'sector_dimensions': sizes, 'elapsed_compute_s': time.perf_counter()-start, 'warnings': warnings,
             'gamma_hz_per_ut': dict(zip(model['isotopes'], validate(model)[2].tolist()))}


def simulate_spin_dynamics(model, settings=None, *, record, directory, cancel, progress):
    from .analysis import plot
    from matplotlib.figure import Figure
    fid, t1, t2, report = compute(model, settings or {}, cancel, progress)
    spectrum = np.fft.fftshift(np.fft.fft(fid, axis=1), axes=1)/len(t2)
    f2 = np.fft.fftshift(np.fft.fftfreq(len(t2), 1/report['settings']['sampling_rate_hz']))
    plot(directory/'fid.png', [(t2, fid[0].real, 'Real'), (t2, fid[0].imag, 'Imaginary')],
         't2 (s)', 'Signal (a.u.)', 'Simulated FID at t1 = 0')
    plot(directory/'spectrum.png', [(f2, abs(spectrum[0]), 'Magnitude')],
         'Frequency (Hz)', '|FFT| / samples (a.u.)', 'Simulated spectrum at t1 = 0')
    arrays = dict(fid=fid, t1_s=t1, t2_s=t2, f2_hz=f2, direct_spectrum=spectrum)
    if len(t1) > 1:
        spec2 = np.fft.fftshift(np.fft.fft(spectrum, axis=0), axes=0)/len(t1)
        f1 = np.fft.fftshift(np.fft.fftfreq(len(t1), report['settings']['t1_step_s']))
        arrays.update(spectrum2d=spec2, f1_hz=f1)
        fig = Figure(figsize=(9, 7), layout='constrained'); ax = fig.add_subplot(111)
        im = ax.pcolormesh(f2, f1, abs(spec2), shading='auto')
        ax.set(xlabel='F2 (Hz)', ylabel='F1 (Hz)', title='Simulated 2D magnitude spectrum')
        fig.colorbar(im, ax=ax, label='Magnitude (a.u.)'); fig.savefig(directory/'spectrum2d.png', dpi=150)
        # Independent display-only logarithmic view makes weak correlations visible
        # when the retained physical DC component dominates the linear color scale.
        from matplotlib.colors import LogNorm
        magnitude = abs(spec2); peak = float(magnitude.max())
        if peak > 0:
            floor = peak*1e-6
            fig = Figure(figsize=(9, 7), layout='constrained'); ax = fig.add_subplot(111)
            im = ax.pcolormesh(f2, f1, np.maximum(magnitude, floor), shading='auto', norm=LogNorm(floor, peak))
            ax.set(xlabel='F2 (Hz)', ylabel='F1 (Hz)', title='Simulated 2D magnitude (log color; floor = peak x 1e-6)')
            fig.colorbar(im, ax=ax, label='Magnitude (a.u., log scale)')
            fig.savefig(directory/'spectrum2d_log.png', dpi=150)
            report['log_display_floor'] = floor
    np.savez_compressed(directory/'simulation.npz', **arrays)
    report['arrays_sha256'] = hashlib.sha256((directory/'simulation.npz').read_bytes()).hexdigest()
    report['model'] = model
    return report
