"""Explicit legacy NMRduino decoding, metadata checks and streaming reads."""
import configparser
import hashlib
import numpy as np
from .storage import source_folder, digest

DECODER = 'legacy_signal_selection_v1: reverse bytes; little-endian int16; [20:-2]; reverse samples'


def decode(payload):
    if len(payload) % 2 or len(payload) < 2*(22+32):
        raise ValueError('Invalid DAT length: need even bytes and at least 32 decoded samples.')
    return np.frombuffer(payload[::-1], dtype='<i2')[20:-2][::-1].astype(np.float64)


def settings(path):
    payload = path.read_bytes()
    parser = configparser.ConfigParser(interpolation=None)
    try:
        text = payload.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = payload.decode('latin-1')
    parser.read_string(text)
    section = next((s for s in parser.sections() if s.lower() == 'nmrduino'), None)
    if section is None:
        raise ValueError(f'{path.name}: missing NMRduino section.')
    fs = float(parser.get(section, 'SampleRate'))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'{path.name}: invalid sampling rate.')
    points = parser.getint(section, 'NumberOfSamples', fallback=0)
    return fs, points, hashlib.sha256(payload).hexdigest()


def inventory(folder, scan_ids=None, cancel=lambda: False, progress=lambda n, total: None):
    folder = source_folder(folder)
    fs, declared, config_hash = settings(folder/'0.ini')
    files = sorted((p for p in folder.glob('*.dat') if p.stem.isdecimal()), key=lambda p: int(p.stem))
    ids = [int(p.stem) for p in files]
    if not files or len(ids) != len(set(ids)):
        raise ValueError('No numbered DAT scans, or duplicate numeric scan IDs.')
    if scan_ids is not None:
        if not scan_ids or any(type(i) is not int for i in scan_ids) or len(set(scan_ids)) != len(scan_ids):
            raise ValueError('scan_ids must contain distinct integer IDs; null selects all.')
        missing = set(scan_ids)-set(ids)
        if missing:
            raise ValueError(f'Scan IDs not found: {sorted(missing)[:20]}')
        selected = set(scan_ids)
        files = [p for p in files if int(p.stem) in selected]
    rows, lengths, missing_configs = [], set(), 0
    for n, path in enumerate(files):
        if cancel():
            raise InterruptedError('Analysis cancelled.')
        stat = path.stat()
        if stat.st_size % 2 or stat.st_size < 108:
            raise ValueError(f'{path.name}: invalid file length.')
        points = stat.st_size//2-22
        lengths.add(points)
        ini = path.with_suffix('.ini')
        ini_hash = None
        if ini.exists():
            rate, _, ini_hash = settings(ini)
            if rate != fs:
                raise ValueError(f'{ini.name}: sampling rate differs from 0.ini.')
        else:
            missing_configs += 1
        rows.append({'scan_id': int(path.stem), 'path': str(path), 'bytes': stat.st_size,
                     'mtime_ns': stat.st_mtime_ns, 'ini_sha256': ini_hash})
        if n % 100 == 0:
            progress(n, len(files))
    if len(lengths) != 1:
        raise ValueError('Selected scans have unequal decoded sample counts.')
    points = lengths.pop()
    manifest = {'folder': str(folder), 'sampling_rate_hz': fs, 'points': points,
                'declared_points': declared, 'decoder': DECODER, 'config_sha256': config_hash,
                'scans': rows, 'missing_per_scan_ini': missing_configs}
    manifest['fingerprint'] = digest(manifest)
    return manifest


def read_scan(row):
    from pathlib import Path
    path = Path(row['path'])
    before = path.stat()
    if before.st_size != row['bytes'] or before.st_mtime_ns != row['mtime_ns']:
        raise ValueError(f'{path.name}: source changed after inspection.')
    payload = path.read_bytes()
    after = path.stat()
    if after.st_mtime_ns != before.st_mtime_ns or len(payload) != before.st_size:
        raise ValueError(f'{path.name}: source changed while reading.')
    return decode(payload), hashlib.sha256(payload).hexdigest()
