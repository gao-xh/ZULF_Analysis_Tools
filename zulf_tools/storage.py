"""Controlled artifact storage and atomic, finite JSON records."""
import hashlib
import json
import os
from pathlib import Path
import platform
import uuid
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / '.analysis'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_folder(folder):
    path = Path(folder).resolve(strict=True)
    config = read_json(PROJECT/'local.json') if (PROJECT/'local.json').exists() else {}
    roots = [Path(p).resolve() for p in config.get('input_roots', [])]
    if not roots or not any(path.is_relative_to(root) for root in roots):
        raise ValueError('Source is outside configured read-only input_roots in local.json.')
    if not path.is_dir() or path == ROOT or path.is_relative_to(ROOT):
        raise ValueError('Source must be an experimental directory outside artifact storage.')
    return path


def artifact(run_id, name='result.json'):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise ValueError('Invalid run ID.')
    return ROOT/'runs'/run_id/name


def begin(operation, parameters):
    import numpy, scipy, matplotlib
    run_id = uuid.uuid4().hex
    path = artifact(run_id)
    path.parent.mkdir(parents=True)
    record = {'run_id': run_id, 'operation': operation, 'parameters': parameters,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'versions': {'tools': '0.1.0', 'python': platform.python_version(),
                           'numpy': numpy.__version__, 'scipy': scipy.__version__,
                           'matplotlib': matplotlib.__version__},
              'status': 'running'}
    record['implementation_sha256'] = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in sorted((PROJECT/'zulf_tools').glob('*.py'))})
    write_json(path, record)
    return record, path.parent


def complete(record, **results):
    record.update(results, status='complete')
    directory = artifact(record['run_id']).parent
    record['artifacts'] = [str(p.resolve()) for p in sorted(directory.iterdir()) if p.is_file()]
    write_json(artifact(record['run_id']), record)
    return record


def get_result(run_id):
    record = read_json(artifact(run_id))
    if record['status'] != 'complete':
        raise ValueError('Requested analysis did not complete.')
    return record
