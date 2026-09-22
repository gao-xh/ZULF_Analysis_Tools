"""Persistent local jobs with cooperative cancellation and progress records."""
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from . import storage


def directory(job_id):
    storage.artifact(job_id)  # Validate opaque IDs before resolving paths.
    return storage.ROOT/'jobs'/job_id


def start_analysis(operation, parameters):
    from .analysis import OPERATIONS
    import inspect
    if operation not in OPERATIONS:
        raise ValueError('Unknown analysis operation.')
    inspect.signature(OPERATIONS[operation]).bind(**parameters, record=None, directory=None, cancel=None, progress=None)
    job_id = uuid.uuid4().hex
    path = directory(job_id)
    path.mkdir(parents=True)
    storage.write_json(path/'request.json', {'operation': operation, 'parameters': parameters})
    storage.write_json(path/'status.json', {'job_id': job_id, 'status': 'queued', 'progress': 0,
                                           'created_utc': datetime.now(timezone.utc).isoformat()})
    with (path/'worker.log').open('wb') as log:
        process = subprocess.Popen([sys.executable, '-m', 'zulf_tools.worker', job_id], cwd=storage.PROJECT,
                                   stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    return {'job_id': job_id, 'status': 'submitted', 'pid': process.pid,
            'next_tool': 'get_job', 'log_path': str(path/'worker.log')}


def get_job(job_id):
    path = directory(job_id)
    status = storage.read_json(path/'status.json')
    if status['status'] == 'complete':
        status['result'] = storage.get_result(status['run_id'])
    status['cancel_requested'] = (path/'cancel').exists()
    return status


def cancel_job(job_id):
    path = directory(job_id)
    status = storage.read_json(path/'status.json')
    if status['status'] in ('queued', 'running'):
        (path/'cancel').touch()
        return {'job_id': job_id, 'status': 'cancellation_requested'}
    return {'job_id': job_id, 'status': status['status']}


def work(job_id):
    from .analysis import execute
    path = directory(job_id)
    request = storage.read_json(path/'request.json')
    status = storage.read_json(path/'status.json')
    status.update(status='running', pid=os.getpid())
    storage.write_json(path/'status.json', status)

    def progress(n, total):
        status.update(completed_units=n, total_units=total, progress=round(100*n/max(1,total), 2),
                      updated_utc=datetime.now(timezone.utc).isoformat())
        storage.write_json(path/'status.json', status)

    try:
        result = execute(**request, cancel=lambda: (path/'cancel').exists(), progress=progress)
        status.update(status='complete', run_id=result['run_id'], progress=100)
    except Exception as exc:
        status.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
    storage.write_json(path/'status.json', status)

