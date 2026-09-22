"""Advance a saved analysis plan by one job submission or observation.

Run as python -m examples.run_workflow PLAN.json --name LOCAL_NAME.
Reinvoke the same command to continue; never restarts a failed or uncertain job.
"""
import argparse
import hashlib
import inspect
import json
import re
from contextlib import contextmanager
from zulf_tools import storage
from zulf_tools.cli import call


def resolve(value, completed):
    if isinstance(value, dict):
        if '$run' in value:
            if set(value) != {'$run'}:
                raise ValueError('A run reference must contain only $run.')
            return completed[value['$run']]['run_id']
        return {k: resolve(v, completed) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, completed) for v in value]
    return value


def validate(plan):
    from zulf_tools.analysis import OPERATIONS
    if set(plan) != {'stages'} or not isinstance(plan['stages'], list) or not plan['stages']:
        raise ValueError('Plan must contain a nonempty stages list.')
    earlier = {}
    for stage in plan['stages']:
        if set(stage) != {'name', 'operation', 'parameters'}:
            raise ValueError('Each stage needs name, operation and parameters.')
        name = stage['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name) or name in earlier:
            raise ValueError('Stage names must be unique lowercase identifiers.')
        if stage['operation'] not in OPERATIONS:
            raise ValueError('Unknown operation.')
        parameters = resolve(stage['parameters'], earlier)
        inspect.signature(OPERATIONS[stage['operation']]).bind(
            **parameters, record=None, directory=None, cancel=None, progress=None)
        earlier[name] = {'run_id': '0' * 32}


@contextmanager
def exclusive(path):
    # A crashed controller leaves this marker deliberately. Inspect its process
    # and state before removing it; elapsed time alone is not proof of death.
    import os
    with path.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def advance(plan, path, invoke=call):
    """One durable tick. Caller serializes access; no polling sleep or retry."""
    validate(plan)
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    state = storage.read_json(path) if path.exists() else {
        'plan_sha256': digest, 'plan': plan, 'stages': {}, 'status': 'ready'}
    if state['plan_sha256'] != digest:
        raise ValueError('Plan changed. Use a new workflow name; prior evidence is retained.')
    for stage in plan['stages']:
        name = stage['name']
        saved = state['stages'].get(name)
        if saved and saved['status'] == 'complete':
            continue
        if saved:
            if saved['status'] in ('dispatching', 'dispatch_error', 'failed', 'cancelled'):
                state['status'] = 'needs_investigation'
                storage.write_json(path, state)
                return state
            # Observation errors propagate, preserving the existing handle.
            job = invoke('get_job', {'job_id': saved['job_id']})
            saved['status'] = job['status']
            saved['observation'] = {k: v for k, v in job.items() if k != 'result'}
            if job['status'] == 'complete':
                saved['run_id'] = job['run_id']
                state['status'] = 'ready'
            elif job['status'] in ('failed', 'cancelled'):
                state['status'] = 'needs_investigation'
            else:
                state['status'] = 'waiting'
            storage.write_json(path, state)
            return state
        request = {'operation': stage['operation'],
                   'parameters': resolve(stage['parameters'], state['stages'])}
        saved = {'status': 'dispatching', 'request': request}
        state['stages'][name] = saved
        state['status'] = 'dispatching'
        storage.write_json(path, state)
        try:
            job = invoke('start_analysis', request)
        except Exception as exc:
            # Submission might have succeeded before the error. Never assume
            # it is safe to retry and launch duplicate expensive work.
            saved.update(status='dispatch_error', error=f'{type(exc).__name__}: {exc}')
            state['status'] = 'needs_investigation'
            storage.write_json(path, state)
            raise
        saved.update(status='submitted', job_id=job['job_id'], submission=job)
        state['status'] = 'waiting'
        storage.write_json(path, state)
        return state
    state['status'] = 'complete'
    state['interpretation'] = 'Execution complete, not scientific acceptance. Inspect figures and evidence flags.'
    storage.write_json(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', args.name):
        parser.error('Use a simple local workflow name, without path separators.')
    directory = storage.ROOT / 'workflows'
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive(directory / (args.name + '.lock')):
        result = advance(storage.read_json(args.plan), directory / (args.name + '.json'))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
