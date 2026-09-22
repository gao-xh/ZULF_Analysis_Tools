"""One JSON request per CLI call, using exactly the same functions as MCP."""
import argparse
import json
import sys
from . import storage


def call(tool, arguments):
    from .analysis import execute, OPERATIONS
    from .jobs import start_analysis, get_job, cancel_job
    if tool in OPERATIONS:
        return execute(tool, arguments)
    tools = {'start_analysis': start_analysis, 'get_job': get_job, 'cancel_job': cancel_job,
             'get_result': storage.get_result}
    if tool not in tools:
        raise ValueError('Unknown tool.')
    return tools[tool](**arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', help='JSON file containing tool and arguments; omit to read stdin')
    args = parser.parse_args()
    try:
        request = storage.read_json(args.request) if args.request else json.load(sys.stdin)
        result = call(request['tool'], request.get('arguments', {}))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}))
        raise SystemExit(1)

if __name__ == '__main__':
    main()

