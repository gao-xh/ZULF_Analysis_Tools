from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zulf_tools import analysis, storage


class ExecutionTests(unittest.TestCase):
    def test_failed_and_cancelled_operations_retain_partial_artifacts_and_cost(self):
        for error, status in [(RuntimeError('numerical failure'), 'failed'),
                              (InterruptedError('cancelled'), 'cancelled')]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                def operation(**kwargs):
                    (kwargs['directory']/'partial.txt').write_text('Best provisional candidate')
                    raise error
                with patch.object(storage, 'ROOT', Path(temp)), patch.dict(analysis.OPERATIONS, {'fixture': operation}):
                    with self.assertRaises(type(error)):
                        analysis.execute('fixture', {})
                    path = next(Path(temp).glob('runs/*/result.json'))
                    r = storage.read_json(path)
                    self.assertEqual(r['status'], status)
                    self.assertEqual(r['error_type'], type(error).__name__)
                    self.assertGreaterEqual(r['execution_wall_seconds'], 0.)
                    self.assertIn(str((path.parent/'partial.txt').resolve()), r['artifacts'])
                    self.assertIn('finished_utc', r)
                    with self.assertRaises(ValueError):
                        storage.get_result(r['run_id'])

    def test_whole_operation_timing_does_not_replace_solver_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            def operation(**kwargs):
                return dict(elapsed_s=123., scientific_result='provisional')
            with patch.object(storage, 'ROOT', Path(temp)), patch.dict(analysis.OPERATIONS, {'fixture': operation}):
                r = analysis.execute('fixture', {})
                self.assertEqual(r['elapsed_s'], 123.)
                self.assertEqual(storage.get_result(r['run_id'])['execution_wall_seconds'], r['execution_wall_seconds'])
                self.assertEqual(r['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
