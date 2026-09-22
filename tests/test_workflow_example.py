import tempfile
import unittest
from pathlib import Path
from examples.run_workflow import advance, exclusive, validate
from zulf_tools import storage


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'state.json'
        self.plan = {'stages': [
            {'name': 'fit', 'operation': 'fit_frequency_decay', 'parameters': {
                'group_run_id': 'a' * 32, 'ranges': [[30, 40]],
                'discovery_groups': [0, 2], 'validation_groups': [1, 3]}},
            {'name': 'review', 'operation': 'review_decay_evidence',
             'parameters': {'fit_run_id': {'$run': 'fit'}}}]}

    def test_resume_uses_same_handle_and_resolves_dependencies(self):
        requests = []
        def invoke(tool, args):
            requests.append((tool, args))
            if tool == 'start_analysis':
                return {'job_id': 'b' * 32}
            return {'status': 'complete', 'run_id': 'c' * 32}
        advance(self.plan, self.path, invoke)
        advance(self.plan, self.path, invoke)
        state = advance(self.plan, self.path, invoke)
        self.assertEqual([r[0] for r in requests], ['start_analysis', 'get_job', 'start_analysis'])
        self.assertEqual(requests[1][1]['job_id'], 'b' * 32)
        self.assertEqual(requests[2][1]['parameters']['fit_run_id'], 'c' * 32)
        self.assertEqual(state['stages']['fit']['run_id'], 'c' * 32)
        advance(self.plan, self.path, invoke)
        self.assertEqual(advance(self.plan, self.path, invoke)['status'], 'complete')

    def test_uncertain_submission_and_failed_job_never_restart(self):
        def fail(*args):
            raise OSError('Lost response after submission')
        with self.assertRaises(OSError):
            advance(self.plan, self.path, fail)
        state = advance(self.plan, self.path, lambda *args: self.fail('Must not resubmit'))
        self.assertEqual(state['status'], 'needs_investigation')
        state['stages']['fit'].update(status='submitted', job_id='b' * 32)
        storage.write_json(self.path, state)
        advance(self.plan, self.path, lambda *args: {'status': 'failed', 'error': 'Numerical failure'})
        advance(self.plan, self.path, lambda *args: self.fail('Failed job must be retained'))

    def test_observation_error_keeps_handle_and_plan_is_immutable(self):
        advance(self.plan, self.path, lambda *args: {'job_id': 'b' * 32})
        before = self.path.read_bytes()
        def fail(*args):
            raise TimeoutError('Temporary observation error')
        with self.assertRaises(TimeoutError):
            advance(self.plan, self.path, fail)
        self.assertEqual(before, self.path.read_bytes())
        self.plan['stages'][0]['parameters']['ranges'] = [[30, 41]]
        with self.assertRaisesRegex(ValueError, 'Plan changed'):
            advance(self.plan, self.path)

    def test_forward_references_and_concurrent_controller_rejected(self):
        with self.assertRaises(KeyError):
            validate({'stages': self.plan['stages'][::-1]})
        lock = self.path.with_suffix('.lock')
        with exclusive(lock):
            with self.assertRaises(FileExistsError):
                with exclusive(lock):
                    pass
        self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main()
