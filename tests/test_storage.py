import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zulf_tools import storage


class StorageTests(unittest.TestCase):
    def test_transient_replacement_failure_preserves_old_until_success(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'status.json';p.write_text('{"value":1}')
            replace=storage.os.replace; calls=[]
            def busy_then_replace(source,target):
                calls.append(1)
                self.assertEqual(json.loads(p.read_text()),{'value':1})
                if len(calls)<3:
                    raise PermissionError('sharing violation')
                replace(source,target)
            with patch.object(storage.os,'replace',side_effect=busy_then_replace),patch.object(storage.time,'sleep'):
                storage.write_json(p,{'value':2})
            self.assertEqual(json.loads(p.read_text()),{'value':2})
            self.assertEqual(len(calls),3)
            self.assertEqual(list(Path(temp).glob('*.tmp')),[])

    def test_permanent_failure_is_bounded_and_nonfinite_does_not_destroy(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'status.json';p.write_text('{"value":1}')
            with patch.object(storage.os,'replace',side_effect=PermissionError('denied')) as replace,patch.object(storage.time,'sleep'):
                with self.assertRaises(PermissionError):
                    storage.write_json(p,{'value':2})
                self.assertEqual(replace.call_count,8)
            with self.assertRaises(ValueError):
                storage.write_json(p,{'value':float('nan')})
            self.assertEqual(json.loads(p.read_text()),{'value':1})
            self.assertEqual(list(Path(temp).glob('*.tmp')),[])


if __name__=='__main__':
    unittest.main()
