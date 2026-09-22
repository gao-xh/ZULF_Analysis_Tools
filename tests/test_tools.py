import asyncio
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import savgol_filter
from zulf_tools import analysis, data, storage, jobs


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.points = 512
        t = np.arange(self.points)/256
        self.originals = []
        for scan in (0,2,8):
            values = np.round(1000*np.exp(-t)*np.cos(2*np.pi*40*t)+scan*10).astype('<i2')
            self.originals.append(values)
            words = np.r_[np.zeros(20,dtype='<i2'), values[::-1], np.zeros(2,dtype='<i2')].astype('<i2')
            (self.source/f'{scan}.dat').write_bytes(words.tobytes()[::-1])
            (self.source/f'{scan}.ini').write_text('[NMRduino]\nSampleRate=256\nNumberOfSamples=512\n')
        self.allowed = patch('zulf_tools.data.source_folder', side_effect=lambda p: Path(p).resolve())
        self.output = patch('zulf_tools.storage.ROOT', self.root/'artifacts')
        self.allowed.start()
        self.output.start()

    def tearDown(self):
        self.allowed.stop()
        self.output.stop()
        self.tmp.cleanup()

    def test_streaming_mean_provenance_and_originals(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        inspected = analysis.execute('inspect_dataset', {'folder':str(self.source)})
        self.assertEqual(inspected['scan_count'],3)
        result = analysis.execute('compute_average', {'folder':str(self.source)})
        actual = np.load(storage.artifact(result['run_id'],'average.npy'))
        np.testing.assert_array_equal(actual,np.mean(self.originals,axis=0))
        manifest = storage.read_json(storage.artifact(result['run_id'],'sources.json'))
        self.assertEqual([r['scan_id'] for r in manifest['scans']],[0,2,8])
        self.assertTrue(all(len(r['sha256'])==64 for r in manifest['scans']))
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()})

    def test_explicit_selection_and_config_mismatch(self):
        m = data.inventory(self.source,[8])
        self.assertEqual([r['scan_id'] for r in m['scans']],[8])
        with self.assertRaises(ValueError):
            data.inventory(self.source,[1])
        with self.assertRaises(ValueError):
            data.inventory(self.source,[2,2])
        (self.source/'2.ini').write_text('[NMRduino]\nSampleRate=512\n')
        with self.assertRaisesRegex(ValueError,'sampling rate differs'):
            data.inventory(self.source)

    def test_bad_files_and_changed_source_rejected(self):
        m = data.inventory(self.source)
        (self.source/'0.dat').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            data.read_scan(m['scans'][0])
        with self.assertRaises(ValueError):
            data.inventory(self.source)

    def test_baseline_subtraction_crop_order_and_time_origin(self):
        x = np.arange(512,dtype=float)**2 + np.cos(np.arange(512))
        actual, baseline, parameters = analysis.recipe(x,256,{'start_s':.125,'end_s':1.,'sg_window':31,'remove_mean':False})
        expected_baseline = savgol_filter(x,31,2,mode='mirror')
        np.testing.assert_allclose(actual,(x-expected_baseline)[32:256])
        self.assertEqual(parameters['actual_start_s'],.125)
        self.assertEqual(len(actual),224)
        with self.assertRaises(ValueError):
            analysis.recipe(x,256,{'sg_window':30})

    def test_comparison_range_peak_and_parent_integrity(self):
        average = analysis.execute('compute_average',{'folder':str(self.source)})
        compared = analysis.execute('compare_preprocessing',{'average_run_id':average['run_id'],
                                    'variants':[{'label':'Raw'},{'label':'Baseline','sg_window':31}]})
        ranges = analysis.execute('inspect_frequency_ranges',{'comparison_run_id':compared['run_id'],'ranges':[[30,50]]})
        peak = ranges['ranges'][0]['candidates'][0]['frequency_hz']
        self.assertAlmostEqual(peak,40,delta=.5)
        self.assertEqual(len(list(storage.artifact(ranges['run_id']).parent.glob('*.png'))),2)
        storage.artifact(average['run_id'],'average.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            analysis.execute('compare_preprocessing',{'average_run_id':average['run_id'],'variants':[{}]})

    def test_cancelled_run_not_accepted(self):
        with self.assertRaises(InterruptedError):
            analysis.execute('compute_average',{'folder':str(self.source)},cancel=lambda:True)
        records = list((self.root/'artifacts'/'runs').glob('*/result.json'))
        self.assertEqual(len(records),1)
        run = storage.read_json(records[0])
        self.assertEqual(run['status'],'cancelled')
        with self.assertRaises(ValueError):
            storage.get_result(run['run_id'])

    def test_job_cancel_and_failure_records(self):
        jid = 'a'*32
        path = jobs.directory(jid)
        storage.write_json(path/'status.json',{'job_id':jid,'status':'queued'})
        storage.write_json(path/'request.json',{'operation':'compute_average','parameters':{'folder':str(self.source)}})
        jobs.cancel_job(jid)
        jobs.work(jid)
        self.assertEqual(jobs.get_job(jid)['status'],'cancelled')

    def test_artifact_paths_cannot_escape(self):
        with self.assertRaises(ValueError):
            storage.artifact('../escape')


class TransportTests(unittest.TestCase):
    def test_stdio_handshake_schema_and_error(self):
        async def check():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            command = StdioServerParameters(command=sys.executable,args=[str(storage.PROJECT/'run_server.py')])
            async with stdio_client(command) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.assertEqual(len(listed.tools),12)
                    bad = await session.call_tool('get_result',{'run_id':'../bad'})
                    self.assertTrue(bad.isError)
        asyncio.run(check())

if __name__ == '__main__':
    unittest.main()
