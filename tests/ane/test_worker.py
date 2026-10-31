# SPDX-License-Identifier: MIT
"""Bundle/weight/shape rejection happens before the worker queries hardware."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('worker', Path(__file__).resolve().parents[2]/'models/ane/worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('model.mil','weights.bin','ggml.bin'): (self.root/name).write_bytes(name.encode())
        (self.root/'package').mkdir()
        (self.root/'package/manifest.json').write_text(json.dumps({'target':'H13'}))
        self.meta = {'schema':'whisper-ane-open-v1','license':'MIT',
                     'input':{'shape':[1,80,3000]},'output':{'shape':[1,1500,384]},
                     'ggml_sha256':worker.sha(self.root/'ggml.bin'),
                     'files':{n:worker.sha(self.root/n) for n in ('model.mil','weights.bin')}}
        self.save()
    def save(self): (self.root/'model.json').write_text(json.dumps(self.meta))
    def verify(self, shape=(80,1500,384)): return worker.verify(self.root,self.root/'ggml.bin',shape)
    def test_valid(self): self.assertEqual(self.verify()['license'],'MIT')
    def test_wrong_weights(self):
        (self.root/'ggml.bin').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError,'ggml weights'): self.verify()
    def test_wrong_shape(self):
        with self.assertRaisesRegex(ValueError,'shape'): self.verify((80,750,384))
    def test_modified_mil(self):
        (self.root/'model.mil').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.verify()
    def test_other_chip(self):
        (self.root/'package/manifest.json').write_text(json.dumps({'target':'H14'}))
        with self.assertRaisesRegex(ValueError,'only H13'): self.verify()
    def test_unknown_schema(self):
        self.meta['schema']='unknown'; self.save()
        with self.assertRaisesRegex(ValueError,'unknown'): self.verify()

if __name__ == '__main__': unittest.main()
