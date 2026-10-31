# SPDX-License-Identifier: MIT
import importlib.util
import array
from pathlib import Path
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('evaluate', Path(__file__).with_name('evaluate.py'))
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
class MetricsTests(unittest.TestCase):
    def test_edit_counts(self):
        # One substitution, one deletion, one insertion; order matters.
        reference = 'a b c d e'.split()
        self.assertEqual(evaluate.errors(reference, 'a x c e f'.split()), 3)
        self.assertEqual(evaluate.errors(reference, []), 5)
        self.assertEqual(evaluate.errors([], ['word']), 1)
        self.assertEqual(evaluate.errors(reference, reference), 0)
    def test_normalization(self):
        self.assertEqual(evaluate.words("Hello, WORLD! Don't."), ['HELLO','WORLD',"DON'T"])

class ParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cpu, self.candidate = [Path(self.tmp.name)/s for s in ('cpu','candidate')]
        self.cpu.mkdir(); self.candidate.mkdir()
        for root in (self.cpu,self.candidate):
            (root/'clip-0.f32').write_bytes(array.array('f',[1.,2.]).tobytes())
    def parity(self): return evaluate.encoder_parity(self.cpu,self.candidate)
    def test_numeric_error(self):
        (self.candidate/'clip-0.f32').write_bytes(array.array('f',[1.25,1.5]).tobytes())
        self.assertEqual(self.parity(), {'max_abs_diff':.5,'elements':2,'tensors':1})
    def test_extra_candidate_window(self):
        (self.candidate/'clip-1.f32').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'capture sets'): self.parity()
    def test_missing_candidate_window(self):
        (self.candidate/'clip-0.f32').unlink()
        with self.assertRaisesRegex(ValueError,'capture sets'): self.parity()
    def test_empty_output(self):
        for root in (self.cpu,self.candidate): (root/'clip-0.f32').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'empty tensor'): self.parity()
    def test_nonfinite(self):
        (self.candidate/'clip-0.f32').write_bytes(array.array('f',[1.,float('nan')]).tobytes())
        with self.assertRaisesRegex(ValueError,'nonfinite'): self.parity()
if __name__ == '__main__': unittest.main()
