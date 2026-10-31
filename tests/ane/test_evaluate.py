# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
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
if __name__ == '__main__': unittest.main()
