# SPDX-License-Identifier: MIT
import importlib.util
import array
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave
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

class EvaluationTests(unittest.TestCase):
    def test_fallback_retires_native_across_processes(self):
        # A fresh CLI process must not undo retirement after a native failure.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'; data.mkdir()
            for key in ('0001','0002'): (data/(key+'.flac')).write_bytes(b'public fixture')
            (data/'clips.trans.txt').write_text('0001 A\n0002 A\n')
            model = root/'model.bin'; model.write_bytes(b'fixture weights')
            out = root/'results'
            attempts = []
            def run(command, **kwargs):
                if command[0] == 'ffmpeg':
                    with wave.open(command[-1],'wb') as wav:
                        wav.setparams((1,2,16000,0,'NONE','not compressed'))
                        wav.writeframes(bytes(320))
                    return subprocess.CompletedProcess(command,0)
                requested = 'WHISPER_ANE_ENCODER' in kwargs['env']
                attempts.append(requested)
                prefix = command[command.index('-of')+1]
                Path(prefix+'.txt').write_text('A')
                Path(prefix+'-0.f32').write_bytes(array.array('f',[1.,2.]).tobytes())
                marker = 'ane: encoder backend=cpu reason=2\n' if requested else ''
                return subprocess.CompletedProcess(command,0,'',marker+'encode time = 1.0 ms\n')
            argv = ['evaluate','--cli',str(root/'cli'),'--model',str(model),
                    '--dataset',str(data),'--bundle',str(root/'bundle'),'--out',str(out)]
            with patch('sys.argv',argv), patch.object(evaluate.subprocess,'run',side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(attempts,[False,False,True,False])
            result = json.loads((out/'results.json').read_text())
            self.assertFalse(result['encoder_parity']['candidate_native'])
            self.assertEqual(result['encoder_parity']['max_abs_diff'],0.)
            self.assertTrue(all(r['fallback'] for r in result['arms']['candidate']['utterances']))
if __name__ == '__main__': unittest.main()
