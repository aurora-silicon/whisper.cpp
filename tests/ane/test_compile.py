# SPDX-License-Identifier: MIT
"""Compiler receipts distinguish invalid MIL interfaces from op coverage gaps."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CoverageTests(unittest.TestCase):
    def test_opset_and_parser_errors_are_not_op_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'probes').mkdir()
            (root/'model.mil').write_text('opset')
            (root/'weights.bin').write_bytes(b'public fixture')
            probes = []
            for op, kind in (('conv','shape'),('relu','syntax')):
                path = 'probes/'+op+'.mil'
                (root/path).write_text(kind)
                probes.append({'name':op,'op':op,'mil':path,'inputs':{'x':[1,64]},'output':[1,64]})
            meta = {'files':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                             for n in ('model.mil','weights.bin')}, 'probes':probes}
            (root/'model.json').write_text(json.dumps(meta))
            compiler = root/'compiler'
            compiler.write_text('''#!/usr/bin/env python3
from pathlib import Path
import sys
kind = Path(sys.argv[sys.argv.index('--mil')+1]).read_text()
code = {'opset':'ane.verify.unsupported-opset', 'shape':'ane.h13.conv-shape',
        'syntax':'mil.parse.expected-token'}[kind]
print('1:1: error ['+code+']: public test diagnostic',file=sys.stderr)
sys.exit(65)
''')
            compiler.chmod(0o700)
            script = Path(__file__).resolve().parents[2]/'models/ane/compile.py'
            run = subprocess.run([sys.executable,str(script),str(root),'--compiler',str(compiler),
                                  '--target','H13'],check=True,capture_output=True,text=True)
            summary = json.loads(run.stdout)
            self.assertEqual((summary['refused'],summary['errors']),(1,2))
            receipt = json.loads((root/'compile-H13/results.json').read_text())
            self.assertEqual([r['outcome'] for r in receipt['records']],['error','refused','error'])
            self.assertTrue(all(r['rc'] == 65 for r in receipt['records']))
            self.assertFalse((root/'package').exists())


if __name__ == '__main__': unittest.main()
