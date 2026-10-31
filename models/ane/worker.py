#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native adapter delegating graph validation/dispatch to the published runner.
The parent owns CPU fallback via libane; this worker never returns CPU output.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import signal


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


def verify(bundle, ggml, shape):
    meta = json.loads((bundle/'model.json').read_text())
    if meta['schema'] != 'whisper-ane-open-v1' or meta['license'] != 'MIT':
        raise ValueError('unknown open-weight bundle')
    if meta['input']['shape'] != [1,shape[0],2*shape[1]] or meta['output']['shape'] != [1,shape[1],shape[2]]:
        raise ValueError('encoder shape differs from bundle')
    if meta['ggml_sha256'] != sha(ggml): raise ValueError('ggml weights differ from bundle')
    for name in ('model.mil','weights.bin'):
        if meta['files'][name] != sha(bundle/name): raise ValueError(f'hash mismatch: {name}')
    manifest = json.loads((bundle/'package/manifest.json').read_text())
    if manifest.get('target') != 'H13': raise ValueError('only H13 has a shared graph runner')
    return meta


def main():
    # The runner owns in-flight work. Ignore terminal interruption in both this
    # worker and its runner, as in libane's existing qualification benchmark.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','ggml','input','output'): p.add_argument('--'+name, type=Path, required=True)
    for name in ('n-mels','n-ctx','n-state'): p.add_argument('--'+name, type=int, required=True)
    a = p.parse_args()
    root = a.bundle.resolve()
    verify(root,a.ggml,(a.n_mels,a.n_ctx,a.n_state))
    import numpy as np
    library = os.environ.get('WHISPER_ANE_LIB', '/usr/lib/libane_python.so')
    compiler = Path(os.environ['WHISPER_ANE_COMPILER']).resolve()
    lib = C.CDLL(library)
    lib.pyane_query_caps.argtypes = [C.c_int,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
    lib.pyane_query_caps.restype = C.c_int
    abi, chip = C.c_uint32(), C.c_uint32()
    rc = lib.pyane_query_caps(0,C.byref(abi),C.byref(chip))
    if rc or (abi.value,chip.value) != (1,13):
        raise OSError(f'H13 runner refused caps rc={rc} ABI={abi.value} chip={chip.value}')
    mel = np.fromfile(a.input,dtype='<f4')
    if mel.size != a.n_mels*2*a.n_ctx or not np.isfinite(mel).all(): raise ValueError('invalid mel')
    inp, out = a.input.parent/'mel.fp16', a.output.parent/'encoder.fp16'
    mel.astype('<f2').tofile(inp)
    command = [sys.executable,str(compiler/'tools/h13_run_linux.py'),str(root/'package'),
               '--mil',str(root/'model.mil'),'--model-root',str(root),'--libane-library',library,
               '--deadline-seconds','120','--input',f'mel={inp}','--output',f'encoder={out}']
    (a.input.parent/'command.json').write_text(json.dumps(command))
    # Preserve the runner's ownership until its normal cleanup completes.
    with (a.input.parent/'runner.log').open('w') as log:
        run = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode: raise RuntimeError(f'native runner rc={run.returncode}')
    result = np.fromfile(out,dtype='<f2').astype('<f4')
    if result.size != a.n_ctx*a.n_state or not np.isfinite(result).all(): raise ValueError('invalid encoder output')
    result.tofile(a.output)

if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(f'ane worker: {error}',file=sys.stderr)
        sys.exit(1)
