#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Open Whisper weights -> coremltools MIL IR -> textual MIL + public weight blob.
No Core ML model compilation or Apple runtime is invoked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def export(args):
    import numpy as np
    import torch
    import coremltools as ct
    import whisper
    from coremltools.converters.mil.mil import types
    from whisper.model import MultiHeadAttention
    MultiHeadAttention.use_sdpa = False
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    model = whisper.load_model(args.model, device='cpu', download_root=str(args.cache))
    encoder = model.encoder.eval()
    sample = torch.zeros(1, 80, 3000)
    traced = torch.jit.trace(encoder, sample)
    # milinternal returns Python IR; it cannot call Apple's model compiler.
    prog = ct.convert(traced, source='pytorch', convert_to='milinternal',
                      inputs=[ct.TensorType(name='mel', shape=sample.shape, dtype=np.float16)],
                      outputs=[ct.TensorType(name='encoder', dtype=np.float16)],
                      minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16)
    fn = prog.functions['main']
    if (len(fn.inputs) != 1 or len(fn.outputs) != 1 or
            list(next(iter(fn.inputs.values())).shape) != [1,80,3000] or
            list(fn.outputs[0].shape) != [1,1500,model.dims.n_audio_state]):
        raise ValueError('converted encoder interface differs from the requested model')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'probes').mkdir()
    payload = bytearray(128)
    names = {}
    for v in fn.inputs.values(): names[v] = 'mel'
    for op in fn.operations:
        if op.blocks: raise ValueError('control flow is not supported by the static exporter')
        for v in op.outputs: names[v] = 'v' + str(len(names))
    if len(fn.outputs) != 1: raise ValueError('expected one encoder output')
    names[fn.outputs[0]] = 'encoder'

    def typ(v):
        dtype = types.builtin_to_string(v.dtype)
        shape = list(v.shape) if types.is_tensor(v.sym_type) else []
        if any(int(d) <= 0 for d in shape):
            raise ValueError('requires positive static dimensions')
        return f'tensor<{dtype}, [{", ".join(str(int(d)) for d in shape)}]>' if types.is_tensor(v.sym_type) else dtype

    lines, consts, records = [], {}, []
    for op in fn.operations:
        if len(op.outputs) != 1: raise ValueError(f'multiple outputs: {op.op_type}')
        v = op.outputs[0]
        name = names[v]
        if op.op_type == 'const':
            value = v.val
            a = np.asarray(value)
            if a.dtype == np.float16 and a.size > 32:
                payload.extend(bytes((-len(payload)) % 64))
                offset = len(payload)
                raw = a.astype('<f2').tobytes()
                payload.extend(struct.pack('<IIQQ', 0xDEADBEEF, 1, len(raw), offset+64) + bytes(40) + raw)
                val = f'BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset}))'
            else:
                val = json.dumps(a.tolist(), allow_nan=False)
            line = f'    {typ(v)} {name} = const()[name = string("{name}"), val = {typ(v)}({val})];'
            consts[v] = line
            lines.append(line)
            continue
        inputs, runtime, deps, attributes, constants = [], {}, [], {}, {}
        for key, value in op.inputs.items():
            if value is None: continue
            sequence = isinstance(value, (tuple, list))
            values = list(value) if sequence else [value]
            refs = []
            for x in values:
                refs.append(names[x])
                if x in consts:
                    constants[names[x]] = [int(d) for d in np.asarray(x.val).shape]
                    # Small control attributes distinguish e.g. transpose axes.
                    value_array = np.asarray(x.val)
                    if value_array.size <= 32:
                        attributes[key] = value_array.tolist()
                    if consts[x] not in deps: deps.append(consts[x])
                else:
                    runtime[names[x]] = x
            inputs.append(key + ' = ' + ('(' + ', '.join(refs) + ')' if sequence else refs[0]))
        line = f'    {typ(v)} {name} = {op.op_type}({", ".join(inputs)})[name = string("{name}")];'
        lines.append(line)
        probe = f'{len(records):04d}-{op.op_type}'
        args_text = ', '.join(typ(x)+' '+n for n,x in runtime.items())
        text = 'program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
        text += f'  func main<ios18>({args_text}) {{\n' + '\n'.join(deps+[line]) + f'\n  }} -> ({name});\n}}\n'
        (root/'probes'/f'{probe}.mil').write_text(text)
        records.append({'name': probe, 'op': op.op_type, 'mil': f'probes/{probe}.mil',
                        'inputs': {n: [int(d) for d in x.shape] for n,x in runtime.items()},
                        'constants': constants,
                        'output': [int(d) for d in v.shape], 'attributes': attributes})
    payload[:8] = struct.pack('<II', 1, 2)
    (root/'weights.bin').write_bytes(payload)
    signature = ', '.join(typ(v)+' '+names[v] for v in fn.inputs.values())
    text = 'program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
    text += f'  func main<ios18>({signature}) {{\n' + '\n'.join(lines) + '\n  } -> (encoder);\n}\n'
    (root/'model.mil').write_text(text)
    checkpoint = args.cache / (args.model+'.pt')
    metadata = {'schema': 'whisper-ane-open-v1', 'model': args.model, 'license': 'MIT',
                'weights_source': 'https://github.com/openai/whisper',
                'checkpoint_sha256': sha(checkpoint), 'ggml_sha256': sha(args.ggml),
                'coremltools': ct.__version__, 'torch': torch.__version__, 'mil_opset':'ios18',
                'input': {'name': 'mel', 'shape': [1,80,3000]},
                'output': {'name': 'encoder', 'shape': [1,1500,model.dims.n_audio_state]},
                'files': {n: sha(root/n) for n in ('model.mil','weights.bin')}, 'probes': records}
    (root/'model.json').write_text(json.dumps(metadata, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['tiny','tiny.en','base','base.en','small','small.en'], required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--ggml', type=Path, required=True, help='matching unquantized whisper.cpp model')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    export(p.parse_args())

if __name__ == '__main__': main()
