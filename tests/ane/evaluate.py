#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired LibriSpeech CPU/native-or-fallback WER, encoder parity, wall/CPU RTF.
Input: unpacked public test-clean directory. No device setup or playback.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import time
import wave


def words(text): return re.findall(r"[A-Z0-9']+", text.upper())


def errors(reference, actual):
    prev = list(range(len(actual)+1))
    for i, a in enumerate(reference, 1):
        row = [i]
        for j, b in enumerate(actual, 1): row.append(min(row[-1]+1, prev[j]+1, prev[j-1]+(a != b)))
        prev = row
    return prev[-1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def encoder_parity(cpu, candidate):
    captures = [{p.name: p for p in root.glob('*.f32')} for root in (cpu, candidate)]
    if not captures[0] or captures[0].keys() != captures[1].keys():
        raise ValueError('encoder capture sets differ or are empty')
    max_diff, elements = 0., 0
    for name in sorted(captures[0]):
        x, y = array.array('f'), array.array('f')
        x.frombytes(captures[0][name].read_bytes())
        y.frombytes(captures[1][name].read_bytes())
        if not x or len(x) != len(y):
            raise ValueError('encoder output shape mismatch or empty tensor')
        if not all(math.isfinite(v) for v in x) or not all(math.isfinite(v) for v in y):
            raise ValueError('nonfinite encoder output')
        max_diff = max(max_diff, max(abs(i-j) for i,j in zip(x,y)))
        elements += len(x)
    return {'max_abs_diff':max_diff, 'elements':elements, 'tensors':len(captures[0])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--require-ane', action='store_true')
    a = p.parse_args()
    if a.limit <= 0: p.error('limit must be positive')
    a.out.mkdir(parents=True, exist_ok=False)
    refs = {}
    for path in sorted(a.dataset.rglob('*.trans.txt')):
        for line in path.read_text().splitlines():
            key, text = line.split(' ',1); refs[key] = text
    clips = sorted(a.dataset.rglob('*.flac'))[:a.limit]
    if not clips: raise ValueError('no LibriSpeech flac clips')
    report = {'model_sha256':digest(a.model), 'threads':a.threads,
              'normalization':"uppercase [A-Z0-9']+ for reference and hypothesis",
              'selection':'lexicographically first flac paths', 'clips':[], 'arms':{}}
    for arm in ('cpu','candidate'):
        dest = a.out/arm; dest.mkdir()
        rows, total_errors, total_words, audio_s, wall_s, cpu_s, encoder_s = [],0,0,0.,0.,0.,0.
        for clip in clips:
            key = clip.stem
            wav = a.out/(key+'.wav')
            if not wav.exists():
                subprocess.run(['ffmpeg','-v','error','-i',str(clip),'-ar','16000','-ac','1',str(wav)],check=True)
            with wave.open(str(wav)) as f: duration = f.getnframes()/f.getframerate()
            env = os.environ.copy()
            env.pop('ANEFORGE_ENCODER',None)
            env.pop('WHISPER_ANE_ENCODER',None)
            if arm == 'candidate': env['WHISPER_ANE_ENCODER'] = str(a.bundle.resolve())
            prefix = (dest/key).resolve()
            env['WHISPER_ENCODER_DUMP'] = str(prefix)
            command = [str(a.cli.resolve()),'-m',str(a.model.resolve()),'-f',str(wav.resolve()),
                       '-ng','-nfa','-t',str(a.threads),'-l','en','-nt','-otxt','-of',str(prefix)]
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.monotonic()
            r = subprocess.run(command,env=env,capture_output=True,text=True)
            elapsed = time.monotonic()-start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            used = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
            (dest/(key+'.log')).write_text(r.stdout+'\n'+r.stderr)
            r.check_returncode()
            hypothesis = Path(str(prefix)+'.txt').read_text()
            ref = words(refs[key]); actual = words(hypothesis)
            count = errors(ref,actual)
            timings = re.search(r'encode time\s*=\s*([0-9.]+)\s*ms',r.stderr)
            enc = float(timings[1])/1000 if timings else None
            native = 'ane: encoder backend=ane ' in r.stderr
            fallback = 'ane: encoder backend=cpu ' in r.stderr
            rows.append({'id':key,'flac_sha256':digest(clip),'audio_s':duration,'wall_s':elapsed,
                         'cpu_s':used,'encoder_s':enc,'errors':count,'words':len(ref),
                         'native':native,'fallback':fallback,'hypothesis':hypothesis.strip()})
            total_errors += count; total_words += len(ref); audio_s += duration
            wall_s += elapsed; cpu_s += used; encoder_s += enc or 0
        report['arms'][arm] = {'utterances':rows,'errors':total_errors,'words':total_words,
                               'wer':total_errors/total_words,'audio_s':audio_s,
                               'wall_s':wall_s,'cpu_s':cpu_s,'encoder_s':encoder_s,
                               'wall_rtf':wall_s/audio_s,'cpu_rtf':cpu_s/audio_s,'encoder_rtf':encoder_s/audio_s}
    report['encoder_parity'] = encoder_parity(a.out/'cpu', a.out/'candidate')
    report['encoder_parity']['candidate_native'] = all(
        r['native'] and not r['fallback'] for r in report['arms']['candidate']['utterances'])
    (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in arm.items() if n != 'utterances'} for k,arm in report['arms'].items()}))
    print(json.dumps(report['encoder_parity']))
    if a.require_ane and not report['encoder_parity']['candidate_native']:
        raise SystemExit('candidate used CPU; not an ANE qualification')

if __name__ == '__main__': main()
