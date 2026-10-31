#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile full graph and isolated MIL ops, preserving every refusal by target."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from export import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--target', choices=['H13','H14','H16G'], required=True)
    p.add_argument('--timeout', type=int, default=120)
    a = p.parse_args()
    root = a.bundle.resolve()
    meta = json.loads((root/'model.json').read_text())
    for name, digest in meta['files'].items():
        if sha(root/name) != digest: raise ValueError(f'hash mismatch: {name}')
    out = root / ('compile-'+a.target)
    out.mkdir(exist_ok=False)
    records = []
    # One probe per operation/input/output geometry; repeated transformer
    # layers share the compiler envelope. Keep the first layer's real weights.
    unique = {}
    for probe in meta['probes']:
        key = json.dumps([probe['op'], list(probe['inputs'].values()), probe['output'], probe.get('attributes')], sort_keys=True)
        unique.setdefault(key, probe)
    probes = [{'name':'full-model','mil':'model.mil'}, *unique.values()]
    for probe in probes:
        name = probe['name']
        package = out/name
        command = [str(a.compiler.resolve()), '--target',a.target,'--output',str(package),
                   '--model-root',str(root),'--mil',str(root/probe['mil'])]
        start = time.monotonic()
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=a.timeout)
            rc, stdout, stderr = run.returncode, run.stdout, run.stderr
        except subprocess.TimeoutExpired as error:
            rc = 124
            stdout, stderr = str(error.stdout or ''), str(error.stderr or '')
        (out/(name+'.log')).write_text(stdout+'\n'+stderr)
        count = None
        if rc == 0 and (package/'manifest.json').is_file():
            count = len(json.loads((package/'manifest.json').read_text()).get('programs', []))
        # Keep full-model artifacts, but not thousands of repeated probe programs.
        if name != 'full-model' and package.exists():
            import shutil
            shutil.rmtree(package)
        records.append({'program_count':count, **probe, 'rc':rc, 'seconds':time.monotonic()-start,
                        'diagnostic':stderr[-4000:], 'command':command})
        (out/'results.json').write_text(json.dumps({'target':a.target,'compiler_sha256':sha(a.compiler),
                                                   'records':records},indent=2)+'\n')
    # Never expose a partial or stale compilation as a runnable bundle.
    if a.target == 'H13' and records[0]['rc'] == 0:
        import shutil
        shutil.copytree(out/'full-model',root/'package')
    print(json.dumps({'target':a.target,'full_model_rc':records[0]['rc'],
                      'refused':sum(r['rc'] != 0 for r in records),'total':len(records)}))

if __name__ == '__main__': main()
