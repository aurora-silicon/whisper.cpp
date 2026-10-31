#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only system power sampling around a lead-authorized evaluation command.
Linux hwmon power*_input is in microwatts. No power/clock policy is changed.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensor', type=Path, help='explicit hwmon power*_input (system power, e.g. PSTR)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--idle-seconds', type=float, default=5)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command: p.error('provide -- COMMAND')
    if not 0 < a.idle_seconds <= 60: p.error('idle seconds must be in (0,60]')
    a.out.mkdir(parents=True, exist_ok=False)
    stop, samples, phase, failures = threading.Event(), [], ['idle_before'], []
    start = time.monotonic()
    def sample():
        while not stop.is_set():
            try:
                value = int(a.sensor.read_text())/1e6 if a.sensor else None
                samples.append((time.monotonic()-start, phase[0], value))
            except (OSError,ValueError) as error:
                failures.append(str(error))
            stop.wait(.1)
    thread = threading.Thread(target=sample)
    thread.start()
    states = {}
    def state(label):
        states[label] = {}
        for name in ('ane_stats','wedged','power/runtime_status'):
            path = Path('/sys/class/accel/accel0/device')/name
            try: states[label][name] = path.read_text().strip()
            except OSError: states[label][name] = None
    state('before')
    try:
        time.sleep(a.idle_seconds)
        phase[0] = 'load'
        with (a.out/'command.log').open('w') as log:
            rc = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        state('after_load')
        phase[0] = 'idle_after'
        time.sleep(a.idle_seconds)
        state('after_idle')
    finally:
        stop.set(); thread.join()
    with (a.out/'power.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['elapsed_s','phase','system_W']); w.writerows(samples)
    report = {'command':command,'rc':rc,'sensor':str(a.sensor) if a.sensor else None,
              'scope':'system power, not isolated ANE rail power','states':states,
              'sensor_errors':sorted(set(failures)), 'phases':{}}
    for part in ('idle_before','load','idle_after'):
        values = [v for _,s,v in samples if s == part and v is not None]
        report['phases'][part] = {'samples':len(values), 'mean_system_W':statistics.mean(values) if values else None}
    report['power_status'] = 'measured' if a.sensor and not failures else 'unavailable'
    (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    raise SystemExit(rc)

if __name__ == '__main__': main()
