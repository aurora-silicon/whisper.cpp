# SPDX-License-Identifier: MIT
# Adapter-only test fixture: never loads a model or touches a device.
import argparse
import os
import struct
p = argparse.ArgumentParser()
for n in ('bundle','ggml','input','output','n-mels','n-ctx','n-state'): p.add_argument('--'+n)
a = p.parse_args()
with open(a.output,'wb') as f:
    f.write(struct.pack('<f',float('nan') if os.environ.get('ANE_TEST_NAN') else 7.) * (int(a.n_ctx)*int(a.n_state)))
