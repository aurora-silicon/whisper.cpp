#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build-host evaluation: no ANE device is accessed for the missing-package arm.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${BUILD_OUT:-$repo/build-ane-evaluate}")
"$repo/models/ane/check-build.sh"
mkdir -p "$out/models" "$out/data"
bash "$repo/models/download-ggml-model.sh" tiny.en "$out/models"
curl --fail --location --retry 3 'https://openslr.trmal.net/resources/12/test-clean.tar.gz' -o "$out/test-clean.tar.gz"
printf '%s  %s\n' '32fa31d27d2e1cad72775fee3f4849a9' "$out/test-clean.tar.gz" | md5sum -c -
# Python's data filter refuses archive links and paths outside this directory.
python3 - "$out" <<'PY'
import sys, tarfile
from pathlib import Path
out = Path(sys.argv[1])
with tarfile.open(out/'test-clean.tar.gz') as archive:
    archive.extractall(out/'data', filter='data')
PY
lscpu > "$out/cpu.txt"
python3 "$repo/tests/ane/evaluate.py" --cli "$out/build/bin/whisper-cli" \
    --model "$out/models/ggml-tiny.en.bin" --dataset "$out/data/LibriSpeech/test-clean" \
    --bundle "$out/absent-bundle" --out "$out/evaluation" --limit 20 --threads 8
# Also exercise a present but invalid package: refusal must recompute the CPU encoder.
mkdir -p "$out/refused-bundle/package"
printf '{}\n' > "$out/refused-bundle/package/manifest.json"
python3 "$repo/tests/ane/evaluate.py" --cli "$out/build/bin/whisper-cli" \
    --model "$out/models/ggml-tiny.en.bin" --dataset "$out/data/LibriSpeech/test-clean" \
    --bundle "$out/refused-bundle" --out "$out/refused-evaluation" --limit 2 --threads 8
