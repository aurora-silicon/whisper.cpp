#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Reproducible Linux export/coverage job. Never submits ANE programs.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${BUILD_OUT:-$repo/build-ane-export}")
mkdir -p "$out"
python3 -m venv "$out/bootstrap"
"$out/bootstrap/bin/pip" install --only-binary=:all: uv==0.9.5
export UV_PYTHON_INSTALL_DIR="$out/python" UV_CACHE_DIR="$out/uv-cache"
"$out/bootstrap/bin/uv" venv --python 3.12 --seed "$out/venv"
export PATH="$out/venv/bin:$PATH"
pip install --only-binary=:all: torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r "$repo/models/ane/requirements.txt"
pip install --only-binary=:all: cmake==4.1.2 ninja==1.13.0
git clone --quiet https://github.com/aurora-silicon/mil-hwx-compiler.git "$out/compiler"
git -C "$out/compiler" checkout --quiet c28fe541697d37d7ec88886bb6b81cef8329dc44
export GNUSTEP_PREFIX="$out/gnustep" GNUSTEP_BUILD_ROOT="$out/gnustep-build" JOBS=8
"$out/compiler/scripts/verify-linux-compiler.sh" build-only
export LD_LIBRARY_PATH="$GNUSTEP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$out/cache" "$out/models"
# Conversion uses tokenizer/mel assets from the installed MIT Whisper package.
assets=$(python3 -c 'import whisper; from pathlib import Path; print(Path(whisper.__file__).parent.parent)')
for model in tiny.en base.en small.en; do
    python3 -c 'import whisper,sys; whisper.load_model(sys.argv[1],device="cpu",download_root=sys.argv[2])' "$model" "$out/cache"
    python3 "$repo/models/convert-pt-to-ggml.py" "$out/cache/$model.pt" "$assets" "$out/models"
    # The upstream converter writes ggml-model.bin regardless of checkpoint name.
    mv "$out/models/ggml-model.bin" "$out/models/ggml-$model.bin"
    python3 "$repo/models/ane/export.py" --model "$model" --cache "$out/cache" \
        --ggml "$out/models/ggml-$model.bin" --output "$out/$model" --threads 8
    for target in H13 H14; do
        python3 "$repo/models/ane/compile.py" "$out/$model" --compiler "$out/compiler/build/mil-hwxc" --target "$target" --timeout 60
    done
    # Coverage receipts and export data retained; large successful probe programs are removed by compile.py.
done
