#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${BUILD_OUT:-$repo/build-ane-check}")
mkdir -p "$out"
if ! command -v cmake >/dev/null; then
    python3 -m venv "$out/build-tools"
    "$out/build-tools/bin/pip" install --only-binary=:all: cmake==4.1.2
    export PATH="$out/build-tools/bin:$PATH"
fi
git clone --quiet https://github.com/aurora-silicon/omarchy-ane.git "$out/libane-src"
git -C "$out/libane-src" checkout --quiet 1cfc4c745c5e1ee5d2c346a5e5e33bb9ada627f1
make -C "$out/libane-src/libane" -j8
make -C "$out/libane-src/bindings/python/dylib" -j8
cmake -S "$repo" -B "$out/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_ANE=ON -DWHISPER_BUILD_TESTS=ON -DGGML_NATIVE=OFF \
    -DLIBANE_INCLUDE_DIR="$out/libane-src/libane" \
    -DLIBANE_LIBRARY="$out/libane-src/libane/libane.a" \
    -DWHISPER_ANE_WORKER_PATH="$repo/models/ane/worker.py"
cmake --build "$out/build" -j8
ctest --test-dir "$out/build" -L unit --output-on-failure
python3 "$repo/tests/ane/test_worker.py"
python3 "$repo/tests/ane/test_evaluate.py"
python3 "$repo/tests/ane/test_compile.py"
python3 "$repo/tests/ane/evaluate.py" --help
