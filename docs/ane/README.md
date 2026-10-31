# Experimental Linux Whisper encoder with libane fallback

Enable with `-DWHISPER_ANE=ON`, then set `WHISPER_ANE_ENCODER` to an open
encoder bundle. The decoder, cross projections and fallback encoder run on CPU.
The option defaults off. It requires libane with the t-3650 fallback API
([runtime PR](https://github.com/aurora-silicon/omarchy-ane/pull/1), commit
`1cfc4c745c5e1ee5d2c346a5e5e33bb9ada627f1`).

The integration reserves the normal CPU convolution and encoder graphs even
when native execution is requested. libane calls the native encoder first;
on refusal, missing hardware, malformed/nonfinite output or runner error, its
CPU callback recomputes both graphs from the original mel. Native output is
committed only when complete and finite. A failed native callback is retired
for the state's lifetime. Missing packages never query hardware. A CPU error
still propagates to the caller.

Set `WHISPER_ANE_KEEP_RECEIPTS=1` to preserve successful public-test worker
inputs, outputs and runner logs as well as failures.

The native adapter delegates validation, physical packing, graph scheduling,
submission and cleanup to the existing `mil-hwx-compiler/tools/h13_run_linux.py`.
It uses a separate runner process per encoder call, including its CPU MIL
reference check. This is a correctness/qualification adapter; it does not yet
keep programs resident between utterances and its timings include process
startup, weight hashing, reference execution, transfers and setup. Do not
interpret those timings as ANE engine latency. No second driver interface,
model scheduler or package format is implemented here. The application retains
CPU graph memory, so native execution does not reduce that memory footprint.

H13 has this native graph adapter. H14 exports and compiler probes are included,
but the existing H13 graph runner refuses H14 and the adapter falls back. H16G
can be probed with `compile.py --target H16G`; it has no enabled Linux graph
execution path here. Unknown chips and ABIs are refused. No generation override,
module reload, driver reset or ownership change is performed. An in-flight
runner is allowed to finish its own cleanup; only its existing pre-program
budget is used, not an unsafe host-side kill timeout. Preserve failed worker
inputs/logs in the reported `/tmp/whisper-ane-*` directory and follow the existing
driver recovery policy before creating another native owner.

## Open export and compilation

Use Linux Python 3.12. Install the pinned CPU PyTorch wheel before the remaining
requirements. `check-export.sh` provisions this environment and builds the public
compiler; it runs only on the build host and never submits ANE work.

```sh
python3.12 -m venv export-venv
. export-venv/bin/activate
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r models/ane/requirements.txt
mkdir -p cache converted
python -c 'import whisper; whisper.load_model("tiny.en",device="cpu",download_root="cache")'
assets=$(python -c 'import whisper; from pathlib import Path; print(Path(whisper.__file__).parent.parent)')
python models/convert-pt-to-ggml.py cache/tiny.en.pt "$assets" converted
python models/ane/export.py --model tiny.en --cache cache \
  --ggml converted/ggml-model.bin --output tiny.en-open
python models/ane/compile.py tiny.en-open --compiler /opt/mil-hwx-compiler/build/mil-hwxc --target H13
python models/ane/compile.py tiny.en-open --compiler /opt/mil-hwx-compiler/build/mil-hwxc --target H14
```

Repeat for `base.en` and `small.en` (multilingual tiny/base/small are also accepted).
Use absent destination directories so reruns cannot pick up stale packages.
The tested compiler source is `aurora-silicon/mil-hwx-compiler` at
`c28fe541697d37d7ec88886bb6b81cef8329dc44`.

`export.py` traces the public MIT Whisper encoder with SDPA disabled and converts
it to coremltools' Python MIL IR (`convert_to="milinternal"`), without loading
any Core ML model. It serializes static operations and fp16 weight blobs using
the documented public MIL syntax. It does not invoke Apple's model compiler,
`xcrun`, `anecompile`, Core ML prediction or import compiled programs. The
[coremltools conversion documentation](https://apple.github.io/coremltools/docs/source/coremltools.converters.convert.html)
describes the MIL IR return type. Our serializer deliberately rejects control
flow and multiple-output operations instead of inventing lowering rules.

Generate the ggml model from the **same checkpoint**, as above. The bundle
records checkpoint, MIL, weight-blob and ggml SHA-256 hashes; the worker verifies
the ggml/MIL/weight hashes and exact `[1,80,3000] -> [1,1500,D]` interface before
querying capabilities. Hash binding detects changed files, not a maliciously
rewritten manifest or arbitrary semantic equivalence. Bundles are trusted local
build artifacts. Quantized ggml checkpoints are outside this comparison.

`compile-TARGET/results.json` retains full-graph and isolated-operation results,
exact shapes, commands, elapsed time and diagnostics. Only exit 65 represents a
compiler refusal; 64 is invocation/syntax usage, 74 is output I/O failure and 124
is a host timeout. Shape probes use the real open weights and share the exported
blob. One probe per op/input/output geometry avoids repeating identical layer
envelopes. Successful isolated packages are removed after recording program
counts. A successful complete H13 package alone is copied to `package/`.

## Build and CPU validation

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_ANE=ON \
  -DLIBANE_INCLUDE_DIR=/opt/omarchy-ane/libane \
  -DLIBANE_LIBRARY=/opt/omarchy-ane/libane/libane.a \
  -DWHISPER_ANE_WORKER_PATH="$PWD/models/ane/worker.py"
cmake --build build -j8
ctest --test-dir build -L unit --output-on-failure
python3 tests/ane/test_worker.py
```

Normal installed builds put the worker in `share/whisper/ane`. Set
`WHISPER_ANE_WORKER` for a relocated installation and `WHISPER_ANE_PYTHON` for the
Python interpreter containing NumPy. `WHISPER_ANE_LIB` names the existing libane
Python shim; `WHISPER_ANE_COMPILER` names the pinned compiler source checkout
containing the runner. Use a native-architecture libane build on the target.

`check-build.sh`, `check-export.sh` and `check-evaluate.sh` are reproducible
buildbox jobs. Invoke them through the controller with `device="buildbox"`,
`capability="execute"`, the task ID and an explicit timeout. They use at most
eight build jobs and do not invoke another buildslot.

The adapter tests use a fake arithmetic-free worker to verify missing packages,
successful handoff, nonfinite rejection, full CPU fallback and native retirement.
They are software checks, not evidence of native inference. Bundle tests reject
changed weights, changed MIL, mismatched dimensions, unknown schema and a
non-H13 package before device access.

## Evaluation and interpretation

Use [LibriSpeech test-clean](https://www.openslr.org/12/) (CC BY 4.0), with its
published archive MD5 `32fa31d27d2e1cad72775fee3f4849a9`. `evaluate.py` selects the
first 20 FLAC paths in lexical order, converts them to mono 16 kHz WAV and runs
CPU and candidate arms with eight threads, English decoding and flash attention
disabled. Audio is decoded as data; nothing is played. Both arms use the same
unquantized ggml weights and decoder settings. The receipt includes clip hashes,
hypotheses, word/edit counts, encoder captures, CPU time and wall time.

```sh
python3 tests/ane/evaluate.py --cli build/bin/whisper-cli \
  --model converted/ggml-model.bin --dataset data/LibriSpeech/test-clean \
  --bundle tiny.en-open --out results-tiny --limit 20 --threads 8
```

WER uses uppercase `[A-Z0-9']+` tokens and Levenshtein distance, with the same
normalization applied to references and hypotheses. It is a small reproducible
subset, not a full-corpus or official Whisper-normalizer score. Wall RTF is
process wall seconds divided by actual audio seconds, including model load and
parity capture. CPU RTF sums process user and system seconds divided by audio
seconds; eight threads can make this larger than wall RTF. Encoder RTF uses
whisper.cpp's encoder timer and includes CPU cross projection and native adapter
overhead. The metric is seconds per audio second; lower is faster.

`WHISPER_ENCODER_DUMP=prefix` captures each encoder result before cross
projection as `prefix-N.f32`, row-major `[1500,D]` float32. Output capture is
explicit and can contain private audio-derived features: preserve it locally,
and publish only public-test results. Evaluation compares every captured tensor
and reports maximum absolute error. Candidate fallback must never be labeled
native parity: the receipt requires native backend markers with no CPU marker.
`--require-ane` returns failure, after saving the receipt, when any candidate
call used CPU. Numerical tolerances for actual ANE outputs must be selected from
measured encoder errors and WER, not inferred from fallback equality.

See [lead commands](M1PRO.md) and the measured [results and compiler gaps](RESULTS.md).

## Licenses and provenance

Whisper source and public weights: [MIT](https://github.com/openai/whisper/blob/main/LICENSE).
coremltools: BSD-3-Clause; PyTorch: BSD-3-Clause; NumPy: BSD-3-Clause;
libane and mil-hwx-compiler: MIT. New integration code follows this repository's
MIT license. No Apple-compiled model/program/firmware bytes are included.
All changes and the PR target only the Aurora whisper.cpp fork.
