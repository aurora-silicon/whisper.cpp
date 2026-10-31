# Verification results

Status: Linux build and software checks pass; no native Whisper encoder
inference result is claimed. The corrected compiler job remains queued through the controller. This page will be updated from their saved receipts before review.

## Software checks

Sixteen host standard-library tests pass: seven bundle/weight/interface
checks, two WER metric checks, five capture/parity checks and one cross-process
native-retirement check and one compiler-classification check. Python syntax, shell syntax and whitespace
checks pass. The optional power sampler runs a command without a sensor and
correctly labels power unavailable. These are software checks only.

The Linux build at `31200669` with `WHISPER_ANE=ON` and `GGML_NATIVE=OFF` passes on x86_64
with GCC 16.1.1. All five unit tests pass, including the adapter fallback and
repeated native-handoff graph tests. All sixteen Python checks also pass on Linux. The build used eight jobs and the pinned libane revision in
[the build instructions](README.md).

## Host CPU and fallback measurements

Measured on an Intel Core i9-13900K Linux x86_64 build host, eight threads,
Release, `GGML_NATIVE=OFF` (AVX2 CPU backend), GPU and flash attention disabled.
The model is unquantized tiny.en; SHA-256
`921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
The dataset is the first 20 lexical LibriSpeech test-clean clips, 156.4701 seconds
and 410 reference words. Both arms use the same CPU decoder and settings.

| Arm | WER | Edits / words | Wall seconds | Wall RTF | CPU RTF | Encoder RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 3.1707% | 13 / 410 | 12.4695 | 0.07969 | 0.49770 | 0.04739 |
| Missing-package CPU fallback | 3.1707% | 13 / 410 | 12.0706 | 0.07714 | 0.48292 | 0.04581 |

All 20 encoder captures (11,520,000 float32 values) match byte for byte, maximum
absolute difference **0.0**. The present-but-invalid-package arm also matches
the CPU on two clips (1,152,000 values, max abs 0.0); both arms have 3 edits in
36 words, WER 8.3333%. Missing/invalid bundles are refused before device access.
After the first fallback, the evaluator requests CPU for the remaining clips.
The receipts include per-clip hypotheses, timing, audio hashes and every encoder
tensor hash: [20-clip receipt](librispeech-tiny-cpu-fallback.json),
[invalid-package receipt](librispeech-tiny-refused-fallback.json).

These are **CPU fallback parity and WER**, not ANE-vs-CPU numerical parity.
`candidate_native=false` in both receipts. This is one paired run on a shared
build host; small timing differences between the CPU arms are noise/cache
effects, not acceleration. Wall RTF includes process/model setup and parity
capture; encoder RTF also includes CPU cross projection. The subset and word
normalizer are described in [README](README.md). No ANE RTF or power was measured.

## Export configuration correction

The first Linux job exported tiny.en, base.en and small.en successfully, but all
H13/H14 full graphs and isolated probes stopped at
`ane.verify.unsupported-opset`: program 1.3 requires `ios18`, while the exporter
requested `ios16`. This was an exporter configuration error, not operation
coverage evidence. Both coremltools conversion and emitted headers now use
`ios18`; a corrected buildbox export/probe run is queued. The initial failed
receipts are preserved locally.

## Coverage handoff in t-3651 terms

The reproducible coverage command is `models/ane/compile.py BUNDLE --compiler
MIL_HWXC --target H13` (repeat for H14). It records the exact op/shape, full-graph
status, compiler diagnostic code, real open weight source and successful program
count. Exit 65 with parser/lexer diagnostics is an export/parser error, not an
operation refusal; timeouts and invocation failures are also errors. Raw job logs remain local; only public model
shapes and diagnostics should be copied into a public coverage report.

Architecture-derived candidate gaps to check against those receipts:

| Area | tiny / base / small encoder geometry | Coverage question |
| --- | --- | --- |
| First Conv1d | `[1,80,3000]`, weights `[D,80,3]`, stride 1 | rank-3 convolution and 3000-step sequence |
| Second Conv1d | `[1,D,3000]`, weights `[D,D,3]`, stride 2 -> `[1,D,1500]` | rank-3 convolution and 1500-step output |
| Q/K/V and attention output projection | `[1,1500,D]`, weights `[D,D]` | large-sequence linear/batched matmul |
| Feed-forward | `D -> 4D -> D` over 1500 tokens | 1536/2048/3072-wide activation and projections |
| Attention | `H=6/8/12`, head width 64, logits `[1,H,1500,1500]` | batched matmul, transpose and 1500-wide softmax |
| Normalization | `[1,1500,D]`, axis -1 with learned scale/bias | affine layer norm and this exact envelope |
| Activation/layout | GELU, casts, reshape and transpose at those shapes | dtype/layout support, not just op-name coverage |

`D=384/512/768`. These rows are model geometry, not measured compiler refusals.
The exact compiler results must replace speculation; an exporter/parser error
must be fixed before declaring a model operation unsupported.

## Remaining native qualification

H13 uses the existing published compiler runner; H14/H16G graph execution is not
enabled. Even after a whole encoder compiles, a lead-run M1 Pro receipt is needed
for native-vs-CPU encoder error, paired WER, native job counters, CPU usage and
power. See [the exact lead payloads](M1PRO.md). The current process-per-call
adapter is deliberately transparent about setup/reference overhead and has no
resident whole-encoder performance claim.
