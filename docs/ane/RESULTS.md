# Verification results

Status: Linux build and software checks pass; no native Whisper encoder
inference result is claimed. Compiler and corpus jobs remain queued through the
controller. This page will be updated from their saved receipts before review.

## Software checks

Fifteen host standard-library tests pass: seven bundle/weight/interface
checks, two WER metric checks, five capture/parity checks and one cross-process
native-retirement check. Python syntax, shell syntax and whitespace
checks pass. The optional power sampler runs a command without a sensor and
correctly labels power unavailable. These are software checks only.

The Linux build with `WHISPER_ANE=ON` and `GGML_NATIVE=OFF` passes on x86_64
with GCC 16.1.1. All five unit tests pass, including the adapter fallback and
repeated native-handoff graph tests. The fourteen Python checks also pass on
Linux. The build used eight jobs and the pinned libane revision in
[the build instructions](README.md).

Real CPU/fallback LibriSpeech measurements are pending the buildbox corpus job.
No host RTF or real encoder max-abs number has been measured yet. No ANE
throughput, WER or power number is available.

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
