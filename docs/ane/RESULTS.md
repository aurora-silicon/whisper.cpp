# Verification results

Status: Linux CPU fallback and compiler coverage are verified. No complete
Whisper encoder package compiles with the pinned compiler, so native encoder
parity, WER, realtime factor and power remain unqualified.

## Software checks

The Linux build at `31200669` with `WHISPER_ANE=ON`, `GGML_NATIVE=OFF`, GCC
16.1.1 and eight jobs passes on x86_64. All five CTest unit cases pass, including
adapter fallback and repeated native-handoff graph reuse. Sixteen Python checks
also pass: seven bundle/interface checks, two WER checks, five parity checks,
one cross-process retirement check and one compiler-classification check.
Python/shell syntax and whitespace checks pass. The no-sensor power-sampler
smoke test runs its command successfully and labels power unavailable.

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

Native qualification must use the checkpoint-matched converted ggml recorded
in [compiler-coverage.json](compiler-coverage.json). This host fallback run used
the downloaded model hash above; no cross-format or native parity is claimed.

## Compiler coverage and t-3651 handoff

Public tiny.en, base.en and small.en all export successfully with coremltools
8.3.0 / CPU PyTorch 2.5.0, using MIL program 1.3 / ios18. All six H13/H14 full
graphs return 65: H13 reports `h13.nonfoldable-transpose`, H14 reports
`h14.conv-outside-envelope`. No runnable encoder package is produced.

Across 120 isolated probes, five compile, 108 refuse by name and seven H13
linear probes reach the 60-second host timeout. Those timeouts are not counted
as operation refusals. Even successful probes expand substantially: a tiny.en
D-to-D linear emits 10,500 programs, and residual adds emit 9,000 / 12,000 /
18,000 programs for tiny/base/small. No native speed follows from compilation.

See [the measured operation/shape handoff](COVERAGE.md) and
[complete public compiler receipt](compiler-coverage.json) for exact shapes,
attributes, diagnostic codes, compile counts and timeout cases. Isolated
view-only reshape refusals are explicitly distinguished from composed-graph
coverage. Raw paths/logs and open weight blobs remain local.

The initial export job used ios16 and stopped at `unsupported-opset`; this
exporter configuration error was corrected in conversion and serialization,
then all models were rerun. Initial failed receipts are retained locally. The
corrected run has no parser/opset errors.

## Remaining native qualification

H13 uses the existing published compiler runner; H14/H16G graph execution is not
enabled. Even after a whole encoder compiles, a lead-run M1 Pro receipt is needed
for native-vs-CPU encoder error, paired WER, native job counters, CPU usage and
power. See [the exact lead payloads](M1PRO.md). The current process-per-call
adapter is deliberately transparent about setup/reference overhead and has no
resident whole-encoder performance claim.
