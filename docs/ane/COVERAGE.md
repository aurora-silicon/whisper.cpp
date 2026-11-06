# Whisper compiler coverage: t-3651 handoff

These are Linux compiler results for public MIT tiny.en, base.en and small.en
encoder weights, exported by coremltools 8.3.0 with CPU PyTorch 2.5.0. The MIL
uses program 1.3 / ios18. The compiler is the Aurora fork at
`c28fe541697d37d7ec88886bb6b81cef8329dc44`.
[The complete public receipt](compiler-coverage.json) records checkpoint, ggml,
MIL and weight-blob hashes, exact runtime/constant shapes, attributes, diagnostic
codes and text, return codes, host elapsed times and program counts. Private
build paths and command arguments have been removed. No device was accessed.

## Full graph and isolated results

All six full graphs return 65. H13 stops at `h13.nonfoldable-transpose` for the
convolution-output tail swap; H14 stops at `h14.conv-outside-envelope`. No
runnable whole-encoder package is produced. Earlier validation order can expose
a layout failure before a convolution failure; isolated probes reveal additional
requirements independently.

Each row below has 20 unique isolated operation/geometry probes, deduplicated
across repeated transformer layers. Counts exclude the full graph.

| Model | Target | Compiled | Refused (65) | Host timeout (124, 60 s) |
| --- | --- | ---: | ---: | ---: |
| tiny.en | H13 | 2 | 16 | 2 |
| base.en | H13 | 2 | 16 | 2 |
| small.en | H13 | 1 | 16 | 3 |
| tiny.en | H14 | 0 | 20 | 0 |
| base.en | H14 | 0 | 20 | 0 |
| small.en | H14 | 0 | 20 | 0 |

There are no parser/opset failures in the corrected run. Seven timeouts are
**not compiler refusals**: H13's two feed-forward linear shapes for every model,
and small.en's D-to-D projection. Their support is undetermined at this budget.
The compile job returns zero after retaining individual failures; this does not
mean that a full graph compiled.

## Exact geometry families to carry into t-3651

`D=384/512/768` and `H=6/8/12` for tiny/base/small respectively. Each symbolic
family below is expanded into exact per-model records in the JSON.

| Family | Exported geometry and attributes | Measured diagnostic / outcome |
| --- | --- | --- |
| Conv1d 1 | `[1,80,3000]`, weight `[D,80,3]`, stride 1 -> `[1,D,3000]` | `h13.conv-outside-envelope`, `h14.conv-outside-envelope` |
| Conv1d 2 | `[1,D,3000]`, weight `[D,D,3]`, stride 2 -> `[1,D,1500]` | Same convolution envelope refusals |
| Conv layout | `[1,D,1500] -> [1,1500,D]`, perm `[0,2,1]` | H13 `nonfoldable-transpose` (also full-graph failure); H14 `outside-parity-envelope` |
| Head layout | Head-splitting/combining transposes around `[1,1500,H,64]` and `[1,H,1500,64]` | H13 `nonfoldable-transpose`; H14 `outside-parity-envelope` |
| Positional add | Runtime `[1,1500,D]` plus constant `[1500,D]` | H13 `invalid-constant-input`; H14 `outside-parity-envelope` |
| Attention scale | Runtime `[1,1500,H,64]` times singleton tensor `[1,1,1,1]`, value 0.353515625 | H13 `invalid-constant-input`; H14 `outside-parity-envelope` |
| Affine layer norm | `[1,1500,D]`, axis -1, gamma/beta `[D]`, fp16 epsilon 0.000010013580322265625 | `h13.norm-outside-envelope`, `h14.norm-outside-envelope` |
| QK matmul | Two `[1,H,1500,64]` tensors, transpose_y=true -> `[1,H,1500,1500]` | H13 `matmul-outside-envelope`; H14 `unsupported-program` (runtime right operand) |
| Attention softmax | `[1,H,1500,1500]`, axis -1 | H13/H14 normalization envelope refusals |
| Scores times V | `[1,H,1500,1500]` times `[1,H,1500,64]` -> `[1,H,1500,64]` | Same batched/runtime matmul refusals |
| GELU | `[1,D,3000]`, `[1,D,1500]`, `[1,1500,4D]`, EXACT | H13 `unary-outside-envelope`; H14 `outside-parity-envelope` |
| Linear with bias | `[1,1500,D]` D-to-D, D-to-4D and 4D-to-D | H14 `linear-bias-unsupported`; H13 results below / timeouts |

For H13, affine layer norm violates more than one current requirement: the
diagnostic asks for no gamma/beta and epsilon exactly 1e-5, while conversion
retains learned scale/bias and rounds epsilon to fp16. The singleton attention
scale is a rank-4 broadcast tensor, whereas the compiler accepts an inline
scalar or a matching tensor. These are concrete importer/envelope requirements,
not evidence that the mathematical primitive is absent.

Isolated head reshapes `[1,1500,D] <-> [1,1500,H,64]` return
`h13.returned-input-alias`, because a view-only function has no producing
program. That is a probe-interface limitation: it does **not** prove that the
reshape cannot fold in a composed graph. Follow-up view probes should include
a producer and consumer. H14 reports `outside-parity-envelope` for these probes.

## Emission size and bounded compilation

| H13 isolated operation | tiny.en programs | base.en programs | small.en programs |
| --- | ---: | ---: | ---: |
| D-to-D linear with bias over 1500 tokens | 10,500 | 3,000 | Timeout at 60 s |
| Runtime residual add `[1,1500,D]` | 9,000 | 12,000 | 18,000 |
| D-to-4D feed-forward linear | Timeout at 60 s | Timeout at 60 s | Timeout at 60 s |
| 4D-to-D feed-forward linear | Timeout at 60 s | Timeout at 60 s | Timeout at 60 s |

These counts are emitted programs, not native timings or numerical parity. The
large expansion motivates bounded planning and program fusion before a
whole-encoder performance claim. Successfully compiled probe packages are
removed after recording counts; public MIL probes and open weight blobs remain
in the local build artifacts. No Apple-compiled program is substituted.

## Draft coverage work

Carry the JSON cases into t-3651 as separate requirements: materialized layout
movement / safe transpose folding; rank-3 Conv1d geometry and long-sequence
tiling; positional/singleton broadcast canonicalization; affine normalization
with its actual epsilon; these batched attention and softmax shapes; large GELU
shapes; H14 bias lowering; and bounded/fused H13 linear emission. Preserve the
existing named refusals until each path has an open CPU fp16 reference and the
required device qualification. No compiler gate is relaxed by this integration.
