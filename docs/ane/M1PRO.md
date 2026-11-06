# M1 Pro lead qualification

These are payload commands for a controller-approved request for explicit device
`m1pro`. They are not an instruction to access a machine directly. This task's
implementation/testing uses only the buildbox. Use the qualified H13 kernel,
existing smoke/bring-up ladder and native aarch64 libane. Preserve driver and
runner failures; do not reload a wedged device or retry its contexts. Runtime
capability checks must report ABI 1 and chip 13. No session/presence checks are
part of this plan. Do not change clock, power, speaker protection or boot policy.

Build whisper.cpp and the fallback-enabled libane on the buildbox for aarch64,
then stage those artifacts and the public test data through the controller's
normal upload capability. Do not copy x86_64 buildbox executables to the Mac.
For an interactive cross build, use a buildbox build slot with an aarch64 Linux
CMake toolchain and an aarch64 libane archive. The ordinary CMake commands in
[README](README.md) work with `-DCMAKE_TOOLCHAIN_FILE=...` and the matching
`LIBANE_LIBRARY`/`LIBANE_INCLUDE_DIR`. Kernel/device setup is outside this patch.

Use `/opt/whisper-ane` as the staged working directory in these payloads. It
contains this fork, `build/bin/whisper-cli` with its matching shared libraries,
`converted/ggml-model.bin`, the reproducible `tiny.en-open` bundle and
`data/LibriSpeech/test-clean`. `/opt/omarchy-ane` is runtime commit
`1cfc4c745c5e1ee5d2c346a5e5e33bb9ada627f1`; `/opt/mil-hwx-compiler` is compiler
commit `c28fe541697d37d7ec88886bb6b81cef8329dc44`. Record the exact kernel and
binary hashes in the private receipt.

First inspect the compile report on the build host. If full-model rc is not zero,
there is no native program to qualify: run the missing-package CPU variant and
report fallback. Do not substitute Apple-compiled programs or override the chip
check. H14 and H16G must use their own future shared graph runner.

After a complete H13 package exists and its existing native ladder is qualified,
use this controller payload to compare encoder output, WER and realtime factor:

```sh
cd /opt/whisper-ane
export WHISPER_ANE_WORKER="$PWD/models/ane/worker.py"
export WHISPER_ANE_PYTHON=/opt/whisper-ane/venv/bin/python
export WHISPER_ANE_KEEP_RECEIPTS=1
export WHISPER_ANE_COMPILER=/opt/mil-hwx-compiler
export WHISPER_ANE_LIB=/opt/omarchy-ane/bindings/python/dylib/libane_python.so
python3 tests/ane/evaluate.py --cli build/bin/whisper-cli \
  --model converted/ggml-model.bin --dataset data/LibriSpeech/test-clean \
  --bundle tiny.en-open --out /tmp/whisper-m1pro-parity-1 \
  --limit 20 --threads 8 --require-ane
cat /tmp/whisper-m1pro-parity-1/results.json
cat /sys/class/accel/accel0/device/ane_stats
cat /sys/class/accel/accel0/device/wedged
cat /sys/class/accel/accel0/device/power/runtime_status
```

Use new output paths for repetitions (at least three paired runs). Record
`encoder_parity.max_abs_diff`, both arms' WER/edit counts, `wall_rtf`, `cpu_rtf`
and `encoder_rtf`. RTF is seconds per audio second. The `candidate_native` flag
must be true; a successful CLI run alone is not proof of native inference.
`ane_stats` job/busy counters must advance during native runs. The native runner
already checks each output against its MIL reference, which is a distinct check
from comparing the complete encoder with whisper.cpp's CPU output.
If one candidate falls back, the evaluator disables native requests for later
clips and finishes on CPU. Inspect the preserved failure before another native
qualification run; do not retry an uncertain or wedged native owner.

The current adapter includes process startup, weight hashing and the runner's
reference calculation on every encoder call. For an engine-only repeat benchmark,
use a public mel tensor and call the existing runner directly after qualification:

```sh
python /opt/mil-hwx-compiler/tools/h13_run_linux.py tiny.en-open/package \
  --mil tiny.en-open/model.mil --model-root tiny.en-open \
  --libane-library "$WHISPER_ANE_LIB" --deadline-seconds 600 \
  --input mel=/tmp/public-mel.fp16 --output encoder=/tmp/public-encoder.fp16 \
  --warmup 3 --iterations 20 --benchmark-json /tmp/whisper-encoder-native.json
```

`public-mel.fp16` must be the exact 480000-byte `[1,80,3000]` fp16 mel tensor
captured from the public clip and preserved by a qualification run (the worker's
`mel.fp16` in the printed receipt directory, retained by
`WHISPER_ANE_KEEP_RECEIPTS=1`). Copy that file to `/tmp/public-mel.fp16`. The 30-second fixed context is padding, not the clip's true
audio duration. Calculate realtime factor using the true clip duration in the
paired receipt. Preserve setup, reference-check and end-to-end timing scopes in
the runner report; never report program-submit time as whole-encoder wall time.

For read-only system power, identify the existing hwmon system-power label:

```sh
for label in /sys/class/hwmon/hwmon*/power*_label; do
  printf '%s: ' "$label"
  cat "$label"
done
```

Choose the system-total PSTR input, not a CPU-only/ANE-only sensor. Substitute
its actual `powerN_input` path for `SENSOR` below. The sampler reads microwatts
and emits watts, records five seconds before and after load and preserves ANE
runtime-PM/wedge/counter snapshots. It does not change policy or gate on those
snapshots. Without an accessible sensor, omit `--sensor`; the paired performance
run still executes and the receipt marks power unavailable.

```sh
python3 tests/ane/power.py --sensor SENSOR --out /tmp/whisper-m1pro-power-1 -- \
  python3 tests/ane/evaluate.py --cli build/bin/whisper-cli \
  --model converted/ggml-model.bin --dataset data/LibriSpeech/test-clean \
  --bundle tiny.en-open --out /tmp/whisper-m1pro-power-eval-1 \
  --limit 20 --threads 8 --require-ane
```

This measures system power across a paired CPU/candidate evaluation, including
model loading and parity capture. For per-arm power, wrap each CLI arm separately
in the same sampler with identical clip/settings. Subtract measured idle system
power from that arm's load average and integrate samples over its measured
interval to estimate system-delta joules. This is not isolated ANE energy. Record
idle-after recovery separately; do not claim an ANE power win from performance
alone. Audio remains data input and no playback is needed.

For example, after staging `public.wav` (a converted public LibriSpeech clip),
these are separate, directly comparable whole-process power payloads. Retain the
logs to confirm the native backend marker and driver job count; the candidate
must not silently be counted as ANE when it falls back.

```sh
python3 tests/ane/power.py --sensor SENSOR --out /tmp/whisper-power-cpu-1 -- \
  env -u WHISPER_ANE_ENCODER build/bin/whisper-cli \
  -m converted/ggml-model.bin -f public.wav -ng -nfa -t 8 -l en -nt
python3 tests/ane/power.py --sensor SENSOR --out /tmp/whisper-power-ane-1 -- \
  env WHISPER_ANE_ENCODER="$PWD/tiny.en-open" build/bin/whisper-cli \
  -m converted/ggml-model.bin -f public.wav -ng -nfa -t 8 -l en -nt
```
