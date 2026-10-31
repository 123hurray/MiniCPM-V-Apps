# Audio8 Android system TTS

## Version 4.6 — pinned prefill output contract and QNN exception cleanup

The pinned slow graph emits only the last position: FP32 logits `[1,1,4097]`
and FP16 hidden `[1,1,896]`, even for a multi-token CPU prefill. Read these
outputs without a prompt-length offset; retain full prompt-length KV updates.
Unexpected shapes still fail with actual and expected dimensions in diagnostics.
QNN registration is now an owned RAII member: both normal teardown and a failed
constructor release sessions before unregistering the provider, then release Env.
Failed registration never unregisters somebody else's provider. Native tests
exercise the shape contract, rejected layouts, teardown order and repeated retries
with fake provider/session objects, without executing ORT/QNN on the build host.
Models, model hashes, compiled-cache paths and signing certificate are unchanged.
The user's v4.5 log confirms HTP initialization/compilation, not successful NPU
token generation, audible output, numerical parity or a speed improvement.

## Version 4.5 — bounded requests and concise status

Automatic voice-selection warmup has been removed, including for old preferences.
The home screen shows only readiness, the latest phase with elapsed time, and preview /
stop / recovery controls. Model, voice and performance details are collapsed by default.
TTS binder operations run off the UI thread. A separate-process watchdog ends only the
synthesis process after 240 seconds of initialization or 180 seconds of synthesis;
stop allows 15 seconds for native cancellation before the same process-only recovery.
A private receiver provides manual recovery without deleting models. Reconnect the
engine after a reset. Initialization phases and CPU AR / codec stages are persisted
before execution, so a failure can be localized. This bounds waiting; it does not
establish a successful NPU run or a measured latency improvement on the phone.

## Experimental branch: audio8-npu-int4

Version 4.4 integrates an experimental ORT/QNN bridge: CPU INT4 prefill, HTP INT4
single-token slow decode, configurable CPU INT4 / HTP fast decode, and the existing
CPU codec. It is not an all-NPU or phone-validated engine. The workflow publishes
separate downloadable models and binds their catalog hash into the APK. Model weights
are not bundled. Existing GGUF and reference files are retained.
See [setup, security constraints, and validation limits](docs/audio8-npu-int4.md).
The stable `audio8-system-tts` branch remains unchanged.

## Version 4.3 — Snapdragon 8 Gen 2 CPU latency optimization

- ARMv8.6-A FP16/dotprod/I8MM kernels and release LTO when supported. This build targets
  SM8550; it is not a portable ARMv8.2 APK.
- Project only the model-metadata semantic token rows and EOS (4096+1 instead of the
  full 155776-token head). Input embeddings remain intact; original token IDs, EOS,
  top-k/top-p/RAS and seed remain unchanged. Unsupported shapes fall back to the full head.
- Reuse the prefill graph only while both shape and target KV allocation match; reuse
  the single codec capacity graph. Turning graph retention off restores release-after-use.
- Independently configurable slow/codec and fast AR threads (default 5/2); optional
  affinity to the five highest-frequency allowed cores, discovered from sysfs instead
  of hard-coded CPU IDs. Severe thermal state caps active slow threads to 3.
- Background model/short-utterance warmup when selecting a voice; a real request cancels
  warmup at an AR frame boundary. Native cancellation between AR frames and fast steps.
- LRU PCM cache in the synthesis process, bounded to 16 MiB / 32 entries / 4 MiB per
  utterance. Keys include text, voice/reference, transcript, rate/pitch and native settings.
  Speech and PCM are not persisted. Settings use atomic files so the separate synthesis
  process sees changes without stale SharedPreferences.
- Phase timings and first PCM / total / cache hit / threads / thermal state diagnostics.
  UI exposes all switches and baseline/recommended presets. Changing native settings
  reloads the session on the next request; it does not mutate an existing graph's shape.

These are implemented optimizations, not measured SM8550 speedup claims. CI tests compare
F32 and Q8 full/compact projection at batch 1 and 4 (including EOS mapping), cancellation,
PCM cache LRU/limits, source+merged TTS manifests, Android Java/native compilation and lint.
On-device end-to-end audio quality, peak PSS and latency still need measurement.

Not silently enabled: INT4 changes the quantization/quality tradeoff; the available LiteRT
export is Preview rather than this Realistic model. QNN/HTP needs a new supported static
quantized graph export and calibration. Existing GGUF CPU graphs do not become NPU graphs
through a build flag. Vulkan/codec correctness is not validated for this app on Adreno 740.
The current build deliberately remains CPU-only. Real streaming codec state is not implemented;
no independent text chunking or overlapping-window approximation is used to hide latency.

Version 4.2 fixes engine discovery: the TTS service filter now declares
`android.intent.category.DEFAULT`, required by Android TtsEngines' use of
`PackageManager.MATCH_DEFAULT_ONLY`. Source and merged manifests are checked in CI.
The app also includes a package-manager discovery check for on-device diagnosis.

Version 4.1 adds an editable HTTPS Hugging Face mirror origin, defaulting to
`https://hf-mirror.com`, with an official-direct shortcut. Both download the identical
pinned model path and must pass the same SHA-256 check. Switching endpoints keeps
the existing partial model. Settings take effect on the next download attempt.
HF-Mirror is a third-party mirror, not a Hugging Face-operated service.

Audio8 release certificate SHA-256 (reuse for every future update):
`42bf8536d97d3a3e6ccb42252a91cd471296045c6455eb7d55d202bb868a8905`.
Private key recovery is kept separately, never in this repository.

This branch replaces the previous multimodal/agent/ASR application with an Android-only
Audio8-TTS-Realistic-0.6B engine. Other branches are unchanged. No model is bundled.

## Scope

- Android 13+, arm64, ARMv8.6-A FP16/dot-product/I8MM; aimed at Snapdragon 8 Gen 2.
- Standalone application ID `com.example.audio8tts`, coexists with the old app.
- Android `TextToSpeechService`, Chinese/English advertised initially, offline synthesis.
- Pinned audio.cpp CPU backend; no untested GPU switch.
- Download/HTTP resume/SHA-256 validation for the 1,429,460,800-byte Q8 model.
- User-imported reference WAV and accurate transcript; no automatic speaker changes.
- Diagnostics export through SAF, including native stderr and available Android exit traces.
- No chat, ASR, agents, shell, Python, camera, vision, or other TTS model.

## Experimental limitations — do not represent this as phone-tested

Cloud compilation is not proof of audible output, quality, memory safety, or real-time
performance on the phone. Validate on an SM8550 before production use.
Current upstream synthesis completes an utterance before returning PCM. A request is never
split independently to work around OOM. Requests over 500 Unicode code points fail explicitly.
Stopping suppresses subsequent PCM immediately and cancels at the next AR/fast-AR boundary.
Model loading, prefill and one in-flight codec compute remain non-interruptible.
Independent pitch control/time-stretch is not implemented;
speech rate uses resampling and changes pitch. Use default 100/100 settings for natural output.
Reference-free output is not guaranteed to preserve the same speaker across requests.
Not yet suitable for low-latency screen-reader navigation.

## Build

Run `bash scripts/prepare-runtime.sh`, then `cd Audio8-Android && bash gradlew :app:assembleRelease`.
Runtime is pinned to `3a0dd44293fc706f5c375ec96d3e6fd9c54b6c45`.
`scripts/audio8-mobile.patch` applies the audited mobile changes to that exact revision;
the setup fails instead of guessing if the patch no longer applies. It is idempotent.
The dedicated GitHub Actions workflow builds on pushes to `audio8-system-tts`.
Only `audio8_tts` is selected in the native model composite.

## Persistent signing

The historical release certificate is documented in the parent branch; its private key is not
recoverable from an APK or certificate fingerprint. Configure repository Actions secrets:
`ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`.
Reuse that same key for all future builds. Without it, the workflow deliberately uploads unsigned
signing inputs only, not an installable APK. Never generate an ephemeral debug signing key.

## Phone validation

1. Install the signed APK, download the model, import an authorized clean reference and transcript.
2. Open system text-to-speech settings and choose Audio8; default speed and pitch.
3. Test short Chinese, English, 100/300/500-character text, queue/flush/stop, and service restart.
4. Test through a second application (e.g. a reader using Android TTS), not only the preview UI.
5. Measure peak PSS, first audible PCM, audio duration/generation duration, temperature over 10 min.
6. Export diagnostics after errors. No claim of completed device testing without results.

## Sources and license

- https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b (Apache-2.0)
- https://huggingface.co/audio-cpp/Audio8-TTS-Realistic-0.6b-GGUF
- https://github.com/0xShug0/audio.cpp
- https://developer.android.com/reference/android/speech/tts/TextToSpeechService

Retain upstream license/NOTICE when distributing runtime or model weights. Voice cloning requires
permission from the speaker. Native/download logs may contain paths or upstream diagnostic data;
review exported logs before sharing.
