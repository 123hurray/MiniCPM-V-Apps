# Audio8 Android system TTS

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

- Android 13+, arm64, ARMv8.2-A dot-product; aimed at Snapdragon 8 Gen 2.
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
Stopping suppresses subsequent PCM immediately, but cannot abort an in-flight native compute;
the next request waits for that compute. Independent pitch control/time-stretch is not implemented;
speech rate uses resampling and changes pitch. Use default 100/100 settings for natural output.
Reference-free output is not guaranteed to preserve the same speaker across requests.
Not yet suitable for low-latency screen-reader navigation.

## Build

Run `bash scripts/prepare-runtime.sh`, then `cd Audio8-Android && bash gradlew :app:assembleRelease`.
Runtime is pinned to `3a0dd44293fc706f5c375ec96d3e6fd9c54b6c45`.
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
