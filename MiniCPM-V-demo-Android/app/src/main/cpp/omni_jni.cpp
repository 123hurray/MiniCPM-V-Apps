// JNI bridge for VoxCPM2 TTS via llama.cpp-omni voxcpm2_runtime.
//
// Exposes a minimal API to Kotlin:
//   - init(baseLmPath, acousticPath)      -> bool
//   - generate(text, cfg, tsteps, out)    -> bool
//   - generateWithClone(text, cfg, tsteps, refWav, out) -> bool
//   - free()

#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "voxcpm2_runtime.h"
#include "llama.h"
#include "diagnostic_log.h"
#include "runtime_config.h"

#define TAG "omni_jni"
#define LOG_I(...) do { __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__); diagnostic_log_printf(ANDROID_LOG_INFO, TAG, __VA_ARGS__); } while (0)
#define LOG_E(...) do { __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__); diagnostic_log_printf(ANDROID_LOG_ERROR, TAG, __VA_ARGS__); } while (0)

static VoxCPM2Runtime * g_runtime = nullptr;
static std::string g_base_lm_path;
static std::string g_acoustic_path;
static std::string g_native_lib_dir;
static bool g_accelerator_active = false;
static bool g_fell_back_to_cpu = false;
static double g_last_generation_ms = 0.0;

namespace {

// The AudioVAE decoder builds a graph for every generated latent patch at once.
// On phones, long CJK input can therefore produce a very large transient graph
// (VoxCPM2 uses roughly text_tokens * 3 + 15 acoustic steps). Keep each graph
// bounded and join the resulting PCM instead of risking a native OOM/abort.
constexpr size_t kMaxTtsChunkCodepoints = 12;
constexpr int    kMaxTtsChunkSteps      = 64;
constexpr int    kInterChunkSilenceMs   = 35;
constexpr int    kChunkFadeMs           = 5;

struct Utf8Unit {
    size_t   begin;
    size_t   end;
    uint32_t codepoint;
};

static std::vector<Utf8Unit> utf8Units(const std::string & text) {
    std::vector<Utf8Unit> result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const size_t begin = i;
        const auto lead = static_cast<uint8_t>(text[i]);
        uint32_t cp = lead;
        size_t width = 1;
        if ((lead & 0xE0u) == 0xC0u) {
            cp = lead & 0x1Fu;
            width = 2;
        } else if ((lead & 0xF0u) == 0xE0u) {
            cp = lead & 0x0Fu;
            width = 3;
        } else if ((lead & 0xF8u) == 0xF0u) {
            cp = lead & 0x07u;
            width = 4;
        }

        if (i + width > text.size()) {
            width = 1;
            cp = lead;
        } else if (width > 1) {
            bool valid = true;
            for (size_t j = 1; j < width; ++j) {
                const auto continuation = static_cast<uint8_t>(text[i + j]);
                if ((continuation & 0xC0u) != 0x80u) {
                    valid = false;
                    break;
                }
                cp = (cp << 6u) | (continuation & 0x3Fu);
            }
            if (!valid) {
                width = 1;
                cp = lead;
            }
        }
        i += width;
        result.push_back({begin, i, cp});
    }
    return result;
}

static bool isBreakCodepoint(uint32_t cp) {
    switch (cp) {
        case ' ': case '\t': case '\n': case '\r':
        case ',': case '.': case '!': case '?': case ';': case ':':
        case 0x3000: // ideographic space
        case 0x3001: // 、
        case 0x3002: // 。
        case 0xFF01: // ！
        case 0xFF0C: // ，
        case 0xFF1A: // ：
        case 0xFF1B: // ；
        case 0xFF1F: // ？
            return true;
        default:
            return false;
    }
}

static bool isWhitespaceCodepoint(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000;
}

static std::vector<std::string> splitTtsText(const std::string & text) {
    const std::vector<Utf8Unit> units = utf8Units(text);
    std::vector<std::string> chunks;
    size_t cursor = 0;
    while (cursor < units.size()) {
        while (cursor < units.size() && isWhitespaceCodepoint(units[cursor].codepoint)) {
            ++cursor;
        }
        if (cursor >= units.size()) break;

        size_t end = std::min(units.size(), cursor + kMaxTtsChunkCodepoints);
        if (end < units.size()) {
            // Prefer a natural boundary in the latter half of the chunk, while
            // still enforcing the hard memory bound.
            const size_t earliest = cursor + kMaxTtsChunkCodepoints / 2;
            for (size_t candidate = end; candidate > earliest; --candidate) {
                if (isBreakCodepoint(units[candidate - 1].codepoint)) {
                    end = candidate;
                    break;
                }
            }
        }

        size_t trimmedEnd = end;
        while (trimmedEnd > cursor && isWhitespaceCodepoint(units[trimmedEnd - 1].codepoint)) {
            --trimmedEnd;
        }
        if (trimmedEnd > cursor) {
            chunks.emplace_back(text.substr(units[cursor].begin,
                                            units[trimmedEnd - 1].end - units[cursor].begin));
        }
        cursor = end;
    }
    return chunks;
}

static void appendPcmChunk(std::vector<float> & output,
                           std::vector<float> chunk,
                           int sampleRate) {
    if (chunk.empty()) return;
    const size_t fadeSamples = std::min(chunk.size() / 2,
        static_cast<size_t>(std::max(0, sampleRate * kChunkFadeMs / 1000)));
    for (size_t i = 0; i < fadeSamples; ++i) {
        const float gain = static_cast<float>(i + 1) / static_cast<float>(fadeSamples + 1);
        chunk[i] *= gain;
        chunk[chunk.size() - 1 - i] *= gain;
    }
    if (!output.empty()) {
        output.insert(output.end(),
                      static_cast<size_t>(std::max(0, sampleRate * kInterChunkSilenceMs / 1000)),
                      0.0f);
    }
    output.insert(output.end(), chunk.begin(), chunk.end());
}

} // namespace

static bool initRuntime(bool forceCpu) {
    if (!g_native_lib_dir.empty()) {
        ggml_backend_load_all_from_path(g_native_lib_dir.c_str());
    } else {
        ggml_backend_load_all();
    }
    runtime_apply_thread_policy();

    const RuntimeBackendMode mode = forceCpu ? RuntimeBackendMode::Cpu : runtime_backend_mode();
    // VoxCPM2 is more than its llama BaseLM: the acoustic model and custom
    // operators share one ggml backend.  Those graphs have not been validated
    // against the Android Vulkan backend, and HTP/Hexagon is not packaged in
    // this APK.  Previously every non-CPU selection called init_best(), which
    // silently chose Vulkan and could abort inside the vendor driver.  Keep
    // speech generation on the optimised ARMv8.6 CPU path until each graph has
    // a tested accelerator implementation.
    const bool acceleratorRequested = mode != RuntimeBackendMode::Cpu;
    const bool useAccelerator = false;
    const int gpuLayers = 0;
    const std::string preferred;
    if (acceleratorRequested) {
        LOG_I("initRuntime: requested mode=%d is not safe for VoxCPM2; using CPU",
              static_cast<int>(mode));
    }

    auto * rt = new VoxCPM2Runtime();
    if (!rt->init(g_base_lm_path, g_acoustic_path, gpuLayers, useAccelerator,
                  runtime_thread_count(), 4096, preferred)) {
        LOG_E("initRuntime: %s init failed: %s",
              useAccelerator ? "accelerator" : "CPU", rt->last_error().c_str());
        delete rt;
        return false;
    }
    g_runtime = rt;
    g_accelerator_active = false;
    g_fell_back_to_cpu = acceleratorRequested;
    LOG_I("initRuntime: backend=%s threads=%d", rt->backend_name().c_str(), runtime_thread_count());
    return true;
}

static std::string jstringToStdString(JNIEnv * env, jstring jStr) {
    if (!jStr) return {};
    const char * chars = env->GetStringUTFChars(jStr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jStr, chars);
    return result;
}

// Read a WAV file as float32 mono PCM.
static bool readWavF32(const std::string & path, std::vector<float> & pcm, int * sampleRate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_E("readWavF32: cannot open %s", path.c_str());
        return false;
    }
    // Minimal WAV header parser
    uint8_t header[44];
    if (fread(header, 1, 44, f) != 44) {
        fclose(f);
        return false;
    }
    int16_t numChannels  = header[22] | (header[23] << 8);
    int32_t sr           = header[24] | (header[25] << 8) | (header[26] << 16) | (header[27] << 24);
    int16_t bitsPerSample = header[34] | (header[35] << 8);
    if (sampleRate) *sampleRate = sr;

    if (bitsPerSample != 16) {
        LOG_I("readWavF32: unsupported bits-per-sample %d, trying anyway", bitsPerSample);
    }

    // Read data chunk
    fseek(f, 0, SEEK_END);
    long totalSize = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    int totalSamples = totalSize / (bitsPerSample / 8);
    if (numChannels > 1) totalSamples /= numChannels;

    pcm.resize(totalSamples);
    std::vector<int16_t> tmp(totalSamples * numChannels);
    fread(tmp.data(), sizeof(int16_t), totalSamples * numChannels, f);
    fclose(f);

    for (int i = 0; i < totalSamples; ++i) {
        float sum = 0.0f;
        for (int c = 0; c < numChannels; ++c) {
            sum += tmp[i * numChannels + c] / 32768.0f;
        }
        pcm[i] = sum / numChannels;
    }
    return true;
}

// Write float32 mono PCM as 16-bit WAV.
static bool writeWavI16(const std::string & path, const std::vector<float> & pcm, int sampleRate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LOG_E("writeWavI16: cannot create %s", path.c_str());
        return false;
    }
    int32_t dataSize = (int32_t)pcm.size() * 2;
    int32_t chunkSize = 36 + dataSize;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&chunkSize, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    int32_t fmtSize = 16;
    fwrite(&fmtSize, 4, 1, f);
    int16_t audioFormat = 1; // PCM
    fwrite(&audioFormat, 2, 1, f);
    int16_t numChannels = 1;
    fwrite(&numChannels, 2, 1, f);
    fwrite(&sampleRate, 4, 1, f);
    int32_t byteRate = sampleRate * numChannels * 2;
    fwrite(&byteRate, 4, 1, f);
    int16_t blockAlign = numChannels * 2;
    fwrite(&blockAlign, 2, 1, f);
    int16_t bitsPerSample = 16;
    fwrite(&bitsPerSample, 2, 1, f);

    // data chunk
    fwrite("data", 1, 4, f);
    fwrite(&dataSize, 4, 1, f);
    for (float sample : pcm) {
        float clamped = std::max(-1.0f, std::min(1.0f, sample));
        int16_t val = (int16_t)(clamped * 32767.0f);
        fwrite(&val, 2, 1, f);
    }
    fclose(f);
    return true;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_minicpm_1v_1demo_TtsEngine_nativeInitOmni(
    JNIEnv * env, jclass /* cls */,
    jstring baseLmPath, jstring acousticPath, jstring nativeLibDir) {

    g_base_lm_path = jstringToStdString(env, baseLmPath);
    g_acoustic_path = jstringToStdString(env, acousticPath);
    g_native_lib_dir = jstringToStdString(env, nativeLibDir);
    g_fell_back_to_cpu = false;

    LOG_I("nativeInitOmni: baseLm=%s acoustic=%s", g_base_lm_path.c_str(), g_acoustic_path.c_str());

    if (g_runtime) {
        g_runtime->free();
        delete g_runtime;
        g_runtime = nullptr;
    }

    if (!initRuntime(false)) {
        if (runtime_backend_mode() == RuntimeBackendMode::Cpu || !initRuntime(true)) {
            return JNI_FALSE;
        }
        g_fell_back_to_cpu = true;
    }

    LOG_I("nativeInitOmni: success");
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_example_minicpm_1v_1demo_TtsEngine_nativeTtsGenerate(
    JNIEnv * env, jclass /* cls */,
    jstring text, jfloat cfgValue, jint timesteps,
    jstring refWavPath, jstring outputPath) {

    if (!g_runtime) {
        LOG_E("nativeTtsGenerate: runtime not initialized");
        return JNI_FALSE;
    }

    auto txt = jstringToStdString(env, text);
    auto refPath = jstringToStdString(env, refWavPath);
    auto outPath = jstringToStdString(env, outputPath);

    VoxCPM2GenerateParams params;
    params.cfg_value           = cfgValue;
    params.inference_timesteps = timesteps;
    params.max_steps           = kMaxTtsChunkSteps;

    std::vector<float> waveform;
    std::vector<float> refPcm;
    const auto started = std::chrono::steady_clock::now();
    runtime_apply_thread_policy();

    if (txt.empty()) {
        LOG_E("nativeTtsGenerate: text is empty");
        diagnostic_log_flush();
        return JNI_FALSE;
    }

    int referenceSampleRate = 0;
    if (!refPath.empty() && !readWavF32(refPath, refPcm, &referenceSampleRate)) {
        LOG_E("nativeTtsGenerate: failed to read reference WAV");
        diagnostic_log_flush();
        return JNI_FALSE;
    }
    if (referenceSampleRate > 0) {
        params.reference_sample_rate = referenceSampleRate;
    }

    const std::vector<std::string> chunks = splitTtsText(txt);
    const int sr = g_runtime->sample_rate();
    LOG_I("nativeTtsGenerate: begin bytes=%zu chunks=%zu maxCodepoints=%zu maxSteps=%d clone=%s",
          txt.size(), chunks.size(), kMaxTtsChunkCodepoints, kMaxTtsChunkSteps,
          refPath.empty() ? "false" : "true");
    diagnostic_log_flush();

    try {
        for (size_t i = 0; i < chunks.size(); ++i) {
            LOG_I("nativeTtsGenerate: chunk %zu/%zu begin bytes=%zu",
                  i + 1, chunks.size(), chunks[i].size());
            diagnostic_log_flush();

            std::vector<float> chunkWaveform = refPath.empty()
                ? g_runtime->generate(chunks[i], params)
                : g_runtime->generate_with_clone(chunks[i], refPcm, params);

            if (chunkWaveform.empty()) {
                LOG_E("nativeTtsGenerate: chunk %zu/%zu failed: %s",
                      i + 1, chunks.size(), g_runtime->last_error().c_str());
                diagnostic_log_flush();
                waveform.clear();
                break;
            }
            LOG_I("nativeTtsGenerate: chunk %zu/%zu complete samples=%zu",
                  i + 1, chunks.size(), chunkWaveform.size());
            appendPcmChunk(waveform, std::move(chunkWaveform), sr);
            diagnostic_log_flush();
        }
    } catch (const std::exception & e) {
        LOG_E("nativeTtsGenerate: C++ exception: %s", e.what());
        diagnostic_log_flush();
        waveform.clear();
    } catch (...) {
        LOG_E("nativeTtsGenerate: unknown C++ exception");
        diagnostic_log_flush();
        waveform.clear();
    }

    if (waveform.empty()) {
        LOG_E("nativeTtsGenerate: generation produced empty waveform: %s",
              g_runtime ? g_runtime->last_error().c_str() : "runtime unavailable");
        diagnostic_log_flush();
        return JNI_FALSE;
    }

    g_last_generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    if (!writeWavI16(outPath, waveform, sr)) {
        LOG_E("nativeTtsGenerate: failed to write output WAV");
        diagnostic_log_flush();
        return JNI_FALSE;
    }

    LOG_I("nativeTtsGenerate: success, %zu samples @ %d Hz -> %s",
          waveform.size(), sr, outPath.c_str());
    diagnostic_log_flush();
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_example_minicpm_1v_1demo_TtsEngine_nativeOmniRuntimeInfo(
    JNIEnv * env, jclass /* cls */) {
    std::string info = "backend=" + (g_runtime ? g_runtime->backend_name() : std::string("none"));
    info += ", accelerator=" + std::string(g_accelerator_active ? "true" : "false");
    info += ", cpuFallback=" + std::string(g_fell_back_to_cpu ? "true" : "false");
    info += ", generationMs=" + std::to_string(static_cast<long long>(g_last_generation_ms));
    return env->NewStringUTF(info.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_TtsEngine_nativeOmniFree(
    JNIEnv * /* env */, jclass /* cls */) {

    LOG_I("nativeOmniFree");
    if (g_runtime) {
        g_runtime->free();
        delete g_runtime;
        g_runtime = nullptr;
    }
    g_accelerator_active = false;
}

} // extern "C"
