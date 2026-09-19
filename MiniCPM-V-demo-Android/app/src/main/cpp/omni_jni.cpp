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
#include <exception>
#include <string>
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
    rt->set_progress_callback([](const char * stage, int current, int total) {
        LOG_I("voxcpm2: stage=%s current=%d total=%d", stage ? stage : "unknown", current, total);
        diagnostic_log_flush();
    });
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
    params.max_steps           = 200;

    std::vector<float> waveform;
    std::vector<float> refPcm;
    const auto started = std::chrono::steady_clock::now();
    runtime_apply_thread_policy();

    int referenceSampleRate = 0;
    if (!refPath.empty()) {
        if (!readWavF32(refPath, refPcm, &referenceSampleRate)) {
            LOG_E("nativeTtsGenerate: failed to read reference WAV");
            diagnostic_log_flush();
            return JNI_FALSE;
        }
        params.reference_sample_rate = referenceSampleRate;
    }

    LOG_I("nativeTtsGenerate: continuous generation begin bytes=%zu clone=%s cfg=%.2f timesteps=%d",
          txt.size(), refPath.empty() ? "false" : "true", cfgValue, timesteps);
    diagnostic_log_flush();

    try {
        waveform = refPath.empty()
            ? g_runtime->generate(txt, params)
            : g_runtime->generate_with_clone(txt, refPcm, params);
    } catch (const std::exception & e) {
        LOG_E("nativeTtsGenerate: C++ exception: %s", e.what());
        diagnostic_log_flush();
        return JNI_FALSE;
    } catch (...) {
        LOG_E("nativeTtsGenerate: unknown C++ exception");
        diagnostic_log_flush();
        return JNI_FALSE;
    }

    if (waveform.empty() && g_accelerator_active) {
        LOG_E("nativeTtsGenerate: accelerator failed (%s); retrying once on CPU",
              g_runtime->last_error().c_str());
        g_runtime->free();
        delete g_runtime;
        g_runtime = nullptr;
        if (initRuntime(true)) {
            g_fell_back_to_cpu = true;
            waveform = refPath.empty()
                ? g_runtime->generate(txt, params)
                : g_runtime->generate_with_clone(txt, refPcm, params);
        }
    }

    if (waveform.empty()) {
        LOG_E("nativeTtsGenerate: generation produced empty waveform: %s",
              g_runtime ? g_runtime->last_error().c_str() : "runtime unavailable");
        diagnostic_log_flush();
        return JNI_FALSE;
    }

    g_last_generation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    int sr = g_runtime->sample_rate();
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
