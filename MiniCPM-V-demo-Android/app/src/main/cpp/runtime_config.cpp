#include "runtime_config.h"

#include <android/log.h>
#include <jni.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <sstream>

#include "ggml-backend.h"
#include "diagnostic_log.h"

namespace {
constexpr const char * TAG = "NativeRuntime";
std::mutex g_mutex;
int g_threads = 4;
RuntimeBackendMode g_mode = RuntimeBackendMode::Auto;
std::vector<int> g_performance_cpus;

const char * mode_name(RuntimeBackendMode mode) {
    switch (mode) {
        case RuntimeBackendMode::Auto: return "auto";
        case RuntimeBackendMode::Cpu: return "cpu";
        case RuntimeBackendMode::Gpu: return "gpu";
        case RuntimeBackendMode::Hexagon: return "hexagon";
    }
    return "unknown";
}
}

void runtime_configure(int threads, const std::vector<int> & performance_cpus, RuntimeBackendMode mode) {
    if (mode == RuntimeBackendMode::Auto || mode == RuntimeBackendMode::Hexagon) {
        __android_log_print(ANDROID_LOG_WARN, TAG,
                            "backend %s has no safe device implementation; forcing CPU",
                            mode_name(mode));
        diagnostic_log_printf(ANDROID_LOG_WARN, TAG,
                              "backend %s has no safe device implementation; forcing CPU",
                              mode_name(mode));
        mode = RuntimeBackendMode::Cpu;
    }
    {
        std::lock_guard<std::mutex> guard(g_mutex);
        g_threads = std::clamp(threads, 1, 8);
        g_mode = mode;
        g_performance_cpus = performance_cpus;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "configured threads=%d mode=%s cpus=%zu",
                        runtime_thread_count(), mode_name(runtime_backend_mode()),
                        runtime_performance_cpus().size());
    diagnostic_log_printf(ANDROID_LOG_INFO, TAG, "configured threads=%d mode=%s cpus=%zu",
                          runtime_thread_count(), mode_name(runtime_backend_mode()),
                          runtime_performance_cpus().size());
    // The affinity mask is inherited by worker threads subsequently created
    // by llama.cpp and ExecuTorch.
    runtime_apply_thread_policy();
}

int runtime_thread_count() {
    std::lock_guard<std::mutex> guard(g_mutex);
    return g_threads;
}

RuntimeBackendMode runtime_backend_mode() {
    std::lock_guard<std::mutex> guard(g_mutex);
    return g_mode;
}

std::vector<int> runtime_performance_cpus() {
    std::lock_guard<std::mutex> guard(g_mutex);
    return g_performance_cpus;
}

bool runtime_apply_thread_policy() {
    const auto cpus = runtime_performance_cpus();
    if (cpus.empty()) return false;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (const int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &mask);
    }
    const bool ok = sched_setaffinity(0, sizeof(mask), &mask) == 0;
    if (!ok) {
        __android_log_print(ANDROID_LOG_WARN, TAG, "sched_setaffinity failed");
    }
    // Android permits an app to make its own worker thread moderately more
    // favourable. Ignore vendor kernels which reject the request.
    setpriority(PRIO_PROCESS, 0, -2);
    return ok;
}

std::string runtime_diagnostics() {
    std::ostringstream out;
    out << "threads=" << runtime_thread_count()
        << ", mode=" << mode_name(runtime_backend_mode())
        << ", affinity=";
    const auto cpus = runtime_performance_cpus();
    for (size_t i = 0; i < cpus.size(); ++i) {
        if (i) out << ',';
        out << cpus[i];
    }
    out << ", backends=";
    const size_t count = ggml_backend_reg_count();
    for (size_t i = 0; i < count; ++i) {
        if (i) out << ',';
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        out << (reg ? ggml_backend_reg_name(reg) : "unknown");
    }
    return out.str();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_NativeRuntime_nativeConfigureRuntime(
        JNIEnv * env, jobject, jint threads, jintArray cpu_ids, jint backend_mode) {
    std::vector<int> cpus;
    if (cpu_ids) {
        const jsize count = env->GetArrayLength(cpu_ids);
        std::vector<jint> values(static_cast<size_t>(count));
        env->GetIntArrayRegion(cpu_ids, 0, count, values.data());
        cpus.assign(values.begin(), values.end());
    }
    const auto mode = backend_mode >= 0 && backend_mode <= 3
        ? static_cast<RuntimeBackendMode>(backend_mode)
        : RuntimeBackendMode::Auto;
    runtime_configure(threads, cpus, mode);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_minicpm_1v_1demo_NativeRuntime_nativeRuntimeDiagnostics(
        JNIEnv * env, jobject) {
    const std::string text = runtime_diagnostics();
    return env->NewStringUTF(text.c_str());
}
