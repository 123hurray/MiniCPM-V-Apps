#include "diagnostic_log.h"

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <initializer_list>

#include <fcntl.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
std::atomic<int> g_log_fd{-1};
std::atomic<bool> g_handlers_installed{false};

char priority_letter(int priority) {
    switch (priority) {
        case ANDROID_LOG_ERROR: return 'E';
        case ANDROID_LOG_WARN: return 'W';
        case ANDROID_LOG_DEBUG: return 'D';
        case ANDROID_LOG_VERBOSE: return 'V';
        default: return 'I';
    }
}

void write_all(int fd, const char * data, size_t size) {
    while (size > 0) {
        const ssize_t written = write(fd, data, size);
        if (written <= 0) return;
        data += written;
        size -= static_cast<size_t>(written);
    }
}

void write_unsigned(int fd, unsigned long value, int base) {
    char buffer[32];
    size_t count = 0;
    do {
        const unsigned digit = value % static_cast<unsigned>(base);
        buffer[count++] = static_cast<char>(digit < 10 ? '0' + digit : 'a' + digit - 10);
        value /= static_cast<unsigned>(base);
    } while (value != 0 && count < sizeof(buffer));
    while (count > 0) write_all(fd, &buffer[--count], 1);
}

void native_signal_handler(int signal_number, siginfo_t * info, void *) {
    const int fd = g_log_fd.load(std::memory_order_relaxed);
    if (fd >= 0) {
        static constexpr char prefix[] = "\n[FATAL] native signal=";
        static constexpr char tid_text[] = " tid=";
        static constexpr char address_text[] = " address=0x";
        static constexpr char suffix[] = "\n";
        write_all(fd, prefix, sizeof(prefix) - 1);
        write_unsigned(fd, static_cast<unsigned long>(signal_number), 10);
        write_all(fd, tid_text, sizeof(tid_text) - 1);
        write_unsigned(fd, static_cast<unsigned long>(syscall(SYS_gettid)), 10);
        write_all(fd, address_text, sizeof(address_text) - 1);
        write_unsigned(fd, reinterpret_cast<unsigned long>(info ? info->si_addr : nullptr), 16);
        write_all(fd, suffix, sizeof(suffix) - 1);
        fsync(fd);
    }

    signal(signal_number, SIG_DFL);
    syscall(SYS_tgkill, getpid(), syscall(SYS_gettid), signal_number);
    _exit(128 + signal_number);
}
}

void diagnostic_log_set_path(const char * path) {
    if (!path || !*path) return;
    const int new_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (new_fd < 0) return;
    const int old_fd = g_log_fd.exchange(new_fd);
    if (old_fd >= 0) close(old_fd);
}

void diagnostic_log_install_crash_handlers() {
    if (g_handlers_installed.exchange(true)) return;
    struct sigaction action {};
    action.sa_sigaction = native_signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_RESETHAND;
    for (const int signal_number : {SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV}) {
        sigaction(signal_number, &action, nullptr);
    }
}

void diagnostic_log_write(int priority, const char * tag, const char * message) {
    const int fd = g_log_fd.load();
    if (fd < 0 || !message) return;
    char line[4096];
    const std::time_t now = std::time(nullptr);
    const int prefix = std::snprintf(line, sizeof(line), "[%lld] [%c] %s: ",
                                     static_cast<long long>(now), priority_letter(priority),
                                     tag ? tag : "native");
    if (prefix < 0) return;
    const size_t offset = static_cast<size_t>(prefix) < sizeof(line)
        ? static_cast<size_t>(prefix) : sizeof(line) - 1;
    const size_t available = sizeof(line) - offset - 2;
    const size_t message_size = std::min(std::strlen(message), available);
    std::memcpy(line + offset, message, message_size);
    size_t total = offset + message_size;
    if (total == 0 || line[total - 1] != '\n') line[total++] = '\n';
    write_all(fd, line, total);
}

void diagnostic_log_printf(int priority, const char * tag, const char * format, ...) {
    if (g_log_fd.load() < 0 || !format) return;
    char message[3072];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    diagnostic_log_write(priority, tag, message);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_NativeRuntime_nativeInitializeDiagnostics(
        JNIEnv * env, jobject, jstring log_path) {
    if (!log_path) return;
    const char * path = env->GetStringUTFChars(log_path, nullptr);
    diagnostic_log_set_path(path);
    env->ReleaseStringUTFChars(log_path, path);
    diagnostic_log_install_crash_handlers();
    diagnostic_log_printf(ANDROID_LOG_INFO, "NativeRuntime", "native diagnostics initialized");
}
