#pragma once

#include <string>
#include <vector>

enum class RuntimeBackendMode : int {
    Auto = 0,
    Cpu = 1,
    Gpu = 2,
    Hexagon = 3,
};

void runtime_configure(int threads, const std::vector<int> & performance_cpus, RuntimeBackendMode mode);
int runtime_thread_count();
RuntimeBackendMode runtime_backend_mode();
std::vector<int> runtime_performance_cpus();
bool runtime_apply_thread_policy();
std::string runtime_diagnostics();

