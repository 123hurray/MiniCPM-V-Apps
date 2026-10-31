#pragma once
#include <cstdint>
#include <memory>
#include <string>

class OnnxAr {
public:
    OnnxAr(const std::string & models, const std::string & libraries,
           const std::string & cache, int threads, int fast_threads, bool fast_npu);
    ~OnnxAr();
    static int forward(void *, int, const int32_t *, int64_t, int64_t, int64_t, int,
                       const float *, float *, int64_t, float *, int64_t) noexcept;
    static const char * error(void *) noexcept;
    void summary() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
