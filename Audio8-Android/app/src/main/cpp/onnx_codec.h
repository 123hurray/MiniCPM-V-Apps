#pragma once
#include <cstdint>
#include <memory>
#include <string>
class OnnxCodec {
    struct Impl; std::unique_ptr<Impl> impl;
public:
    OnnxCodec(const std::string &,const std::string &,const std::string &);
    ~OnnxCodec();
    void prepare(int64_t frames);
    static int decode(void *,const int32_t *,int64_t,int64_t,const float **,int64_t *);
    static const char * error(void *);
};
