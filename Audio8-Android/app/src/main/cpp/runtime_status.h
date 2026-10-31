#pragma once
#include <chrono>
#include <cstdio>
#include <string>
#include <unistd.h>
inline std::string audio8_status_path;
inline void audio8_stage(const std::string & stage) {
    const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::fprintf(stderr,"Audio8 stage timestamp_ms=%lld %s\n",static_cast<long long>(ms),stage.c_str());
    if(audio8_status_path.empty()) return;
    const std::string temporary=audio8_status_path+".tmp";
    if(FILE * file=std::fopen(temporary.c_str(),"w")) {
        std::fprintf(file,"%lld\n%s",static_cast<long long>(ms),stage.c_str());
        std::fflush(file);fsync(fileno(file));std::fclose(file);
        std::rename(temporary.c_str(),audio8_status_path.c_str());
    }
}
