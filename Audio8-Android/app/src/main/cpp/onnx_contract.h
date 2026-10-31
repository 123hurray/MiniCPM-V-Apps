#pragma once
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void require_last_token_output(const std::vector<int64_t> & shape,int64_t width,const char * name) {
    // The pinned graph slices [:, -1:, :] BEFORE norm/head. Prefill input length
    // only controls KV deltas; it does not control the logits/hidden output length.
    if(shape==std::vector<int64_t>{1,1,width}) return;
    std::ostringstream message;message<<"Audio8 "<<name<<" shape=[";
    for(size_t i=0;i<shape.size();++i)message<<(i?",":"")<<shape[i];
    message<<"] expected=[1,1,"<<width<<"]";
    throw std::runtime_error(message.str());
}
