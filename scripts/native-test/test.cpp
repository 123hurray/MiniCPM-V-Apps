#include "engine/community_models/audio8_tts/mobile_math.h"
#include "engine/community_models/audio8_tts/mobile.h"
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace engine::models::audio8_tts;
void require(bool value,const char * message) { if(!value) throw std::runtime_error(message); }
void run(ggml_type type,int batch) {
    constexpr int dim=32,vocab=155776,start=151678,end=155773,eos=151645;
    auto * ctx=ggml_init({16*1024*1024,nullptr,true});
    auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,2);
    auto * w=ggml_new_tensor_2d(ctx,type,dim,vocab);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,dim,batch);
    auto * full=ggml_mul_mat(ctx,w,x);
    auto * speech=ggml_mul_mat(ctx,mobile_head_rows(ctx,w,start,end-start+1),x);
    auto * stop=ggml_mul_mat(ctx,mobile_head_rows(ctx,w,eos,1),x);
    auto * compact=ggml_concat(ctx,speech,stop,0);
    auto * graph=ggml_new_graph_custom(ctx,2048,false);
    ggml_build_forward_expand(graph,full); ggml_build_forward_expand(graph,compact);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    require(buffer!=nullptr,"allocation");
    std::vector<float> weights(dim*vocab),input(dim*batch);
    for(size_t i=0;i<weights.size();i++) weights[i]=std::sin(float(i)*0.019f)*0.2f;
    for(size_t i=0;i<input.size();i++) input[i]=std::cos(float(i)*0.13f);
    std::vector<unsigned char> bytes(ggml_nbytes(w));
    if(type==GGML_TYPE_F32) std::memcpy(bytes.data(),weights.data(),bytes.size());
    else require(ggml_quantize_chunk(type,weights.data(),bytes.data(),0,vocab,dim,nullptr)==bytes.size(),"quantize");
    ggml_backend_tensor_set(w,bytes.data(),0,bytes.size());
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
    std::vector<float> a(vocab*batch),b(4097*batch);
    ggml_backend_tensor_get(full,a.data(),0,a.size()*sizeof(float));
    ggml_backend_tensor_get(compact,b.data(),0,b.size()*sizeof(float));
    float error=0;
    for(int t=0;t<batch;t++) {
        std::vector<float> slice(b.begin()+t*4097,b.begin()+(t+1)*4097);
        auto expanded=mobile_expand_logits(slice,vocab,start,end,eos);
        for(int i=0;i<vocab;i++) {
            if((i>=start && i<=end) || i==eos) error=std::max(error,std::abs(expanded[i]-a[t*vocab+i]));
            else require(std::isinf(expanded[i]) && expanded[i]<0,"masked token");
        }
    }
    require(error<0.0001f,"compact projection differs from baseline");
    bool rejected=false;
    try { mobile_head_rows(ctx,w,-1,1); } catch(const std::exception&) { rejected=true; }
    require(rejected,"bounds validation");
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
    std::printf("head equivalence type=%s batch=%d max_error=%.8g\n",ggml_type_name(type),batch,error);
}
int main() {
    try {
        run(GGML_TYPE_F32,1); run(GGML_TYPE_Q8_0,1);
        run(GGML_TYPE_F32,4); run(GGML_TYPE_Q8_0,4);
        mobile_options.cancelled=[](void*){return 1;};
        bool cancelled=false;
        try { mobile_check_cancelled(); } catch(const std::exception&) { cancelled=true; }
        require(cancelled,"cancellation");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
