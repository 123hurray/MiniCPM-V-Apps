#include "engine/community_models/audio8_tts/mobile_math.h"
#include "engine/community_models/audio8_tts/mobile.h"
#include "engine/community_models/audio8_tts/mobile_codec.h"
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
void codec_test(int threads,int channels,int outputs,int frames,int kernel,int stride,ggml_type type=GGML_TYPE_F32) {
    auto * ctx=ggml_init({16*1024*1024,nullptr,true});auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,threads);
    auto * w=ggml_new_tensor_3d(ctx,type,kernel,outputs,channels);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,frames,channels);
    auto * a=ggml_conv_transpose_1d(ctx,w,x,stride,0,1);
    auto * b=mobile_conv_transpose(ctx,w,x,stride);
    auto * alpha=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,channels);
    auto * s=ggml_sin(ctx,ggml_mul(ctx,x,ggml_reshape_2d(ctx,alpha,1,channels)));
    auto * old_snake=ggml_add(ctx,x,ggml_div(ctx,ggml_mul(ctx,s,s),ggml_reshape_2d(ctx,alpha,1,channels)));
    auto * new_snake=mobile_snake(ctx,x,alpha);
    auto * graph=ggml_new_graph_custom(ctx,2048,false);
    for(auto * tensor:{a,b,old_snake,new_snake})ggml_build_forward_expand(graph,tensor);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer!=nullptr,"codec allocation");
    std::vector<float> weights(kernel*outputs*channels),values(frames*channels),scales(channels);
    for(size_t i=0;i<weights.size();++i)weights[i]=std::sin(float(i)*0.031f)*0.1f;
    for(size_t i=0;i<values.size();++i)values[i]=std::cos(float(i)*0.013f)*1.5f;
    for(size_t i=0;i<scales.size();++i)scales[i]=0.05f+float(i%17)*0.11f;
    if(type==GGML_TYPE_F32)ggml_backend_tensor_set(w,weights.data(),0,weights.size()*sizeof(float));
    else {std::vector<ggml_fp16_t> half(weights.size());ggml_fp32_to_fp16_row(weights.data(),half.data(),weights.size());ggml_backend_tensor_set(w,half.data(),0,half.size()*sizeof(ggml_fp16_t));}
    ggml_backend_tensor_set(x,values.data(),0,values.size()*sizeof(float));
    ggml_backend_tensor_set(alpha,scales.data(),0,scales.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"codec compute");
    float maximum=0;
    for(auto pair:{std::make_pair(a,b),std::make_pair(old_snake,new_snake)}) {
        std::vector<float> first(ggml_nelements(pair.first)),second(first.size());
        ggml_backend_tensor_get(pair.first,first.data(),0,first.size()*sizeof(float));
        ggml_backend_tensor_get(pair.second,second.data(),0,second.size()*sizeof(float));
        for(size_t i=0;i<first.size();++i) {
            float error=std::abs(first[i]-second[i]);maximum=std::max(maximum,error);
            require(std::isfinite(second[i]) && error<0.00003f+std::abs(first[i])*0.00005f,"codec kernel mismatch");
        }
    }
    std::printf("codec parity threads=%d C=%d O=%d T=%d K=%d S=%d max_error=%.8g\n",threads,channels,outputs,frames,kernel,stride,maximum);
    ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);
}
int main() {
    try {
        run(GGML_TYPE_F32,1); run(GGML_TYPE_Q8_0,1);
        run(GGML_TYPE_F32,4); run(GGML_TYPE_Q8_0,4);
        for(int threads:{1,5})for(int stride:{2,4,8}) {
            codec_test(threads,32,16,1,2*stride,stride);
            codec_test(threads,64,32,19,2*stride,stride);
            codec_test(threads,64,32,19,2*stride,stride,GGML_TYPE_F16);
        }
        mobile_options.cancelled=[](void*){return 1;};
        bool cancelled=false;
        try { mobile_check_cancelled(); } catch(const std::exception&) { cancelled=true; }
        require(cancelled,"cancellation");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
