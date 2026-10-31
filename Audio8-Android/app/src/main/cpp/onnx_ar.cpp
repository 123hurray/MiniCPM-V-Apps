#include "onnx_ar.h"
#include "network_guard.h"
#include "runtime_status.h"
#include "onnx_contract.h"
#include "provider_registration.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using Half=Ort::Float16_t;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
struct State {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names,output_names;
    std::vector<const char *> outputs;
    explicit State(Ort::Env & env,const std::string & path,const Ort::SessionOptions & options) {
        session=std::make_unique<Ort::Session>(env,path.c_str(),options);
        Ort::AllocatorWithDefaultOptions allocator;
        for(size_t i=0;i<session->GetInputCount();++i) input_names.emplace_back(session->GetInputNameAllocated(i,allocator).get());
        for(size_t i=0;i<session->GetOutputCount();++i) output_names.emplace_back(session->GetOutputNameAllocated(i,allocator).get());
        for(auto & name:output_names) outputs.push_back(name.c_str());
    }
};
struct Feeds {
    Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::vector<std::string> names;
    std::vector<Ort::Value> tensors;
    template<class T> void add(const std::string & name,T * data,size_t count,std::initializer_list<int64_t> dims) {
        names.push_back(name);
        tensors.push_back(Ort::Value::CreateTensor<T>(memory,data,count,dims.begin(),dims.size()));
    }
    std::vector<Ort::Value> run(State & state) {
        if(names.size()!=state.input_names.size()) throw std::runtime_error("Audio8 ORT input contract mismatch");
        std::vector<const char *> ptrs;for(auto & s:names) ptrs.push_back(s.c_str());
        return state.session->Run(Ort::RunOptions{nullptr},ptrs.data(),tensors.data(),tensors.size(),state.outputs.data(),state.outputs.size());
    }
};
const Half * half_data(const Ort::Value & value,size_t count) {
    auto info=value.GetTensorTypeAndShapeInfo();
    if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 || info.GetElementCount()!=count)
        throw std::runtime_error("Audio8 ORT FP16 tensor contract mismatch");
    return value.GetTensorData<Half>();
}
const float * float_data(const Ort::Value & value,size_t count) {
    auto info=value.GetTensorTypeAndShapeInfo();
    if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetElementCount()!=count)
        throw std::runtime_error("Audio8 ORT FP32 logits contract mismatch");
    return value.GetTensorData<float>();
}
}

struct OnnxAr::Impl {
    Ort::Env env{nullptr};
    ProviderRegistration<Ort::Env> registration{env};
    std::unique_ptr<State> prefill,slow,fast;
    std::vector<Ort::ConstEpDevice> devices;
    std::vector<std::vector<Half>> slow_cache,fast_cache;
    std::string last_error;
    bool fast_npu;
    double prefill_ms=0,slow_ms=0,fast_ms=0;
    int64_t slow_calls=0,fast_calls=0;

    Impl(const std::string & models,const std::string & libraries,const std::string & cache,int threads,int fast_threads,bool npu)
        : fast_npu(npu) {
        audio8_stage("检查推理进程隔离");
        require_offline_synthesis_process();
        const std::string dsp=libraries+";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp";
        if(setenv("ADSP_LIBRARY_PATH",dsp.c_str(),1)!=0) throw std::runtime_error("Cannot configure Qualcomm DSP library path");
        audio8_stage("初始化 ONNX Runtime");
        env=Ort::Env(ORT_LOGGING_LEVEL_WARNING,"audio8-offline");
        env.DisableTelemetryEvents();
        audio8_stage("注册 QNN 后端");
        registration.acquire("QNNExecutionProvider",libraries+"/libonnxruntime_providers_qnn.so");
        audio8_stage("查找 NPU 设备");
        for(auto device:env.GetEpDevices())
            if(std::string(device.EpName())=="QNNExecutionProvider" && device.Device().Type()==OrtHardwareDeviceType_NPU) devices.push_back(device);
        if(devices.empty()) throw std::runtime_error("QNN HTP device not found; NPU mode will not fall back to CPU");
        std::filesystem::create_directories(cache);
        auto cpu_options=[&](int n) {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(std::clamp(n,1,8));options.SetInterOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            return options;
        };
        auto npu_state=[&](const std::string & name) {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
            const std::string context=cache+"/"+name+"_ctx.onnx";
            const bool cached=std::filesystem::is_regular_file(context);
            if(!cached) {
                options.AddConfigEntry("ep.context_enable","1");
                options.AddConfigEntry("ep.context_file_path",context.c_str());
            }
            std::unordered_map<std::string,std::string> config={
                {"backend_path",libraries+"/libQnnHtp.so"},{"soc_model","43"},{"htp_arch","73"},
                {"htp_performance_mode","sustained_high_performance"},
                {"htp_graph_finalization_optimization_mode","3"},
                {"profiling_level","basic"},{"profiling_file_path",cache+"/qnn-"+name+".csv"}};
            options.AppendExecutionProvider_V2(env,devices,config);
            audio8_stage(std::string(cached?"加载 NPU 缓存 · ":"编译 NPU 图 · ")+name);
            auto state=std::make_unique<State>(env,cached?context:models+"/"+name+"_static.onnx",options);
            std::fprintf(stderr,"Audio8 QNN initialized part=%s backend=HTP soc=43 arch=73 strict_no_cpu_fallback=1 cached_context=%d\n",name.c_str(),cached);
            return state;
        };
        audio8_stage("加载 CPU INT4 预填充模型");
        prefill=std::make_unique<State>(env,models+"/slow_ar_int4.onnx",cpu_options(threads));
        slow=npu_state("slow");
        audio8_stage("加载 Fast AR");
        fast=fast_npu?npu_state("fast"):std::make_unique<State>(env,models+"/fast_ar_int4.onnx",cpu_options(fast_threads));
        if(prefill->outputs.size()!=50 || slow->outputs.size()!=50 || fast->outputs.size()!=9)
            throw std::runtime_error("Audio8 ORT output count does not match pinned 24/4-layer model");
        std::fprintf(stderr,"Audio8 AR placement prefill=CPU_INT4 slow_decode=QNN_HTP_INT4 fast=%s codec=CPU telemetry=disabled internet_sockets=denied\n",fast_npu?"QNN_HTP_INT4":"CPU_INT4");
    }
    ~Impl() {
        fast.reset();slow.reset();prefill.reset();
    }
    void update_cache(std::vector<std::vector<Half>> & cache,const std::vector<Ort::Value> & outputs,
                      size_t offset,int64_t steps,int64_t position,int64_t capacity) {
        if(position<0 || steps<1 || position+steps>capacity || outputs.size()!=offset+cache.size())
            throw std::runtime_error("Audio8 KV update out of bounds");
        for(size_t i=0;i<cache.size();++i) {
            const Half * delta=half_data(outputs[offset+i],static_cast<size_t>(2*steps*64));
            for(int head=0;head<2;++head)
                std::copy_n(delta+head*steps*64,static_cast<size_t>(steps*64),cache[i].data()+(head*capacity+position)*64);
        }
    }
    void add_cache(Feeds & feeds,std::vector<std::vector<Half>> & cache,int layers,int capacity) {
        for(int i=0;i<layers;++i) for(int kv=0;kv<2;++kv) {
            auto & values=cache[static_cast<size_t>(2*i+kv)];
            feeds.add("cache_"+std::string(kv?"value_":"key_")+std::to_string(i),values.data(),values.size(),{1,2,capacity,64});
        }
    }
    void forward(int phase,const int32_t * input,int64_t rows,int64_t steps,int64_t position,int codebook,
                 const float * hidden,float * logits,int64_t logits_count,float * hidden_out,int64_t hidden_count) {
        auto start=Clock::now();
        Feeds feeds;
        if(phase==0 || phase==1) {
            if(!input || rows!=11 || steps<1 || position<0 || position+steps>2048 ||
               (phase==1 && steps!=1) || logits_count!=155776 || !hidden_out || hidden_count!=896)
                throw std::runtime_error("Audio8 slow forward contract mismatch");
            if(phase==0) {
                prefill_ms=slow_ms=fast_ms=0;slow_calls=fast_calls=0;
                slow_cache.assign(48,std::vector<Half>(2*2048*64,Half::FromBits(0)));
            }
            if(slow_cache.size()!=48) throw std::runtime_error("Audio8 prefill has not initialized KV cache");
            std::vector<int64_t> codes(input,input+rows*steps),positions(static_cast<size_t>(steps));
            for(int64_t i=0;i<steps;++i) positions[static_cast<size_t>(i)]=position+i;
            feeds.add("codes",codes.data(),codes.size(),{1,11,steps});
            feeds.add("input_pos",positions.data(),positions.size(),{steps});
            add_cache(feeds,slow_cache,24,2048);
            auto outputs=feeds.run(phase==0?*prefill:*slow);
            require_last_token_output(outputs[0].GetTensorTypeAndShapeInfo().GetShape(),4097,"slow logits");
            require_last_token_output(outputs[1].GetTensorTypeAndShapeInfo().GetShape(),896,"slow hidden");
            const float * values=float_data(outputs[0],4097);
            std::fill_n(logits,static_cast<size_t>(logits_count),-std::numeric_limits<float>::infinity());
            for(int i=0;i<4096;++i) logits[151678+i]=values[i];
            logits[151645]=values[4096];
            const Half * h=half_data(outputs[1],896);
            for(int i=0;i<896;++i) hidden_out[i]=h[i].ToFloat();
            update_cache(slow_cache,outputs,2,steps,position,2048);
            if(phase==0) prefill_ms=elapsed(start); else { slow_ms+=elapsed(start);++slow_calls; }
        } else if(phase==2) {
            if(!hidden || logits_count!=4096 || codebook<0 || codebook>=10 || position!=codebook || (codebook>0 && !input))
                throw std::runtime_error("Audio8 fast forward contract mismatch");
            if(codebook==0) fast_cache.assign(8,std::vector<Half>(2*10*64,Half::FromBits(0)));
            if(fast_cache.size()!=8) throw std::runtime_error("Audio8 fast KV state not initialized");
            std::array<Half,896> hidden_values;
            for(int i=0;i<896;++i) hidden_values[static_cast<size_t>(i)]=Half(hidden[i]);
            int64_t token=codebook==0?0:input[0],pos=position;
            bool use_hidden=codebook==0;
            feeds.add("slow_hidden",hidden_values.data(),hidden_values.size(),{1,1,896});
            feeds.add("token_id",&token,1,{1,1});
            feeds.add("use_slow_hidden",&use_hidden,1,{1});
            feeds.add("input_pos",&pos,1,{1});
            add_cache(feeds,fast_cache,4,10);
            auto outputs=feeds.run(*fast);
            const float * values=float_data(outputs[0],4096);
            std::copy_n(values,4096,logits);
            update_cache(fast_cache,outputs,1,1,position,10);
            fast_ms+=elapsed(start);++fast_calls;
        } else throw std::runtime_error("Unknown Audio8 forward phase");
    }
};

OnnxAr::OnnxAr(const std::string & models,const std::string & libraries,const std::string & cache,
               int threads,int fast_threads,bool fast_npu)
    : impl_(std::make_unique<Impl>(models,libraries,cache,threads,fast_threads,fast_npu)) {}
OnnxAr::~OnnxAr()=default;
int OnnxAr::forward(void * user,int phase,const int32_t * codes,int64_t rows,int64_t steps,int64_t position,int codebook,
                    const float * hidden,float * logits,int64_t logits_count,float * hidden_out,int64_t hidden_count) noexcept {
    auto & impl=*static_cast<OnnxAr *>(user)->impl_;
    try { impl.forward(phase,codes,rows,steps,position,codebook,hidden,logits,logits_count,hidden_out,hidden_count);return 0; }
    catch(const std::exception & e) { impl.last_error=e.what();std::fprintf(stderr,"Audio8 ORT error phase=%d: %s\n",phase,e.what());return 1; }
}
const char * OnnxAr::error(void * user) noexcept { return static_cast<OnnxAr *>(user)->impl_->last_error.c_str(); }
void OnnxAr::summary() const {
    const auto & i=*impl_;
    std::fprintf(stderr,"Audio8 ORT timings prefill_cpu_ms=%.2f slow_htp_ms=%.2f fast_ms=%.2f slow_calls=%lld fast_calls=%lld fast_backend=%s\n",
        i.prefill_ms,i.slow_ms,i.fast_ms,static_cast<long long>(i.slow_calls),static_cast<long long>(i.fast_calls),i.fast_npu?"HTP":"CPU");
}
