#include "onnx_codec.h"
#include "network_guard.h"
#include "provider_registration.h"
#include "runtime_status.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
}
struct OnnxCodec::Impl {
    Ort::Env env{nullptr};
    ProviderRegistration<Ort::Env> registration{env};
    std::unique_ptr<Ort::Session> session;
    std::vector<Ort::ConstEpDevice> devices;
    std::vector<int64_t> input;
    std::vector<float> pcm;
    std::string model,libraries,cache,last_error;
    int64_t session_frames=0;
    Impl(const std::string & m,const std::string & l,const std::string & c):model(m),libraries(l),cache(c) {
        require_offline_synthesis_process();
        const auto dsp=l+";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp";
        if(setenv("ADSP_LIBRARY_PATH",dsp.c_str(),1))throw std::runtime_error("Cannot set DSP path");
        env=Ort::Env(ORT_LOGGING_LEVEL_WARNING,"audio8-codec-offline");env.DisableTelemetryEvents();
        registration.acquire("QNNExecutionProvider",l+"/libonnxruntime_providers_qnn.so");
        for(auto device:env.GetEpDevices())
            if(std::string(device.EpName())=="QNNExecutionProvider" && device.Device().Type()==OrtHardwareDeviceType_NPU)devices.push_back(device);
        if(devices.empty())throw std::runtime_error("Codec QNN NPU unavailable; CPU fallback disabled");
        std::filesystem::create_directories(cache);
    }
    void prepare(int64_t frames) {
        if(session && session_frames==frames)return;
        session.reset();session_frames=0;
        Ort::SessionOptions options;options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
        Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options,"batch",1));
        Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options,"frames",frames));
        Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options,"samples",frames*2048));
        // Exact-length graphs preserve attention and boundary behavior; no padded frames or independent chunks.
        const auto context=cache+"/codec-f"+std::to_string(frames)+"-qnn251-v1.onnx";
        bool cached=std::filesystem::is_regular_file(context);
        if(!cached){options.AddConfigEntry("ep.context_enable","1");options.AddConfigEntry("ep.context_embed_mode","1");options.AddConfigEntry("ep.context_file_path",context.c_str());}
        std::unordered_map<std::string,std::string> config={
            {"backend_path",libraries+"/libQnnHtp.so"},{"soc_model","43"},{"htp_arch","73"},
            {"enable_htp_fp16_precision","1"},{"htp_performance_mode","sustained_high_performance"},
            {"htp_graph_finalization_optimization_mode","3"},
            {"profiling_level","basic"},{"profiling_file_path",cache+"/codec-qnn.csv"}};
        options.AppendExecutionProvider_V2(env,devices,config);
        audio8_stage(cached?"加载 codec NPU 缓存":"编译 codec NPU 图（实验）");auto start=Clock::now();
        try {session=std::make_unique<Ort::Session>(env,(cached?context:model+"/codec_decoder_fp16.onnx").c_str(),options);}
        catch(...){if(cached){std::error_code ec;std::filesystem::remove(context,ec);}throw;}
        if(session->GetInputCount()!=1 || session->GetOutputCount()!=1)throw std::runtime_error("Codec I/O count mismatch");
        Ort::AllocatorWithDefaultOptions allocator;
        if(std::string(session->GetInputNameAllocated(0,allocator).get())!="codes" || std::string(session->GetOutputNameAllocated(0,allocator).get())!="audio")throw std::runtime_error("Codec I/O names mismatch");
        session_frames=frames;
        // Bound context storage: retain only the most recent exact-length graph.
        for(auto & entry:std::filesystem::directory_iterator(cache)) {
            auto name=entry.path().filename().string();
            if(name.rfind("codec-f",0)==0 && entry.path()!=context){std::error_code ec;std::filesystem::remove(entry.path(),ec);}
        }
        std::fprintf(stderr,"Audio8 codec NPU prepare_ms=%.2f frames=%lld context_hit=%d strict_no_cpu_fallback=1 precision=fp16_attempt\n",ms(start),(long long)frames,cached);
    }
    void run(const int32_t * codes,int64_t books,int64_t frames,const float ** out,int64_t * count) {
        if(!codes || books!=10 || frames<1 || frames>4096)throw std::runtime_error("Codec codes shape out of bounds");
        auto total=Clock::now();prepare(frames);auto copy=Clock::now();
        input.assign(codes,codes+books*frames);
        for(auto value:input)if(value<0 || value>=4096)throw std::runtime_error("Codec code outside vocabulary");
        std::array<int64_t,3> dims{1,books,frames};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto tensor=Ort::Value::CreateTensor<int64_t>(memory,input.data(),input.size(),dims.data(),dims.size());
        double input_ms=ms(copy);const char * in="codes",*output="audio";
        audio8_stage("解码音频 · NPU codec");auto start=Clock::now();
        auto values=session->Run(Ort::RunOptions{nullptr},&in,&tensor,1,&output,1);double run_ms=ms(start);start=Clock::now();
        auto info=values.at(0).GetTensorTypeAndShapeInfo();
        if(info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape()!=std::vector<int64_t>{1,1,frames*2048})throw std::runtime_error("Codec PCM shape or dtype mismatch");
        const float * data=values[0].GetTensorData<float>();pcm.assign(data,data+frames*2048);
        for(float value:pcm)if(!std::isfinite(value))throw std::runtime_error("Non-finite NPU codec output");
        *out=pcm.data();*count=pcm.size();
        std::fprintf(stderr,"Audio8 codec backend=QNN_HTP input_ms=%.2f run_ms=%.2f output_ms=%.2f total_ms=%.2f frames=%lld audio_ms=%.2f rtf=%.4f\n",input_ms,run_ms,ms(start),ms(total),(long long)frames,frames*2048.0/44.1,run_ms/(frames*2048.0/44.1));
    }
};
OnnxCodec::OnnxCodec(const std::string & m,const std::string & l,const std::string & c):impl(std::make_unique<Impl>(m,l,c)){}
OnnxCodec::~OnnxCodec()=default;
int OnnxCodec::decode(void * p,const int32_t * c,int64_t b,int64_t f,const float ** o,int64_t * n){auto & self=*static_cast<OnnxCodec*>(p)->impl;try{self.run(c,b,f,o,n);return 0;}catch(const std::exception & e){self.last_error=e.what();std::fprintf(stderr,"Audio8 codec NPU failed: %s\n",e.what());return -1;}}
const char * OnnxCodec::error(void * p){return static_cast<OnnxCodec*>(p)->impl->last_error.c_str();}
