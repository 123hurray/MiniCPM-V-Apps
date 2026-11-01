#include "onnx_codec.h"
#include "network_guard.h"
#include "provider_registration.h"
#include "runtime_status.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void memory_log(int part) {
    std::ifstream f("/proc/self/status");std::string line;
    while(std::getline(f,line))if(line.rfind("VmRSS:",0)==0 || line.rfind("VmPeak:",0)==0)
        std::fprintf(stderr,"Audio8 codec stage=%d %s\n",part,line.c_str());
}
struct Tensor {
    std::vector<uint8_t> bytes;std::vector<int64_t> shape;ONNXTensorElementDataType type;
    explicit Tensor(const Ort::Value & value) {
        auto info=value.GetTensorTypeAndShapeInfo();shape=info.GetShape();type=info.GetElementType();
        size_t width=type==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16?2:type==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT?4:type==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64?8:type==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32?4:type==ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL?1:0;
        if(!width || info.GetElementCount()>256*1024*1024/width)throw std::runtime_error("Codec boundary tensor exceeds memory contract");
        bytes.resize(info.GetElementCount()*width);std::memcpy(bytes.data(),value.GetTensorRawData(),bytes.size());
    }
    Tensor()=default;
};
constexpr int parts=14;
}
struct OnnxCodec::Impl {
    Ort::Env env{nullptr};ProviderRegistration<Ort::Env> registration{env};
    std::unique_ptr<Ort::Session> session;std::vector<Ort::ConstEpDevice> devices;
    std::vector<float> pcm;std::string model,libraries,cache,last_error;
    Impl(const std::string & m,const std::string & l,const std::string & c):model(m),libraries(l),cache(c+"/parts-v1") {
        require_offline_synthesis_process();
        const auto dsp=l+";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp";
        if(setenv("ADSP_LIBRARY_PATH",dsp.c_str(),1))throw std::runtime_error("Cannot set DSP path");
        env=Ort::Env(ORT_LOGGING_LEVEL_WARNING,"audio8-codec-offline");env.DisableTelemetryEvents();
        registration.acquire("QNNExecutionProvider",l+"/libonnxruntime_providers_qnn.so");
        for(auto d:env.GetEpDevices())if(std::string(d.EpName())=="QNNExecutionProvider" && d.Device().Type()==OrtHardwareDeviceType_NPU)devices.push_back(d);
        if(devices.empty())throw std::runtime_error("Codec QNN NPU unavailable; CPU fallback disabled");
        std::filesystem::create_directories(cache);
    }
    std::string stem(int64_t capacity,int part)const {
        char b[80];std::snprintf(b,sizeof(b),"codec-%lld-part%02d",(long long)capacity,part);return b;
    }
    void load(int64_t capacity,int part,bool compile) {
        session.reset(); // Exactly one QNN graph at a time.
        auto name=stem(capacity,part),context=cache+"/"+name+".onnx";
        bool hit=std::filesystem::is_regular_file(context);
        if(!hit && !compile)throw std::runtime_error("请先准备此容量的全部 NPU 子图，缺少 "+name);
        Ort::SessionOptions options;options.SetIntraOpNumThreads(1);options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
        if(!hit){options.AddConfigEntry("ep.context_enable","1");options.AddConfigEntry("ep.context_embed_mode","1");options.AddConfigEntry("ep.context_file_path",context.c_str());}
        std::unordered_map<std::string,std::string> config={
            {"backend_path",libraries+"/libQnnHtp.so"},{"soc_model","43"},{"htp_arch","73"},
            {"enable_htp_fp16_precision","1"},{"htp_performance_mode","sustained_high_performance"},
            {"htp_graph_finalization_optimization_mode","0"},
            {"profiling_level","basic"},{"profiling_file_path",cache+"/"+name+".csv"}};
        options.AppendExecutionProvider_V2(env,devices,config);
        audio8_stage(std::string(hit?"加载":"编译")+" NPU codec 子图 "+std::to_string(part+1)+"/14");memory_log(part);
        auto start=Clock::now();
        try{session=std::make_unique<Ort::Session>(env,(hit?context:model+"/"+name+".onnx").c_str(),options);}
        catch(const std::exception & e){std::error_code ec;std::filesystem::remove(context,ec);throw std::runtime_error(name+": "+e.what());}
        std::fprintf(stderr,"Audio8 codec NPU stage=%d capacity=%lld prepare_ms=%.2f context_hit=%d strict_no_cpu_fallback=1\n",part,(long long)capacity,ms(start),hit);
    }
    void prepare(int64_t capacity){
        if(capacity!=128 && capacity!=256)throw std::runtime_error("Invalid codec capacity");
        for(int i=0;i<parts;++i){load(capacity,i,true);session.reset();}
        std::ofstream marker(cache+"/ready-"+std::to_string(capacity));marker<<"parts-v1";
        if(!marker)throw std::runtime_error("Cannot persist codec preparation marker");
    }
    void run(const int32_t * codes,int64_t books,int64_t frames,const float ** out,int64_t * count) {
        if(!codes || books!=10 || frames<1 || frames>256)throw std::runtime_error("Codec codes shape out of bounds");
        auto total=Clock::now();const int64_t capacity=frames<=128?128:256;
        if(!std::filesystem::is_regular_file(cache+"/ready-"+std::to_string(capacity)))throw std::runtime_error("请先准备全部 NPU codec 子图");
        std::vector<int64_t> padded(books*capacity,0);
        for(int64_t b=0;b<books;++b)for(int64_t t=0;t<frames;++t){auto c=codes[b*frames+t];if(c<0 || c>=4096)throw std::runtime_error("Invalid codec token");padded[b*capacity+t]=c;}
        std::map<std::string,Tensor> live;
        auto & first=live["codes"];first.shape={1,books,capacity};first.type=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        first.bytes.resize(padded.size()*sizeof(int64_t));std::memcpy(first.bytes.data(),padded.data(),first.bytes.size());padded.clear();padded.shrink_to_fit();
        double compute=0,copy=0;
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        for(int part=0;part<parts;++part){
            load(capacity,part,false);auto start=Clock::now();
            Ort::AllocatorWithDefaultOptions allocator;
            std::vector<std::string> in_names,out_names;std::vector<const char *> ins,outs;std::vector<Ort::Value> feeds;
            for(size_t i=0;i<session->GetInputCount();++i)in_names.emplace_back(session->GetInputNameAllocated(i,allocator).get());
            for(size_t i=0;i<session->GetOutputCount();++i)out_names.emplace_back(session->GetOutputNameAllocated(i,allocator).get());
            for(auto & name:in_names){
                auto found=live.find(name);if(found==live.end())throw std::runtime_error("Missing codec boundary "+name);
                auto & t=found->second;
                ins.push_back(name.c_str());feeds.push_back(Ort::Value::CreateTensor(memory,t.bytes.data(),t.bytes.size(),t.shape.data(),t.shape.size(),t.type));
            }
            for(auto & name:out_names)outs.push_back(name.c_str());
            audio8_stage("NPU codec 解码子图 "+std::to_string(part+1)+"/14");
            auto values=session->Run(Ort::RunOptions{nullptr},ins.data(),feeds.data(),feeds.size(),outs.data(),outs.size());double run_ms=ms(start);compute+=run_ms;start=Clock::now();
            // The partitioner verifies large boundary tensors have one downstream consumer.
            std::vector<Tensor> copied;for(auto & value:values)copied.emplace_back(value);
            values.clear();feeds.clear();
            for(auto & name:in_names)if(live.at(name).bytes.size()>4096)live.erase(name);
            for(size_t i=0;i<out_names.size();++i)live.insert_or_assign(out_names[i],std::move(copied[i]));copy+=ms(start);session.reset();
            std::fprintf(stderr,"Audio8 codec NPU stage=%d run_ms=%.2f live_tensors=%zu\n",part,run_ms,live.size());
        }
        auto found=live.find("audio");if(found==live.end() || found->second.type!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || found->second.shape!=std::vector<int64_t>{1,1,capacity*2048})throw std::runtime_error("Invalid codec PCM contract");
        pcm.resize(frames*2048);std::memcpy(pcm.data(),found->second.bytes.data(),pcm.size()*sizeof(float));
        for(float v:pcm)if(!std::isfinite(v))throw std::runtime_error("Non-finite codec PCM");
        *out=pcm.data();*count=pcm.size();
        std::fprintf(stderr,"Audio8 codec backend=QNN_HTP parts=14 run_ms=%.2f boundary_copy_ms=%.2f total_ms=%.2f frames=%lld audio_ms=%.2f\n",compute,copy,ms(total),(long long)frames,frames*2048.0/44.1);
    }
};
OnnxCodec::OnnxCodec(const std::string & m,const std::string & l,const std::string & c):impl(std::make_unique<Impl>(m,l,c)){}
OnnxCodec::~OnnxCodec()=default;
void OnnxCodec::prepare(int64_t frames){impl->prepare(frames);}
int OnnxCodec::decode(void * p,const int32_t * c,int64_t b,int64_t f,const float ** o,int64_t * n){auto & self=*static_cast<OnnxCodec*>(p)->impl;try{self.run(c,b,f,o,n);return 0;}catch(const std::exception & e){self.last_error=e.what();return -1;}}
const char * OnnxCodec::error(void * p){return static_cast<OnnxCodec*>(p)->impl->last_error.c_str();}
