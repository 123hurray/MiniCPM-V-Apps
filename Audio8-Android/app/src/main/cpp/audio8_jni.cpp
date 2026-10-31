#include <jni.h>
#include <audiocpp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <codecvt>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <fstream>
#include <chrono>

namespace {
std::string utf8(JNIEnv* e, jstring v) {
    if (!v) return {};
    const jchar* p=e->GetStringChars(v,nullptr);
    if (!p) throw std::runtime_error("String allocation failed");
    std::u16string s(reinterpret_cast<const char16_t*>(p), e->GetStringLength(v));
    e->ReleaseStringChars(v,p);
    return std::wstring_convert<std::codecvt_utf8_utf16<char16_t>,char16_t>{}.to_bytes(s);
}
std::string quote(const std::string& s) {
    std::string o="\"";
    const char* hex="0123456789abcdef";
    for (unsigned char c:s) {
        if (c=='"' || c=='\\') { o+='\\'; o+=c; }
        else if(c<32) { o+="\\u00"; o+=hex[c>>4]; o+=hex[c&15]; }
        else o+=c;
    }
    return o+'"';
}
void check(audiocpp_status status) { if(status!=AUDIOCPP_OK) throw std::runtime_error(audiocpp_last_error()); }
void fail(JNIEnv* e,const std::exception& x) { if(!e->ExceptionCheck()) e->ThrowNew(e->FindClass("java/lang/IllegalStateException"),x.what()); }
struct Engine {
    audiocpp_registry* registry=nullptr;
    audiocpp_model* model=nullptr;
    audiocpp_session* session=nullptr;
    ~Engine(){ audiocpp_session_free(session); audiocpp_model_free(model); audiocpp_registry_free(registry); }
};
struct AffinityScope {
    cpu_set_t previous{}; bool changed=false;
    explicit AffinityScope(bool enabled) {
        if (!enabled || sched_getaffinity(0,sizeof(previous),&previous)) return;
        std::vector<std::pair<long,int>> cores;
        for(int i=0;i<CPU_SETSIZE;i++) if(CPU_ISSET(i,&previous)) {
            long freq=0; std::ifstream f("/sys/devices/system/cpu/cpu"+std::to_string(i)+"/cpufreq/cpuinfo_max_freq"); f>>freq;
            if(freq>0) cores.emplace_back(freq,i);
        }
        if(cores.size()<4) return;
        std::sort(cores.rbegin(),cores.rend()); cpu_set_t selected; CPU_ZERO(&selected);
        for(size_t i=0;i<std::min<size_t>(5,cores.size());i++) CPU_SET(cores[i].second,&selected);
        changed=sched_setaffinity(0,sizeof(selected),&selected)==0;
        fprintf(stderr,"Audio8 affinity enabled=%d candidates=%zu\n",changed,cores.size());
    }
    ~AffinityScope(){if(changed) sched_setaffinity(0,sizeof(previous),&previous);}
};
struct MobileScope {
    JNIEnv * env; jobject sink; jmethodID cancelled;
    MobileScope(JNIEnv * e,jobject s,bool compact,bool graphs,int slow,int fast)
        :env(e),sink(s) {
        jclass cls=e->GetObjectClass(s); cancelled=e->GetMethodID(cls,"cancelled","()Z"); e->DeleteLocalRef(cls);
        if(!cancelled) throw std::runtime_error("Cancellation callback missing");
        audiocpp_audio8_mobile_options(compact,graphs,slow,fast,[](void * p){
            auto * self=static_cast<MobileScope*>(p);
            return int(self->env->ExceptionCheck() || self->env->CallBooleanMethod(self->sink,self->cancelled));
        },this);
    }
    ~MobileScope(){audiocpp_audio8_mobile_options(0,0,0,0,nullptr,nullptr);}
};
}
extern "C" JNIEXPORT jlong JNICALL Java_com_example_audio8tts_NativeEngine_load(JNIEnv* e,jclass,jstring path,jint threads,jboolean affinity) {
    try {
        std::string file=utf8(e,path);
        std::string log=file.substr(0,file.find_last_of('/'))+"/diagnostics-native.txt";
        int fd=open(log.c_str(),O_WRONLY|O_CREAT|O_APPEND,0600);
        if(fd>=0){dup2(fd,STDERR_FILENO);close(fd);setvbuf(stderr,nullptr,_IONBF,0);}
        AffinityScope affinityScope(affinity);
        auto h=std::make_unique<Engine>();
        check(audiocpp_registry_create(nullptr,&h->registry));
        audiocpp_model_config config{"audio8_tts",nullptr,nullptr,nullptr};
        check(audiocpp_model_load(h->registry,file.c_str(),&config,nullptr,&h->model));
        audiocpp_backend_config backend{"cpu",0,std::clamp(int(threads),1,8)};
        check(audiocpp_session_create(h->model,"tts","offline",&backend,nullptr,&h->session));
        return reinterpret_cast<jlong>(h.release());
    } catch(const std::exception& x) { fail(e,x);return 0; }
}
extern "C" JNIEXPORT void JNICALL Java_com_example_audio8tts_NativeEngine_close(JNIEnv*,jclass,jlong handle) { delete reinterpret_cast<Engine*>(handle); }
extern "C" JNIEXPORT void JNICALL Java_com_example_audio8tts_NativeEngine_synthesize(JNIEnv* e,jclass,jlong handle,jstring text,jstring ref,jstring transcript,jint rate,jint pitch,jboolean compact,jboolean graphs,jint threads,jint fastThreads,jboolean affinity,jobject sink) {
    try {
        auto* h=reinterpret_cast<Engine*>(handle); if(!h) throw std::runtime_error("Model not loaded");
        AffinityScope affinityScope(affinity);
        MobileScope mobile(e,sink,compact,graphs,std::clamp(int(threads),1,8),std::clamp(int(fastThreads),1,8));
        std::unique_ptr<audiocpp_request,decltype(&audiocpp_request_free)> request(audiocpp_request_create(),audiocpp_request_free);
        if (!request) throw std::bad_alloc();
        std::string input=utf8(e,text),reference=utf8(e,ref),words=utf8(e,transcript);
        check(audiocpp_request_set_text(request.get(),input.c_str(),nullptr));
        check(audiocpp_request_set_option(request.get(),"seed","42"));
        check(audiocpp_request_set_option(request.get(),"text_chunk_size","1000000"));
        check(audiocpp_request_set_option(request.get(),"max_tokens","4096"));
        if(!reference.empty()) {
            std::string references="[{\"audio\":"+quote(reference)+",\"text\":"+quote(words)+"}]";
            check(audiocpp_request_set_option(request.get(),"multi_reference_cond",references.c_str()));
        }
        audiocpp_result* raw=nullptr;
        check(audiocpp_session_run(h->session,request.get(),&raw));
        std::unique_ptr<audiocpp_result,decltype(&audiocpp_result_free)> result(raw,audiocpp_result_free);
        const float* samples=nullptr; size_t frames=0; int sampleRate=0,channels=0;
        check(audiocpp_result_audio(result.get(),&samples,&frames,&sampleRate,&channels));
        if (!samples || frames==0 || sampleRate!=44100 || channels!=1) throw std::runtime_error("Missing or unexpected PCM output");
        if (pitch!=100) fprintf(stderr,"Audio8: independent pitch is not supported; requested=%d\n",pitch);
        const double step=std::clamp(double(rate)/100.0,0.5,2.0);
        jclass cls=e->GetObjectClass(sink); jmethodID pcm=e->GetMethodID(cls,"pcm","([B)Z");
        if(!pcm) throw std::runtime_error("PCM callback missing");
        std::vector<jbyte> block; block.reserve(8192);
        auto flush=[&]() {
            jbyteArray bytes=e->NewByteArray(static_cast<jsize>(block.size()));
            if(!bytes) throw std::bad_alloc();
            e->SetByteArrayRegion(bytes,0,static_cast<jsize>(block.size()),block.data());
            bool ok=e->CallBooleanMethod(sink,pcm,bytes);e->DeleteLocalRef(bytes);block.clear();
            return ok && !e->ExceptionCheck();
        };
        for(double pos=0;pos<frames;pos+=step) {
            size_t i=static_cast<size_t>(pos); float a=std::isfinite(samples[i])?samples[i]:0.f;
            float b=std::isfinite(samples[std::min(i+1,frames-1)])?samples[std::min(i+1,frames-1)]:0.f;
            float v=a+(b-a)*float(pos-i);
            int16_t s=static_cast<int16_t>(std::clamp(v,-1.f,1.f)*32767.f);
            block.push_back(static_cast<jbyte>(s&255));block.push_back(static_cast<jbyte>((uint16_t(s)>>8)&255));
            if(block.size()==8192 && !flush()) return;
        }
        if(!block.empty()) flush();
    } catch(const std::exception& x) { fail(e,x); }
}
