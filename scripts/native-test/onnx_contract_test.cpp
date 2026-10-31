// No ORT/QNN headers, runtime loading or provider execution in this test.
#include "onnx_contract.h"
#include "provider_registration.h"
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

struct Counts { bool registered=false;int acquired=0,released=0,sessions=0; };
struct FakeEnv {
    Counts & c;
    void RegisterExecutionProviderLibrary(const char *,const std::string &) {
        if(c.registered)throw std::runtime_error("already registered");
        c.registered=true;++c.acquired;
    }
    void UnregisterExecutionProviderLibrary(const char *) {
        assert(c.registered && c.sessions==0);c.registered=false;++c.released;
    }
};
struct FakeSession {
    Counts & c;
    explicit FakeSession(Counts & v):c(v){assert(c.registered);++c.sessions;}
    ~FakeSession(){assert(c.registered);--c.sessions;}
};
struct Model {
    FakeEnv env;
    ProviderRegistration<FakeEnv> registration{env};
    std::unique_ptr<FakeSession> session;
    explicit Model(Counts & c,int fail):env{c} {
        registration.acquire("QNNExecutionProvider","fake-provider");
        if(fail==1)throw std::runtime_error("failed before session");
        session=std::make_unique<FakeSession>(c);
        if(fail==2)throw std::runtime_error("failed after session");
    }
};
int main() {
    // The same last-position contract holds for prefill of any input length.
    for(int input_steps:{1,26,512}) {
        (void)input_steps;
        require_last_token_output({1,1,4097},4097,"slow logits");
        require_last_token_output({1,1,896},896,"slow hidden");
    }
    for(auto shape:std::vector<std::vector<int64_t>>{{1,26,4097},{1,4097},{2,1,4097},{1,1,4096},{1,0,4097}}) {
        bool rejected=false;
        try {require_last_token_output(shape,4097,"slow logits");}
        catch(const std::runtime_error & e){rejected=true;assert(std::string(e.what()).find("expected=[1,1,4097]")!=std::string::npos);}
        assert(rejected);
    }
    Counts c;
    // Retry after either constructor failure must not encounter a leaked registration.
    for(int attempt:{1,2,0,2,0}) {
        try { Model model(c,attempt);assert(attempt==0); }
        catch(const std::runtime_error &){assert(attempt!=0);}
        assert(!c.registered && c.sessions==0 && c.acquired==c.released);
    }
    // A registration failure must not unregister a provider owned by someone else.
    c.registered=true;
    try {Model model(c,0);assert(false);}catch(const std::runtime_error &){}
    assert(c.registered && c.acquired==c.released);
}
