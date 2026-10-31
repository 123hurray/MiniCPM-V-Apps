#pragma once
#include <cstdio>
#include <exception>
#include <string>

// Declare after Env and before sessions: session teardown -> unregister -> Env.
// Member RAII also runs if the containing model constructor throws midway.
template<class Environment> class ProviderRegistration {
    Environment & env_;
    std::string name_;
    bool owned_=false;
public:
    explicit ProviderRegistration(Environment & env):env_(env) { }
    ProviderRegistration(const ProviderRegistration &)=delete;
    ProviderRegistration & operator=(const ProviderRegistration &)=delete;
    void acquire(const std::string & name,const std::string & path) {
        name_=name;
        env_.RegisterExecutionProviderLibrary(name.c_str(),path);
        owned_=true;
    }
    ~ProviderRegistration() noexcept {
        if(!owned_)return;
        try { env_.UnregisterExecutionProviderLibrary(name_.c_str()); }
        catch(const std::exception & e) { std::fprintf(stderr,"Audio8 provider cleanup error: %s\n",e.what()); }
        catch(...) { std::fprintf(stderr,"Audio8 provider cleanup error: unknown\n"); }
    }
};
