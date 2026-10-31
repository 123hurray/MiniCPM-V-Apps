#pragma once
#include <cerrno>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#if defined(__ANDROID__) && defined(__aarch64__)
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstddef>
#endif

inline void require_offline_synthesis_process() {
#if defined(__ANDROID__) && defined(__aarch64__)
    static std::once_flag once;
    std::call_once(once, [] {
        // Restrict this :synthesis process, not the separate UI/downloader process.
        // UNIX RPC sockets remain available to the local Qualcomm driver.
        sock_filter rules[] = {
            BPF_STMT(BPF_LD|BPF_W|BPF_ABS, offsetof(seccomp_data,arch)),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_AARCH64,1,0),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
            BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(seccomp_data,nr)),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_socket,0,4),
            BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(seccomp_data,args[0])),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AF_INET,1,0),
            BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AF_INET6,0,1),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ERRNO|EPERM),
            BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        };
        sock_fprog program{static_cast<unsigned short>(sizeof(rules)/sizeof(rules[0])),rules};
        if(prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)!=0 ||
           syscall(__NR_seccomp,SECCOMP_SET_MODE_FILTER,SECCOMP_FILTER_FLAG_TSYNC,&program)!=0)
            throw std::runtime_error(std::string("Cannot enforce offline NPU synthesis process: ")+std::strerror(errno));
    });
#else
    throw std::runtime_error("NPU offline guard requires Android arm64");
#endif
}
