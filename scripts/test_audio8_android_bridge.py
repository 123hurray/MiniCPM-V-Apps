"""Source/contract checks only: never load ORT or QNN on the build host."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
CPP=ROOT/'Audio8-Android/app/src/main/cpp'
JAVA=ROOT/'Audio8-Android/app/src/main/java/com/example/audio8tts'

class AndroidBridgeTests(unittest.TestCase):
    def test_codec_thread_and_layout_controls(self):
        jni=(CPP/'audio8_jni.cpp').read_text()
        self.assertIn('audiocpp_audio8_codec_options(codec,codecFast)',jni)
        settings=(JAVA/'PerformanceSettings.java').read_text()
        self.assertIn('number(p,"codec_threads",5)',settings)
        self.assertIn('bool(p,"codec_fast",true)',settings)
        patch=(ROOT/'scripts/audio8-mobile.patch').read_text()
        self.assertIn('frame_capacity_ == frames',patch)
        self.assertIn('mobile_codec_threads(threads_)',patch)
        self.assertIn('mobile_conv_transpose',patch)
        self.assertIn('mobile_snake_kernel',patch)

    def test_native_output_contract_and_registration_lifetime(self):
        with tempfile.TemporaryDirectory(prefix='audio8-contract-') as directory:
            executable=str(Path(directory)/'contract-test')
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(CPP),
                str(ROOT/'scripts/native-test/onnx_contract_test.cpp'),'-o',executable],check=True)
            subprocess.run([executable],check=True)
        source=(CPP/'onnx_ar.cpp').read_text()
        self.assertIn('float_data(outputs[0],4097)',source)
        self.assertIn('half_data(outputs[1],896)',source)
        self.assertNotIn('(steps-1)*4097',source)
        self.assertIn('update_cache(slow_cache,outputs,2,steps,position,2048)',source)
        self.assertLess(source.index('Ort::Env env{nullptr}'),source.index('ProviderRegistration<Ort::Env>'))
        self.assertLess(source.index('ProviderRegistration<Ort::Env>'),source.index('std::unique_ptr<State>'))
    def test_bounded_requests_and_no_automatic_warmup(self):
        service=(JAVA/'Audio8Service.java').read_text()
        self.assertNotIn('warmup_begin',service)
        self.assertNotIn('scheduleWarmup',service)
        self.assertIn('armWatchdog(s.loadTimeout',service)
        self.assertIn('armWatchdog(s.synthesisTimeout',service)
        self.assertIn('},15,TimeUnit.SECONDS)',service)
        self.assertIn('killProcess(android.os.Process.myPid())',service)
        ui=(JAVA/'MainActivity.java').read_text()
        self.assertIn('speechWorker.execute',ui)
        self.assertIn('root.setVisibility(android.view.View.GONE)',ui)
        self.assertIn('EngineControlReceiver.class',ui)
        self.assertIn('native-status.txt',ui)
        manifest=(ROOT/'Audio8-Android/app/src/main/AndroidManifest.xml').read_text()
        self.assertIn('android:name=".EngineControlReceiver" android:process=":synthesis" android:exported="false"',manifest)

    def test_npu_placement_and_fail_closed(self):
        source=(CPP/'onnx_ar.cpp').read_text()
        self.assertLess(source.index('require_offline_synthesis_process();'),source.index('env=Ort::Env'))
        self.assertLess(source.index('env.DisableTelemetryEvents();'),source.index('registration.acquire'))
        self.assertIn('"session.disable_cpu_ep_fallback","1"',source)
        self.assertIn('OrtHardwareDeviceType_NPU',source)
        self.assertNotIn('enable_htp_prepare_only',source)
        self.assertIn('float_data(outputs[0]',source)
        self.assertIn('half_data(outputs[1]',source)
        self.assertIn('prefill=CPU_INT4 slow_decode=QNN_HTP_INT4',source)

    def test_callback_bound_for_load_and_synthesis(self):
        source=(CPP/'audio8_jni.cpp').read_text()
        self.assertEqual(source.count('ExternalScope external(h->ar.get());'),2)
        self.assertIn('OnnxAr::forward',source)
        self.assertIn('audiocpp_audio8_external_forward(nullptr,nullptr,nullptr)',source)
        patch=(ROOT/'scripts/audio8-mobile.patch').read_text()
        self.assertIn('if (mobile_options.forward) return generate_external(prompt, options);',patch)
        self.assertIn('External Audio8 AR callback is not bound',patch)

    def test_catalog_is_pinned_and_no_path_traversal(self):
        source=(JAVA/'NpuModelStore.java').read_text()
        self.assertIn('hash(bytes).equals(BuildConfig.NPU_MANIFEST_SHA256)',source)
        self.assertIn('REQUIRED.contains(name)',source)
        self.assertIn('found.containsAll(REQUIRED)',source)
        self.assertIn('hash(part).equals(e.hash())',source)
        self.assertIn('DownloadEndpoint.matchesRange',source)
        self.assertIn('uri.getRawUserInfo()!=null',source)

    def test_synthesis_process_is_separate(self):
        manifest=(ROOT/'Audio8-Android/app/src/main/AndroidManifest.xml').read_text()
        self.assertIn('android:process=":synthesis"',manifest)
        self.assertIn('android.intent.category.DEFAULT',manifest)

    def test_network_bpf_instructions(self):
        source=(CPP/'network_guard.h').read_text()
        body=source.split('sock_filter rules[] = {',1)[1].split('};',1)[0]
        actual=re.sub(r'\s+','',body)
        expected='''
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
        '''
        self.assertEqual(actual,re.sub(r'\s+','',expected))
        self.assertIn('SECCOMP_FILTER_FLAG_TSYNC',source)
        # Simulate the verified branches; do not install a filter on the host.
        def allows(arm64,syscall,family):
            if not arm64:return False
            if syscall!='socket':return True
            return family not in ('AF_INET','AF_INET6')
        for family in ('AF_INET','AF_INET6'):self.assertFalse(allows(True,'socket',family))
        self.assertTrue(allows(True,'socket','AF_UNIX'))
        self.assertTrue(allows(True,'ioctl','unused'))

if __name__=='__main__':unittest.main(verbosity=2)
