package com.example.audio8tts;

import android.media.AudioFormat;
import android.os.PowerManager;
import android.os.SystemClock;
import android.speech.tts.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public final class Audio8Service extends TextToSpeechService {
    private final AtomicLong generation=new AtomicLong();
    private final ScheduledExecutorService watchdog=Executors.newSingleThreadScheduledExecutor();
    private final AtomicLong operation=new AtomicLong();
    private final AtomicBoolean inNative=new AtomicBoolean();
    private final ExecutorService warmer=Executors.newSingleThreadExecutor();
    private final Object lock=new Object();
    private final PcmCache cache=new PcmCache();
    private LiteCodecRuntime liteCodec;
    private long engine;
    private String engineKey="", current="zho";
    private volatile boolean destroyed;
    private volatile boolean trimRequested;
    private static final List<Locale> LOCALES=List.of(Locale.SIMPLIFIED_CHINESE,Locale.US);
    @Override protected String[] onGetLanguage() { return new String[]{current,"",""}; }
    private boolean modelsReady() {var s=new PerformanceSettings(this);return ModelStore.ready(this) && (s.codecBackend.equals("litert")?LiteCodecStore.ready(this):!s.codecNpu||CodecModelStore.ready(this));}

    @Override protected int onIsLanguageAvailable(String lang,String country,String variant) {
        if (!(lang.equals("zho") || lang.equals("chi") || lang.equals("zh") || lang.equals("eng") || lang.equals("en"))) return TextToSpeech.LANG_NOT_SUPPORTED;
        return modelsReady() ? TextToSpeech.LANG_AVAILABLE : TextToSpeech.LANG_MISSING_DATA;
    }
    @Override protected int onLoadLanguage(String lang,String country,String variant) {
        int status=onIsLanguageAvailable(lang,country,variant);
        if (status>=0) current=lang; return status;
    }
    @Override public List<Voice> onGetVoices() {
        List<Voice> voices=new ArrayList<>();
        for (Locale l:LOCALES) voices.add(new Voice("audio8-"+l.toLanguageTag(),l,Voice.QUALITY_HIGH,Voice.LATENCY_VERY_HIGH,false,Set.of()));
        return voices;
    }
    @Override public String onGetDefaultVoiceNameFor(String l,String c,String v) {
        if (onIsLanguageAvailable(l,c,v)==TextToSpeech.LANG_NOT_SUPPORTED) return null;
        return l.startsWith("en") ? "audio8-en-US" : "audio8-zh-CN";
    }
    @Override public int onIsValidVoiceName(String name) { return onGetVoices().stream().anyMatch(v -> v.getName().equals(name)) ? TextToSpeech.SUCCESS : TextToSpeech.ERROR; }
    @Override public int onLoadVoice(String name) {
        int status=modelsReady() ? onIsValidVoiceName(name) : TextToSpeech.ERROR;
        return status;
    }
    private void ensureEngine(PerformanceSettings s) {
        if(!s.codecBackend.equals("litert") && liteCodec!=null){liteCodec.close();liteCodec=null;}
        if (engine!=0 && !engineKey.equals(s.engineKey())) {
            RuntimeStatus.write(this,"切换引擎，释放旧模型");armWatchdog(60,"旧模型释放超时");
            NativeEngine.close(engine);engine=0;cache.clear();disarmWatchdog();
        }
        if (engine==0) {
            long start=SystemClock.elapsedRealtime();
            RuntimeStatus.write(this,"正在加载模型");
            if(s.npu && !android.os.Build.SOC_MODEL.toUpperCase(Locale.ROOT).contains("SM8550"))
                throw new IllegalStateException("NPU 实验模型仅针对 SM8550；当前 SoC="+android.os.Build.SOC_MODEL);
            if(s.npu && !NpuModelStore.ready(this)) throw new IllegalStateException("请先在设置中下载 NPU INT4 模型");
            if(s.codecNpu && !android.os.Build.SOC_MODEL.toUpperCase(Locale.ROOT).contains("SM8550"))throw new IllegalStateException("codec NPU 实验仅支持 SM8550");
            if(s.codecNpu && !CodecModelStore.ready(this))throw new IllegalStateException("请先下载 codec ONNX 模型");
            armWatchdog(s.loadTimeout,"模型初始化超时");
            engine=NativeEngine.load(ModelStore.model(this).getPath(),s.threads,s.affinity,
                s.npu?NpuModelStore.dir(this).getPath():"",getApplicationInfo().nativeLibraryDir,
                NpuModelStore.runtimeCache(this).getPath(),s.fastThreads,s.fastNpu,
                s.codecNpu?CodecModelStore.dir(this).getPath():"");
            engineKey=s.engineKey();
            disarmWatchdog();
            ModelStore.log(this,"model_load_ms="+(SystemClock.elapsedRealtime()-start)+" config="+engineKey);
        }
    }
    private int activeThreads(PerformanceSettings s) {
        PowerManager power=getSystemService(PowerManager.class);
        int thermal=power==null ? 0 : power.getCurrentThermalStatus();
        int threads=thermal>=PowerManager.THERMAL_STATUS_SEVERE ? Math.min(s.threads,3) : s.threads;
        ModelStore.log(this,"threads="+threads+" fast_threads="+Math.min(threads,s.fastThreads)+" thermal="+thermal);
        return threads;
    }
    private void armWatchdog(int seconds,String reason) {
        long id=operation.incrementAndGet();inNative.set(true);
        watchdog.schedule(() -> {
            if(operation.get()!=id || !inNative.get()) return;
            RuntimeStatus.write(this,reason+"；语音进程已结束，请重新连接引擎");
            android.os.Process.killProcess(android.os.Process.myPid());
        },seconds,TimeUnit.SECONDS);
    }
    private void disarmWatchdog() { inNative.set(false);operation.incrementAndGet(); }
    @Override protected void onStop() {
        generation.incrementAndGet();
        if(inNative.get()) {
            long id=operation.get();
            RuntimeStatus.write(this,"正在停止；原生计算超过 15 秒未退出将重启语音进程");
            watchdog.schedule(() -> {
                if(operation.get()==id && inNative.get()) {
                    RuntimeStatus.write(this,"停止超时；语音进程已结束，请重新连接引擎");
                    android.os.Process.killProcess(android.os.Process.myPid());
                }
            },15,TimeUnit.SECONDS);
        } else RuntimeStatus.write(this,"已停止");
    }
    @Override protected void onSynthesizeText(SynthesisRequest request,SynthesisCallback callback) {
        long ticket=generation.incrementAndGet(), start=SystemClock.elapsedRealtime();
        RuntimeStatus.write(this,"请求已接收");
        synchronized(lock) {
            try {
                if (destroyed || generation.get()!=ticket) return;
                releaseIfTrimRequested();
                if(request.getParams().getBoolean("com.example.audio8tts.codec_benchmark",false)){
                    if(callback.start(44100,AudioFormat.ENCODING_PCM_16BIT,1)!=TextToSpeech.SUCCESS)return;
                    armWatchdog(240,"codec 基准超时");benchmarkCodec(ticket);disarmWatchdog();callback.done();RuntimeStatus.write(this,"codec 基准完成，请导出基准包");return;
                }
                if (!modelsReady()) { callback.error(TextToSpeech.ERROR_NOT_INSTALLED_YET); return; }
                if (onIsLanguageAvailable(request.getLanguage(),request.getCountry(),request.getVariant())<0) { callback.error(TextToSpeech.ERROR_INVALID_REQUEST); return; }
                int prepare=request.getParams().getInt("com.example.audio8tts.prepare_codec_frames",0);
                if(prepare!=0) {
                    if(prepare!=128 && prepare!=256)throw new IllegalArgumentException("不支持的 codec 容量");
                    PerformanceSettings settings=new PerformanceSettings(this);
                    if(!settings.codecNpu)throw new IllegalStateException("请先启用并保存 codec NPU");
                    // Preparing NPU graphs must not retain the full GGUF/AR/CPU codec weights.
                    if(engine!=0){NativeEngine.close(engine);engine=0;engineKey="";}
                    cache.clear();
                    ModelStore.log(this,"codec_prepare_ar_released=true");
                    if(callback.start(44100,AudioFormat.ENCODING_PCM_16BIT,1)!=TextToSpeech.SUCCESS)return;
                    ModelStore.log(this,"codec_prepare_begin capacity="+prepare);
                    armWatchdog(240,"NPU codec 准备超时");
                    NativeEngine.prepareCodec(CodecModelStore.dir(this).getPath(),getApplicationInfo().nativeLibraryDir,NpuModelStore.runtimeCache(this).getPath()+"/preview01-codec-317c12d",prepare);disarmWatchdog();
                    callback.done();RuntimeStatus.write(this,"NPU codec 图已准备，接下来可试听");
                    ModelStore.log(this,"codec_prepare_done capacity="+prepare+" total_ms="+(SystemClock.elapsedRealtime()-start));return;
                }
                String text=request.getCharSequenceText().toString().strip();
                if (text.isEmpty() || text.codePointCount(0,text.length())>500) { callback.error(TextToSpeech.ERROR_INVALID_REQUEST); return; }
                PerformanceSettings s=new PerformanceSettings(this);
                File ref=ModelStore.reference(this), transcript=ModelStore.transcript(this);
                String words=transcript.isFile() ? ModelStore.text(transcript) : "";
                boolean clone=ref.isFile() && !words.isBlank();
                String key=text+"\u0000"+request.getSpeechRate()+":"+request.getPitch()+":"+request.getVoiceName()+":"+s.engineKey()+":"+clone+":"+ref.lastModified()+":"+ref.length()+":"+words;
                if (!s.cache) cache.clear();
                byte[] hit=s.cache && !CodecFixtures.enabled(this) ? cache.get(key) : null;
                if (hit==null) ensureEngine(s);
                if (destroyed || generation.get()!=ticket) return;
                if (callback.start(44100,AudioFormat.ENCODING_PCM_16BIT,1)!=TextToSpeech.SUCCESS) return;
                ByteArrayOutputStream collected=new ByteArrayOutputStream();
                final boolean[] complete={true}, collect={s.cache}, first={true};
                NativeEngine.Sink sink=new NativeEngine.Sink() {
                    public boolean cancelled() { return destroyed || generation.get()!=ticket; }
                    public void captureCodes(int[] codes,int frames){CodecFixtures.capture(Audio8Service.this,codes,frames);}
                    public float[] codec(int[] codes,int frames){
                        RuntimeStatus.write(Audio8Service.this,"解码音频 · LiteRT INT8");
                        if(liteCodec==null)liteCodec=new LiteCodecRuntime(Audio8Service.this,"cpu_int8");
                        return liteCodec.decode(codes,frames,s.liteThreads,s.codecWindows,()->cancelled());
                    }
                    public boolean pcm(byte[] data) {
                        if (cancelled()) { complete[0]=false; return false; }
                        if (first[0]) { first[0]=false; ModelStore.log(Audio8Service.this,"first_pcm_ms="+(SystemClock.elapsedRealtime()-start)); }
                        for (int offset=0;offset<data.length;) {
                            int count=Math.min(callback.getMaxBufferSize(),data.length-offset)&~1;
                            if (cancelled() || count==0 || callback.audioAvailable(data,offset,count)!=TextToSpeech.SUCCESS) { complete[0]=false; return false; }
                            offset+=count;
                        }
                        if (collect[0]) {
                            if (collected.size()+data.length>PcmCache.ENTRY_LIMIT) { collect[0]=false; collected.reset(); }
                            else collected.write(data,0,data.length);
                        }
                        return true;
                    }
                };
                ModelStore.log(this,"synthesis_begin chars="+text.length()+" cache_hit="+(hit!=null));
                if (hit!=null) { collect[0]=false; sink.pcm(hit); }
                else {
                    int threads=activeThreads(s);
                    RuntimeStatus.write(this,"正在生成语音");
                    armWatchdog(s.synthesisTimeout,"语音合成超时");
                    PowerManager power=getSystemService(PowerManager.class);
                    int codec=power!=null && power.getCurrentThermalStatus()>=PowerManager.THERMAL_STATUS_SEVERE?Math.min(s.codecThreads,3):s.codecThreads;
                    NativeEngine.synthesize(engine,text,clone ? ref.getPath() : "",clone ? words : "",request.getSpeechRate(),request.getPitch(),s.compact,s.graphs,threads,Math.min(threads,s.fastThreads),codec,s.codecFast,s.affinity,s.codecBackend.equals("litert"),sink);
                    disarmWatchdog();
                }
                if (!sink.cancelled() && complete[0]) {
                    if (hit==null && collect[0]) cache.put(key,collected.toByteArray());
                    callback.done(); RuntimeStatus.write(this,"合成已完成");ModelStore.log(this,"synthesis_done total_ms="+(SystemClock.elapsedRealtime()-start)+" target_10000ms_met="+((SystemClock.elapsedRealtime()-start)<=10000)+" cache_hit="+(hit!=null));
                } else { RuntimeStatus.write(this,"已停止");ModelStore.log(this,"synthesis_stopped total_ms="+(SystemClock.elapsedRealtime()-start)); }
                releaseIfTrimRequested();
            } catch (Throwable e) {
                disarmWatchdog();
                if (destroyed || ticket!=generation.get()) { RuntimeStatus.write(this,"已停止");ModelStore.log(this,"synthesis_cancelled ms="+(SystemClock.elapsedRealtime()-start)); return; }
                ModelStore.log(this,"synthesis_error "+android.util.Log.getStackTraceString(e));
                RuntimeStatus.write(this,"合成失败："+e.getMessage());
                callback.error(TextToSpeech.ERROR_SYNTHESIS);
                if (engine!=0) { NativeEngine.close(engine); engine=0; }
            }
        }
    }
    private void benchmarkCodec(long ticket) throws Exception {
        if(liteCodec!=null){liteCodec.close();liteCodec=null;}
        if(engine!=0){NativeEngine.close(engine);engine=0;engineKey="";}
        File fixture=CodecFixtures.latest(this);int[] codes=CodecContract.read(fixture.toPath());int frames=codes.length/10;
        if(frames>128)throw new IllegalArgumentException("请先记录一条不超过128帧的短句用于完整对照");
        org.json.JSONObject report=new org.json.JSONObject();org.json.JSONArray rows=new org.json.JSONArray();
        report.put("fixture",fixture.getName()).put("frames",frames).put("audio_ms",frames*2048.0/44.1).put("device",android.os.Build.MODEL).put("soc",android.os.Build.SOC_MODEL).put("inference_only",true);
        float[] int8=null;long best=Long.MAX_VALUE;int bestThreads=4;
        try {
        for(int threads:new int[]{4,2,5}){
            RuntimeStatus.write(this,"基准 · LiteRT INT8 · "+threads+"线程");
            try(LiteCodecRuntime runtime=new LiteCodecRuntime(this,"cpu_int8")){
                long begin=SystemClock.elapsedRealtime();float[] cold=runtime.decode(codes,frames,threads,false,()->destroyed||generation.get()!=ticket);long coldMs=SystemClock.elapsedRealtime()-begin;
                begin=SystemClock.elapsedRealtime();float[] warm=runtime.decode(codes,frames,threads,false,()->destroyed||generation.get()!=ticket);long warmMs=SystemClock.elapsedRealtime()-begin;
                double repeat=CodecContract.correlation(cold,warm);if(repeat<0.9999)throw new IllegalStateException("相同音频码重复输出不稳定");
                if(threads==4)int8=warm;
                if(warmMs<best){best=warmMs;bestThreads=threads;}
                int[] altered=codes.clone();for(int q=0;q<10;q++)for(int t=0;t<frames;t++)altered[q*frames+t]=(altered[q*frames+t]+1)%(q==0?4096:1024);
                float[] other=runtime.decode(altered,frames,threads,false,()->destroyed||generation.get()!=ticket);double diff=0;for(int i=0;i<warm.length;i++){double d=warm[i]-other[i];diff+=d*d;}diff/=warm.length;
                if(diff<1e-10)throw new IllegalStateException("codec 对不同输入返回相同音频");
                rows.put(new org.json.JSONObject().put("backend","litert_int8_cpu").put("threads",threads).put("cold_ms",coldMs).put("warm_ms",warmMs).put("repeat_correlation",repeat).put("input_sensitivity_mse",diff).put("rms",CodecContract.rms(warm)));
            }
        }
        report.put("recommended_litert_threads",bestThreads);
        if(LiteFloatStore.ready(this))for(String mode:new String[]{"cpu_fp16","gpu_fp16"}){
            RuntimeStatus.write(this,"基准 · LiteRT "+mode);
            try(LiteCodecRuntime runtime=new LiteCodecRuntime(this,mode)){
                long begin=SystemClock.elapsedRealtime();float[] pcm=runtime.decode(codes,frames,4,false,()->destroyed||generation.get()!=ticket);long elapsed=SystemClock.elapsedRealtime()-begin;
                rows.put(new org.json.JSONObject().put("backend",mode).put("cold_ms",elapsed).put("correlation_int8_cpu",CodecContract.correlation(int8,pcm)).put("production_enabled",false));
            }catch(Throwable e){rows.put(new org.json.JSONObject().put("backend",mode).put("error",String.valueOf(e.getMessage())).put("production_enabled",false));}
        }
        RuntimeStatus.write(this,"基准 · 现有 GGML codec，同一份音频码");
        ensureEngine(new PerformanceSettings(this,"ggml"));
        for(int run=0;run<2;run++){
            long begin=SystemClock.elapsedRealtime();float[] baseline=NativeEngine.baselineCodec(engine,codes,frames,5);long elapsed=SystemClock.elapsedRealtime()-begin;
            rows.put(new org.json.JSONObject().put("backend","ggml_cpu").put("run",run).put("decode_ms",elapsed).put("correlation_litert_int8",CodecContract.correlation(baseline,int8)).put("rms",CodecContract.rms(baseline)));
        }
        ModelStore.log(this,"codec_benchmark_complete recommended_threads="+bestThreads+" best_warm_ms="+best);
        report.put("complete",true);
        } finally {
            report.put("results",rows).put("audio_quality_listening_required",true);
            ModelStore.saveText(new File(ModelStore.dir(this),"codec-benchmark.json"),report.toString(2));
        }
    }
    private void releaseIfTrimRequested() {
        if (!trimRequested) return;
        trimRequested=false; cache.clear();if(liteCodec!=null){liteCodec.close();liteCodec=null;}
        if (engine!=0) { NativeEngine.close(engine); engine=0; }
        ModelStore.log(this,"memory_pressure_released_idle_session");
    }
    @Override public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        if (level==android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW ||
            level==android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level>=android.content.ComponentCallbacks2.TRIM_MEMORY_MODERATE) {
            trimRequested=true;
            try { warmer.execute(() -> { synchronized(lock) { releaseIfTrimRequested(); } }); }
            catch (RejectedExecutionException ignored) { }
        }
    }
    @Override public void onDestroy() {
        destroyed=true; onStop(); warmer.shutdownNow();super.onDestroy();
        // Do not block the main thread waiting for an uninterruptible codec kernel.
        new Thread(() -> { synchronized(lock) {
            armWatchdog(60,"语音进程退出超时");
            if (engine!=0) { NativeEngine.close(engine); engine=0; }
            if(liteCodec!=null){liteCodec.close();liteCodec=null;}
            cache.clear();
            disarmWatchdog();watchdog.shutdown();
        } },"audio8-release").start();
    }
}
