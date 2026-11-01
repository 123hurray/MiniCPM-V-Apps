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
    private long engine;
    private String engineKey="", current="zho";
    private volatile boolean destroyed;
    private volatile boolean trimRequested;
    private static final List<Locale> LOCALES=List.of(Locale.SIMPLIFIED_CHINESE,Locale.US);
    @Override protected String[] onGetLanguage() { return new String[]{current,"",""}; }
    private boolean modelsReady() { return ModelStore.ready(this) && (!new PerformanceSettings(this).codecNpu || CodecModelStore.ready(this)); }
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
                byte[] hit=s.cache ? cache.get(key) : null;
                if (hit==null) ensureEngine(s);
                if (destroyed || generation.get()!=ticket) return;
                if (callback.start(44100,AudioFormat.ENCODING_PCM_16BIT,1)!=TextToSpeech.SUCCESS) return;
                ByteArrayOutputStream collected=new ByteArrayOutputStream();
                final boolean[] complete={true}, collect={s.cache}, first={true};
                NativeEngine.Sink sink=new NativeEngine.Sink() {
                    public boolean cancelled() { return destroyed || generation.get()!=ticket; }
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
                    NativeEngine.synthesize(engine,text,clone ? ref.getPath() : "",clone ? words : "",request.getSpeechRate(),request.getPitch(),s.compact,s.graphs,threads,Math.min(threads,s.fastThreads),codec,s.codecFast,s.affinity,sink);
                    disarmWatchdog();
                }
                if (!sink.cancelled() && complete[0]) {
                    if (hit==null && collect[0]) cache.put(key,collected.toByteArray());
                    callback.done(); RuntimeStatus.write(this,"合成已完成");ModelStore.log(this,"synthesis_done total_ms="+(SystemClock.elapsedRealtime()-start));
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
    private void releaseIfTrimRequested() {
        if (!trimRequested) return;
        trimRequested=false; cache.clear();
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
            cache.clear();
            disarmWatchdog();watchdog.shutdown();
        } },"audio8-release").start();
    }
}
