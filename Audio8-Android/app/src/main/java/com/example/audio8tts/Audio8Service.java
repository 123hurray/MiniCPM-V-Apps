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
    private final AtomicBoolean warming=new AtomicBoolean();
    private final ExecutorService warmer=Executors.newSingleThreadExecutor();
    private final Object lock=new Object();
    private final PcmCache cache=new PcmCache();
    private long engine;
    private String engineKey="", current="zho";
    private volatile boolean destroyed;
    private static final List<Locale> LOCALES=List.of(Locale.SIMPLIFIED_CHINESE,Locale.US);
    @Override protected String[] onGetLanguage() { return new String[]{current,"",""}; }
    @Override protected int onIsLanguageAvailable(String lang,String country,String variant) {
        if (!(lang.equals("zho") || lang.equals("chi") || lang.equals("zh") || lang.equals("eng") || lang.equals("en"))) return TextToSpeech.LANG_NOT_SUPPORTED;
        return ModelStore.ready(this) ? TextToSpeech.LANG_AVAILABLE : TextToSpeech.LANG_MISSING_DATA;
    }
    @Override protected int onLoadLanguage(String lang,String country,String variant) {
        int status=onIsLanguageAvailable(lang,country,variant);
        if (status>=0) { current=lang; scheduleWarmup(); } return status;
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
        int status=ModelStore.ready(this) ? onIsValidVoiceName(name) : TextToSpeech.ERROR;
        if (status==TextToSpeech.SUCCESS) scheduleWarmup(); return status;
    }
    private void ensureEngine(PerformanceSettings s) {
        if (engine!=0 && !engineKey.equals(s.engineKey())) { NativeEngine.close(engine); engine=0; cache.clear(); }
        if (engine==0) {
            long start=SystemClock.elapsedRealtime();
            engine=NativeEngine.load(ModelStore.model(this).getPath(),s.threads,s.affinity);
            engineKey=s.engineKey();
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
    private void scheduleWarmup() {
        if (destroyed || !ModelStore.ready(this) || !new PerformanceSettings(this).prewarm || !warming.compareAndSet(false,true)) return;
        long ticket=generation.get();
        warmer.execute(() -> {
            try { synchronized(lock) {
                if (destroyed || generation.get()!=ticket) return;
                PerformanceSettings s=new PerformanceSettings(this);
                if (engine!=0 && engineKey.equals(s.engineKey())) return;
                ensureEngine(s);
                if (generation.get()!=ticket || destroyed) return;
                ModelStore.log(this,"warmup_begin");
                NativeEngine.synthesize(engine,"你好。","","",100,100,s.compact,s.graphs,s.threads,s.fastThreads,s.affinity,new NativeEngine.Sink() {
                    public boolean pcm(byte[] data) { return !cancelled(); }
                    public boolean cancelled() { return destroyed || generation.get()!=ticket; }
                });
                ModelStore.log(this,"warmup_done");
            } } catch (Throwable e) { ModelStore.log(this,"warmup_end "+e.getMessage()); }
            finally { warming.set(false); }
        });
    }
    @Override protected void onStop() { generation.incrementAndGet(); }
    @Override protected void onSynthesizeText(SynthesisRequest request,SynthesisCallback callback) {
        // Interrupt a background warmup before waiting for the native session lock.
        long ticket=generation.incrementAndGet(), start=SystemClock.elapsedRealtime();
        synchronized(lock) {
            try {
                if (destroyed || generation.get()!=ticket) return;
                if (!ModelStore.ready(this)) { callback.error(TextToSpeech.ERROR_NOT_INSTALLED_YET); return; }
                if (onIsLanguageAvailable(request.getLanguage(),request.getCountry(),request.getVariant())<0) { callback.error(TextToSpeech.ERROR_INVALID_REQUEST); return; }
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
                    NativeEngine.synthesize(engine,text,clone ? ref.getPath() : "",clone ? words : "",request.getSpeechRate(),request.getPitch(),s.compact,s.graphs,threads,Math.min(threads,s.fastThreads),s.affinity,sink);
                }
                if (!sink.cancelled() && complete[0]) {
                    if (hit==null && collect[0]) cache.put(key,collected.toByteArray());
                    callback.done(); ModelStore.log(this,"synthesis_done total_ms="+(SystemClock.elapsedRealtime()-start));
                } else ModelStore.log(this,"synthesis_stopped total_ms="+(SystemClock.elapsedRealtime()-start));
            } catch (Throwable e) {
                if (destroyed || ticket!=generation.get()) { ModelStore.log(this,"synthesis_cancelled ms="+(SystemClock.elapsedRealtime()-start)); return; }
                ModelStore.log(this,"synthesis_error "+android.util.Log.getStackTraceString(e));
                callback.error(TextToSpeech.ERROR_SYNTHESIS);
                if (engine!=0) { NativeEngine.close(engine); engine=0; }
            }
        }
    }
    @Override public void onDestroy() {
        destroyed=true; onStop(); warmer.shutdownNow(); super.onDestroy();
        // Do not block the main thread waiting for an uninterruptible codec kernel.
        new Thread(() -> { synchronized(lock) {
            if (engine!=0) { NativeEngine.close(engine); engine=0; }
            cache.clear();
        } },"audio8-release").start();
    }
}
