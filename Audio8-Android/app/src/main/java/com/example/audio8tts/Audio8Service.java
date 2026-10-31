package com.example.audio8tts;

import android.media.AudioFormat;
import android.speech.tts.*;
import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

public final class Audio8Service extends TextToSpeechService {
    private final AtomicLong generation = new AtomicLong();
    private long engine;
    private String current = "zho";
    private static final List<Locale> LOCALES = List.of(Locale.SIMPLIFIED_CHINESE, Locale.US);
    @Override protected String[] onGetLanguage() { return new String[]{current, "", ""}; }
    @Override protected int onIsLanguageAvailable(String lang, String country, String variant) {
        if (!(lang.equals("zho") || lang.equals("chi") || lang.equals("zh") || lang.equals("eng") || lang.equals("en"))) return TextToSpeech.LANG_NOT_SUPPORTED;
        return ModelStore.ready(this) ? TextToSpeech.LANG_AVAILABLE : TextToSpeech.LANG_MISSING_DATA;
    }
    @Override protected int onLoadLanguage(String lang, String country, String variant) {
        int status = onIsLanguageAvailable(lang, country, variant);
        if (status >= 0) current = lang; return status;
    }
    @Override public List<Voice> onGetVoices() {
        List<Voice> voices = new ArrayList<>();
        for (Locale l : LOCALES) voices.add(new Voice("audio8-" + l.toLanguageTag(), l, Voice.QUALITY_HIGH, Voice.LATENCY_VERY_HIGH, false, Set.of()));
        return voices;
    }
    @Override public String onGetDefaultVoiceNameFor(String l, String c, String v) {
        if (onIsLanguageAvailable(l,c,v) == TextToSpeech.LANG_NOT_SUPPORTED) return null;
        return l.startsWith("en") ? "audio8-en-US" : "audio8-zh-CN";
    }
    @Override public int onIsValidVoiceName(String name) { return onGetVoices().stream().anyMatch(v -> v.getName().equals(name)) ? TextToSpeech.SUCCESS : TextToSpeech.ERROR; }
    @Override public int onLoadVoice(String name) { return ModelStore.ready(this) ? onIsValidVoiceName(name) : TextToSpeech.ERROR; }
    @Override protected void onStop() { generation.incrementAndGet(); }
    @Override protected void onSynthesizeText(SynthesisRequest request, SynthesisCallback callback) {
        long ticket = generation.get();
        try {
            if (!ModelStore.ready(this)) { callback.error(TextToSpeech.ERROR_NOT_INSTALLED_YET); return; }
            if (onIsLanguageAvailable(request.getLanguage(),request.getCountry(),request.getVariant()) < 0) { callback.error(TextToSpeech.ERROR_INVALID_REQUEST); return; }
            String text = request.getCharSequenceText().toString().strip();
            // Never truncate or independently re-synthesize long text with a random voice.
            if (text.isEmpty() || text.codePointCount(0,text.length()) > 500) { ModelStore.log(this,"rejected utterance length=" + text.length()); callback.error(TextToSpeech.ERROR_INVALID_REQUEST); return; }
            ModelStore.log(this,"synthesis_begin chars=" + text.length() + " rate=" + request.getSpeechRate());
            if (engine == 0) engine = NativeEngine.load(ModelStore.model(this).getPath(), 5);
            if (ticket != generation.get()) return;
            if (callback.start(44100, AudioFormat.ENCODING_PCM_16BIT, 1) != TextToSpeech.SUCCESS) return;
            File ref = ModelStore.reference(this), transcript = ModelStore.transcript(this);
            boolean clone = ref.isFile() && transcript.isFile() && !ModelStore.text(transcript).isBlank();
            NativeEngine.synthesize(engine, text, clone ? ref.getPath() : "", clone ? ModelStore.text(transcript) : "",
                request.getSpeechRate(), request.getPitch(), pcm -> {
                    if (generation.get() != ticket) return false;
                    for (int offset = 0; offset < pcm.length;) {
                        if (generation.get() != ticket) return false;
                        int count = Math.min(callback.getMaxBufferSize(), pcm.length - offset);
                        count &= ~1;
                        if (count == 0 || callback.audioAvailable(pcm, offset, count) != TextToSpeech.SUCCESS) return false;
                        offset += count;
                    }
                    return true;
                });
            if (ticket == generation.get()) { callback.done(); ModelStore.log(this,"synthesis_done"); }
            else ModelStore.log(this,"synthesis_stopped");
        } catch (Throwable e) {
            ModelStore.log(this, "synthesis_error " + android.util.Log.getStackTraceString(e));
            if (ticket == generation.get()) callback.error(TextToSpeech.ERROR_SYNTHESIS);
            if (engine != 0) { NativeEngine.close(engine); engine = 0; }
        }
    }
    @Override public void onDestroy() { onStop(); super.onDestroy(); /* Process owns native state; don't free while synthesis is active. */ }
}
