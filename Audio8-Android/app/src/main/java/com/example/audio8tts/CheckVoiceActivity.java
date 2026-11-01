package com.example.audio8tts;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import java.util.ArrayList;
import java.util.List;
public final class CheckVoiceActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state); var settings=new PerformanceSettings(this); boolean ready = ModelStore.ready(this) && (settings.codecBackend.equals("litert")?LiteCodecStore.ready(this):!settings.codecNpu||CodecModelStore.ready(this));
        Intent result = new Intent();
        ArrayList<String> voices = new ArrayList<>(List.of("zho-CHN", "eng-USA"));
        result.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, ready ? voices : new ArrayList<>());
        result.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, ready ? new ArrayList<>() : voices);
        setResult(ready ? TextToSpeech.Engine.CHECK_VOICE_DATA_PASS : TextToSpeech.Engine.CHECK_VOICE_DATA_FAIL, result); finish();
    }
}
