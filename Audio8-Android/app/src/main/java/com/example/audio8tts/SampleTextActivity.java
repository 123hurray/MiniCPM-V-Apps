package com.example.audio8tts;
import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
public final class SampleTextActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        setResult(TextToSpeech.LANG_AVAILABLE, new Intent().putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, "你好，这是 Audio8 离线语音合成引擎。")); finish();
    }
}
