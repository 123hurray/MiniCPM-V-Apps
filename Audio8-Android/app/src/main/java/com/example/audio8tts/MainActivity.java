package com.example.audio8tts;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.*;
import android.os.*;
import android.speech.tts.TextToSpeech;
import android.widget.*;
import java.io.*;
import java.util.Locale;
import java.util.concurrent.*;

public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private TextView status;
    private EditText text, referenceText;
    private Button download;
    private TextToSpeech tts;
    private boolean ttsReady;
    private void message(String s) { runOnUiThread(() -> { if (!isDestroyed()) status.setText(s); }); }
    private Button button(LinearLayout root, String title, Runnable action) {
        Button b = new Button(this); b.setText(title); root.addView(b); b.setOnClickListener(v -> action.run()); return b;
    }
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this); LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL); root.setPadding(32,48,32,32); scroll.addView(root); setContentView(scroll);
        TextView title = new TextView(this); title.setText("Audio8 系统语音 · 4.0"); title.setTextSize(26); root.addView(title);
        TextView note = new TextView(this);
        note.setText("Audio8-TTS-Realistic-0.6B · 本地 CPU / ARM64\n首次下载约 1.43 GB，至少预留 3 GB 存储。无需 root。下载后可在系统文字转语音设置中选择本引擎。\n实验版：每次请求整段合成后播放，不是低延迟帧流式；单次最多 500 字，不截断、不随机拆段。停止立即丢弃播放，当前原生计算完成后才能处理下一条。\n建议系统语速和音调保持默认；当前语速使用重采样，不能独立保留音调。为跨请求保持音色，请导入你有权使用的参考 WAV 和准确文本。"); root.addView(note);
        status = new TextView(this); status.setText(ModelStore.ready(this) ? "模型已就绪" : "模型未下载"); root.addView(status);
        download = button(root, "下载 / 续传模型", () -> {
            download.setEnabled(false);
            worker.execute(() -> { try { ModelStore.download(this,this::message); } catch (Exception e) { ModelStore.log(this,"download_error " + e); message("下载失败：" + e.getMessage()); } finally { runOnUiThread(() -> download.setEnabled(true)); } });
        });
        text = new EditText(this); text.setHint("输入试听文本（最多 500 字）"); text.setText("你好，这是在手机本地运行的 Audio8 系统语音。"); root.addView(text);
        button(root,"通过系统 TTS 接口试听", () -> {
            if (!ttsReady) { message("系统引擎初始化中，请稍后重试"); return; }
            tts.setLanguage(Locale.SIMPLIFIED_CHINESE); tts.setSpeechRate(1f); tts.setPitch(1f);
            if (tts.speak(text.getText().toString(), TextToSpeech.QUEUE_FLUSH, null, "preview") != TextToSpeech.SUCCESS) message("提交失败，请导出诊断");
        });
        button(root,"停止", () -> { if (tts != null) tts.stop(); });
        button(root,"打开系统文字转语音设置", () -> {
            try { startActivity(new Intent("com.android.settings.TTS_SETTINGS")); } catch (ActivityNotFoundException e) { startActivity(new Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS)); }
        });
        referenceText = new EditText(this); referenceText.setHint("参考音频中逐字准确的文本"); root.addView(referenceText);
        try { if (ModelStore.transcript(this).isFile()) referenceText.setText(ModelStore.text(ModelStore.transcript(this))); } catch (IOException ignored) { }
        button(root,"导入参考 WAV（建议 3～10 秒）", () -> {
            Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT).setType("audio/*").addCategory(Intent.CATEGORY_OPENABLE); startActivityForResult(i,1);
        });
        button(root,"保存参考文本", () -> { try { ModelStore.saveText(ModelStore.transcript(this),referenceText.getText().toString()); message("已保存；下次合成生效"); } catch (IOException e) { message(e.toString()); } });
        button(root,"关闭参考音色", () -> { try { ModelStore.saveText(ModelStore.transcript(this),""); referenceText.setText(""); message("参考音色已关闭；无参考模式不保证跨请求音色一致"); } catch (IOException e) { message(e.toString()); } });
        button(root,"导出诊断到任意目录", () -> startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("text/plain").putExtra(Intent.EXTRA_TITLE,"audio8-diagnostics.txt").addCategory(Intent.CATEGORY_OPENABLE),2));
        tts = new TextToSpeech(this, result -> {
            ttsReady = result == TextToSpeech.SUCCESS;
            if (!ttsReady) message("引擎未初始化；先下载模型，再重新打开应用");
        },getPackageName());
        tts.setOnUtteranceProgressListener(new android.speech.tts.UtteranceProgressListener() {
            public void onStart(String id) { message("正在合成 / 播放…"); }
            public void onDone(String id) { message("试听结束"); }
            @Deprecated public void onError(String id) { message("合成失败，请导出诊断"); }
            public void onError(String id,int code) { message("合成失败，错误码=" + code + "；请导出诊断"); }
        });
    }
    @Override protected void onActivityResult(int request,int result,Intent data) {
        super.onActivityResult(request,result,data);
        if (result != RESULT_OK || data == null || data.getData() == null) return;
        final android.net.Uri uri = data.getData();
        worker.execute(() -> {
            try {
                if (request == 1) { ModelStore.importReference(this,uri); message("参考 WAV 已导入，请填写并保存准确文本"); }
                if (request == 2) {
                    try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                        if (out == null) throw new IOException("无法写入诊断");
                        out.write(("Audio8 4.0 " + Build.MANUFACTURER + " " + Build.MODEL + " SDK=" + Build.VERSION.SDK_INT + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        for (String name : new String[]{"diagnostics.txt","diagnostics-native.txt"}) {
                            File f = new File(ModelStore.dir(this),name);
                            if (f.exists()) try (InputStream in = new FileInputStream(f)) { in.transferTo(out); }
                        }
                        if (Build.VERSION.SDK_INT >= 30) {
                            ActivityManager am = (ActivityManager)getSystemService(ACTIVITY_SERVICE);
                            for (android.app.ApplicationExitInfo info : am.getHistoricalProcessExitReasons(getPackageName(),0,5)) {
                                out.write(("\nexit: " + info.getProcessName() + " reason=" + info.getReason() + " " + info.getDescription() + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
                                try (InputStream trace = info.getTraceInputStream()) { if (trace != null) trace.transferTo(out); }
                            }
                        }
                    }
                    message("诊断已导出；可能包含文件路径及运行时数据，分享前请检查");
                }
            } catch (Exception e) { message(e.toString()); }
        });
    }
    @Override public void onDestroy() { if (tts != null) { tts.stop(); tts.shutdown(); } worker.shutdown(); super.onDestroy(); }
}
