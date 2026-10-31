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
    private final ExecutorService speechWorker=Executors.newSingleThreadExecutor();
    private final Handler ui=new Handler(Looper.getMainLooper());
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
        TextView title = new TextView(this); title.setText("Audio8 · 4.5"); title.setTextSize(26); root.addView(title);
        TextView note = new TextView(this);
        note.setText("Audio8-TTS-Realistic-0.6B · SoC="+Build.SOC_MODEL+"\nCPU 基础模型约 1.43 GB，已有文件保留。NPU 另下载约 0.9 GB 模型，请预留至少 5 GB 空间。模型不在 APK 中。\nNPU 实验路径：主干解码 HTP INT4；预填充、codec 保留 CPU。Fast AR 可选 CPU INT4 / NPU。不是全模型 NPU。\n首次 NPU 编译可能需要数分钟，后续复用缓存。无法启动时会报错，不会静默改用 CPU。推理进程禁止新建互联网 socket；下载由独立界面进程完成。\n仍整段生成后播放，不随机拆段；停止在 AR 步间生效，不能中断单次加载或 codec 计算。语速和音调建议默认。"); root.addView(note);
        status = new TextView(this); status.setText(ModelStore.ready(this) ? "模型已就绪" : "模型未下载"); root.addView(status);
        PerformanceSettings perf = new PerformanceSettings(this);
        TextView perfTitle = new TextView(this); perfTitle.setText("性能设置（保存后下次请求生效；改变原生配置会重载模型）"); root.addView(perfTitle);
        CheckBox compact = option(root,"只计算有效语音词表（4096 + EOS）",perf.compact);
        CheckBox graphs = option(root,"保留计算图（减少重复构图，增加常驻内存）",perf.graphs);
        CheckBox affinity = option(root,"优先使用高频 CPU 核心",perf.affinity);
        CheckBox prewarm = option(root,"自动预热已停用（避免阻塞试听）",false);prewarm.setEnabled(false);
        CheckBox cache = option(root,"重复文本内存音频缓存（最多 16 MiB，不落盘）",perf.cache);
        CheckBox npu = option(root,"启用主干 NPU INT4（SM8550 实验模式，需另下模型）",perf.npu);
        CheckBox fastNpu = option(root,"Fast AR 也用 NPU（可与 CPU INT4 对比）",perf.fastNpu);
        fastNpu.setEnabled(npu.isChecked());
        npu.setOnCheckedChangeListener((b,checked) -> fastNpu.setEnabled(checked));
        TextView threadLabels = new TextView(this);threadLabels.setText("下方第一个数：CPU 主干/预填充与 codec 线程；第二个数：Fast AR CPU 线程。它们不是 NPU 核心数。");root.addView(threadLabels);
        EditText threads = new EditText(this); threads.setHint("主干 / codec 线程数 1～8"); threads.setInputType(2); threads.setText(""+perf.threads); root.addView(threads);
        EditText fastThreads = new EditText(this); fastThreads.setHint("小型 Fast AR 线程数 1～8"); fastThreads.setInputType(2); fastThreads.setText(""+perf.fastThreads); root.addView(fastThreads);
        button(root,"保存性能设置", () -> {
            try {
                int n=Integer.parseInt(threads.getText().toString()), f=Integer.parseInt(fastThreads.getText().toString());
                if(n<1 || n>8 || f<1 || f>8) throw new IllegalArgumentException("线程数需为 1～8");
                PerformanceSettings.save(this,compact.isChecked(),graphs.isChecked(),affinity.isChecked(),prewarm.isChecked(),cache.isChecked(),n,f,npu.isChecked(),fastNpu.isChecked());
                speechWorker.execute(() -> {if(tts!=null)tts.stop();});
                message("已保存，下次合成重载模型。若刚下载完 NPU 模型，请点下方重新连接系统引擎。");
            } catch(Exception e) { message("保存失败："+e.getMessage()); }
        });
        button(root,"恢复 4.2 对照设置", () -> {
            try {
                PerformanceSettings.save(this,false,false,false,false,false,5,5,false,false);
                compact.setChecked(false); graphs.setChecked(false); affinity.setChecked(false); prewarm.setChecked(false); cache.setChecked(false);
                npu.setChecked(false);fastNpu.setChecked(false);threads.setText("5"); fastThreads.setText("5"); message("已保存 CPU 对照设置；NPU 关闭。");
            } catch(Exception e) { message(e.getMessage()); }
        });
        button(root,"恢复推荐优化设置", () -> {
            try {
                PerformanceSettings.save(this,true,true,true,false,true,5,2,false,false);
                compact.setChecked(true); graphs.setChecked(true); affinity.setChecked(true); prewarm.setChecked(false); cache.setChecked(true);
                npu.setChecked(false);fastNpu.setChecked(false);threads.setText("5"); fastThreads.setText("2"); message("已恢复推荐 CPU 优化设置；NPU 关闭。");
            } catch(Exception e) { message(e.getMessage()); }
        });
        TextView npuStatus=new TextView(this);npuStatus.setText(NpuModelStore.ready(this)?"NPU 模型已就绪":"NPU 模型未下载");root.addView(npuStatus);
        EditText npuOrigin=new EditText(this);npuOrigin.setSingleLine(true);npuOrigin.setText(NpuModelStore.origin(this));npuOrigin.setHint("NPU Release 下载基础地址，可填写 HTTPS 代理路径");root.addView(npuOrigin);
        button(root,"下载 / 续传 NPU INT4 模型", () -> {
            try { NpuModelStore.saveOrigin(this,npuOrigin.getText().toString()); }
            catch(Exception e) { message(e.getMessage());return; }
            worker.execute(() -> {
                try { NpuModelStore.download(this,this::message);runOnUiThread(() -> npuStatus.setText("NPU 模型已就绪，请重新连接系统引擎")); }
                catch(Exception e) { ModelStore.log(this,"npu_download_error "+e);message("NPU 下载失败："+e.getMessage()); }
            });
        });
        button(root,"重新连接系统语音引擎", () -> recreate());
        TextView sourceNote = new TextView(this);
        sourceNote.setText("Hugging Face 下载域名（默认第三方 HF-Mirror，仅影响模型下载）\n切换源保留下载进度，文件仍校验固定 SHA-256。填写域名即可，无需模型路径或 Token。"); root.addView(sourceNote);
        EditText endpoint = new EditText(this); endpoint.setSingleLine(true);
        endpoint.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_VARIATION_URI);
        endpoint.setHint(DownloadEndpoint.MIRROR); endpoint.setText(ModelStore.endpoint(this)); root.addView(endpoint);
        button(root,"保存自定义下载域名", () -> {
            try { endpoint.setText(ModelStore.saveEndpoint(this, endpoint.getText().toString())); message("已保存，下次下载生效"); }
            catch (Exception e) { message(e.getMessage()); }
        });
        button(root,"使用 HF-Mirror 镜像", () -> {
            try { endpoint.setText(ModelStore.saveEndpoint(this, DownloadEndpoint.MIRROR)); message("已切换至 HF-Mirror，下次下载生效"); }
            catch (IOException e) { message(e.getMessage()); }
        });
        button(root,"使用 Hugging Face 官方直连", () -> {
            try { endpoint.setText(ModelStore.saveEndpoint(this, DownloadEndpoint.OFFICIAL)); message("已切换至官方直连，下次下载生效"); }
            catch (IOException e) { message(e.getMessage()); }
        });
        download = button(root, "下载 / 续传模型", () -> {
            try { endpoint.setText(ModelStore.saveEndpoint(this, endpoint.getText().toString())); }
            catch (Exception e) { message(e.getMessage()); return; }
            download.setEnabled(false);
            worker.execute(() -> { try { ModelStore.download(this,this::message); } catch (Exception e) { ModelStore.log(this,"download_error " + e); message("下载失败：" + e.getMessage()); } finally { runOnUiThread(() -> download.setEnabled(true)); } });
        });
        text = new EditText(this); text.setHint("输入试听文本（最多 500 字）"); text.setText("你好，这是在手机本地运行的 Audio8 系统语音。"); root.addView(text);
        Button preview=button(root,"试听", () -> {
            if (!ttsReady) { message("系统引擎初始化中，请稍后重试"); return; }
            String input=text.getText().toString();message("已提交，等待语音引擎");
            speechWorker.execute(() -> {
                tts.setLanguage(Locale.SIMPLIFIED_CHINESE);tts.setSpeechRate(1f);tts.setPitch(1f);
                if(tts.speak(input,TextToSpeech.QUEUE_FLUSH,null,"preview")!=TextToSpeech.SUCCESS) message("提交失败，请重新连接引擎");
            });
        });
        Button stop=button(root,"停止", () -> {message("正在请求停止");speechWorker.execute(() -> {if(tts!=null)tts.stop();});});
        button(root,"检查系统 TTS 注册", () -> {
            Intent service = new Intent(TextToSpeech.Engine.INTENT_ACTION_TTS_SERVICE).setPackage(getPackageName());
            var matches = getPackageManager().queryIntentServices(service, android.content.pm.PackageManager.MATCH_DEFAULT_ONLY);
            boolean discoverable = matches.stream().anyMatch(r -> r.serviceInfo != null && r.serviceInfo.exported && r.serviceInfo.name.equals(Audio8Service.class.getName()));
            String report = "系统默认条件匹配：" + (discoverable ? "通过" : "未找到引擎") + "；模型：" + (ModelStore.ready(this) ? "已下载" : "未下载（不影响引擎出现在列表中）");
            ModelStore.log(this, "tts_discovery " + report); message(report);
        });
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
        arrangeHome(scroll,root,title,note,preview,stop);
    }
    private void arrangeHome(ScrollView scroll,LinearLayout root,TextView title,TextView note,Button preview,Button stop) {
        LinearLayout home=new LinearLayout(this);home.setOrientation(LinearLayout.VERTICAL);home.setPadding(32,48,32,32);
        for(android.view.View v:new android.view.View[]{title,status,text,preview,stop})root.removeView(v);
        root.removeView(note);root.setPadding(0,0,0,0);
        scroll.removeView(root);scroll.addView(home);home.addView(title);
        TextView models=new TextView(this);home.addView(models);
        TextView phase=new TextView(this);phase.setTextSize(18);phase.setPadding(0,24,0,24);home.addView(phase);
        phase.setMaxLines(3);phase.setEllipsize(android.text.TextUtils.TruncateAt.END);
        status.setMaxLines(2);status.setEllipsize(android.text.TextUtils.TruncateAt.END);
        home.addView(status);home.addView(text);home.addView(preview);home.addView(stop);
        button(home,"引擎无响应？重启语音进程",() -> {
            sendBroadcast(new Intent(this,EngineControlReceiver.class));
            message("已请求重启；模型保留。稍后点重新连接。");
        });
        button(home,"重新连接引擎",this::recreate);
        Button expand=button(home,"模型、音色与高级设置 ▸",() -> {});
        root.setVisibility(android.view.View.GONE);home.addView(root);
        expand.setOnClickListener(v -> {
            boolean open=root.getVisibility()==android.view.View.VISIBLE;
            root.setVisibility(open?android.view.View.GONE:android.view.View.VISIBLE);
            expand.setText(open?"模型、音色与高级设置 ▸":"收起高级设置 ▾");
        });
        TextView limits=new TextView(this);limits.setText("加载最长 4 分钟，合成最长 3 分钟。超时会结束语音进程，不删除模型。NPU 首次编译尚未真机验证。");root.addView(limits);
        ui.post(new Runnable(){public void run(){
            if(isDestroyed())return;
            PerformanceSettings s=new PerformanceSettings(MainActivity.this);
            models.setText((s.npu?"NPU INT4":"CPU")+" · 基础模型"+(ModelStore.ready(MainActivity.this)?"就绪":"未下载")+(s.npu?" · NPU模型"+(NpuModelStore.ready(MainActivity.this)?"就绪":"未下载"):""));
            java.io.File a=new java.io.File(ModelStore.dir(MainActivity.this),"service-status.txt"),b=new java.io.File(ModelStore.dir(MainActivity.this),"native-status.txt");
            phase.setText(RuntimeStatus.read(MainActivity.this,b.lastModified()>a.lastModified()?"native-status.txt":"service-status.txt"));
            ui.postDelayed(this,1000);
        }});
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
                        out.write(("Audio8 4.5 " + Build.MANUFACTURER + " " + Build.MODEL + " SoC="+Build.SOC_MODEL+" SDK=" + Build.VERSION.SDK_INT + " config="+new PerformanceSettings(this).engineKey()+" catalog="+BuildConfig.NPU_MANIFEST_SHA256+"\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        for (String name : new String[]{"diagnostics.txt","diagnostics-native.txt","service-status.txt","native-status.txt"}) {
                            File f = new File(ModelStore.dir(this),name);
                            if (f.exists()) try (InputStream in = new FileInputStream(f)) { in.transferTo(out); }
                        }
                        File[] profiles=NpuModelStore.runtimeCache(this).listFiles((d,name)->name.endsWith(".csv"));
                        if(profiles!=null) for(File file:profiles) {
                            out.write(("\nQNN profile: "+file.getName()+"\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
                            try(InputStream in=new FileInputStream(file)) { out.write(in.readNBytes(2*1024*1024)); }
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
    @Override public void onDestroy() {
        ui.removeCallbacksAndMessages(null);
        speechWorker.execute(() -> {if(tts!=null){tts.stop();tts.shutdown();}});
        speechWorker.shutdown();worker.shutdown();super.onDestroy();
    }
    private CheckBox option(LinearLayout root,String label,boolean checked) {
        CheckBox box=new CheckBox(this); box.setText(label); box.setChecked(checked); root.addView(box); return box;
    }
}
