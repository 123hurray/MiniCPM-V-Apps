package com.example.audio8tts;
import android.content.Context;
import java.io.File;
final class RuntimeStatus {
    static void write(Context c,String phase) {
        try { ModelStore.saveText(new File(ModelStore.dir(c),"service-status.txt"),System.currentTimeMillis()+"\n"+phase); }
        catch(Exception e) { android.util.Log.w("Audio8","Cannot persist phase",e); }
        ModelStore.log(c,"stage="+phase);
    }
    static String read(Context c,String name) {
        try {
            String[] parts=ModelStore.text(new File(ModelStore.dir(c),name)).split("\n",2);
            long age=Math.max(0,(System.currentTimeMillis()-Long.parseLong(parts[0]))/1000);
            String phase=parts[1];
            if(phase.contains("完成") || phase.contains("失败") || phase.contains("超时") || phase.contains("已结束") || phase.contains("已重启") || phase.equals("模型就绪")) return phase;
            return phase+" · 已用 "+age+" 秒";
        } catch(Exception ignored) { return "尚未开始，点击试听"; }
    }
}
