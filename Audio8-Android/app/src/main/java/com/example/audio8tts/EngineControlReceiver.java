package com.example.audio8tts;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
public final class EngineControlReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context c,Intent i) {
        if (!android.app.Application.getProcessName().endsWith(":synthesis")) return;
        RuntimeStatus.write(c,"语音进程已重启；请重新连接引擎");
        android.os.Process.killProcess(android.os.Process.myPid());
    }
}
