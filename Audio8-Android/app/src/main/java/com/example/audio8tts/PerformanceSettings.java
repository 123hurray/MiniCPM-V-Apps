package com.example.audio8tts;

import android.content.Context;

final class PerformanceSettings {
    final boolean compact, graphs, affinity, prewarm, cache;
    final int threads, fastThreads;
    PerformanceSettings(Context context) {
        // Separate service process: read an atomically replaced file, not cached SharedPreferences.
        java.util.Properties p = new java.util.Properties();
        try (var in = new java.io.FileInputStream(file(context))) { p.load(in); }
        catch (java.io.IOException ignored) { }
        compact = bool(p,"compact",true); graphs = bool(p,"graphs",true);
        affinity = bool(p,"affinity",true); prewarm = bool(p,"prewarm",true);
        cache = bool(p,"cache",true);
        threads = number(p,"threads",5); fastThreads = number(p,"fast_threads",2);
    }
    private static boolean bool(java.util.Properties p,String k,boolean d) { return Boolean.parseBoolean(p.getProperty(k,Boolean.toString(d))); }
    private static int number(java.util.Properties p,String k,int d) {
        try { return Math.max(1,Math.min(8,Integer.parseInt(p.getProperty(k,""+d)))); }
        catch (NumberFormatException e) { return d; }
    }
    static java.io.File file(Context c) { return new java.io.File(ModelStore.dir(c),"performance.properties"); }
    static void save(Context c,boolean compact,boolean graphs,boolean affinity,boolean prewarm,boolean cache,int threads,int fast) throws java.io.IOException {
        var p = new java.util.Properties();
        p.setProperty("compact",""+compact); p.setProperty("graphs",""+graphs);
        p.setProperty("affinity",""+affinity); p.setProperty("prewarm",""+prewarm); p.setProperty("cache",""+cache);
        p.setProperty("threads",""+threads); p.setProperty("fast_threads",""+fast);
        android.util.AtomicFile f = new android.util.AtomicFile(file(c));
        var out = f.startWrite();
        try { p.store(out,"Audio8 performance"); f.finishWrite(out); }
        catch (java.io.IOException e) { f.failWrite(out); throw e; }
    }
    String engineKey() { return compact+":"+graphs+":"+affinity+":"+threads+":"+fastThreads; }
}
