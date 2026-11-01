package com.example.audio8tts;

final class NativeEngine {
    static { System.loadLibrary("audio8_jni"); }
    interface Sink { boolean pcm(byte[] data); boolean cancelled(); void captureCodes(int[] codes,int frames); float[] codec(int[] codes,int frames); }
    // A handle is owned exclusively by the synthesis thread.
    static native long load(String path, int threads, boolean affinity, String npuModels,
                            String libraries, String cache, int fastThreads, boolean fastNpu, String codecModels);
    static native void synthesize(long handle, String text, String reference,
                                  String transcript, int rate, int pitch, boolean compact,
                                  boolean graphs, int threads, int fastThreads, int codecThreads, boolean codecFast, boolean affinity, boolean useJavaCodec, Sink sink);
    static native void prepareCodec(String models,String libraries,String cache,int frames);
    static native float[] baselineCodec(long handle,int[] codes,int frames,int threads);
    static native void close(long handle);
}
