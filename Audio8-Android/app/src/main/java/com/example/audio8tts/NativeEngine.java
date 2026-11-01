package com.example.audio8tts;

final class NativeEngine {
    static { System.loadLibrary("audio8_jni"); }
    interface Sink { boolean pcm(byte[] data); boolean cancelled(); }
    // A handle is owned exclusively by the synthesis thread.
    static native long load(String path, int threads, boolean affinity, String npuModels,
                            String libraries, String cache, int fastThreads, boolean fastNpu, String codecModels);
    static native void synthesize(long handle, String text, String reference,
                                  String transcript, int rate, int pitch, boolean compact,
                                  boolean graphs, int threads, int fastThreads, int codecThreads, boolean codecFast, boolean affinity, Sink sink);
    static native void prepareCodec(long handle,int frames);
    static native void close(long handle);
}
