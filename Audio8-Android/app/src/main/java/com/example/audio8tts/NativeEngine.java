package com.example.audio8tts;

final class NativeEngine {
    static { System.loadLibrary("audio8_jni"); }
    interface Sink { boolean pcm(byte[] data); boolean cancelled(); }
    // A handle is owned exclusively by the synthesis thread.
    static native long load(String path, int threads, boolean affinity);
    static native void synthesize(long handle, String text, String reference,
                                  String transcript, int rate, int pitch, boolean compact,
                                  boolean graphs, int threads, int fastThreads, boolean affinity, Sink sink);
    static native void close(long handle);
}
