package com.example.audio8tts;

final class NativeEngine {
    static { System.loadLibrary("audio8_jni"); }
    interface Sink { boolean pcm(byte[] data); }
    // A handle is owned exclusively by the synthesis thread.
    static native long load(String path, int threads);
    static native void synthesize(long handle, String text, String reference,
                                  String transcript, int rate, int pitch, Sink sink);
    static native void close(long handle);
}
