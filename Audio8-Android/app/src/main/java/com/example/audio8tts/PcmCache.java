package com.example.audio8tts;
import java.util.LinkedHashMap;

/** Bounded process-local cache; no speech text or PCM is written to disk. */
final class PcmCache {
    private static final int LIMIT=16*1024*1024;
    static final int ENTRY_LIMIT=4*1024*1024;
    private final LinkedHashMap<String,byte[]> entries=new LinkedHashMap<>(16,0.75f,true);
    private int bytes;
    byte[] get(String key) { return entries.get(key); }
    void clear() { entries.clear(); bytes=0; }
    void put(String key,byte[] pcm) {
        if (pcm.length>ENTRY_LIMIT || pcm.length==0) return;
        byte[] previous=entries.put(key,pcm); bytes+=pcm.length-(previous==null ? 0 : previous.length);
        while (bytes>LIMIT || entries.size()>32) {
            var oldest=entries.entrySet().iterator(); var entry=oldest.next();
            bytes-=entry.getValue().length; oldest.remove();
        }
    }
}
