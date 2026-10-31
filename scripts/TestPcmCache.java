package com.example.audio8tts;
public class TestPcmCache {
    static void require(boolean ok) { if(!ok) throw new AssertionError("PCM cache invariant"); }
    public static void main(String[] args) {
        PcmCache c=new PcmCache();
        require(c.get("missing")==null);
        for(int i=0;i<4;i++) c.put(""+i,new byte[PcmCache.ENTRY_LIMIT]);
        require(c.get("0")!=null);
        c.put("4",new byte[PcmCache.ENTRY_LIMIT]);
        require(c.get("1")==null && c.get("0")!=null);
        c.put("large",new byte[PcmCache.ENTRY_LIMIT+1]); require(c.get("large")==null);
        c.put("4",new byte[]{42}); require(c.get("4")[0]==42);
        c.clear(); require(c.get("0")==null);
        for(int i=0;i<33;i++) c.put(""+i,new byte[]{1});
        require(c.get("0")==null && c.get("32")!=null);
        System.out.println("PCM cache bounds, LRU, replacement and clear passed");
    }
}
