package com.example.audio8tts;

public final class DownloadEndpointTest {
    private static void check(boolean value) { if (!value) throw new AssertionError(); }
    private static void rejected(String value) {
        try { DownloadEndpoint.normalize(value); throw new AssertionError("Accepted invalid endpoint: " + value); }
        catch (IllegalArgumentException expected) { }
    }
    public static void main(String[] args) {
        check(DownloadEndpoint.normalize(" HTTPS://HF-MIRROR.COM/ ").equals(DownloadEndpoint.MIRROR));
        check(DownloadEndpoint.normalize("https://cdn.example.com:8443/").equals("https://cdn.example.com:8443"));
        String official = DownloadEndpoint.modelUrl(DownloadEndpoint.OFFICIAL);
        check(DownloadEndpoint.modelUrl(DownloadEndpoint.MIRROR).equals(official.replace("https://huggingface.co", "https://hf-mirror.com")));
        check(official.contains("/resolve/b57015b9e916ede150f4b168b4b882be37ad723f/"));
        for (String invalid : new String[]{"", "http://hf-mirror.com", "hf-mirror.com", "https://user:password@mirror.example", "https://mirror.example/model", "https://mirror.example/?token=x", "https://mirror.example/#x", "https://mirror.example:0", "https://mirror.example:65536"}) rejected(invalid);
        rejected(null);
        check(DownloadEndpoint.matchesRange("bytes 10-99/100", 10, 100));
        check(!DownloadEndpoint.matchesRange(null, 10, 100));
        check(!DownloadEndpoint.matchesRange("bytes 11-99/100", 10, 100));
        check(!DownloadEndpoint.matchesRange("bytes 10-100/100", 10, 100));
        check(!DownloadEndpoint.matchesRange("bytes 10-99/101", 10, 100));
        check(!DownloadEndpoint.matchesRange("bytes 10-99/*", 10, 100));
        check(!DownloadEndpoint.matchesRange("bytes 10-9999999999999999999999/100", 10, 100));
        System.out.println("Download endpoint and resume validation tests passed");
    }
}
