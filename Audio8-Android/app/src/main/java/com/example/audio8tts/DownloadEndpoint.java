package com.example.audio8tts;

import java.net.URI;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Origin-only mirror configuration. Model revision, path and hash never change. */
public final class DownloadEndpoint {
    public static final String MIRROR = "https://hf-mirror.com";
    public static final String OFFICIAL = "https://huggingface.co";
    private static final String MODEL_PATH = "/audio-cpp/Audio8-TTS-Realistic-0.6b-GGUF/resolve/b57015b9e916ede150f4b168b4b882be37ad723f/audio8-tts-realistic-0.6b-q8_0.gguf";
    private static final Pattern RANGE = Pattern.compile("bytes ([0-9]+)-([0-9]+)/([0-9]+)");
    public static String normalize(String value) {
        if (value == null) throw new IllegalArgumentException("请填写 HTTPS 镜像域名");
        try {
            URI uri = new URI(value.trim());
            String path = uri.getRawPath();
            if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null ||
                uri.getRawUserInfo() != null || uri.getRawQuery() != null || uri.getRawFragment() != null ||
                (path != null && !path.isEmpty() && !path.equals("/")) ||
                uri.getPort() == 0 || uri.getPort() > 65535) {
                throw new IllegalArgumentException("只填写 HTTPS 域名，例如 https://hf-mirror.com；不要填写模型路径、密码或参数");
            }
            return "https://" + uri.getHost().toLowerCase(Locale.ROOT) + (uri.getPort() == -1 ? "" : ":" + uri.getPort());
        } catch (java.net.URISyntaxException e) {
            throw new IllegalArgumentException("镜像域名格式无效", e);
        }
    }
    public static String modelUrl(String endpoint) { return normalize(endpoint) + MODEL_PATH; }
    public static String previewModelUrl() { return "https://github.com/123hurray/MiniCPM-V-Apps/releases/download/audio8-preview01-q8-v1/audio8-preview-0.1b-q8_0.gguf"; }
    public static boolean matchesRange(String header, long offset, long size) {
        if (header == null || offset < 0 || size <= offset) return false;
        Matcher match = RANGE.matcher(header);
        if (!match.matches()) return false;
        try {
            long start = Long.parseLong(match.group(1)), end = Long.parseLong(match.group(2));
            return start == offset && end >= start && end < size && Long.parseLong(match.group(3)) == size;
        } catch (NumberFormatException e) { return false; }
    }
}
