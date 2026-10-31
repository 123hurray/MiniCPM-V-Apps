package com.example.audio8tts;

import android.content.Context;
import android.net.Uri;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;

final class ModelStore {
    static final long SIZE = 851018912L;
    static final String HASH = "fdab2e5c14e2cd00d198b752c52cb3d92097ee18fd54f4ef4eed99ee5d8c512e";
    static String endpoint(Context c) {
        String stored = c.getSharedPreferences("downloads", Context.MODE_PRIVATE).getString("hf_endpoint", DownloadEndpoint.MIRROR);
        try { return DownloadEndpoint.normalize(stored); } catch (IllegalArgumentException e) { return DownloadEndpoint.MIRROR; }
    }
    static String saveEndpoint(Context c, String value) throws IOException {
        String normalized = DownloadEndpoint.normalize(value);
        if (!c.getSharedPreferences("downloads", Context.MODE_PRIVATE).edit().putString("hf_endpoint", normalized).commit()) throw new IOException("无法保存下载域名");
        return normalized;
    }
    interface Progress { void update(String message); }
    static File dir(Context c) { File d = new File(c.getFilesDir(), "audio8"); d.mkdirs(); return d; }
    static File model(Context c) { return new File(dir(c), "preview-0.1b-q8.gguf"); }
    static boolean ready(Context c) { return model(c).length() == SIZE; }
    static File reference(Context c) { return new File(dir(c), "reference.wav"); }
    static File transcript(Context c) { return new File(dir(c), "reference.txt"); }
    static synchronized void log(Context c, String message) {
        try (FileWriter w = new FileWriter(new File(dir(c), "diagnostics.txt"), true)) {
            w.write(java.time.Instant.now() + " " + message + "\n"); w.flush();
        } catch (IOException ignored) { }
    }
    static String text(File file) throws IOException { return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8); }
    static void saveText(File file, String value) throws IOException {
        File tmp = new File(file.getPath() + ".tmp");
        Files.write(tmp.toPath(), value.getBytes(StandardCharsets.UTF_8));
        Files.move(tmp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
    static void importReference(Context c, Uri uri) throws IOException {
        File tmp = new File(dir(c), "reference.part");
        try (InputStream in = c.getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(tmp)) {
            if (in == null) throw new IOException("无法读取音频");
            byte[] b = new byte[65536]; long total = 0; int n;
            while ((n = in.read(b)) != -1) { total += n; if (total > 10_000_000) throw new IOException("参考 WAV 不得超过 10MB"); out.write(b, 0, n); }
        }
        try (InputStream in = new FileInputStream(tmp)) {
            byte[] h = in.readNBytes(12);
            if (h.length != 12 || !new String(h, 0, 4, StandardCharsets.US_ASCII).equals("RIFF") ||
                !new String(h, 8, 4, StandardCharsets.US_ASCII).equals("WAVE")) throw new IOException("请选择 PCM WAV 文件");
        }
        Files.move(tmp.toPath(), reference(c).toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
    static void download(Context c, Progress p) throws Exception {
        String origin = "https://github.com";
        p.update("连接下载源：" + origin);
        log(c, "download_source=" + origin);
        File part = new File(dir(c), "preview-0.1b-q8.part"); long offset = part.length();
        if (offset > SIZE) throw new IOException("下载临时文件过大，请清除应用数据后重试");
        if (offset < SIZE) {
            HttpURLConnection conn = (HttpURLConnection) new URL(DownloadEndpoint.previewModelUrl()).openConnection();
            conn.setConnectTimeout(30000); conn.setReadTimeout(60000);
            conn.setRequestProperty("Accept-Encoding", "identity");
            if (offset > 0) conn.setRequestProperty("Range", "bytes=" + offset + "-");
            try {
                int status = conn.getResponseCode();
                if (status != 200 && status != 206) throw new IOException(origin + " 返回 HTTP " + status + "，可切换下载源后续传");
                if (status == 200) offset = 0;
                else if (!DownloadEndpoint.matchesRange(conn.getHeaderField("Content-Range"), offset, SIZE)) throw new IOException("断点响应不匹配，原有进度已保留；请切换下载源重试");
                try (InputStream in = conn.getInputStream(); FileOutputStream out = new FileOutputStream(part, offset > 0)) {
                    byte[] b = new byte[262144]; long total = offset, reported = 0; int n;
                    while ((n = in.read(b)) != -1) {
                        total += n; if (total > SIZE) throw new IOException("模型大小异常"); out.write(b, 0, n);
                        long now = System.currentTimeMillis();
                        if (now - reported > 500) { p.update("下载 " + (total * 100 / SIZE) + "%（可断点续传）"); reported = now; }
                    }
                    out.getFD().sync();
                }
            } finally { conn.disconnect(); }
        }
        if (part.length() != SIZE) throw new IOException("文件不完整，请重试以续传");
        p.update("校验 SHA-256…");
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new FileInputStream(part)) { byte[] b = new byte[262144]; int n; while ((n = in.read(b)) != -1) md.update(b, 0, n); }
        StringBuilder hex = new StringBuilder(); for (byte b : md.digest()) hex.append(String.format("%02x", b & 255));
        if (!HASH.equals(hex.toString())) { Files.delete(part.toPath()); throw new IOException("校验失败，损坏的临时模型已删除，请重新下载"); }
        Files.move(part.toPath(), model(c).toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        log(c, "model_download_verified sha256=" + HASH);
        p.update("模型已就绪");
    }
}
