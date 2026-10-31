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
    static final long SIZE = 1429460800L;
    static final String HASH = "e8a2402b50341b12316f94e78afd8b69d38010c5d3ba3878d64ca8f9ea90a930";
    static final String URL_MODEL = "https://huggingface.co/audio-cpp/Audio8-TTS-Realistic-0.6b-GGUF/resolve/b57015b9e916ede150f4b168b4b882be37ad723f/audio8-tts-realistic-0.6b-q8_0.gguf";
    interface Progress { void update(String message); }
    static File dir(Context c) { File d = new File(c.getFilesDir(), "audio8"); d.mkdirs(); return d; }
    static File model(Context c) { return new File(dir(c), "realistic-q8.gguf"); }
    static boolean ready(Context c) { return model(c).length() == SIZE; }
    static File reference(Context c) { return new File(dir(c), "reference.wav"); }
    static File transcript(Context c) { return new File(dir(c), "reference.txt"); }
    static synchronized void log(Context c, String message) {
        try (FileWriter w = new FileWriter(new File(dir(c), "diagnostics.txt"), true)) {
            w.write(java.time.Instant.now() + " " + message + "\n"); w.flush();
        } catch (IOException ignored) { }
    }
    static String text(File file) throws IOException { return Files.readString(file.toPath(), StandardCharsets.UTF_8); }
    static void saveText(File file, String value) throws IOException {
        File tmp = new File(file.getPath() + ".tmp");
        Files.writeString(tmp.toPath(), value, StandardCharsets.UTF_8);
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
        File part = new File(dir(c), "model.part"); long offset = part.length();
        if (offset > SIZE) throw new IOException("下载临时文件过大，请清除应用数据后重试");
        if (offset < SIZE) {
            HttpURLConnection conn = (HttpURLConnection) new URL(URL_MODEL).openConnection();
            conn.setConnectTimeout(30000); conn.setReadTimeout(60000);
            if (offset > 0) conn.setRequestProperty("Range", "bytes=" + offset + "-");
            try {
                int status = conn.getResponseCode();
                if (status != 200 && status != 206) throw new IOException("HTTP " + status);
                if (status == 200) offset = 0;
                else if (!conn.getHeaderField("Content-Range").startsWith("bytes " + offset + "-")) throw new IOException("断点响应不匹配");
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
