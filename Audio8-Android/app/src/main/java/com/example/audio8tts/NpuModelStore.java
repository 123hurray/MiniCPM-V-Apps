package com.example.audio8tts;

import android.content.Context;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import org.json.*;

final class NpuModelStore {
    private static final Set<String> REQUIRED=Set.of("slow_ar_int4.onnx","slow_ar_int4.onnx.data","fast_ar_int4.onnx","fast_ar_int4.onnx.data",
        "slow_static.onnx","slow_static.onnx.data","fast_static.onnx","fast_static.onnx.data");
    private static final Set<String> META=Set.of("SOURCE-LICENSE","SOURCE-NOTICE","SOURCE-README.md","SOURCE-provenance.json","build-manifest.json");
    record Entry(String name,long size,String hash) { }
    static File dir(Context c) { File d=new File(ModelStore.dir(c),"npu-int4-v1"); d.mkdirs(); return d; }
    static File runtimeCache(Context c) {
        File d=new File(ModelStore.dir(c),"qnn-cache/"+BuildConfig.NPU_MANIFEST_SHA256);d.mkdirs();return d;
    }
    private static String hex(byte[] bytes) { StringBuilder s=new StringBuilder();for(byte b:bytes)s.append(String.format(java.util.Locale.ROOT,"%02x",b&255));return s.toString(); }
    private static String hash(byte[] bytes) throws Exception { return hex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
    private static String hash(File file) throws Exception {
        MessageDigest digest=MessageDigest.getInstance("SHA-256");
        try(InputStream in=new FileInputStream(file)) { byte[] bytes=new byte[262144];int n;while((n=in.read(bytes))!=-1)digest.update(bytes,0,n); }
        return hex(digest.digest());
    }
    static String origin(Context c) {
        try { return normalize(ModelStore.text(new File(ModelStore.dir(c),"npu-origin.txt"))); }
        catch(Exception ignored) { return BuildConfig.NPU_RELEASE_BASE; }
    }
    static String normalize(String text) throws Exception {
        URI uri=new URI(text.strip());
        if(!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost()==null || uri.getRawUserInfo()!=null || uri.getRawQuery()!=null || uri.getRawFragment()!=null)
            throw new IOException("NPU 下载源必须是无账号、查询参数和片段的 HTTPS 基础路径");
        return uri.toString().replaceAll("/+$","");
    }
    static void saveOrigin(Context c,String text) throws Exception { ModelStore.saveText(new File(ModelStore.dir(c),"npu-origin.txt"),normalize(text)); }
    private static List<Entry> catalog(byte[] bytes) throws Exception {
        if(!BuildConfig.NPU_MANIFEST_SHA256.matches("[0-9a-f]{64}") || !hash(bytes).equals(BuildConfig.NPU_MANIFEST_SHA256))
            throw new IOException("NPU 模型清单与 APK 固定 SHA-256 不符");
        JSONObject root=new JSONObject(new String(bytes,StandardCharsets.UTF_8));
        if(root.getInt("schema")!=1 || !root.getString("soc").equals("SM8550")) throw new IOException("不支持的 NPU 模型清单");
        JSONArray array=root.getJSONArray("files"); List<Entry> out=new ArrayList<>();Set<String> found=new HashSet<>();
        for(int i=0;i<array.length();i++) {
            JSONObject item=array.getJSONObject(i);String name=item.getString("name"),digest=item.getString("sha256");long size=item.getLong("size");
            if((!REQUIRED.contains(name) && !META.contains(name)) || !found.add(name) || !digest.matches("[0-9a-f]{64}") || size<1 || size>2_000_000_000L)
                throw new IOException("NPU 模型清单条目无效");
            out.add(new Entry(name,size,digest));
        }
        if(!found.containsAll(REQUIRED)) throw new IOException("NPU 模型清单缺少文件");
        return out;
    }
    static boolean ready(Context c) {
        try {
            File root=dir(c);
            if(!ModelStore.text(new File(root,"ready.sha256")).equals(BuildConfig.NPU_MANIFEST_SHA256)) return false;
            File manifest=new File(root,"npu-catalog.json");if(manifest.length()>262144) return false;
            for(Entry e:catalog(Files.readAllBytes(manifest.toPath()))) if(new File(root,e.name()).length()!=e.size()) return false;
            return true;
        } catch(Exception e) { return false; }
    }
    static void download(Context c,ModelStore.Progress progress) throws Exception {
        if(!BuildConfig.NPU_MANIFEST_SHA256.matches("[0-9a-f]{64}")) throw new IOException("此构建未绑定 NPU 模型清单");
        String source=origin(c);File root=dir(c); byte[] bytes;
        HttpURLConnection catalogConnection=(HttpURLConnection)new URL(source+"/npu-catalog.json").openConnection();
        catalogConnection.setConnectTimeout(30000);catalogConnection.setReadTimeout(60000);
        try(InputStream in=catalogConnection.getInputStream()) { bytes=in.readNBytes(262145); }
        finally { catalogConnection.disconnect(); }
        if(bytes.length>262144)throw new IOException("NPU 模型清单过大");
        List<Entry> entries=catalog(bytes);long total=entries.stream().mapToLong(Entry::size).sum(),done=0;
        for(Entry e:entries) {
            File target=new File(root,e.name());
            if(target.length()==e.size() && hash(target).equals(e.hash())) { done+=e.size();continue; }
            File part=new File(root,e.name()+".part");long offset=part.length();
            if(offset>e.size()) throw new IOException("NPU 临时文件超长："+e.name());
            if(offset<e.size()) {
                HttpURLConnection conn=(HttpURLConnection)new URL(source+"/"+e.name()).openConnection();
                conn.setConnectTimeout(30000);conn.setReadTimeout(60000);conn.setRequestProperty("Accept-Encoding","identity");
                if(offset>0)conn.setRequestProperty("Range","bytes="+offset+"-");
                try {
                    int status=conn.getResponseCode();
                    if(status==200)offset=0;
                    else if(status!=206 || !DownloadEndpoint.matchesRange(conn.getHeaderField("Content-Range"),offset,e.size())) throw new IOException("NPU 下载响应无效："+status);
                    try(InputStream in=conn.getInputStream();FileOutputStream out=new FileOutputStream(part,offset>0)) {
                        byte[] block=new byte[262144];int count;long received=offset,last=0;
                        while((count=in.read(block))!=-1) {
                            received+=count;if(received>e.size())throw new IOException("NPU 文件超长");out.write(block,0,count);
                            long now=System.currentTimeMillis();if(now-last>500) { progress.update("NPU 下载 "+((done+received)*100/total)+"% · "+e.name());last=now; }
                        }
                        out.getFD().sync();
                    }
                } finally { conn.disconnect(); }
            }
            if(part.length()!=e.size()) throw new IOException("NPU 文件未完整，请续传："+e.name());
            if(!hash(part).equals(e.hash())) { Files.delete(part.toPath());throw new IOException("NPU 文件校验失败，损坏临时文件已删除："+e.name()); }
            Files.move(part.toPath(),target.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);done+=e.size();
        }
        ModelStore.saveText(new File(root,"npu-catalog.json"),new String(bytes,StandardCharsets.UTF_8));
        ModelStore.saveText(new File(root,"ready.sha256"),BuildConfig.NPU_MANIFEST_SHA256);
        ModelStore.log(c,"npu_models_verified catalog_sha256="+BuildConfig.NPU_MANIFEST_SHA256);
        progress.update("NPU INT4 模型已就绪；仍需保留基础 GGUF 供 tokenizer / codec 使用");
    }
}
