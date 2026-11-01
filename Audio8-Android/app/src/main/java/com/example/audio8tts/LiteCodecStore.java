package com.example.audio8tts;

import android.content.Context;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;

final class LiteCodecStore {
    static final String REVISION="56fcb491ebf03eb3a1a66615fedf6744015dd4c7";
    private static final String[] NAMES={"codec_decoder_int8_T128.tflite"};
    private static final long[] SIZES={132429760L};
    private static final String[] HASHES={"8f7bb7d518bf77e33c3a729aeae55d96b79c592b465b60bd9f78f0143619d80b"};
    static File dir(Context c){File f=new File(ModelStore.dir(c),"codec-litert-56fcb49");f.mkdirs();return f;}
    static boolean ready(Context c){
        try {
            if(!ModelStore.text(new File(dir(c),"verified.txt")).equals(REVISION))return false;
            for(int i=0;i<NAMES.length;i++)if(new File(dir(c),NAMES[i]).length()!=SIZES[i])return false;
            return true;
        }catch(IOException e){return false;}
    }
    private static String hash(File f)throws Exception{
        var digest=MessageDigest.getInstance("SHA-256");
        try(var in=new FileInputStream(f)){byte[] b=new byte[262144];int n;while((n=in.read(b))!=-1)digest.update(b,0,n);}
        var s=new StringBuilder();for(byte b:digest.digest())s.append(String.format("%02x",b&255));return s.toString();
    }
    static void download(Context c,ModelStore.Progress progress)throws Exception{
        for(int i=0;i<NAMES.length;i++){
            File target=new File(dir(c),NAMES[i]),part=new File(dir(c),NAMES[i]+".part");
            if(target.length()==SIZES[i] && hash(target).equals(HASHES[i]))continue;
            long offset=part.length();if(offset>SIZES[i])throw new IOException("codec 临时文件超长");
            if(offset<SIZES[i]){
                String url=ModelStore.endpoint(c)+"/litert-community/Audio8-TTS-Preview-0.6b/resolve/"+REVISION+"/"+NAMES[i];
                var conn=(HttpURLConnection)new URL(url).openConnection();
                conn.setConnectTimeout(30000);conn.setReadTimeout(60000);conn.setRequestProperty("Accept-Encoding","identity");
                if(offset>0)conn.setRequestProperty("Range","bytes="+offset+"-");
                try{
                    int status=conn.getResponseCode();if(status==200)offset=0;
                    else if(status!=206 || !DownloadEndpoint.matchesRange(conn.getHeaderField("Content-Range"),offset,SIZES[i]))throw new IOException("codec 下载响应不匹配 HTTP "+status);
                    try(var in=conn.getInputStream();var out=new FileOutputStream(part,offset>0)){
                        byte[] b=new byte[262144];int n;long total=offset,reported=0;
                        while((n=in.read(b))!=-1){total+=n;if(total>SIZES[i])throw new IOException("codec 文件超长");out.write(b,0,n);long now=System.currentTimeMillis();if(now-reported>500){progress.update(NAMES[i]+" "+total*100/SIZES[i]+"%");reported=now;}}
                        out.getFD().sync();
                    }
                }finally{conn.disconnect();}
            }
            if(part.length()!=SIZES[i])throw new IOException("codec 文件不完整，重试续传");
            progress.update("校验 codec SHA-256");
            if(!hash(part).equals(HASHES[i])){Files.delete(part.toPath());throw new IOException("codec SHA-256 不匹配");}
            Files.move(part.toPath(),target.toPath(),StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
        }
        ModelStore.saveText(new File(dir(c),"verified.txt"),REVISION);progress.update("codec 模型已校验；LiteRT INT8 codec 可在下次请求中加载");
    }
}
