package com.example.audio8tts;

import android.content.Context;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;

final class CodecModelStore {
    static final String REVISION="317c12d4e0da83847b594fcf8bd74bf2c76615ec";
    private static final String[] NAMES={"codec-static-128.onnx","codec-static-256.onnx","codec_decoder_fp16.onnx.data"};
    private static final long[] SIZES={506620L,506620L,260741440L};
    private static final String[] HASHES={"470953b43ef44c5507d63dcca7082233c13867af556772355fb6b77b2bef4305","9474a6e7cef058b6858170a1e0bb64d51e22e041cf0c7ad1b9688152fdcec27e","b9bc968a84f41b86a9a3d19bc6c42b00b5de576512a87501d0c66ff7f896c954"};
    static File dir(Context c){File f=new File(ModelStore.dir(c),"codec-preview01-317c12d");f.mkdirs();return f;}
    static boolean enabled(Context c){try{return ModelStore.text(new File(ModelStore.dir(c),"codec-npu.txt")).equals("true");}catch(IOException e){return false;}}
    static void enable(Context c,boolean value)throws IOException{ModelStore.saveText(new File(ModelStore.dir(c),"codec-npu.txt"),Boolean.toString(value));}
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
                String url=NAMES[i].endsWith(".data") ? ModelStore.endpoint(c)+"/Edge0/audio8-TTS-0.1B-ONNX-INT8/resolve/"+REVISION+"/"+NAMES[i] : "https://github.com/123hurray/MiniCPM-V-Apps/releases/download/audio8-android-v5.0/"+NAMES[i];
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
        ModelStore.saveText(new File(dir(c),"verified.txt"),REVISION);progress.update("codec 模型已校验；NPU 能否编译仍需真机验证");
    }
}
