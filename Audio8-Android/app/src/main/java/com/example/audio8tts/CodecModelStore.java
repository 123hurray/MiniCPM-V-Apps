package com.example.audio8tts;

import android.content.Context;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;

final class CodecModelStore {
    static final String REVISION="317c12d4e0da83847b594fcf8bd74bf2c76615ec";
    private static final String[] NAMES={"codec-128-part00.onnx","codec-128-part01.onnx","codec-128-part02.onnx","codec-128-part03.onnx","codec-128-part04.onnx","codec-128-part05.onnx","codec-128-part06.onnx","codec-128-part07.onnx","codec-128-part08.onnx","codec-128-part09.onnx","codec-128-part10.onnx","codec-128-part11.onnx","codec-128-part12.onnx","codec-128-part13.onnx","codec-256-part00.onnx","codec-256-part01.onnx","codec-256-part02.onnx","codec-256-part03.onnx","codec-256-part04.onnx","codec-256-part05.onnx","codec-256-part06.onnx","codec-256-part07.onnx","codec-256-part08.onnx","codec-256-part09.onnx","codec-256-part10.onnx","codec-256-part11.onnx","codec-256-part12.onnx","codec-256-part13.onnx","codec_decoder_fp16.onnx.data"};
    private static final long[] SIZES={297316L,8870L,5212L,5212L,13754L,9351L,9356L,12456L,7055L,7055L,10331L,5892L,5900L,2611L,297316L,8870L,5212L,5212L,13754L,9351L,9356L,12456L,7055L,7055L,10331L,5892L,5900L,2611L,260741440L};
    private static final String[] HASHES={"8be6338fc8b9c339588221cb62712623b50eeb132a8b6f151575b3aa49e5ffa5","e059fedb9067827420fd8cfa7ce540f504db6cc754a04fbd9eb699afe1d19ce4","d1c7f03db09ffc4cf3c11f3c5698e5e7e28e816a6a47b21443ed45a07cacdb7a","eb2c0a8c25a6112c0e1fc75ef788d3844b1c4b911e8142c33353aef29251efd4","9b850c1d86363e8efec9114a4c32545d0dfb2e927845ab18fd39cc4a2cdc04a4","e226955db65798929e71ebb4f654b099bf8f81c00408fe4819bd6271dfe54411","ecdeec7a819fc696c37b1a01b9400627de7c7c5798f7e80dcf9417743900885c","607a02aeacc55d8512dd46c7da4f535aa7f966ba12e18bc92269c95029131a1a","22daf353bb980465d0ed9a0cde948b7245eefce073f13b818550cad8f9327128","333ae3886f528f490ac4a8562046cfc24b0116975ad9d38f039d8172ad6f4aa2","d52b609ecb16572d28f9ecdbd5a5faf760ac85b5f00adafac0f67650b133e9b0","09bfc1c0bc12f981541a5c97c7869fb0d03ee538ba062791feb02cb812c66cf9","f4e53e322e0049443524529c07fffa12d93a58cfc46f77af66359bab5cb23902","e64bf59f28ec39381cdb81928bb022a2ab71d732bb30c666c9b32c069ed4d434","ee680e69fdb5bcaa5b6905486ab8fe11bb520a9f06d0b491c512cc6a34f686b5","101dcf08fad945b552db0ac4bdb4469b6c0840915ccd77d1c90d36dc588a9461","7483ced8eaa20920be77715d86ecc1eafb78ba3eafb75121a001d418bb95b45c","c366cb690daee54bb28d683b2c1d6dae3832e9d4b2a337286a981c00ec41493c","e3c8285aad0669b522a9cfe83af2de2cf1416de034ecc6e7b86477a3a2ff92ce","cf1406a5ee8d08b8a5203f1165c846f001c4e1ed58382253622af9144b354223","6a1d98cc24adca30a167464d022557aa023b16a48c047e03f9386fd0b43ac878","99fdeeb66c521e99c420436bc2f90cf857adf115dd28d59f50f0669e49793b7e","38b37e9dc83981b6870c112223f6aa491309a63e01495988ee4752adfd7d625d","7ae0a15dfa90cdaf5cf2ad1e1ddf9c9d3343c171163ded11851cd6bb061aee14","7be336619c95441af6e97b2c5e156f81d3d6d65a4139f523110e4db23833c916","fcb71e5ab5050d3f551ab2fcd154c443c9b2e664a03c4211cd40d3e649eec677","7dbe97c3ee8fb7d050758cf06a36b457726f4167873fa171e6848af176e21ff3","785c86d1b61fbb189d7ac5972ca5cdd4df389117409c71e93bc1c1308ef71a89","b9bc968a84f41b86a9a3d19bc6c42b00b5de576512a87501d0c66ff7f896c954"};
    static File dir(Context c){File f=new File(ModelStore.dir(c),"codec-preview01-317c12d");f.mkdirs();return f;}
    static boolean enabled(Context c){return CodecBackend.get(c).equals("qnn");}
    static void enable(Context c,boolean value)throws IOException{CodecBackend.save(c,value?"qnn":"litert");}
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
                String url=NAMES[i].endsWith(".data") ? ModelStore.endpoint(c)+"/Edge0/audio8-TTS-0.1B-ONNX-INT8/resolve/"+REVISION+"/"+NAMES[i] : "https://github.com/123hurray/MiniCPM-V-Apps/releases/download/audio8-android-v5.1/"+NAMES[i];
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
