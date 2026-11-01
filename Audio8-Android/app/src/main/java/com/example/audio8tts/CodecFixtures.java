package com.example.audio8tts;
import android.content.Context;
import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;
import java.util.zip.*;
final class CodecFixtures {
    static File dir(Context c){File f=new File(ModelStore.dir(c),"codec-fixtures");f.mkdirs();return f;}
    static boolean enabled(Context c){return new File(ModelStore.dir(c),"capture-codes.enabled").exists();}
    static void enabled(Context c,boolean enabled)throws IOException{File f=new File(ModelStore.dir(c),"capture-codes.enabled");if(enabled)ModelStore.saveText(f,"enabled");else Files.deleteIfExists(f.toPath());}
    static void capture(Context c,int[] codes,int frames){if(!enabled(c))return;try{
        byte[] data=CodecContract.encode(codes,frames);byte[] hash=MessageDigest.getInstance("SHA-256").digest(data);StringBuilder name=new StringBuilder();for(byte b:hash)name.append(String.format("%02x",b&255));
        Path f=new File(dir(c),name+".a8c").toPath();if(!Files.exists(f)){Path tmp=new File(dir(c),"pending.tmp").toPath();Files.write(tmp,data);Files.move(tmp,f,StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);}Files.setLastModifiedTime(f,java.nio.file.attribute.FileTime.fromMillis(System.currentTimeMillis()));
        File[] files=dir(c).listFiles((d,n)->n.endsWith(".a8c"));if(files!=null&&files.length>64){Arrays.sort(files,Comparator.comparingLong(File::lastModified));for(int i=0;i<files.length-64;i++)Files.delete(files[i].toPath());}
    }catch(Exception e){ModelStore.log(c,"capture_codes_error "+e.getClass().getSimpleName());}}
    static File latest(Context c)throws IOException{File[] a=dir(c).listFiles((d,n)->n.endsWith(".a8c"));if(a==null||a.length==0)throw new IOException("请先开启记录并试听一次，再运行基准");Arrays.sort(a,Comparator.comparingLong(File::lastModified).reversed());return a[0];}
    static void export(Context c,OutputStream stream)throws IOException{try(var zip=new ZipOutputStream(stream)){
        zip.putNextEntry(new ZipEntry("README.txt"));zip.write("真实音频码可以还原语音。格式 A8C1：12字节头（magic、LE int32码本数、帧数），随后码本优先LE int32数据。没有保存输入文本。QDQ校准至少需要16份不同语音码。".getBytes(java.nio.charset.StandardCharsets.UTF_8));zip.closeEntry();
        File[] files=dir(c).listFiles((d,n)->n.endsWith(".a8c"));if(files!=null)for(File f:files){zip.putNextEntry(new ZipEntry(f.getName()));Files.copy(f.toPath(),zip);zip.closeEntry();}
        File report=new File(ModelStore.dir(c),"codec-benchmark.json");if(report.isFile()){zip.putNextEntry(new ZipEntry("codec-benchmark.json"));Files.copy(report.toPath(),zip);zip.closeEntry();}
    }}
}
