package com.example.audio8tts;
import android.content.Context;
import java.io.*;
final class CodecBackend {
    static String get(Context c){try{String s=ModelStore.text(new File(ModelStore.dir(c),"codec-backend.txt")).trim();if(s.equals("ggml")||s.equals("qnn")||s.equals("litert"))return s;}catch(IOException ignored){}return "litert";}
    static void save(Context c,String backend)throws IOException{if(!backend.equals("litert")&&!backend.equals("ggml")&&!backend.equals("qnn"))throw new IOException("未知 codec 后端");ModelStore.saveText(new File(ModelStore.dir(c),"codec-backend.txt"),backend);}
    static int threads(Context c){try{return Math.max(1,Math.min(8,Integer.parseInt(ModelStore.text(new File(ModelStore.dir(c),"litert-threads.txt")).trim())));}catch(Exception e){return 4;}}
    static void threads(Context c,int n)throws IOException{if(n<1||n>8)throw new IOException("线程数需为1～8");ModelStore.saveText(new File(ModelStore.dir(c),"litert-threads.txt"),""+n);}
    static boolean windows(Context c){return new File(ModelStore.dir(c),"codec-windows.enabled").isFile();}
    static void windows(Context c,boolean value)throws IOException{File f=new File(ModelStore.dir(c),"codec-windows.enabled");if(value)ModelStore.saveText(f,"enabled");else java.nio.file.Files.deleteIfExists(f.toPath());}
}
