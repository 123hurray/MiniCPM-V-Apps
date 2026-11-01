package com.example.audio8tts;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
final class CodecContract {
    static final int BOOKS=10, CAPACITY=128, HOP=2048;
    static void validate(int[] codes,int frames){
        if(frames<1||frames>4096||codes.length!=BOOKS*frames)throw new IllegalArgumentException("音频码形状不符");
        for(int q=0;q<BOOKS;q++)for(int t=0;t<frames;t++)if(codes[q*frames+t]<0||codes[q*frames+t]>=(q==0?4096:1024))throw new IllegalArgumentException("音频码超出码本范围 q="+q);
    }
    static int[] window(int[] codes,int frames,int start){
        if(start<0||start>=frames)throw new IllegalArgumentException("无效窗口");
        int[] out=new int[BOOKS*CAPACITY];int n=Math.min(CAPACITY,frames-start);
        for(int q=0;q<BOOKS;q++)System.arraycopy(codes,q*frames+start,out,q*CAPACITY,n);return out;
    }
    static byte[] encode(int[] codes,int frames){validate(codes,frames);var b=ByteBuffer.allocate(12+codes.length*4).order(ByteOrder.LITTLE_ENDIAN);b.put(new byte[]{65,56,67,49});b.putInt(BOOKS);b.putInt(frames);for(int v:codes)b.putInt(v);return b.array();}
    static int[] read(Path path)throws IOException{
        if(Files.size(path)>12+10*4096*4)throw new IOException("音频码文件过大");
        var b=ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
        if(b.remaining()<12||b.getInt()!=0x31433841||b.getInt()!=BOOKS)throw new IOException("音频码文件格式无效");
        int n=b.getInt();if(n<1||n>4096||b.remaining()!=n*BOOKS*4)throw new IOException("音频码文件长度不符");
        int[] codes=new int[n*BOOKS];for(int i=0;i<codes.length;i++)codes[i]=b.getInt();validate(codes,n);return codes;
    }
    static double rms(float[] x){double s=0;for(float v:x){if(!Float.isFinite(v))throw new IllegalArgumentException("非有限 PCM");s+=(double)v*v;}return Math.sqrt(s/Math.max(1,x.length));}
    static double correlation(float[] a,float[] b){if(a.length!=b.length)throw new IllegalArgumentException("PCM 长度不同");double sa=0,sb=0,aa=0,bb=0,ab=0;for(int i=0;i<a.length;i++){sa+=a[i];sb+=b[i];aa+=(double)a[i]*a[i];bb+=(double)b[i]*b[i];ab+=(double)a[i]*b[i];}double n=a.length;return (ab-sa*sb/n)/Math.sqrt(Math.max(1e-30,(aa-sa*sa/n)*(bb-sb*sb/n)));}
}
