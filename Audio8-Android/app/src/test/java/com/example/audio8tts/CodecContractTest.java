package com.example.audio8tts;
import java.nio.file.*;
import java.util.Arrays;
public class CodecContractTest {
    static void check(boolean ok){if(!ok)throw new AssertionError();}
    static void rejects(Runnable run){try{run.run();}catch(IllegalArgumentException e){return;}throw new AssertionError("accepted invalid input");}
    public static void main(String[] args)throws Exception{
        int[] codes=new int[10*150];for(int q=0;q<10;q++)for(int t=0;t<150;t++)codes[q*150+t]=q*80+t;
        int[] win=CodecContract.window(codes,150,100);
        for(int q=0;q<10;q++){check(win[q*128]==q*80+100);check(win[q*128+49]==q*80+149);check(win[q*128+50]==0);}
        Path p=Files.createTempFile("audio8-fixture",".a8c");
        try{Files.write(p,CodecContract.encode(codes,150));check(Arrays.equals(codes,CodecContract.read(p)));Files.write(p,new byte[11]);try{CodecContract.read(p);throw new AssertionError();}catch(java.io.IOException expected){}}
        finally{Files.delete(p);}
        rejects(()->CodecContract.validate(new int[11],1));
        codes[150]=1024;rejects(()->CodecContract.validate(codes,150));
        rejects(()->CodecContract.rms(new float[]{Float.NaN}));
        check(Math.abs(CodecContract.correlation(new float[]{1,2,3},new float[]{2,4,6})-1)<1e-9);
        System.out.println("Codec contract tests passed");
    }
}
