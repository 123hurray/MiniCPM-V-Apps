import java.net.URLClassLoader;
import java.nio.file.Files;
import javax.tools.ToolProvider;
public final class TestCodecContract {
    public static void main(String[] args) throws Exception {
        var output=Files.createTempDirectory("audio8-codec-contract");
        if(ToolProvider.getSystemJavaCompiler().run(null,System.out,System.err,"-d",output.toString(),
            "Audio8-Android/app/src/main/java/com/example/audio8tts/CodecContract.java",
            "Audio8-Android/app/src/test/java/com/example/audio8tts/CodecContractTest.java")!=0)throw new AssertionError("compile failed");
        try(var loader=new URLClassLoader(new java.net.URL[]{output.toUri().toURL()})){
            loader.loadClass("com.example.audio8tts.CodecContractTest").getMethod("main",String[].class).invoke(null,(Object)new String[0]);
        }
    }
}
