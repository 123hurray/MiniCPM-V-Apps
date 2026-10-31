import java.net.URLClassLoader;
import java.nio.file.Files;
import javax.tools.ToolProvider;
public final class TestDownloadEndpoint {
    public static void main(String[] args) throws Exception {
        var output = Files.createTempDirectory("audio8-download-tests");
        int result = ToolProvider.getSystemJavaCompiler().run(null, System.out, System.err, "-d", output.toString(),
            "Audio8-Android/app/src/main/java/com/example/audio8tts/DownloadEndpoint.java",
            "Audio8-Android/app/src/test/java/com/example/audio8tts/DownloadEndpointTest.java");
        if (result != 0) throw new IllegalStateException("Test compilation failed");
        try (var loader = new URLClassLoader(new java.net.URL[]{output.toUri().toURL()})) {
            loader.loadClass("com.example.audio8tts.DownloadEndpointTest").getMethod("main", String[].class).invoke(null, (Object)new String[0]);
        }
    }
}
