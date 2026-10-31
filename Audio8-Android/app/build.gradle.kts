plugins { id("com.android.application") }
val npuCatalog = providers.environmentVariable("AUDIO8_NPU_MANIFEST_SHA256").orElse("").get()
require(npuCatalog.isEmpty() || npuCatalog.matches(Regex("[0-9a-f]{64}")))
android {
    namespace = "com.example.audio8tts"
    compileSdk = 36
    ndkVersion = "27.0.12077973"
    defaultConfig {
        applicationId = "com.example.audio8tts"
        minSdk = 33
        targetSdk = 36
        versionCode = 45
        versionName = "4.9"
        buildConfigField("String", "NPU_MANIFEST_SHA256", "\"$npuCatalog\"")
        buildConfigField("String", "NPU_RELEASE_BASE", "\"https://github.com/123hurray/MiniCPM-V-Apps/releases/download/audio8-sm8550-int4-v1\"")
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake {
            cppFlags += listOf("-std=c++17", "-O3")
            arguments += "-DANDROID_STL=c++_shared"
            targets += listOf("audio8_jni", "audiocpp")
        } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    buildTypes { release { isMinifyEnabled = false } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    buildFeatures { buildConfig = true }
    packaging { jniLibs {
        useLegacyPackaging = true
        pickFirsts += listOf("**/libonnxruntime.so", "**/libc++_shared.so")
        excludes += listOf("**/libQnnGpu.so", "**/libQnnDsp*.so", "**/libQnnHtpV68*.so",
            "**/libQnnHtpV69*.so", "**/libQnnHtpV75*.so", "**/libQnnHtpV79*.so", "**/libQnnHtpV81*.so")
    } }
}
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
    implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.7.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.51.0")
}
