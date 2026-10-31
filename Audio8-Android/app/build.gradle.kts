plugins { id("com.android.application") }
android {
    namespace = "com.example.audio8tts"
    compileSdk = 36
    ndkVersion = "27.0.12077973"
    defaultConfig {
        applicationId = "com.example.audio8tts"
        minSdk = 33
        targetSdk = 36
        versionCode = 37
        versionName = "4.1"
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
    packaging { jniLibs { useLegacyPackaging = false } }
}
