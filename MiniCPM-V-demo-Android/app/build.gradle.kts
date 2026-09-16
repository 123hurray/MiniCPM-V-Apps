plugins {
    alias(libs.plugins.android.application)
    id("org.jetbrains.kotlin.plugin.serialization")
    id("com.chaquo.python")
}

val enableVulkan = providers.gradleProperty("MINICPMV_ENABLE_VULKAN")
    .map(String::toBoolean)
    .orElse(true)
val openClSdkRoot = providers.environmentVariable("OPENCL_SDK_ROOT")
val enableOpenCl = providers.gradleProperty("MINICPMV_ENABLE_OPENCL")
    .map(String::toBoolean)
    .orElse(openClSdkRoot.isPresent)
val hexagonSdkRoot = providers.environmentVariable("HEXAGON_SDK_ROOT")
val hexagonToolsRoot = providers.environmentVariable("HEXAGON_TOOLS_ROOT")
val enableHexagon = providers.gradleProperty("MINICPMV_ENABLE_HEXAGON")
    .map(String::toBoolean)
    .orElse(hexagonSdkRoot.isPresent && hexagonToolsRoot.isPresent)

android {
    namespace = "com.example.minicpm_v_demo"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }
    ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.example.minicpm_v_demo"
        // Koog 1.2 targets JVM 17 and its stable dependency graph uses
        // MethodHandle APIs available from Android 8.0 onward.
        // This sets Android 8.0 (Oreo) as the minimum supported release.
        minSdk = 26
        targetSdk = 36
        versionCode = 28
        versionName = "3.6"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters.add("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DBUILD_SHARED_LIBS=ON"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DLLAMA_BUILD_TOOLS=ON"
                arguments += "-DLLAMA_OPENSSL=OFF"

                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_LLAMAFILE=ON"
                // Keep the fallback usable on every arm64 Android device.
                // SM8550-class devices load the separately packaged v8.6
                // implementation before this baseline SONAME resolves.
                arguments += "-DGGML_CPU_ARM_ARCH=armv8-a"
                arguments += "-DGGML_VULKAN=${if (enableVulkan.get()) "ON" else "OFF"}"
                arguments += "-DGGML_OPENCL=${if (enableOpenCl.get()) "ON" else "OFF"}"
                arguments += "-DGGML_HEXAGON=${if (enableHexagon.get()) "ON" else "OFF"}"
                arguments += "-DLLAMA_CURL=OFF"
                if (enableVulkan.get()) {
                    // Android cross-compilation does not search host package
                    // roots, even though shader generation runs on the host.
                    arguments += "-DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers"
                    arguments += "-DVulkan_INCLUDE_DIR=/tmp/minicpm-vulkan-headers/include"
                }
                if (enableOpenCl.get()) {
                    arguments += "-DCMAKE_PREFIX_PATH=${openClSdkRoot.get()}"
                    arguments += "-DGGML_OPENCL_USE_ADRENO_KERNELS=ON"
                }
                if (enableHexagon.get()) {
                    arguments += "-DHEXAGON_SDK_ROOT=${hexagonSdkRoot.get()}"
                    arguments += "-DHEXAGON_TOOLS_ROOT=${hexagonToolsRoot.get()}"
                    arguments += "-DPREBUILT_LIB_DIR=android_aarch64"
                }
            }
        }
    }

    // Release signing config. Credentials live in ~/.gradle/gradle.properties
    // (outside any git repo, only readable by your local mac account).
    // If those properties aren't set the release build still works but produces
    // an unsigned apk - useful e.g. on CI without secrets.
    signingConfigs {
        create("release") {
            val keystorePath = providers.gradleProperty("MINICPMV_KEYSTORE").orNull
            if (!keystorePath.isNullOrBlank()) {
                storeFile = file(keystorePath)
                storePassword = providers.gradleProperty("MINICPMV_KEYSTORE_PASSWORD").orNull
                keyAlias = providers.gradleProperty("MINICPMV_KEY_ALIAS").orNull
                keyPassword = providers.gradleProperty("MINICPMV_KEY_PASSWORD").orNull
            }
        }
    }

    buildTypes {
        release {
            // Keep ProGuard/R8 disabled: the app calls native JNI symbols and
            // shrinking the Kotlin side has no measurable benefit here, while
            // an over-aggressive shrinker is the most common cause of crashes
            // in apps with lots of JNI bindings.
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Only attach the signing config when the keystore actually exists,
            // so contributors without the secret can still run :assembleRelease
            // (it'll produce an unsigned apk in that case).
            val signingCfg = signingConfigs.getByName("release")
            if (signingCfg.storeFile?.exists() == true) {
                signingConfig = signingCfg
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    androidResources {
        noCompress.add("gguf")
        noCompress.add("bin")
        noCompress.add("pte")
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.activity.ktx)
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")

    // Mature Kotlin-native agent runtime. A custom LLMClient below bridges
    // Koog's tool loop to the already-loaded on-device MiniCPM text model.
    implementation("ai.koog:koog-agents:1.2.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.10.0")

    // MiniMind-O speech-native runtime.  XNNPACK executes the separately
    // downloaded, dynamic-int8 .pte models; no model weight is bundled in APK.
    implementation("org.pytorch:executorch-android:1.4.0")

    // Markdown rendering for AI streaming responses (headings, bold, lists, code, etc.)
    implementation("io.noties.markwon:core:4.6.2")
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}

// ---------------------------------------------------------------------------
// Dynamic CPU dispatch: build an additional libggml-cpu.so optimised for
// ARMv8.6-a (i8mm + bf16) and package it alongside the baseline build.
// At runtime the Kotlin CpuFeatures helper detects hardware capabilities
// and pre-loads the best variant before the rest of the native chain.
// ---------------------------------------------------------------------------

fun runCmd(vararg args: String) {
    val proc = ProcessBuilder(*args).inheritIO().start()
    val rc = proc.waitFor()
    if (rc != 0) error("Command failed (rc=$rc): ${args.joinToString(" ")}")
}

val sdkRoot: String = System.getenv("ANDROID_HOME")
    ?: file("../local.properties").takeIf { it.exists() }?.readLines()
        ?.firstOrNull { it.startsWith("sdk.dir=") }?.substringAfter("=")
    ?: error("Cannot locate Android SDK — set ANDROID_HOME or local.properties")

tasks.register("buildGgmlCpu_v86") {
    group = "native"
    description = "Build libggml-cpu optimised for armv8.6-a+i8mm+bf16"

    val destSo = file("src/main/jniLibs/arm64-v8a/libggml-cpu-v86.so")
    outputs.file(destSo)

    doLast {
        val cmake = "$sdkRoot/cmake/4.1.2/bin/cmake"
        val toolchain = "$sdkRoot/ndk/27.0.12077973/build/cmake/android.toolchain.cmake"
        val bd = File(project.layout.buildDirectory.asFile.get(), "v86-cmake/arm64-v8a")
        bd.mkdirs()

        runCmd(
            cmake,
            "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
            "-DANDROID_ABI=arm64-v8a",
            "-DANDROID_PLATFORM=android-24",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DLLAMA_BUILD_COMMON=ON",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DGGML_NATIVE=OFF",
            "-DGGML_LLAMAFILE=ON",
            "-DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16",
            "-S", file("src/main/cpp").absolutePath,
            "-B", bd.absolutePath,
        )

        runCmd(
            cmake,
            "--build", bd.absolutePath,
            "--target", "ggml-cpu",
            "-j", Runtime.getRuntime().availableProcessors().toString(),
        )

        val builtSo = fileTree(bd).matching { include("**/libggml-cpu.so") }.singleFile
        destSo.parentFile.mkdirs()
        builtSo.copyTo(destSo, overwrite = true)
        logger.lifecycle("Copied v86 ggml-cpu -> ${destSo.absolutePath} (${destSo.length() / 1024}K)")
    }
}

afterEvaluate {
    listOf("Debug", "Release").forEach { buildType ->
        tasks.findByName("merge${buildType}JniLibFolders")?.dependsOn("buildGgmlCpu_v86")
    }
}
