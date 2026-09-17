package com.example.minicpm_v_demo

import android.app.Activity
import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.os.PowerManager
import android.os.Process
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor
import java.util.zip.ZipFile

/**
 * Process-wide native runtime bootstrap and mobile performance policy.
 *
 * Loading the selected CPU implementation here is important: both the chat
 * and VoxCPM2 entry points depend on libggml-cpu, so the optimised SONAME must
 * win before either JNI bridge loads its dependency chain.
 */
object NativeRuntime {
    private const val TAG = "NativeRuntime"
    private const val PREFS = "native_runtime"
    private const val KEY_BACKEND = "backend"
    private const val MAX_DIAGNOSTIC_LOG_BYTES = 4L * 1024L * 1024L

    enum class BackendMode(val storedValue: String, val nativeValue: Int) {
        AUTO("auto", 0),
        CPU("cpu", 1),
        GPU("gpu", 2),
        HEXAGON("hexagon", 3),
    }

    private val lock = Any()
    @Volatile private var loaded = false
    @Volatile private var loadError: String? = null
    @Volatile private var thermalStatus = PowerManager.THERMAL_STATUS_NONE
    @Volatile private var configuredThreads = 4
    @Volatile private var selectedVariant = "baseline"
    @Volatile private var diagnosticLogFile: File? = null
    @Volatile private var liveDiagnosticPath: String? = null
    private var thermalListener: PowerManager.OnThermalStatusChangedListener? = null

    private external fun nativeConfigureRuntime(
        threads: Int,
        performanceCpuIds: IntArray,
        backendMode: Int,
    )
    private external fun nativeInitializeDiagnostics(logPath: String, publicLogFd: Int)
    private external fun nativeRuntimeDiagnostics(): String

    fun initialize(context: Context) {
        synchronized(lock) {
            if (!loaded) {
                val diagnosticFile = prepareDiagnosticFile(context.applicationContext)
                appendDiagnosticEvent(
                    diagnosticFile,
                    "process start sdk=${Build.VERSION.SDK_INT} soc=${deviceSocName()} " +
                        "hardware=${Build.HARDWARE}",
                )
                val variant = CpuFeatures.bestGgmlCpuVariant()
                selectedVariant = variant ?: "baseline"
                try {
                    if (variant != null) {
                        try {
                            System.loadLibrary("ggml-cpu-$variant")
                            Log.i(TAG, "Pre-loaded optimised ggml-cpu variant: $variant")
                        } catch (error: UnsatisfiedLinkError) {
                            selectedVariant = "baseline (v86 unavailable)"
                            Log.w(TAG, "Optimised CPU backend unavailable; using baseline", error)
                        }
                    }
                    System.loadLibrary("minicpm_v_demo")
                    val liveLog = createLiveDiagnosticFile(context.applicationContext)
                    liveDiagnosticPath = liveLog.second
                    nativeInitializeDiagnostics(diagnosticFile.absolutePath, liveLog.first)
                    loaded = true
                } catch (error: UnsatisfiedLinkError) {
                    loadError = error.message
                    Log.e(TAG, "Unable to load native runtime", error)
                    throw error
                }
            }
            configure(context)
            registerThermalListener(context.applicationContext)
        }
    }

    fun prepareForInference(context: Context) {
        initialize(context.applicationContext)
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_MORE_FAVORABLE)
        } catch (error: RuntimeException) {
            Log.w(TAG, "Unable to raise inference thread priority", error)
        }
        configure(context.applicationContext)
    }

    fun backendMode(context: Context): BackendMode {
        val preferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val stored = preferences.getString(KEY_BACKEND, BackendMode.CPU.storedValue)
        val requested = BackendMode.entries.firstOrNull { it.storedValue == stored } ?: BackendMode.CPU
        if (requested == BackendMode.AUTO || requested == BackendMode.HEXAGON) {
            Log.w(TAG, "Migrating unavailable backend preference ${requested.storedValue} to CPU")
            preferences.edit().putString(KEY_BACKEND, BackendMode.CPU.storedValue).apply()
            return BackendMode.CPU
        }
        return requested
    }

    fun setBackendMode(context: Context, mode: BackendMode) {
        val effective = if (mode == BackendMode.GPU) BackendMode.GPU else BackendMode.CPU
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_BACKEND, effective.storedValue)
            .apply()
        appendDiagnosticEvent(
            prepareDiagnosticFile(context.applicationContext),
            "backend preference requested=${mode.storedValue} effective=${effective.storedValue}",
        )
        if (loaded) configure(context.applicationContext)
    }

    fun enableSustainedPerformanceMode(activity: Activity) {
        val power = activity.getSystemService(PowerManager::class.java) ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && power.isSustainedPerformanceModeSupported) {
            activity.window.setSustainedPerformanceMode(true)
            Log.i(TAG, "Sustained performance mode enabled")
        }
    }

    fun diagnostics(context: Context): String {
        val profile = CpuFeatures.deviceProfile()
        val libraries = packagedLibraries(context)
        val native = if (loaded) {
            runCatching { nativeRuntimeDiagnostics() }.getOrElse { "native error: ${it.message}" }
        } else {
            "native runtime not loaded"
        }
        return buildString {
            appendLine("SoC: ${profile.soc}")
            appendLine("Hardware: ${Build.HARDWARE} / ${Build.BOARD}")
            appendLine("ABI: ${Build.SUPPORTED_ABIS.joinToString()}")
            appendLine("CPU: ${profile.coreCount} cores; performance cores=${profile.performanceCpuIds.joinToString()}")
            appendLine("Max frequencies: ${profile.maxFrequenciesKHz.joinToString { "${it / 1000}MHz" }}")
            appendLine("Features: dotprod=${CpuFeatures.hasDotprod}, fp16=${CpuFeatures.hasFp16}, i8mm=${CpuFeatures.hasI8mm}, bf16=${CpuFeatures.hasBf16}, sve2=${CpuFeatures.hasSve2}")
            appendLine("CPU library: $selectedVariant")
            appendLine("Threads: $configuredThreads (thermal=${thermalName(thermalStatus)})")
            appendLine("Preference: ${backendMode(context).storedValue}")
            appendLine("Packaged backends: CPU=true, Vulkan=${"libggml-vulkan.so" in libraries}, OpenCL=${"libggml-opencl.so" in libraries}, Hexagon=${"libggml-hexagon.so" in libraries}")
            appendLine("Diagnostic log: ${prepareDiagnosticFile(context).absolutePath}")
            liveDiagnosticPath?.let { appendLine("Live public log: $it") }
            previousExitSummary(context)?.let { appendLine("Previous exit: $it") }
            append("Native: $native")
            loadError?.let { append("\nLoad error: $it") }
            TtsEngine.lastPerformanceSummary()?.let { append("\nLast TTS: $it") }
        }
    }

    fun shortSummary(context: Context): String {
        val profile = CpuFeatures.deviceProfile()
        val policy = if (backendMode(context) == BackendMode.GPU) "GPU-SAFE" else "CPU"
        return "$policy / ${profile.recommendedThreads}T"
    }

    fun exportDiagnostics(context: Context): String {
        val name = "minicpm-diagnostics-${
            SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        }.txt"
        val current = prepareDiagnosticFile(context.applicationContext)
        val previous = File(current.parentFile, "runtime.previous.log")
        val header = buildString {
            appendLine("MiniCPM Android diagnostics")
            appendLine("Exported: ${Date()}")
            appendLine()
            appendLine(diagnostics(context))
            appendLine()
            appendLine("===== runtime.previous.log =====")
        }.toByteArray()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, name)
                put(MediaStore.Downloads.MIME_TYPE, "text/plain")
                put(
                    MediaStore.Downloads.RELATIVE_PATH,
                    "${Environment.DIRECTORY_DOWNLOADS}/MiniCPMLogs",
                )
            }
            val resolver = context.contentResolver
            val uri = checkNotNull(
                resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values),
            ) { "无法创建日志文件" }
            try {
                checkNotNull(resolver.openOutputStream(uri, "w")).use { output ->
                    output.write(header)
                    if (previous.isFile) previous.inputStream().use { it.copyTo(output) }
                    output.write("\n===== runtime.log =====\n".toByteArray())
                    if (current.isFile) current.inputStream().use { it.copyTo(output) }
                }
            } catch (error: Exception) {
                resolver.delete(uri, null, null)
                throw error
            }
            return "Download/MiniCPMLogs/$name"
        }

        val targetDirectory = checkNotNull(
            context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),
        ) { "外部存储不可用" }
        val target = File(targetDirectory, name)
        target.outputStream().use { output ->
            output.write(header)
            if (previous.isFile) previous.inputStream().use { it.copyTo(output) }
            output.write("\n===== runtime.log =====\n".toByteArray())
            if (current.isFile) current.inputStream().use { it.copyTo(output) }
        }
        return target.absolutePath
    }

    private fun configure(context: Context) {
        if (!loaded) return
        val profile = CpuFeatures.deviceProfile()
        configuredThreads = when {
            thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> maxOf(2, profile.recommendedThreads - 2)
            thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE -> maxOf(2, profile.recommendedThreads - 1)
            else -> profile.recommendedThreads
        }
        nativeConfigureRuntime(
            configuredThreads,
            profile.performanceCpuIds.toIntArray(),
            backendMode(context).nativeValue,
        )
    }

    private fun prepareDiagnosticFile(context: Context): File {
        diagnosticLogFile?.let { return it }
        val directory = File(checkNotNull(context.getExternalFilesDir(null)), "diagnostics")
        directory.mkdirs()
        val current = File(directory, "runtime.log")
        if (current.length() > MAX_DIAGNOSTIC_LOG_BYTES) {
            val previous = File(directory, "runtime.previous.log")
            if (previous.exists()) previous.delete()
            current.renameTo(previous)
        }
        diagnosticLogFile = current
        return current
    }

    private fun createLiveDiagnosticFile(context: Context): Pair<Int, String> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return -1 to "unavailable"
        return runCatching {
            val name = "minicpm-live-${
                SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
            }.txt"
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, name)
                put(MediaStore.Downloads.MIME_TYPE, "text/plain")
                put(
                    MediaStore.Downloads.RELATIVE_PATH,
                    "${Environment.DIRECTORY_DOWNLOADS}/MiniCPMLogs",
                )
            }
            val uri = checkNotNull(
                context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values),
            ) { "无法创建实时日志" }
            val descriptor = checkNotNull(
                context.contentResolver.openFileDescriptor(uri, "wa"),
            ) { "无法打开实时日志" }
            descriptor.detachFd() to "Download/MiniCPMLogs/$name"
        }.onFailure { Log.w(TAG, "Unable to create public live diagnostic log", it) }
            .getOrElse { -1 to "unavailable" }
    }

    private fun packagedLibraries(context: Context): Set<String> {
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val result = nativeDir.listFiles()?.mapTo(mutableSetOf()) { it.name } ?: mutableSetOf()
        val apkPaths = buildList {
            add(context.applicationInfo.sourceDir)
            context.applicationInfo.splitSourceDirs?.let(::addAll)
        }
        apkPaths.forEach { path ->
            runCatching {
                ZipFile(path).use { zip ->
                    val entries = zip.entries()
                    while (entries.hasMoreElements()) {
                        val entry = entries.nextElement().name
                        if (entry.startsWith("lib/arm64-v8a/") && entry.endsWith(".so")) {
                            result += entry.substringAfterLast('/')
                        }
                    }
                }
            }.onFailure { Log.w(TAG, "Unable to inspect packaged libraries in $path", it) }
        }
        return result
    }

    private fun deviceSocName(): String = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        Build.SOC_MODEL
    } else {
        Build.HARDWARE
    }

    private fun appendDiagnosticEvent(file: File, message: String) {
        runCatching {
            file.appendText("[${System.currentTimeMillis()}] [K] NativeRuntime: $message\n")
        }.onFailure { Log.w(TAG, "Unable to append diagnostic event", it) }
    }

    private fun previousExitSummary(context: Context): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return null
        return runCatching {
            val activityManager = context.getSystemService(ActivityManager::class.java)
            activityManager?.getHistoricalProcessExitReasons(null, 0, 5)
                ?.firstOrNull()
                ?.let { info ->
                    "reason=${exitReasonName(info.reason)}(${info.reason}), status=${info.status}, " +
                        "importance=${info.importance}, timestamp=${Date(info.timestamp)}, " +
                        "description=${info.description ?: "-"}"
                }
        }.getOrNull()
    }

    private fun exitReasonName(reason: Int): String = when (reason) {
        ApplicationExitInfo.REASON_CRASH -> "crash"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "native_crash"
        ApplicationExitInfo.REASON_ANR -> "anr"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "low_memory"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "resource_usage"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "user_requested"
        else -> "other"
    }

    private fun registerThermalListener(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || thermalListener != null) return
        val power = context.getSystemService(PowerManager::class.java) ?: return
        thermalStatus = power.currentThermalStatus
        val listener = PowerManager.OnThermalStatusChangedListener { status ->
            thermalStatus = status
            Log.i(TAG, "Thermal state changed to ${thermalName(status)}")
            configure(context)
        }
        thermalListener = listener
        power.addThermalStatusListener(Executor { command -> command.run() }, listener)
    }

    private fun thermalName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "none"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
        else -> status.toString()
    }
}
