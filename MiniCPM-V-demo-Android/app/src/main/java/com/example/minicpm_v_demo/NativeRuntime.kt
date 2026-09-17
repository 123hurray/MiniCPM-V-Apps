package com.example.minicpm_v_demo

import android.app.Activity
import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.Process
import android.util.Log
import java.io.File
import java.util.concurrent.Executor

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
    private var thermalListener: PowerManager.OnThermalStatusChangedListener? = null

    private external fun nativeConfigureRuntime(
        threads: Int,
        performanceCpuIds: IntArray,
        backendMode: Int,
    )
    private external fun nativeRuntimeDiagnostics(): String

    fun initialize(context: Context) {
        synchronized(lock) {
            if (!loaded) {
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
        val stored = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_BACKEND, BackendMode.AUTO.storedValue)
        return BackendMode.entries.firstOrNull { it.storedValue == stored } ?: BackendMode.AUTO
    }

    fun setBackendMode(context: Context, mode: BackendMode) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_BACKEND, mode.storedValue)
            .apply()
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
        val nativeDir = File(context.applicationInfo.nativeLibraryDir)
        val libraries = nativeDir.listFiles()?.map { it.name }?.toSet().orEmpty()
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
            append("Native: $native")
            loadError?.let { append("\nLoad error: $it") }
            TtsEngine.lastPerformanceSummary()?.let { append("\nLast TTS: $it") }
        }
    }

    fun shortSummary(context: Context): String {
        val profile = CpuFeatures.deviceProfile()
        val policy = when (backendMode(context)) {
            BackendMode.GPU -> "GPU-4L"
            BackendMode.AUTO -> "AUTO→CPU"
            BackendMode.HEXAGON -> "HEXAGON→CPU"
            BackendMode.CPU -> "CPU"
        }
        return "$policy / ${profile.recommendedThreads}T"
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
