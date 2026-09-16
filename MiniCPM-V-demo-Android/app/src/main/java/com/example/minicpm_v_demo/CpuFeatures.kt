package com.example.minicpm_v_demo

import android.util.Log
import android.os.Build
import java.io.File

object CpuFeatures {
    private const val TAG = "CpuFeatures"

    private val features: Set<String> by lazy { readFeatures() }

    val hasI8mm: Boolean get() = "i8mm" in features
    val hasBf16: Boolean get() = "bf16" in features
    val hasSve2: Boolean get() = "sve2" in features
    val hasDotprod: Boolean get() = "asimddp" in features
    val hasFp16: Boolean get() = "fphp" in features

    data class DeviceProfile(
        val soc: String,
        val coreCount: Int,
        val maxFrequenciesKHz: List<Int>,
        val performanceCpuIds: List<Int>,
        val recommendedThreads: Int,
    )

    private val profile: DeviceProfile by lazy { readDeviceProfile() }

    fun deviceProfile(): DeviceProfile = profile

    /**
     * Returns the best available ggml-cpu library name for the current CPU.
     * The returned string can be passed to [System.load] after prepending
     * the native library directory and "lib" prefix.
     */
    fun bestGgmlCpuVariant(): String? {
        if (hasI8mm && hasBf16) return "v86"
        return null
    }

    fun summary(): String = buildString {
        append("CPU features: ${features.joinToString()}\n")
        append("dotprod=$hasDotprod fp16=$hasFp16 i8mm=$hasI8mm bf16=$hasBf16 sve2=$hasSve2\n")
        append("Selected ggml-cpu variant: ${bestGgmlCpuVariant() ?: "baseline"}")
    }

    private fun readFeatures(): Set<String> = try {
        val featureSets = File("/proc/cpuinfo").readLines()
            .filter { it.startsWith("Features") }
            .map {
                it.substringAfter(":").trim().split("\\s+".toRegex()).filter(String::isNotBlank).toSet()
            }
        // Instruction-specific binaries must only use features shared by all
        // visible cores. This is safer than trusting the first core on a
        // heterogeneous mobile SoC.
        featureSets.reduceOrNull { common, next -> common intersect next }
            ?: emptySet<String>().also { Log.w(TAG, "No Features line in /proc/cpuinfo") }
    } catch (e: Exception) {
        Log.e(TAG, "Failed to read /proc/cpuinfo", e)
        emptySet()
    }

    private fun readDeviceProfile(): DeviceProfile {
        val coreCount = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val frequencies = (0 until coreCount).map { cpu ->
            sequenceOf(
                "/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq",
                "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq",
            ).mapNotNull { path ->
                runCatching {
                    File(path).takeIf(File::canRead)?.readText()?.trim()?.toIntOrNull()
                }.getOrNull()
            }
                .firstOrNull() ?: 0
        }
        val highest = frequencies.maxOrNull().orEmptyFrequency()
        val performance = if (highest > 0) {
            frequencies.indices.filter { frequencies[it] >= (highest * 70L / 100L) }
        } else {
            (0 until coreCount).toList().takeLast(minOf(4, coreCount))
        }.ifEmpty { (0 until coreCount).toList() }

        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL.takeUnless { it.isBlank() } ?: Build.HARDWARE
        } else {
            Build.HARDWARE
        }
        val isSnapdragon8Gen2 = listOf(soc, Build.HARDWARE, Build.BOARD)
            .any { value ->
                value.contains("SM8550", ignoreCase = true) ||
                    value.contains("8 Gen 2", ignoreCase = true) ||
                    value.contains("kalama", ignoreCase = true)
            }
        val threads = when {
            isSnapdragon8Gen2 -> minOf(5, performance.size).coerceAtLeast(2)
            performance.size >= 4 -> minOf(4, performance.size)
            else -> minOf(4, coreCount)
        }
        return DeviceProfile(soc, coreCount, frequencies, performance, threads.coerceAtLeast(1))
    }

    private fun Int?.orEmptyFrequency(): Int = this ?: 0
}
