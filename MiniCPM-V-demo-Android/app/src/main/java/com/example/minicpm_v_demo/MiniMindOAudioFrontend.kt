package com.example.minicpm_v_demo

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin

/** Kaldi-compatible 80-bin fbank followed by SenseVoice LFR(7, 6). */
object MiniMindOAudioFrontend {
    const val SAMPLE_RATE = 16_000
    const val MAX_LFR_FRAMES = 256
    private const val FRAME_SIZE = 400
    private const val FRAME_SHIFT = 160
    private const val FFT_SIZE = 512
    private const val MEL_BINS = 80
    private const val LFR_M = 7
    private const val LFR_N = 6

    data class Features(val values: FloatArray, val frameCount: Int)

    private val window = FloatArray(FRAME_SIZE) { index ->
        (0.54 - 0.46 * cos(2.0 * PI * index / (FRAME_SIZE - 1))).toFloat()
    }
    private val melFilters = makeMelFilters()

    fun compute(pcm: ShortArray): Features {
        if (pcm.size < FRAME_SIZE) return Features(FloatArray(MAX_LFR_FRAMES * 560), 1)
        val frameCount = 1 + (pcm.size - FRAME_SIZE) / FRAME_SHIFT
        val fbanks = Array(frameCount) { FloatArray(MEL_BINS) }
        val real = FloatArray(FFT_SIZE)
        val imag = FloatArray(FFT_SIZE)
        val power = FloatArray(FFT_SIZE / 2 + 1)

        for (frame in 0 until frameCount) {
            java.util.Arrays.fill(real, 0f)
            java.util.Arrays.fill(imag, 0f)
            val offset = frame * FRAME_SHIFT
            var mean = 0f
            for (i in 0 until FRAME_SIZE) mean += pcm[offset + i].toFloat()
            mean /= FRAME_SIZE
            var previous = pcm[offset].toFloat() - mean
            for (i in 0 until FRAME_SIZE) {
                val sample = pcm[offset + i].toFloat() - mean
                // Kaldi replicates the first sample for pre-emphasis, so the
                // first value is (1 - 0.97) * sample rather than unfiltered.
                val emphasized = if (i == 0) 0.03f * sample else sample - 0.97f * previous
                real[i] = emphasized * window[i]
                previous = sample
            }
            fft(real, imag)
            for (bin in power.indices) {
                power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin]
            }
            for (mel in 0 until MEL_BINS) {
                var energy = 0f
                val filter = melFilters[mel]
                for (bin in power.indices) energy += power[bin] * filter[bin]
                fbanks[frame][mel] = ln(max(energy, 1.1920929e-7f).toDouble()).toFloat()
            }
        }

        val lfrCount = minOf((frameCount + LFR_N - 1) / LFR_N, MAX_LFR_FRAMES)
        val result = FloatArray(MAX_LFR_FRAMES * MEL_BINS * LFR_M)
        for (outFrame in 0 until lfrCount) {
            val center = outFrame * LFR_N
            for (stack in 0 until LFR_M) {
                val source = (center + stack - (LFR_M - 1) / 2).coerceIn(0, frameCount - 1)
                System.arraycopy(
                    fbanks[source],
                    0,
                    result,
                    (outFrame * LFR_M + stack) * MEL_BINS,
                    MEL_BINS,
                )
            }
        }
        return Features(result, lfrCount.coerceAtLeast(1))
    }

    private fun makeMelFilters(): Array<FloatArray> {
        fun mel(hz: Double) = 1127.0 * ln(1.0 + hz / 700.0)
        val low = mel(20.0)
        val high = mel(SAMPLE_RATE / 2.0)
        val points = DoubleArray(MEL_BINS + 2) { i ->
            low + (high - low) * i / (MEL_BINS + 1)
        }
        val bins = DoubleArray(FFT_SIZE / 2 + 1) { bin ->
            mel(bin * SAMPLE_RATE.toDouble() / FFT_SIZE)
        }
        return Array(MEL_BINS) { index ->
            FloatArray(bins.size) { bin ->
                when {
                    bins[bin] <= points[index] || bins[bin] >= points[index + 2] -> 0f
                    bins[bin] < points[index + 1] ->
                        ((bins[bin] - points[index]) / (points[index + 1] - points[index])).toFloat()
                    else -> ((points[index + 2] - bins[bin]) /
                        (points[index + 2] - points[index + 1])).toFloat()
                }
            }
        }
    }

    private fun fft(real: FloatArray, imag: FloatArray) {
        var j = 0
        for (i in 1 until FFT_SIZE) {
            var bit = FFT_SIZE shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
                val ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
        }
        var length = 2
        while (length <= FFT_SIZE) {
            val angle = -2.0 * PI / length
            val stepReal = cos(angle).toFloat()
            val stepImag = sin(angle).toFloat()
            var start = 0
            while (start < FFT_SIZE) {
                var wr = 1f
                var wi = 0f
                for (k in 0 until length / 2) {
                    val even = start + k
                    val odd = even + length / 2
                    val tr = wr * real[odd] - wi * imag[odd]
                    val ti = wr * imag[odd] + wi * real[odd]
                    real[odd] = real[even] - tr
                    imag[odd] = imag[even] - ti
                    real[even] += tr
                    imag[even] += ti
                    val nextWr = wr * stepReal - wi * stepImag
                    wi = wr * stepImag + wi * stepReal
                    wr = nextWr
                }
                start += length
            }
            length = length shl 1
        }
    }
}
