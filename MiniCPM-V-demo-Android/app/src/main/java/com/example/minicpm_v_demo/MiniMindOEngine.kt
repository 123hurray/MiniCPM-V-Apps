package com.example.minicpm_v_demo

import android.content.Context
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.Closeable
import java.util.PriorityQueue
import kotlin.coroutines.coroutineContext
import kotlin.math.exp
import kotlin.random.Random
import kotlinx.coroutines.ensureActive

/** Speech-native MiniMind-O generation pipeline for ExecuTorch/XNNPACK. */
class MiniMindOEngine(private val context: Context) : Closeable {
    companion object {
        private const val CONTEXT = 1024
        private const val AUDIO_PAD = 2049
        private const val TEXT_EOS = 2
        private const val ENTER_TOKEN = 201
        private const val MAX_NEW_TOKENS = 256
        private const val MIMI_WINDOW_FRAMES = 6
        private const val MIMI_OVERLAP_FRAMES = 2
        private const val MIMI_SAMPLES_PER_FRAME = 1920
    }

    private var main: Module? = null
    private var senseVoice: Module? = null
    private var mimi: Module? = null
    private var tokenizer: MiniMindOTokenizer? = null

    fun load(onStatus: (String) -> Unit = {}) {
        check(MiniMindOModelStore.isComplete(context)) { "MiniMind-O 模型尚未下载完整" }
        if (main != null) return
        onStatus("加载 MiniMind-O 主干…")
        main = Module.load(
            MiniMindOModelStore.file(context, "minimind-o-main-int8.pte").absolutePath
        ).also { it.loadMethod("forward") }
        onStatus("加载 SenseVoice…")
        senseVoice = Module.load(
            MiniMindOModelStore.file(context, "minimind-o-sensevoice-int8.pte").absolutePath
        ).also { it.loadMethod("forward") }
        onStatus("加载 Mimi 声码器…")
        mimi = Module.load(
            MiniMindOModelStore.file(context, "minimind-o-mimi-int8.pte").absolutePath
        ).also { it.loadMethod("forward") }
        tokenizer = MiniMindOTokenizer(
            MiniMindOModelStore.file(context, "tokenizer.json")
        )
        onStatus("模型已就绪")
    }

    suspend fun respond(
        pcm16k: ShortArray,
        onText: (String) -> Unit,
        onAudio: (FloatArray, Int) -> Unit,
        onStatus: (String) -> Unit,
    ) {
        val mainModule = checkNotNull(main) { "模型未加载" }
        val senseModule = checkNotNull(senseVoice) { "SenseVoice 未加载" }
        val mimiModule = checkNotNull(mimi) { "Mimi 未加载" }
        val tok = checkNotNull(tokenizer) { "Tokenizer 未加载" }

        coroutineContext.ensureActive()
        onStatus("提取语音特征…")
        val frontend = MiniMindOAudioFrontend.compute(pcm16k)
        val senseOutputs = senseModule.forward(
            EValue.from(
                Tensor.fromBlob(
                    frontend.values,
                    longArrayOf(1, MiniMindOAudioFrontend.MAX_LFR_FRAMES.toLong(), 560),
                )
            ),
            EValue.from(Tensor.fromBlob(longArrayOf(frontend.frameCount.toLong()), longArrayOf(1))),
        )
        val projected = senseOutputs.last().toTensor().dataAsFloatArray
        check(projected.size >= frontend.frameCount * 768) {
            "SenseVoice 输出形状异常：${senseOutputs.last().toTensor().shape().contentToString()}"
        }

        val audioPrompt = "<|audio_pad|>".repeat(frontend.frameCount)
        val prompt = "<|im_start|>user\n$audioPrompt<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        val promptIds = tok.encode(prompt)
        check(promptIds.size < CONTEXT - MAX_NEW_TOKENS) { "本轮语音过长" }

        val prefillFeatures = FloatArray(promptIds.size * 768)
        val prefillMask = FloatArray(promptIds.size)
        var audioFrame = 0
        for (index in promptIds.indices) {
            if (promptIds[index] == 16 && audioFrame < frontend.frameCount) {
                System.arraycopy(projected, audioFrame * 768, prefillFeatures, index * 768, 768)
                prefillMask[index] = 1f
                audioFrame++
            }
        }
        val padAudio = LongArray(8 * promptIds.size) { AUDIO_PAD.toLong() }
        onStatus("MiniMind-O 正在回答…")
        var outputs = forwardMain(
            mainModule,
            LongArray(promptIds.size) { promptIds[it].toLong() },
            padAudio,
            prefillFeatures,
            prefillMask,
            LongArray(promptIds.size) { it.toLong() },
        )

        val generatedText = ArrayList<Int>()
        val audioCodes = Array(8) { ArrayList<Int>() }
        val audioStops = IntArray(8) { -1 }
        val playableFrames = ArrayList<IntArray>()
        var lastDecodeFrameCount = 0
        var textFinished = false
        var firstFinished = true
        var nextAudioInput = IntArray(8) { AUDIO_PAD }
        var position = promptIds.size
        var emittedAudio = false

        for (step in 0 until MAX_NEW_TOKENS) {
            coroutineContext.ensureActive()
            var textToken = if (textFinished) {
                if (firstFinished) ENTER_TOKEN.also { firstFinished = false } else 0
            } else {
                sample(outputs.first, 50, 0.90f, 0.75f)
            }
            val audioStep = step - 1
            for (layer in 0 until 8) {
                val code = if (audioStep < layer) {
                    AUDIO_PAD
                } else {
                    sample(outputs.second[layer], 50, 1f, 0.20f)
                }
                audioCodes[layer] += code
                if (audioStops[layer] < 0 && code >= 2048) audioStops[layer] = audioCodes[layer].lastIndex
            }

            if (!textFinished) {
                if (textToken == TEXT_EOS) {
                    textFinished = true
                } else {
                    generatedText += textToken
                    onText(cleanText(tok.decode(generatedText)))
                }
            }

            if (audioStep >= 7) {
                val frame = IntArray(8) { layer ->
                    audioCodes[layer][step - 7 + layer]
                }
                val active = (0 until 8).count { layer ->
                    audioStops[layer] < 0 || step - 7 + layer < audioStops[layer]
                }
                if (active == 8) {
                    // The first delayed frame contains AUDIO_PAD in the early
                    // codebooks. Upstream maps those placeholders to code 0
                    // before Mimi decoding; dropping the whole frame makes
                    // short replies end before a decodable window exists.
                    playableFrames += IntArray(8) { layer ->
                        frame[layer].takeIf { it in 0 until 2048 } ?: 0
                    }
                }
            }
            val shouldDecode = if (lastDecodeFrameCount == 0) {
                playableFrames.size >= 6
            } else {
                playableFrames.size - lastDecodeFrameCount >= 4
            }
            if (shouldDecode) {
                val window = playableFrames.takeLast(6)
                val pcm = decodeMimi(mimiModule, window)
                val drop = if (lastDecodeFrameCount == 0) 0 else
                    MIMI_OVERLAP_FRAMES * MIMI_SAMPLES_PER_FRAME
                if (!emittedAudio) {
                    emittedAudio = true
                    onStatus("MiniMind-O 正在播放语音…")
                }
                onAudio(pcm, drop)
                lastDecodeFrameCount = playableFrames.size
            }

            if (textFinished && audioStops.all { it >= 0 }) break
            nextAudioInput = IntArray(8) { AUDIO_PAD }
            for (layer in 0 until minOf(audioStep + 1, 8)) {
                nextAudioInput[layer] = audioCodes[layer].last()
            }
            outputs = forwardMain(
                mainModule,
                longArrayOf(textToken.toLong()),
                LongArray(8) { nextAudioInput[it].toLong() },
                FloatArray(768),
                floatArrayOf(0f),
                longArrayOf(position.toLong()),
            )
            position++
        }

        // Fixed-shape mobile Mimi accepts six frames. Flush the 1..5 frames
        // left at end of generation by padding the causal tail, then expose
        // only samples belonging to real new frames. Without this path short
        // answers produce text but never reach AudioTrack.
        if (playableFrames.size > lastDecodeFrameCount) {
            val newFrames = playableFrames.size - lastDecodeFrameCount
            val overlap = if (lastDecodeFrameCount > 0) {
                minOf(MIMI_OVERLAP_FRAMES, lastDecodeFrameCount)
            } else {
                0
            }
            val start = lastDecodeFrameCount - overlap
            val window = playableFrames.subList(start, playableFrames.size).toMutableList()
            while (window.size < MIMI_WINDOW_FRAMES) window += window.last()
            val pcm = decodeMimi(mimiModule, window.take(MIMI_WINDOW_FRAMES))
            val drop = overlap * MIMI_SAMPLES_PER_FRAME
            val end = minOf(pcm.size, drop + newFrames * MIMI_SAMPLES_PER_FRAME)
            if (!emittedAudio) onStatus("MiniMind-O 正在播放语音…")
            onAudio(pcm.copyOfRange(0, end), drop)
            emittedAudio = true
        }
        if (!emittedAudio) onStatus("本轮未生成可播放语音")
        onStatus("请继续说话")
    }

    private data class MainOutputs(
        val first: FloatArray,
        val second: Array<FloatArray>,
    )

    private fun forwardMain(
        module: Module,
        textIds: LongArray,
        audioIds: LongArray,
        audioFeatures: FloatArray,
        audioMask: FloatArray,
        positions: LongArray,
    ): MainOutputs {
        val sequence = textIds.size
        val mask = FloatArray(sequence * CONTEXT)
        for (row in 0 until sequence) {
            val position = positions[row].toInt()
            for (column in position + 1 until CONTEXT) mask[row * CONTEXT + column] = -1.0e9f
        }
        val result = module.forward(
            EValue.from(Tensor.fromBlob(textIds, longArrayOf(1, sequence.toLong()))),
            EValue.from(Tensor.fromBlob(audioIds, longArrayOf(1, 8, sequence.toLong()))),
            EValue.from(Tensor.fromBlob(audioFeatures, longArrayOf(1, sequence.toLong(), 768))),
            EValue.from(Tensor.fromBlob(audioMask, longArrayOf(1, sequence.toLong()))),
            EValue.from(Tensor.fromBlob(positions, longArrayOf(sequence.toLong()))),
            EValue.from(Tensor.fromBlob(mask, longArrayOf(sequence.toLong(), CONTEXT.toLong()))),
        )
        val text = result[result.size - 2].toTensor().dataAsFloatArray
        val audioFlat = result.last().toTensor().dataAsFloatArray
        check(audioFlat.size == 8 * 2112) { "Talker 输出大小异常：${audioFlat.size}" }
        return MainOutputs(text, Array(8) { layer ->
            audioFlat.copyOfRange(layer * 2112, (layer + 1) * 2112)
        })
    }

    private fun decodeMimi(module: Module, frames: List<IntArray>): FloatArray {
        val flattened = LongArray(8 * 6)
        for (layer in 0 until 8) for (frame in 0 until 6) {
            flattened[layer * 6 + frame] = frames[frame][layer].toLong()
        }
        return module.forward(
            EValue.from(Tensor.fromBlob(flattened, longArrayOf(1, 8, 6)))
        ).last().toTensor().dataAsFloatArray
    }

    private fun sample(
        logits: FloatArray,
        topK: Int,
        topP: Float,
        temperature: Float,
    ): Int {
        data class Entry(val score: Float, val id: Int)
        val heap = PriorityQueue<Entry>(compareBy { it.score })
        for (id in logits.indices) {
            val entry = Entry(logits[id] / temperature, id)
            if (heap.size < topK) heap += entry
            else if (entry.score > heap.peek().score) {
                heap.poll()
                heap += entry
            }
        }
        val choices = heap.toList().sortedByDescending { it.score }
        val max = choices.first().score
        val weights = DoubleArray(choices.size) { exp((choices[it].score - max).toDouble()) }
        val total = weights.sum()
        var cumulative = 0.0
        var keep = weights.size
        if (topP < 1f) {
            for (i in weights.indices) {
                cumulative += weights[i] / total
                if (cumulative >= topP) {
                    keep = i + 1
                    break
                }
            }
        }
        val keptTotal = (0 until keep).sumOf { weights[it] }
        var target = Random.nextDouble() * keptTotal
        for (i in 0 until keep) {
            target -= weights[i]
            if (target <= 0.0) return choices[i].id
        }
        return choices[keep - 1].id
    }

    private fun cleanText(value: String): String = value
        .replace("<think>\n\n</think>\n\n", "")
        .replace("<|im_end|>", "")
        .trim()

    override fun close() {
        main?.close()
        senseVoice?.close()
        mimi?.close()
        main = null
        senseVoice = null
        mimi = null
        tokenizer = null
        System.gc()
    }
}
