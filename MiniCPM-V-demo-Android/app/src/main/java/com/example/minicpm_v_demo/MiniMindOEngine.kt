package com.example.minicpm_v_demo

import android.content.Context
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.PriorityQueue
import kotlin.coroutines.coroutineContext
import kotlin.math.exp
import kotlin.math.sqrt
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
        private const val HISTORY_TURNS = 4
        private const val AUDIO_REPETITION_PENALTY = 1.05f
        private const val VOICE_SAMPLE_COUNT_16K = 64_000
        private const val VOICE_SAMPLE_COUNT_24K = 96_000
        private const val VOICE_MAGIC = 0x4d4d4f56
    }

    private data class ConversationTurn(
        val projectedAudio: FloatArray,
        val audioFrames: Int,
        val assistantText: String,
    )

    private var main: Module? = null
    private var senseVoice: Module? = null
    private var mimi: Module? = null
    private var tokenizer: MiniMindOTokenizer? = null
    private val history = ArrayDeque<ConversationTurn>()
    @Volatile private var voiceCodes: LongArray? = null
    @Volatile private var voiceFrames = 0

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
        loadSavedVoice()
        onStatus("模型已就绪")
    }

    fun hasVoiceClone(): Boolean = voiceCodes != null && voiceFrames > 0

    fun createVoiceClone(
        pcm16k: ShortArray,
        onStatus: (String) -> Unit = {},
    ): Int {
        check(pcm16k.size >= VOICE_SAMPLE_COUNT_16K) { "参考语音不足 4 秒" }
        var energy = 0.0
        var peak = 0
        for (index in 0 until VOICE_SAMPLE_COUNT_16K) {
            val value = pcm16k[index].toInt()
            energy += value.toDouble() * value
            peak = maxOf(peak, kotlin.math.abs(value))
        }
        val rms = sqrt(energy / VOICE_SAMPLE_COUNT_16K)
        check(rms >= 260.0) { "参考语音太轻，请靠近麦克风重录" }
        check(peak < 32_600) { "参考语音有爆音，请离麦克风稍远后重录" }

        onStatus("正在提取克隆音色…")
        val waveform = FloatArray(VOICE_SAMPLE_COUNT_24K)
        for (index in waveform.indices) {
            val source = index * 2f / 3f
            val left = source.toInt().coerceAtMost(VOICE_SAMPLE_COUNT_16K - 1)
            val right = (left + 1).coerceAtMost(VOICE_SAMPLE_COUNT_16K - 1)
            val fraction = source - left
            waveform[index] = (
                pcm16k[left] * (1f - fraction) + pcm16k[right] * fraction
            ) / 32768f
        }
        val encoder = Module.load(
            MiniMindOModelStore.file(context, "minimind-o-mimi-encoder-int8.pte").absolutePath
        )
        val (shape, raw) = try {
            encoder.loadMethod("forward")
            val output = encoder.forward(
                EValue.from(
                    Tensor.fromBlob(waveform, longArrayOf(1, 1, waveform.size.toLong()))
                )
            ).last().toTensor()
            output.shape() to output.dataAsLongArray
        } finally {
            encoder.close()
        }
        check(shape.size == 3 && shape[0] == 1L && shape[1] >= 8L) {
            "Mimi encoder 输出形状异常：${shape.contentToString()}"
        }
        val frames = shape[2].toInt()
        val codes = LongArray(8 * frames)
        for (layer in 0 until 8) {
            System.arraycopy(raw, layer * frames, codes, layer * frames, frames)
        }
        saveVoice(codes, frames)
        voiceCodes = codes
        voiceFrames = frames
        return frames
    }

    fun deleteVoiceClone() {
        voiceCodes = null
        voiceFrames = 0
        voiceFile().delete()
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

        val previousTurns = synchronized(history) { history.toMutableList() }
        fun composePrompt(turns: List<ConversationTurn>): String = buildString {
            for (turn in turns) {
                append("<|im_start|>user\n")
                append("<|audio_pad|>".repeat(turn.audioFrames))
                append("<|im_end|>\n<|im_start|>assistant\n")
                append(turn.assistantText)
                append("<|im_end|>\n")
            }
            append("<|im_start|>user\n")
            append("<|audio_pad|>".repeat(frontend.frameCount))
            append("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        }
        var retainedTurns = previousTurns
        var prompt = composePrompt(retainedTurns)
        var promptIds = tok.encode(prompt)
        while (retainedTurns.isNotEmpty() && promptIds.size >= CONTEXT - MAX_NEW_TOKENS) {
            retainedTurns = retainedTurns.drop(1).toMutableList()
            prompt = composePrompt(retainedTurns)
            promptIds = tok.encode(prompt)
        }
        check(promptIds.size < CONTEXT - MAX_NEW_TOKENS) { "本轮语音过长" }

        val prefillFeatures = FloatArray(promptIds.size * 768)
        val prefillMask = FloatArray(promptIds.size)
        val audioSegments = retainedTurns.map { it.projectedAudio to it.audioFrames } +
            listOf(projected to frontend.frameCount)
        var segmentIndex = 0
        var audioFrame = 0
        for (index in promptIds.indices) {
            while (segmentIndex < audioSegments.size && audioFrame >= audioSegments[segmentIndex].second) {
                segmentIndex++
                audioFrame = 0
            }
            if (promptIds[index] == 16 && segmentIndex < audioSegments.size) {
                val (segment, frames) = audioSegments[segmentIndex]
                check(audioFrame < frames)
                System.arraycopy(segment, audioFrame * 768, prefillFeatures, index * 768, 768)
                prefillMask[index] = 1f
                audioFrame++
            }
        }
        val padAudio = LongArray(8 * promptIds.size) { AUDIO_PAD.toLong() }
        val activeVoice = voiceCodes
        val activeVoiceFrames = voiceFrames
        if (activeVoice != null && activeVoiceFrames > 0) {
            val usedFrames = minOf(activeVoiceFrames, promptIds.size)
            val destinationStart = promptIds.size - usedFrames
            val sourceStart = activeVoiceFrames - usedFrames
            for (layer in 0 until 8) for (frame in 0 until usedFrames) {
                padAudio[layer * promptIds.size + destinationStart + frame] =
                    activeVoice[layer * activeVoiceFrames + sourceStart + frame]
            }
        }
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
                    sample(
                        outputs.second[layer],
                        50,
                        1f,
                        0.20f,
                        audioCodes[layer].takeLast(3),
                        AUDIO_REPETITION_PENALTY,
                    )
                }
                audioCodes[layer] += code
                // AUDIO_PAD is used before a delayed codebook starts. It is
                // not an end token. Upstream records a stop only for a code
                // that was actually sampled after this layer became active.
                if (audioStep >= layer && audioStops[layer] < 0 && code >= 2048) {
                    audioStops[layer] = audioCodes[layer].lastIndex
                }
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
        val finalText = cleanText(tok.decode(generatedText))
        if (finalText.isNotBlank()) {
            val completed = ConversationTurn(
                projected.copyOf(frontend.frameCount * 768),
                frontend.frameCount,
                finalText,
            )
            synchronized(history) {
                history += completed
                while (history.size > HISTORY_TURNS) history.removeFirst()
            }
        }
        onStatus(
            if (!emittedAudio) "请继续说话（本轮无语音，有效帧 ${playableFrames.size}）"
            else "请继续说话（上一轮语音 ${playableFrames.size} 帧）"
        )
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
        recentTokens: List<Int> = emptyList(),
        repetitionPenalty: Float = 1f,
    ): Int {
        data class Entry(val score: Float, val id: Int)
        val recent = recentTokens.toHashSet()
        val heap = PriorityQueue<Entry>(compareBy { it.score })
        for (id in logits.indices) {
            var score = logits[id] / temperature
            if (repetitionPenalty != 1f && id in recent) {
                score = if (score > 0f) score / repetitionPenalty else score * repetitionPenalty
            }
            val entry = Entry(score, id)
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

    @Synchronized
    fun clearHistory() {
        history.clear()
    }

    private fun voiceFile(): File = File(MiniMindOModelStore.directory(context), "voice-clone.bin")

    private fun saveVoice(codes: LongArray, frames: Int) {
        DataOutputStream(voiceFile().outputStream().buffered()).use { output ->
            output.writeInt(VOICE_MAGIC)
            output.writeInt(frames)
            for (code in codes) output.writeShort(code.toInt())
        }
    }

    private fun loadSavedVoice() {
        val file = voiceFile()
        if (!file.isFile) return
        try {
            DataInputStream(file.inputStream().buffered()).use { input ->
                check(input.readInt() == VOICE_MAGIC)
                val frames = input.readInt()
                check(frames in 1..128)
                val codes = LongArray(8 * frames) { input.readUnsignedShort().toLong() }
                check(codes.all { it in 0 until 2048 })
                voiceCodes = codes
                voiceFrames = frames
            }
        } catch (_: Throwable) {
            file.delete()
            voiceCodes = null
            voiceFrames = 0
        }
    }

    override fun close() {
        main?.close()
        senseVoice?.close()
        mimi?.close()
        main = null
        senseVoice = null
        mimi = null
        tokenizer = null
        clearHistory()
        System.gc()
    }
}
