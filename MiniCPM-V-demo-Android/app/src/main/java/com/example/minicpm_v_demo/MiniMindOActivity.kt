package com.example.minicpm_v_demo

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.sqrt

class MiniMindOActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var scroll: ScrollView
    private lateinit var progress: LinearProgressIndicator
    private lateinit var downloadButton: MaterialButton
    private lateinit var conversationButton: MaterialButton
    private lateinit var voiceButton: MaterialButton
    private lateinit var playInputButton: MaterialButton

    private var engine: MiniMindOEngine? = null
    private var recorder: AudioRecord? = null
    private var recordThread: Thread? = null
    @Volatile private var sessionActive = false
    @Volatile private var generating = false
    @Volatile private var playbackExpectedUntilNanos = 0L
    private var generationJob: Job? = null
    private var echoCanceler: AcousticEchoCanceler? = null
    private var audioTrack: AudioTrack? = null
    private var inputPlayer: MediaPlayer? = null
    private var lastInputFile: File? = null
    private var assistantPrefixLength = 0
    private enum class MicAction { CONVERSATION, VOICE_CLONE }
    private var pendingMicAction = MicAction.CONVERSATION

    private val micPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            if (pendingMicAction == MicAction.VOICE_CLONE) recordVoiceClone()
            else startConversation()
        }
        else Toast.makeText(this, R.string.minimind_o_permission, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NativeRuntime.enableSustainedPerformanceMode(this)
        setContentView(R.layout.activity_minimind_o)
        findViewById<Toolbar>(R.id.toolbar).setNavigationOnClickListener { finish() }
        status = findViewById(R.id.tv_status)
        transcript = findViewById(R.id.tv_transcript)
        scroll = findViewById(R.id.scroll_transcript)
        progress = findViewById(R.id.progress_model)
        downloadButton = findViewById(R.id.btn_download_model)
        conversationButton = findViewById(R.id.btn_conversation)
        voiceButton = findViewById(R.id.btn_voice_clone)
        playInputButton = findViewById(R.id.btn_play_last_input)

        downloadButton.setOnClickListener {
            if (MiniMindOModelStore.isComplete(this)) confirmDeleteModel() else startDownload()
        }
        conversationButton.setOnClickListener {
            if (sessionActive) stopConversation() else ensureMicAndStart()
        }
        voiceButton.setOnClickListener { ensureMicAndClone() }
        voiceButton.setOnLongClickListener {
            if (MiniMindOModelStore.hasVoiceClone(this)) confirmDeleteVoice()
            true
        }
        playInputButton.setOnClickListener { playLastModelInput() }
        playInputButton.setOnLongClickListener {
            confirmDeleteInputRecordings()
            true
        }
        restoreLastInputRecording()
        observeDownloads()
        refreshModelState()
    }

    private fun observeDownloads() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                ModelDownloadController.status.collect { state ->
                    when (state) {
                        is ModelDownloadController.Status.Running -> {
                            progress.visibility = View.VISIBLE
                            status.text = state.message.ifBlank { getString(R.string.minimind_o_downloading) }
                            downloadButton.isEnabled = false
                            conversationButton.isEnabled = false
                            voiceButton.isEnabled = false
                        }
                        ModelDownloadController.Status.Completed -> {
                            progress.visibility = View.GONE
                            ModelDownloadController.acknowledge()
                            refreshModelState()
                        }
                        is ModelDownloadController.Status.Failed -> {
                            progress.visibility = View.GONE
                            downloadButton.isEnabled = true
                            voiceButton.isEnabled = MiniMindOModelStore.isComplete(this@MiniMindOActivity)
                            status.text = getString(R.string.minimind_o_error, state.message)
                            ModelDownloadController.acknowledge()
                        }
                        ModelDownloadController.Status.Cancelled -> {
                            progress.visibility = View.GONE
                            ModelDownloadController.acknowledge()
                            refreshModelState()
                        }
                        ModelDownloadController.Status.Idle -> Unit
                    }
                }
            }
        }
    }

    private fun refreshModelState() {
        val complete = MiniMindOModelStore.isComplete(this)
        conversationButton.isEnabled = complete
        voiceButton.isEnabled = complete && !sessionActive
        downloadButton.isEnabled = true
        downloadButton.setText(if (complete) R.string.minimind_o_delete else R.string.minimind_o_download)
        status.setText(if (complete) R.string.minimind_o_ready else R.string.minimind_o_checking)
        voiceButton.setText(
            if (MiniMindOModelStore.hasVoiceClone(this)) R.string.minimind_o_voice_rerecord
            else R.string.minimind_o_voice_record
        )
    }

    private fun startDownload() {
        ModelDownloadController.acknowledge()
        progress.visibility = View.VISIBLE
        status.setText(R.string.minimind_o_downloading)
        ModelDownloadService.startMiniMindO(applicationContext)
    }

    private fun confirmDeleteModel() {
        AlertDialog.Builder(this)
            .setTitle(R.string.minimind_o_delete)
            .setMessage("将删除已下载的 MiniMind-O 模型，APK 不受影响。")
            .setPositiveButton(R.string.delete) { _, _ ->
                stopConversation()
                engine?.close()
                engine = null
                MiniMindOModelStore.delete(this)
                refreshModelState()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun ensureMicAndStart() {
        pendingMicAction = MicAction.CONVERSATION
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) startConversation() else micPermission.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun ensureMicAndClone() {
        if (sessionActive) return
        pendingMicAction = MicAction.VOICE_CLONE
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) recordVoiceClone() else micPermission.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun recordVoiceClone() {
        if (!MiniMindOModelStore.isComplete(this) || sessionActive) return
        voiceButton.isEnabled = false
        conversationButton.isEnabled = false
        progress.visibility = View.VISIBLE
        status.setText(R.string.minimind_o_voice_recording)
        lifecycleScope.launch(Dispatchers.IO) {
            var cloneRecorder: AudioRecord? = null
            try {
                val minBuffer = AudioRecord.getMinBufferSize(
                    MiniMindOAudioFrontend.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                )
                cloneRecorder = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    MiniMindOAudioFrontend.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    max(minBuffer, 4096),
                )
                check(cloneRecorder.state == AudioRecord.STATE_INITIALIZED) {
                    "参考语音录音初始化失败"
                }
                val reference = ShortArray(64_000)
                var offset = 0
                cloneRecorder.startRecording()
                while (offset < reference.size) {
                    val count = cloneRecorder.read(reference, offset, reference.size - offset)
                    check(count > 0) { "参考语音录制失败：$count" }
                    offset += count
                }
                cloneRecorder.stop()
                // The encoder is used only while creating the voice. Release
                // the conversation stack first so lower-memory phones never
                // hold both large ExecuTorch programs at once.
                engine?.close()
                val activeEngine = MiniMindOEngine(applicationContext).also { engine = it }
                val frames = activeEngine.createVoiceClone(reference) { message ->
                    runOnUiThread { status.text = message }
                }
                withContext(Dispatchers.Main) {
                    status.text = getString(R.string.minimind_o_voice_ready, frames)
                    voiceButton.setText(R.string.minimind_o_voice_rerecord)
                }
            } catch (error: Throwable) {
                withContext(Dispatchers.Main) {
                    status.text = getString(R.string.minimind_o_error, error.message)
                }
            } finally {
                try { cloneRecorder?.stop() } catch (_: Throwable) {}
                cloneRecorder?.release()
                withContext(Dispatchers.Main) {
                    progress.visibility = View.GONE
                    voiceButton.isEnabled = MiniMindOModelStore.isComplete(this@MiniMindOActivity)
                    conversationButton.isEnabled = MiniMindOModelStore.isComplete(this@MiniMindOActivity)
                }
            }
        }
    }

    private fun confirmDeleteVoice() {
        AlertDialog.Builder(this)
            .setTitle(R.string.minimind_o_voice_delete)
            .setMessage(R.string.minimind_o_voice_delete_message)
            .setPositiveButton(R.string.delete) { _, _ ->
                engine?.deleteVoiceClone()
                MiniMindOModelStore.file(this, "voice-clone.bin").delete()
                voiceButton.setText(R.string.minimind_o_voice_record)
                status.setText(R.string.minimind_o_ready)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun startConversation() {
        if (sessionActive || !MiniMindOModelStore.isComplete(this)) return
        conversationButton.isEnabled = false
        progress.visibility = View.VISIBLE
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // Avoid keeping a GGUF model and the ~460 MB MiniMind-O stack
                // resident at the same time. MainActivity reloads its selected
                // model when this screen is closed.
                val llama = LlamaEngine.getInstance(applicationContext)
                if (llama.state.value is LlamaState.ModelReady) llama.unloadModel()
                val activeEngine = engine ?: MiniMindOEngine(applicationContext).also { engine = it }
                activeEngine.load { message -> runOnUiThread { status.text = message } }
                withContext(Dispatchers.Main) {
                    progress.visibility = View.GONE
                    conversationButton.isEnabled = true
                    conversationButton.setText(R.string.minimind_o_stop)
                    voiceButton.isEnabled = false
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
                beginRecordingLoop()
            } catch (error: Throwable) {
                withContext(Dispatchers.Main) {
                    progress.visibility = View.GONE
                    conversationButton.isEnabled = true
                    status.text = getString(R.string.minimind_o_error, error.message)
                }
            }
        }
    }

    private fun beginRecordingLoop() {
        val minBuffer = AudioRecord.getMinBufferSize(
            MiniMindOAudioFrontend.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val activeRecorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            MiniMindOAudioFrontend.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            max(minBuffer, 4096),
        )
        check(activeRecorder.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord 初始化失败" }
        recorder = activeRecorder
        if (AcousticEchoCanceler.isAvailable()) {
            echoCanceler = AcousticEchoCanceler.create(activeRecorder.audioSessionId)?.apply { enabled = true }
        }
        sessionActive = true
        activeRecorder.startRecording()
        runOnUiThread { status.setText(R.string.minimind_o_listening) }
        recordThread = Thread({ recordingLoop(activeRecorder) }, "MiniMindO-AudioRecord").apply { start() }
    }

    private fun recordingLoop(activeRecorder: AudioRecord) {
        val chunk = ShortArray(1024)
        // Keep a full second before VAD commits. This is deliberately longer
        // than the trigger window so the model receives the beginning of a
        // sentence instead of only the audio after speech detection.
        val ring = ShortArray(MiniMindOAudioFrontend.SAMPLE_RATE)
        var ringPosition = 0
        var ringCount = 0
        val speech = ShortArray(16 * MiniMindOAudioFrontend.SAMPLE_RATE)
        var speechSize = 0
        var voicedSamples = 0
        var silentSamples = 0
        var speaking = false
        var noiseFloor = 180.0

        while (sessionActive) {
            val read = activeRecorder.read(chunk, 0, chunk.size)
            if (read <= 0) continue
            val duringPlayback = System.nanoTime() < playbackExpectedUntilNanos
            var energy = 0.0
            for (i in 0 until read) energy += chunk[i].toDouble() * chunk[i]
            val rms = sqrt(energy / read)
            // AEC quality varies by vendor. During playback use a stricter
            // threshold to reject loudspeaker echo, but keep VAD active so a
            // nearby user can still interrupt the assistant in real time.
            val threshold = if (duringPlayback) {
                max(1100.0, noiseFloor * 4.0)
            } else {
                max(420.0, noiseFloor * 2.2)
            }
            val voiced = rms > threshold
            if (!duringPlayback && !speaking && !voiced) {
                noiseFloor = noiseFloor * 0.98 + rms * 0.02
            }
            if (duringPlayback && !speaking && !voiced) ringCount = 0

            if (voiced) {
                voicedSamples += read
                silentSamples = 0
                val triggerSamples = if (duringPlayback) 3072 else 1024
                if (!speaking && voicedSamples >= triggerSamples) {
                    speaking = true
                    val start = (ringPosition - ringCount + ring.size) % ring.size
                    for (i in 0 until ringCount) {
                        if (speechSize < speech.size) speech[speechSize++] = ring[(start + i) % ring.size]
                    }
                    if (duringPlayback) {
                        if (generating) generationJob?.cancel()
                        audioTrack?.pause()
                        audioTrack?.flush()
                        audioTrack?.play()
                        playbackExpectedUntilNanos = 0L
                        generating = false
                    }
                    runOnUiThread { status.setText(R.string.minimind_o_speaking) }
                }
            } else if (speaking) {
                silentSamples += read
            } else {
                voicedSamples = 0
            }

            if (speaking) {
                val copy = minOf(read, speech.size - speechSize)
                if (copy > 0) {
                    System.arraycopy(chunk, 0, speech, speechSize, copy)
                    speechSize += copy
                }
                if (silentSamples >= 12_800 || speechSize == speech.size) {
                    val trim = minOf(silentSamples, 10_240)
                    val turn = speech.copyOf((speechSize - trim).coerceAtLeast(2048))
                    speaking = false
                    voicedSamples = 0
                    silentSamples = 0
                    speechSize = 0
                    submitTurn(turn)
                    ringPosition = 0
                    ringCount = 0
                }
            } else {
                for (i in 0 until read) {
                    ring[ringPosition] = chunk[i]
                    ringPosition = (ringPosition + 1) % ring.size
                    if (ringCount < ring.size) ringCount++
                }
            }
        }
    }

    private fun submitTurn(pcm: ShortArray) {
        generationJob?.cancel()
        generating = true
        lastInputFile = runCatching { saveModelInput(pcm) }.getOrNull()
        runOnUiThread {
            transcript.append("\n\n${getString(R.string.minimind_o_user_turn)}\n")
            transcript.append(getString(R.string.minimind_o_assistant))
            assistantPrefixLength = transcript.length()
            playInputButton.isEnabled = lastInputFile?.isFile == true
            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            try {
                engine?.respond(
                    pcm,
                    onText = { value ->
                        runOnUiThread {
                            transcript.text = transcript.text.substring(0, assistantPrefixLength) + value
                            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
                        }
                    },
                    onAudio = { values, drop -> play(values, drop) },
                    onStatus = { message -> runOnUiThread { status.text = message } },
                )
            } catch (_: kotlinx.coroutines.CancellationException) {
                // Barge-in intentionally cancels the previous response.
            } catch (error: Throwable) {
                runOnUiThread { status.text = getString(R.string.minimind_o_error, error.message) }
            } finally {
                generating = false
            }
        }
    }

    private fun recordingsDirectory(): File = File(
        getExternalFilesDir(null) ?: filesDir,
        "minimind-o/recordings",
    ).apply { mkdirs() }

    private fun saveModelInput(pcm: ShortArray): File {
        val target = File(recordingsDirectory(), "model-input-${System.currentTimeMillis()}.wav")
        FileOutputStream(target).use { output ->
            val dataBytes = pcm.size * 2
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
                put("RIFF".toByteArray(Charsets.US_ASCII))
                putInt(dataBytes + 36)
                put("WAVEfmt ".toByteArray(Charsets.US_ASCII))
                putInt(16)
                putShort(1.toShort())
                putShort(1.toShort())
                putInt(MiniMindOAudioFrontend.SAMPLE_RATE)
                putInt(MiniMindOAudioFrontend.SAMPLE_RATE * 2)
                putShort(2.toShort())
                putShort(16.toShort())
                put("data".toByteArray(Charsets.US_ASCII))
                putInt(dataBytes)
            }
            output.write(header.array())
            val samples = ByteBuffer.allocate(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
            pcm.forEach { samples.putShort(it) }
            output.write(samples.array())
        }
        recordingsDirectory().listFiles { file ->
            file.name.startsWith("model-input-") && file.extension.equals("wav", true)
        }?.sortedByDescending(File::lastModified)?.drop(20)?.forEach(File::delete)
        return target
    }

    private fun restoreLastInputRecording() {
        lastInputFile = recordingsDirectory().listFiles { file ->
            file.name.startsWith("model-input-") && file.extension.equals("wav", true)
        }?.maxByOrNull(File::lastModified)
        playInputButton.isEnabled = lastInputFile?.isFile == true
    }

    private fun playLastModelInput() {
        val file = lastInputFile?.takeIf(File::isFile) ?: return
        inputPlayer?.release()
        inputPlayer = MediaPlayer().apply {
            setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            setDataSource(file.absolutePath)
            setOnCompletionListener { player ->
                player.release()
                if (inputPlayer === player) inputPlayer = null
                playInputButton.setText(R.string.minimind_o_play_input)
            }
            prepare()
            start()
        }
        playInputButton.setText(R.string.minimind_o_playing_input)
        status.text = getString(R.string.minimind_o_playing_saved, file.name)
    }

    private fun confirmDeleteInputRecordings() {
        if (lastInputFile == null) return
        AlertDialog.Builder(this)
            .setTitle(R.string.minimind_o_delete_inputs)
            .setMessage(R.string.minimind_o_delete_inputs_message)
            .setPositiveButton(R.string.delete) { _, _ ->
                inputPlayer?.release()
                inputPlayer = null
                recordingsDirectory().listFiles()?.forEach(File::delete)
                lastInputFile = null
                playInputButton.isEnabled = false
                playInputButton.setText(R.string.minimind_o_play_input)
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    @Synchronized
    private fun play(values: FloatArray, drop: Int) {
        val track = audioTrack ?: AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(24_000)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(
                max(
                    AudioTrack.getMinBufferSize(
                        24_000,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                    ),
                    24_000 * 2,
                )
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            .also {
                check(it.state == AudioTrack.STATE_INITIALIZED) { "AudioTrack 初始化失败" }
                it.setVolume(1f)
                audioTrack = it
                it.play()
            }
        if (track.playState != AudioTrack.PLAYSTATE_PLAYING) track.play()
        val start = drop.coerceIn(0, values.size)
        val pcm = ShortArray(values.size - start) { index ->
            (values[start + index].coerceIn(-1f, 1f) * Short.MAX_VALUE)
                .toInt()
                .toShort()
        }
        val peak = pcm.maxOfOrNull { kotlin.math.abs(it.toInt()) } ?: 0
        val durationNanos = pcm.size * 1_000_000_000L / 24_000L
        val now = System.nanoTime()
        playbackExpectedUntilNanos = maxOf(now, playbackExpectedUntilNanos) + durationNanos + 100_000_000L
        runOnUiThread { status.text = "正在播放：${pcm.size} samples，峰值 $peak" }
        var written = 0
        while (written < pcm.size && generating && playbackExpectedUntilNanos != 0L) {
            // Small writes bound barge-in latency. A blocking write of the
            // whole waveform can otherwise keep playing after cancellation.
            val count = track.write(
                pcm,
                written,
                minOf(2048, pcm.size - written),
                AudioTrack.WRITE_BLOCKING,
            )
            if (!generating) return
            check(count > 0) { "AudioTrack 写入失败：$count" }
            written += count
        }
    }

    private fun stopConversation() {
        sessionActive = false
        generationJob?.cancel()
        generationJob = null
        recorder?.let {
            try { it.stop() } catch (_: Throwable) {}
            it.release()
        }
        recorder = null
        recordThread = null
        echoCanceler?.release()
        echoCanceler = null
        audioTrack?.let {
            try { it.stop() } catch (_: Throwable) {}
            it.release()
        }
        audioTrack = null
        playbackExpectedUntilNanos = 0L
        generating = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (::conversationButton.isInitialized) {
            conversationButton.setText(R.string.minimind_o_start)
            voiceButton.isEnabled = MiniMindOModelStore.isComplete(this)
            playInputButton.isEnabled = lastInputFile?.isFile == true
            status.setText(if (MiniMindOModelStore.isComplete(this)) R.string.minimind_o_ready else R.string.minimind_o_checking)
        }
    }

    override fun onDestroy() {
        stopConversation()
        inputPlayer?.release()
        inputPlayer = null
        engine?.close()
        engine = null
        super.onDestroy()
    }
}
