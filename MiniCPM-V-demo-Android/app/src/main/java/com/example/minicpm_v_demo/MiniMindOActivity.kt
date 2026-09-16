package com.example.minicpm_v_demo

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
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
import kotlin.math.max
import kotlin.math.sqrt

class MiniMindOActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private lateinit var transcript: TextView
    private lateinit var scroll: ScrollView
    private lateinit var progress: LinearProgressIndicator
    private lateinit var downloadButton: MaterialButton
    private lateinit var conversationButton: MaterialButton

    private var engine: MiniMindOEngine? = null
    private var recorder: AudioRecord? = null
    private var recordThread: Thread? = null
    @Volatile private var sessionActive = false
    @Volatile private var generating = false
    private var generationJob: Job? = null
    private var echoCanceler: AcousticEchoCanceler? = null
    private var audioTrack: AudioTrack? = null
    private var assistantPrefixLength = 0

    private val micPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startConversation()
        else Toast.makeText(this, R.string.minimind_o_permission, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_minimind_o)
        findViewById<Toolbar>(R.id.toolbar).setNavigationOnClickListener { finish() }
        status = findViewById(R.id.tv_status)
        transcript = findViewById(R.id.tv_transcript)
        scroll = findViewById(R.id.scroll_transcript)
        progress = findViewById(R.id.progress_model)
        downloadButton = findViewById(R.id.btn_download_model)
        conversationButton = findViewById(R.id.btn_conversation)

        downloadButton.setOnClickListener {
            if (MiniMindOModelStore.isComplete(this)) confirmDeleteModel() else startDownload()
        }
        conversationButton.setOnClickListener {
            if (sessionActive) stopConversation() else ensureMicAndStart()
        }
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
                        }
                        ModelDownloadController.Status.Completed -> {
                            progress.visibility = View.GONE
                            ModelDownloadController.acknowledge()
                            refreshModelState()
                        }
                        is ModelDownloadController.Status.Failed -> {
                            progress.visibility = View.GONE
                            downloadButton.isEnabled = true
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
        downloadButton.isEnabled = true
        downloadButton.setText(if (complete) R.string.minimind_o_delete else R.string.minimind_o_download)
        status.setText(if (complete) R.string.minimind_o_ready else R.string.minimind_o_checking)
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
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) startConversation() else micPermission.launch(Manifest.permission.RECORD_AUDIO)
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
        val ring = ShortArray(4800)
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
            var energy = 0.0
            for (i in 0 until read) energy += chunk[i].toDouble() * chunk[i]
            val rms = sqrt(energy / read)
            val voiced = rms > max(550.0, noiseFloor * 2.8)
            if (!speaking && !voiced) noiseFloor = noiseFloor * 0.98 + rms * 0.02

            if (voiced) {
                voicedSamples += read
                silentSamples = 0
                if (!speaking && voicedSamples >= 2048) {
                    speaking = true
                    val start = (ringPosition - ringCount + ring.size) % ring.size
                    for (i in 0 until ringCount) {
                        if (speechSize < speech.size) speech[speechSize++] = ring[(start + i) % ring.size]
                    }
                    if (generating) {
                        generationJob?.cancel()
                        audioTrack?.pause()
                        audioTrack?.flush()
                        audioTrack?.play()
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
        runOnUiThread {
            transcript.append("\n\n${getString(R.string.minimind_o_user_turn)}\n")
            transcript.append(getString(R.string.minimind_o_assistant))
            assistantPrefixLength = transcript.length()
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
        var written = 0
        while (written < pcm.size) {
            val count = track.write(pcm, written, pcm.size - written, AudioTrack.WRITE_BLOCKING)
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
        generating = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (::conversationButton.isInitialized) {
            conversationButton.setText(R.string.minimind_o_start)
            status.setText(if (MiniMindOModelStore.isComplete(this)) R.string.minimind_o_ready else R.string.minimind_o_checking)
        }
    }

    override fun onDestroy() {
        stopConversation()
        engine?.close()
        engine = null
        super.onDestroy()
    }
}
