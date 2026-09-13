package com.example.minicpm_v_demo

import android.Manifest
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
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
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.security.MessageDigest

class ModelManagerActivity : AppCompatActivity() {

    private lateinit var tvModelStatus: TextView
    private lateinit var btnDownload: MaterialButton
    private lateinit var btnImport: MaterialButton
    private lateinit var btnLoadModel: MaterialButton
    private lateinit var btnDeleteModel: MaterialButton
    private lateinit var progressDownload: LinearProgressIndicator
    private lateinit var recyclerModels: RecyclerView
    private lateinit var tvLanguageValue: TextView

    private lateinit var engine: LlamaEngine
    private lateinit var modelAdapter: ModelAdapter

    private val modelFilePicker =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isNotEmpty()) importModelFiles(uris)
        }

    // Android 13+: POST_NOTIFICATIONS is a runtime permission. We need it
    // for the foreground download service's progress notification (without
    // a notification the OS will outright kill the foreground service).
    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            // We start the download regardless of grant: the service still
            // posts a notification, the user just won't see it. Foreground
            // service itself is allowed without the permission.
            if (!granted) {
                Toast.makeText(
                    this,
                    getString(R.string.toast_notification_denied),
                    Toast.LENGTH_LONG
                ).show()
            }
            startDownloadService()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_manager)

        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        toolbar.setNavigationOnClickListener { finish() }

        tvModelStatus = findViewById(R.id.tv_model_status)
        btnDownload = findViewById(R.id.btn_download)
        btnImport = findViewById(R.id.btn_import)
        btnLoadModel = findViewById(R.id.btn_load_model)
        btnDeleteModel = findViewById(R.id.btn_delete_model)
        progressDownload = findViewById(R.id.progress_download)
        recyclerModels = findViewById(R.id.recycler_models)
        tvLanguageValue = findViewById(R.id.tv_language_value)

        engine = LlamaEngine.getInstance(applicationContext)

        setupModelList()
        updateLoadButtonState()
        observeEngineState()
        observeDownloadStatus()
        updateLanguageDisplay()

        btnDownload.setOnClickListener { onDownloadClicked() }
        btnImport.setOnClickListener { onImportClicked() }
        btnLoadModel.setOnClickListener { loadSelectedModel() }
        btnDeleteModel.setOnClickListener { confirmDeleteModel() }
        findViewById<View>(R.id.btn_language).setOnClickListener { showLanguagePicker() }
    }

    private fun setupModelList() {
        val selectedModel = LlamaEngine.getSelectedModel(this)
        modelAdapter = ModelAdapter(
            models = ModelInfo.AVAILABLE_MODELS,
            selectedModelId = selectedModel.id,
            onModelSelected = { model ->
                val previousModelId = LlamaEngine.getSelectedModel(this).id
                LlamaEngine.setSelectedModel(this, model.id)
                updateLoadButtonState()

                if (previousModelId != model.id) {
                    val wasLoaded = engine.state.value is LlamaState.ModelReady
                    if (wasLoaded) {
                        reloadSelectedModel()
                    } else {
                        tvModelStatus.text = getString(R.string.switched_to_load, model.displayName)
                    }
                }
            }
        )

        recyclerModels.layoutManager = LinearLayoutManager(this)
        recyclerModels.adapter = modelAdapter
    }

    private fun reloadSelectedModel() {
        val model = LlamaEngine.getSelectedModel(this)
        val modelPath = LlamaEngine.modelPath(applicationContext)

        if (!File(modelPath).exists()) {
            tvModelStatus.text = getString(R.string.switched_to_download, model.displayName)
            lifecycleScope.launch(Dispatchers.IO) {
                try { engine.unloadModel() } catch (_: Exception) {}
                withContext(Dispatchers.Main) { updateLoadButtonState() }
            }
            return
        }

        btnLoadModel.isEnabled = false
        btnDownload.isEnabled = false
        tvModelStatus.text = getString(R.string.switching_to, model.displayName)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                engine.unloadModel()
                val mmprojPath = LlamaEngine.mmprojPath(applicationContext)
                val mmprojArg = mmprojPath?.let { if (File(it).exists()) it else null }
                engine.loadModel(modelPath, mmprojArg)
                LlamaEngine.markModelSwitched(applicationContext)
                withContext(Dispatchers.Main) {
                    updateLoadButtonState()
                    Toast.makeText(this@ModelManagerActivity, getString(R.string.toast_load_success, model.displayName), Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error reloading model", e)
                engine.resetToInitialized()
                withContext(Dispatchers.Main) {
                    tvModelStatus.text = getString(R.string.toast_load_failed, e.message)
                    updateLoadButtonState()
                }
            }
        }
    }

    private fun observeEngineState() {
        lifecycleScope.launch {
            engine.state.collect { state ->
                when (state) {
                    is LlamaState.Uninitialized -> {
                        tvModelStatus.text = getString(R.string.status_uninitialized)
                    }
                    is LlamaState.Initializing -> {
                        tvModelStatus.text = getString(R.string.status_initializing)
                    }
                    is LlamaState.Initialized -> {
                        tvModelStatus.text = getString(R.string.status_initialized)
                        updateLoadButtonState()
                    }
                    is LlamaState.LoadingModel -> {
                        tvModelStatus.text = getString(R.string.status_loading)
                        btnLoadModel.isEnabled = false
                        btnDownload.isEnabled = false
                    }
                    is LlamaState.ModelReady -> {
                        tvModelStatus.text = getString(R.string.status_ready)
                        btnLoadModel.isEnabled = true
                        btnDownload.isEnabled = true
                        updateLoadButtonState()
                    }
                    is LlamaState.ProcessingSystemPrompt,
                    is LlamaState.ProcessingUserPrompt -> {
                        tvModelStatus.text = getString(R.string.status_generating)
                    }
                    is LlamaState.PrefillingImage -> {
                        tvModelStatus.text = getString(R.string.status_prefilling_image)
                    }
                    is LlamaState.Generating -> {
                        tvModelStatus.text = getString(R.string.status_generating)
                    }
                    is LlamaState.UnloadingModel -> {
                        tvModelStatus.text = getString(R.string.status_unloading)
                    }
                    is LlamaState.Error -> {
                        tvModelStatus.text = getString(R.string.status_error, state.exception.message)
                        btnLoadModel.isEnabled = true
                        btnDownload.isEnabled = true
                    }
                }
            }
        }
    }

    private fun updateLoadButtonState() {
        val exists = LlamaEngine.modelsExist(this)
        val isReady = engine.state.value is LlamaState.ModelReady
        btnLoadModel.isEnabled = exists
        btnDeleteModel.visibility = if (exists) View.VISIBLE else View.GONE
        btnLoadModel.text = when {
            isReady -> getString(R.string.reload_model)
            exists -> getString(R.string.load_model)
            else -> getString(R.string.no_model_file)
        }
    }

    private fun onDownloadClicked() {
        if (ModelDownloadController.isRunning) {
            Toast.makeText(this, R.string.toast_downloading, Toast.LENGTH_SHORT).show()
            return
        }
        if (LlamaEngine.modelsExist(this)) {
            Toast.makeText(this, R.string.toast_already_downloaded, Toast.LENGTH_SHORT).show()
            return
        }

        // Android 13+ needs runtime POST_NOTIFICATIONS so the foreground
        // service notification is actually visible. Lower OS versions get
        // the permission for free at install time and just fall through.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }

        startDownloadService()
    }

    private fun onImportClicked() {
        if (ModelDownloadController.isRunning) {
            Toast.makeText(this, R.string.toast_downloading, Toast.LENGTH_SHORT).show()
            return
        }

        val model = LlamaEngine.getSelectedModel(this)
        val expected = requiredFiles(model).joinToString("\n") { "• ${it.fileName}" }
        AlertDialog.Builder(this)
            .setTitle(R.string.import_model_files)
            .setMessage(getString(R.string.import_model_hint, model.displayName, expected))
            .setPositiveButton(R.string.select_files) { _, _ ->
                modelFilePicker.launch(arrayOf("application/octet-stream", "application/x-gguf", "*/*"))
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun importModelFiles(uris: List<Uri>) {
        val model = LlamaEngine.getSelectedModel(this)
        btnImport.isEnabled = false
        btnDownload.isEnabled = false
        btnLoadModel.isEnabled = false
        progressDownload.visibility = View.VISIBLE
        tvModelStatus.text = getString(R.string.importing_model)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val required = requiredFiles(model)
                val selected = uris.map { uri -> uri to queryDisplayName(uri) }
                val unknown = selected.map { it.second }.filter { name ->
                    required.none { name == it.fileName || name == it.remoteName }
                }
                if (unknown.isNotEmpty()) {
                    throw IOException(getString(R.string.import_unrecognized_files, unknown.joinToString()))
                }

                val duplicated = selected.mapNotNull { (_, name) ->
                    required.firstOrNull { name == it.fileName || name == it.remoteName }?.fileName
                }.groupingBy { it }.eachCount().filterValues { it > 1 }.keys
                if (duplicated.isNotEmpty()) {
                    throw IOException(getString(R.string.import_duplicate_files, duplicated.joinToString()))
                }

                if (engine.state.value is LlamaState.ModelReady) engine.unloadModel()

                val modelDir = File(LlamaEngine.modelDirFor(applicationContext, model))
                if (!modelDir.exists() && !modelDir.mkdirs()) {
                    throw IOException(getString(R.string.import_create_dir_failed))
                }

                selected.forEachIndexed { index, (uri, sourceName) ->
                    val spec = required.first { sourceName == it.fileName || sourceName == it.remoteName }
                    withContext(Dispatchers.Main) {
                        tvModelStatus.text = getString(
                            R.string.importing_model_progress,
                            index + 1,
                            selected.size,
                            spec.fileName
                        )
                    }
                    copyAndVerify(uri, File(modelDir, spec.fileName), spec.md5)
                }

                val missing = required.filterNot { File(modelDir, it.fileName).exists() }
                withContext(Dispatchers.Main) {
                    updateLoadButtonState()
                    if (missing.isEmpty()) {
                        tvModelStatus.text = getString(R.string.import_complete_status)
                        Toast.makeText(this@ModelManagerActivity, R.string.import_complete, Toast.LENGTH_SHORT).show()
                    } else {
                        tvModelStatus.text = getString(
                            R.string.import_partial_status,
                            missing.joinToString { it.fileName }
                        )
                        Toast.makeText(this@ModelManagerActivity, R.string.import_partial, Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model import failed", e)
                withContext(Dispatchers.Main) {
                    tvModelStatus.text = getString(R.string.import_failed, e.message ?: e::class.java.simpleName)
                    Toast.makeText(this@ModelManagerActivity, tvModelStatus.text, Toast.LENGTH_LONG).show()
                }
            } finally {
                withContext(Dispatchers.Main) {
                    progressDownload.visibility = View.GONE
                    btnImport.isEnabled = true
                    btnDownload.isEnabled = true
                    updateLoadButtonState()
                }
            }
        }
    }

    private fun queryDisplayName(uri: Uri): String {
        var cursor: Cursor? = null
        return try {
            cursor = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) cursor.getString(0)
            else uri.lastPathSegment?.substringAfterLast('/') ?: throw IOException(getString(R.string.import_unknown_name))
        } finally {
            cursor?.close()
        }
    }

    private fun copyAndVerify(uri: Uri, target: File, expectedMd5: String?) {
        val temp = File(target.parentFile, "${target.name}.importing")
        temp.delete()
        try {
            contentResolver.openInputStream(uri)?.use { input ->
                temp.outputStream().buffered().use { output -> input.copyTo(output) }
            } ?: throw IOException(getString(R.string.import_open_failed, target.name))

            if (temp.length() == 0L) throw IOException(getString(R.string.import_empty_file, target.name))
            if (!expectedMd5.isNullOrBlank()) {
                val actual = fileMd5(temp)
                if (!actual.equals(expectedMd5, ignoreCase = true)) {
                    throw IOException(getString(R.string.import_md5_failed, target.name))
                }
            }

            if (target.exists() && !target.delete()) throw IOException(getString(R.string.import_replace_failed, target.name))
            if (!temp.renameTo(target)) {
                temp.copyTo(target, overwrite = true)
                if (!temp.delete()) Log.w(TAG, "Could not delete import temp file: ${temp.absolutePath}")
            }
        } catch (e: Exception) {
            temp.delete()
            throw e
        }
    }

    private fun fileMd5(file: File): String {
        val digest = MessageDigest.getInstance("MD5")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun requiredFiles(model: ModelInfo): List<ModelFileSpec> = buildList {
        add(ModelFileSpec(model.ggufFileName, model.ggufRemoteName, model.ggufMd5))
        model.mmprojFileName?.let { add(ModelFileSpec(it, model.mmprojRemoteName, model.mmprojMd5)) }
        model.acousticFileName?.let { add(ModelFileSpec(it, model.acousticRemoteName, model.acousticMd5)) }
    }

    private fun startDownloadService() {
        // Drop any prior terminal status so the UI re-enters Running cleanly.
        ModelDownloadController.acknowledge()

        btnDownload.isEnabled = false
        btnLoadModel.isEnabled = false
        progressDownload.visibility = View.VISIBLE
        tvModelStatus.text = getString(R.string.status_downloading)

        ModelDownloadService.start(applicationContext)
    }

    /**
     * Mirrors the foreground service's [ModelDownloadController] state into
     * the UI. We use repeatOnLifecycle(STARTED) so we don't burn cycles
     * collecting while the Activity is in the background, but we still
     * pick up any progress that arrived during that window when we resume.
     */
    private fun observeDownloadStatus() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                ModelDownloadController.status.collect { status ->
                    when (status) {
                        is ModelDownloadController.Status.Idle -> {
                            progressDownload.visibility = View.GONE
                            btnDownload.isEnabled = true
                            updateLoadButtonState()
                        }
                        is ModelDownloadController.Status.Running -> {
                            progressDownload.visibility = View.VISIBLE
                            btnDownload.isEnabled = false
                            btnLoadModel.isEnabled = false
                            tvModelStatus.text = status.message
                        }
                        is ModelDownloadController.Status.Completed -> {
                            progressDownload.visibility = View.GONE
                            tvModelStatus.text = getString(R.string.download_complete_status)
                            Toast.makeText(this@ModelManagerActivity, R.string.download_complete_toast, Toast.LENGTH_SHORT).show()
                            btnDownload.isEnabled = true
                            updateLoadButtonState()
                            ModelDownloadController.acknowledge()
                        }
                        is ModelDownloadController.Status.Cancelled -> {
                            progressDownload.visibility = View.GONE
                            tvModelStatus.text = getString(R.string.download_cancelled)
                            btnDownload.isEnabled = true
                            updateLoadButtonState()
                            ModelDownloadController.acknowledge()
                        }
                        is ModelDownloadController.Status.Failed -> {
                            Log.w(TAG, "Download failed: ${status.message}")
                            progressDownload.visibility = View.GONE
                            tvModelStatus.text = getString(R.string.download_failed, status.message)
                            Toast.makeText(
                                this@ModelManagerActivity,
                                getString(R.string.download_failed, status.message),
                                Toast.LENGTH_LONG
                            ).show()
                            btnDownload.isEnabled = true
                            updateLoadButtonState()
                            ModelDownloadController.acknowledge()
                        }
                    }
                }
            }
        }
    }

    private fun loadSelectedModel() {
        val model = LlamaEngine.getSelectedModel(applicationContext)
        // TTS models are loaded on-demand by TtsActivity, not via LlamaEngine.
        if (model.isTts) {
            LlamaEngine.markModelSwitched(applicationContext)
            finish() // return to parent; MainActivity will redirect
            return
        }

        val currentState = engine.state.value
        if (currentState is LlamaState.LoadingModel) {
            Toast.makeText(this, R.string.toast_already_loading, Toast.LENGTH_SHORT).show()
            return
        }

        val modelPath = LlamaEngine.modelPath(applicationContext)
        val mmprojPath = LlamaEngine.mmprojPath(applicationContext)

        if (!File(modelPath).exists()) {
            Toast.makeText(this, R.string.toast_model_not_found, Toast.LENGTH_LONG).show()
            return
        }

        val isReload = currentState is LlamaState.ModelReady

        btnLoadModel.isEnabled = false
        btnDownload.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                if (isReload) {
                    Log.i(TAG, "Unloading model for reload...")
                    engine.unloadModel()
                }

                val mmprojArg = mmprojPath?.let { path ->
                    if (File(path).exists()) path else null
                }
                engine.loadModel(modelPath, mmprojArg)
                // No default system prompt: aligned with iOS opt-r1. See
                // MainActivity.clearChat() for the rationale.

                withContext(Dispatchers.Main) {
                    btnLoadModel.isEnabled = true
                    btnDownload.isEnabled = true
                    updateLoadButtonState()
                    Toast.makeText(this@ModelManagerActivity, R.string.toast_model_loaded, Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                engine.resetToInitialized()
                withContext(Dispatchers.Main) {
                    tvModelStatus.text = getString(R.string.toast_load_failed, e.message)
                    btnLoadModel.isEnabled = true
                    btnDownload.isEnabled = true
                    updateLoadButtonState()
                }
            }
        }
    }

    private fun confirmDeleteModel() {
        val model = LlamaEngine.getSelectedModel(this)
        val fileList = buildString {
            append("• ${model.ggufFileName}")
            if (model.mmprojFileName != null) {
                append("\n• ${model.mmprojFileName}")
            }
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_dialog_title)
            .setMessage(getString(R.string.delete_dialog_message, model.displayName, fileList))
            .setPositiveButton(R.string.delete) { _, _ -> deleteModelFiles() }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun deleteModelFiles() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val modelPath = LlamaEngine.modelPath(applicationContext)
                val mmprojPath = LlamaEngine.mmprojPath(applicationContext)

                var deleted = false
                File(modelPath).let { if (it.exists()) { it.delete(); deleted = true } }
                mmprojPath?.let { File(it) }?.let { if (it.exists()) { it.delete(); deleted = true } }
                // Also delete the acoustic GGUF for TTS models
                val acousticPath = LlamaEngine.acousticPath(applicationContext)
                acousticPath?.let { File(it) }?.let { if (it.exists()) { it.delete(); deleted = true } }

                withContext(Dispatchers.Main) {
                    updateLoadButtonState()
                    if (deleted) {
                        tvModelStatus.text = getString(R.string.model_files_deleted)
                        Toast.makeText(this@ModelManagerActivity, R.string.model_files_deleted, Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this@ModelManagerActivity, R.string.toast_no_files_to_delete, Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error deleting model", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@ModelManagerActivity, getString(R.string.toast_delete_failed, e.message), Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun updateLanguageDisplay() {
        tvLanguageValue.text = LocaleManager.currentLanguage(this).displayName
    }

    private fun showLanguagePicker() {
        val current = LocaleManager.currentLanguage(this)
        val options = LocaleManager.AppLanguage.entries.toTypedArray()
        val items = options.map {
            if (it == current) "${it.displayName}  \u2713" else it.displayName
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle(R.string.language_picker_title)
            .setItems(items) { _, which ->
                val picked = options[which]
                if (picked != current) {
                    LocaleManager.setLanguageAndRestart(this, picked)
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    companion object {
        private val TAG = ModelManagerActivity::class.java.simpleName
    }

    private data class ModelFileSpec(val fileName: String, val remoteName: String?, val md5: String?)
}
