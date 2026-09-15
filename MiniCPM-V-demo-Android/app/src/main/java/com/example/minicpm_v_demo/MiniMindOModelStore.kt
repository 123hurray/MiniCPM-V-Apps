package com.example.minicpm_v_demo

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/** Downloaded MiniMind-O runtime assets. No weights are packaged in the APK. */
object MiniMindOModelStore {
    const val MODEL_ID = "minimind-o-3o-int8"

    data class Asset(
        val fileName: String,
        val url: String,
        val sha256: String,
    )

    // These are Android-specific ExecuTorch exports derived from the
    // Apache-2.0 MiniMind-O checkpoints. They live in a release rather than
    // the APK so an install/update never duplicates ~460 MB of model data.
    val assets = listOf(
        Asset(
            "minimind-o-main-int8.pte",
            "https://github.com/OpenBMB/MiniCPM-V-Apps/releases/download/minimind-o-android-v1/minimind-o-main-int8.pte",
            "016af26199668483fca2f20b603ddb0889977de7d8eb00ea7734673ee7f832e6",
        ),
        Asset(
            "minimind-o-sensevoice-int8.pte",
            "https://github.com/OpenBMB/MiniCPM-V-Apps/releases/download/minimind-o-android-v1/minimind-o-sensevoice-int8.pte",
            "d7cf94ac2a26bb7ab4947182ffbfa9ea2b4c6fb8ba28692d71568ebe8b41eac3",
        ),
        Asset(
            "minimind-o-mimi-int8.pte",
            "https://github.com/OpenBMB/MiniCPM-V-Apps/releases/download/minimind-o-android-v1/minimind-o-mimi-int8.pte",
            "c1c51e4bd0fb0b87efa7087a08fa126aa08326527e0cb474bcb765b5ea58cdb1",
        ),
        Asset(
            "tokenizer.json",
            "https://huggingface.co/jingyaogong/minimind-3o/resolve/main/tokenizer.json",
            "71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668",
        ),
    )

    fun directory(context: Context): File =
        File(context.filesDir, "models/$MODEL_ID")

    fun file(context: Context, name: String): File = File(directory(context), name)

    fun isComplete(context: Context): Boolean = assets.all { asset ->
        val candidate = file(context, asset.fileName)
        candidate.isFile && candidate.length() > 0L &&
            (!isRealDigest(asset.sha256) || sha256(candidate) == asset.sha256)
    }

    fun delete(context: Context) {
        directory(context).deleteRecursively()
    }

    suspend fun downloadAll(
        context: Context,
        onProgress: (String) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val dir = directory(context)
        if (!dir.exists() && !dir.mkdirs()) {
            throw IOException("无法创建模型目录：${dir.absolutePath}")
        }
        for ((index, asset) in assets.withIndex()) {
            coroutineContext.ensureActive()
            download(asset, file(context, asset.fileName)) { done, total ->
                val percent = if (total > 0L) (done * 100L / total).coerceIn(0L, 100L) else -1L
                onProgress(
                    if (percent >= 0) "${index + 1}/${assets.size} ${asset.fileName}  $percent%"
                    else "${index + 1}/${assets.size} ${asset.fileName}  ${done / 1_048_576} MB"
                )
            }
        }
    }

    private suspend fun download(
        asset: Asset,
        target: File,
        progress: (Long, Long) -> Unit,
    ) {
        if (target.isFile && target.length() > 0L) {
            if (!isRealDigest(asset.sha256) || sha256(target) == asset.sha256) return
            target.delete()
        }
        val temp = File(target.parentFile, "${target.name}.tmp")
        var resume = if (temp.isFile) temp.length() else 0L
        val connection = (URL(asset.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "MiniCPM-V-Android/3.1")
            if (resume > 0L) setRequestProperty("Range", "bytes=$resume-")
        }
        try {
            val code = connection.responseCode
            if (code == HttpURLConnection.HTTP_OK && resume > 0L) {
                temp.delete()
                resume = 0L
            } else if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL) {
                throw IOException("HTTP $code: ${asset.url}")
            }
            val bodyLength = connection.contentLengthLong.coerceAtLeast(0L)
            val total = if (bodyLength > 0L) resume + bodyLength else -1L
            connection.inputStream.buffered().use { input ->
                FileOutputStream(temp, resume > 0L).buffered().use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    var written = resume
                    while (true) {
                        coroutineContext.ensureActive()
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        written += count
                        progress(written, total)
                    }
                }
            }
            if (isRealDigest(asset.sha256)) {
                val actual = sha256(temp)
                if (actual != asset.sha256) {
                    temp.delete()
                    throw IOException("${asset.fileName} SHA-256 校验失败")
                }
            }
            if (target.exists() && !target.delete()) throw IOException("无法替换 ${target.name}")
            if (!temp.renameTo(target)) {
                temp.copyTo(target, overwrite = true)
                temp.delete()
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun isRealDigest(value: String): Boolean =
        value.length == 64 && value.all { it in '0'..'9' || it in 'a'..'f' }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
