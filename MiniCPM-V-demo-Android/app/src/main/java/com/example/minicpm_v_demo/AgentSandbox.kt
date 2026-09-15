package com.example.minicpm_v_demo

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

/**
 * App-level workspace used by the local agent tools.
 *
 * File APIs enforce canonical-path confinement. Shell commands are additionally
 * restricted to a small command set and reject path/expansion syntax which can
 * escape the workspace. Python applies the same root policy inside the embedded
 * runtime. Android's per-app UID sandbox remains the outer security boundary.
 */
class AgentSandbox(context: Context) {
    val root: File = File(context.filesDir, WORKSPACE_DIR).canonicalFile.apply {
        check(exists() || mkdirs()) { "Cannot create Agent workspace: $absolutePath" }
    }

    private val appContext = context.applicationContext

    suspend fun listFiles(path: String, recursive: Boolean): String = withContext(Dispatchers.IO) {
        val target = resolve(path)
        require(target.exists()) { "Path does not exist: $path" }
        if (target.isFile) return@withContext relative(target)

        val files = if (recursive) target.walkTopDown().drop(1).take(MAX_LIST_ENTRIES) else {
            target.listFiles().orEmpty().sortedBy { it.name }.asSequence()
        }
        val lines = files.map { file ->
            val suffix = if (file.isDirectory) "/" else " (${file.length()} bytes)"
            relative(file) + suffix
        }.toList()
        if (lines.isEmpty()) "(empty workspace)" else lines.joinToString("\n")
    }

    suspend fun readFile(path: String, maxChars: Int): String = withContext(Dispatchers.IO) {
        val target = resolve(path)
        require(target.isFile) { "Not a file: $path" }
        val limit = maxChars.coerceIn(1, MAX_READ_CHARS)
        target.bufferedReader().use { reader ->
            val buffer = CharArray(limit + 1)
            val count = reader.read(buffer)
            if (count < 0) return@withContext ""
            val text = String(buffer, 0, minOf(count, limit))
            if (count > limit) "$text\n…(truncated)" else text
        }
    }

    suspend fun writeFile(path: String, content: String, append: Boolean): String = withContext(Dispatchers.IO) {
        require(content.toByteArray().size <= MAX_WRITE_BYTES) {
            "One write is limited to ${MAX_WRITE_BYTES / 1024} KiB"
        }
        val target = resolve(path)
        require(target != root) { "A file path is required" }
        target.parentFile?.let { parent ->
            check(parent.exists() || parent.mkdirs()) { "Cannot create parent directory" }
        }
        val existingBytes = if (append) 0L else target.takeIf { it.isFile }?.length() ?: 0L
        check(workspaceSize() - existingBytes + content.toByteArray().size <= MAX_WORKSPACE_BYTES) {
            "Agent workspace limit is ${MAX_WORKSPACE_BYTES / 1024 / 1024} MiB"
        }
        if (append) target.appendText(content) else target.writeText(content)
        "Wrote ${content.length} characters to ${relative(target)}"
    }

    suspend fun runShell(command: String, timeoutSeconds: Int): String = coroutineScope {
        val normalizedCommand = normalizeShellCommand(command)
        validateShellCommand(normalizedCommand)
        val timeout = timeoutSeconds.coerceIn(1, MAX_TIMEOUT_SECONDS)
        val process = withContext(Dispatchers.IO) {
            ProcessBuilder("/system/bin/sh", "-c", normalizedCommand)
                .directory(root)
                .redirectErrorStream(true)
                .apply {
                    environment().clear()
                    environment()["PATH"] = "/system/bin:/system/xbin"
                    environment()["HOME"] = root.absolutePath
                    environment()["TMPDIR"] = root.absolutePath
                    environment()["LANG"] = "C.UTF-8"
                }
                .start()
        }
        val output = async(Dispatchers.IO) { readProcessOutput(process) }
        val completedExit = withTimeoutOrNull(timeout * 1000L) {
            withContext(Dispatchers.IO) { process.waitFor() }
        }
        if (completedExit == null) {
            // Process.waitFor(timeout, unit) and destroyForcibly are API 26;
            // use the API-24-safe methods because this app supports Android 7.
            process.destroy()
            withContext(Dispatchers.IO) { process.waitFor() }
        }
        val text = output.await()
        val exit = completedExit ?: 124
        "exit_code=$exit\n${text.ifBlank { "(no output)" }}"
    }

    suspend fun runPython(code: String, timeoutSeconds: Int): String = withContext(Dispatchers.IO) {
        require(code.length <= MAX_PYTHON_CHARS) { "Python source is too large" }
        val timeout = timeoutSeconds.coerceIn(1, MAX_TIMEOUT_SECONDS)
        ensurePythonStarted()
        Python.getInstance()
            .getModule("agent_runtime")
            .callAttr("run_code", code, root.absolutePath, timeout, MAX_OUTPUT_CHARS)
            .toString()
    }

    private fun ensurePythonStarted() {
        if (Python.isStarted()) return
        synchronized(Python::class.java) {
            if (!Python.isStarted()) Python.start(AndroidPlatform(appContext))
        }
    }

    private fun resolve(path: String): File {
        val requested = path.trim().ifEmpty { "." }
        require(!File(requested).isAbsolute) { "Absolute paths are not allowed" }
        val target = File(root, requested).canonicalFile
        require(target == root || target.path.startsWith(root.path + File.separator)) {
            "Path escapes the Agent workspace"
        }
        return target
    }

    private fun relative(file: File): String =
        file.relativeTo(root).path.ifEmpty { "." }

    private fun workspaceSize(): Long =
        root.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    private fun readProcessOutput(process: Process): String {
        val out = StringBuilder()
        process.inputStream.bufferedReader().use { reader ->
            val buffer = CharArray(2048)
            while (true) {
                val count = reader.read(buffer)
                if (count < 0) break
                if (out.length < MAX_OUTPUT_CHARS) {
                    out.append(buffer, 0, minOf(count, MAX_OUTPUT_CHARS - out.length))
                }
            }
        }
        if (out.length >= MAX_OUTPUT_CHARS) out.append("\n…(truncated)")
        return out.toString()
    }

    internal fun validateShellCommand(command: String) {
        require(command.isNotBlank()) { "Command is empty" }
        require(command.length <= MAX_SHELL_CHARS) { "Command is too long" }
        require(!command.contains('\n') && !command.contains('\r') && !command.contains('\u0000')) {
            "Multiline commands are not allowed"
        }
        require(".." !in command && '~' !in command && '$' !in command && '`' !in command && '\\' !in command) {
            "Path traversal and shell expansion are not allowed"
        }
        require(command.none { it == '*' || it == '?' || it == '[' || it == ']' || it == '{' || it == '}' }) {
            "Shell glob expansion is not allowed"
        }
        require(!ABSOLUTE_PATH.containsMatchIn(command)) { "Absolute paths are not allowed" }
        require(!DANGEROUS_TOKEN.containsMatchIn(command)) { "Command contains a blocked operation" }

        command.split(Regex("(?:&&|\\|\\||[;|])"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .forEach { segment ->
                val executable = segment.substringBefore(' ').substringBefore('\t')
                require(executable in ALLOWED_COMMANDS) { "Command not allowed: $executable" }
            }
    }

    companion object {
        const val WORKSPACE_DIR = "agent_workspace"
        private const val MAX_LIST_ENTRIES = 500
        private const val MAX_READ_CHARS = 32_000
        private const val MAX_WRITE_BYTES = 512 * 1024
        private const val MAX_WORKSPACE_BYTES = 16L * 1024 * 1024
        private const val MAX_OUTPUT_CHARS = 32_000
        private const val MAX_PYTHON_CHARS = 64_000
        private const val MAX_SHELL_CHARS = 2_000
        private const val MAX_TIMEOUT_SECONDS = 30

        private val ABSOLUTE_PATH = Regex("(?:^|[\\s=<>])/[A-Za-z0-9._-]")
        private val DANGEROUS_TOKEN = Regex(
            "(?:^|\\s)(?:sh|bash|zsh|su|run-as|toybox|am|pm|cmd|mount|ln|chmod|chown|env|xargs)(?:\\s|$)|-(?:exec|ok)(?:\\s|$)"
        )
        private val ALLOWED_COMMANDS = setOf(
            "pwd", "ls", "find", "cat", "head", "tail", "wc", "sort", "uniq",
            "grep", "cut", "tr", "date", "echo", "printf", "mkdir", "touch",
            "cp", "mv", "rm", "rmdir", "du"
        )

        internal fun normalizeShellCommand(command: String): String =
            command.trimEnd('\r', '\n')
    }
}
