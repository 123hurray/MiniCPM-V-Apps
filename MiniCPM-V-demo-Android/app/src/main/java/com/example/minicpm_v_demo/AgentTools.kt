package com.example.minicpm_v_demo

import ai.koog.agents.core.tools.SimpleTool
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.serialization.typeToken
import kotlinx.serialization.Serializable

internal class AgentTools(
    private val sandbox: AgentSandbox,
    private val onActivity: suspend (String) -> Unit,
) {
    @Serializable
    data class ProtocolErrorArgs(
        val message: String,
    )

    @Serializable
    data class ListFilesArgs(
        @property:LLMDescription("Relative directory inside the Agent workspace, usually .")
        val path: String = ".",
        @property:LLMDescription("Whether to recursively list descendants")
        val recursive: Boolean = false,
    )

    @Serializable
    data class ReadFileArgs(
        @property:LLMDescription("Relative file path inside the Agent workspace")
        val path: String,
        @property:LLMDescription("Maximum number of characters to return, at most 32000")
        val maxChars: Int = 12000,
    )

    @Serializable
    data class WriteFileArgs(
        @property:LLMDescription("Relative file path inside the Agent workspace")
        val path: String,
        @property:LLMDescription("UTF-8 text to write")
        val content: String,
        @property:LLMDescription("Append instead of replacing the file")
        val append: Boolean = false,
    )

    @Serializable
    data class ShellArgs(
        @property:LLMDescription("Single-line shell command using workspace-relative paths")
        val command: String,
        @property:LLMDescription("Timeout from 1 to 30 seconds")
        val timeoutSeconds: Int = 15,
    )

    @Serializable
    data class PythonArgs(
        @property:LLMDescription("Python source code. Read and write only workspace-relative paths")
        val code: String,
        @property:LLMDescription("Timeout from 1 to 30 seconds")
        val timeoutSeconds: Int = 15,
    )

    private val listFiles = object : SimpleTool<ListFilesArgs>(
        argsType = typeToken<ListFilesArgs>(),
        name = "list_files",
        description = "List files in the private Agent workspace. Paths must be relative.",
    ) {
        override suspend fun execute(args: ListFilesArgs): String {
            onActivity(name)
            return sandbox.listFiles(args.path, args.recursive)
        }
    }

    private val readFile = object : SimpleTool<ReadFileArgs>(
        argsType = typeToken<ReadFileArgs>(),
        name = "read_file",
        description = "Read a UTF-8 text file from the private Agent workspace.",
    ) {
        override suspend fun execute(args: ReadFileArgs): String {
            onActivity(name)
            return sandbox.readFile(args.path, args.maxChars)
        }
    }

    private val writeFile = object : SimpleTool<WriteFileArgs>(
        argsType = typeToken<WriteFileArgs>(),
        name = "write_file",
        description = "Write or append UTF-8 text in the private Agent workspace.",
    ) {
        override suspend fun execute(args: WriteFileArgs): String {
            onActivity(name)
            return sandbox.writeFile(args.path, args.content, args.append)
        }
    }

    private val runShell = object : SimpleTool<ShellArgs>(
        argsType = typeToken<ShellArgs>(),
        name = "run_shell",
        description = "Run an allowlisted Android shell command in the private Agent workspace.",
    ) {
        override suspend fun execute(args: ShellArgs): String {
            onActivity(name)
            return sandbox.runShell(args.command, args.timeoutSeconds)
        }
    }

    private val runPython = object : SimpleTool<PythonArgs>(
        argsType = typeToken<PythonArgs>(),
        name = "run_python",
        description = "Run a Python 3.12 script in the private Agent workspace.",
    ) {
        override suspend fun execute(args: PythonArgs): String {
            onActivity(name)
            return sandbox.runPython(args.code, args.timeoutSeconds)
        }
    }

    /**
     * Internal recovery path. MiniCpmKoogClient emits this call when the local
     * model attempted a malformed tool call. Koog then returns this output to
     * the model like any other tool result so it can correct the call and keep
     * reasoning within the same bounded agent run.
     */
    private val protocolError = object : SimpleTool<ProtocolErrorArgs>(
        argsType = typeToken<ProtocolErrorArgs>(),
        name = PROTOCOL_ERROR_TOOL,
        description = "Internal tool-call protocol recovery. Never call this tool directly.",
    ) {
        override suspend fun execute(args: ProtocolErrorArgs): String = """
            TOOL_CALL_FORMAT_ERROR
            ${args.message}
            Correct the tool call and try again now. Reply with only one valid JSON object when calling a tool.
            Do not claim that the requested tool ran until you receive its actual result.
        """.trimIndent()
    }

    val registry: ToolRegistry = ToolRegistry {
        tool(listFiles)
        tool(readFile)
        tool(writeFile)
        tool(runShell)
        tool(runPython)
        tool(protocolError)
    }

    companion object {
        const val PROTOCOL_ERROR_TOOL = "tool_protocol_error"
    }
}
