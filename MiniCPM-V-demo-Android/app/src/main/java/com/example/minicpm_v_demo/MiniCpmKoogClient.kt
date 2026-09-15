package com.example.minicpm_v_demo

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.Prompt
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.MessagePart
import ai.koog.prompt.message.ResponseMetaInfo
import kotlinx.coroutines.flow.collect
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import java.util.concurrent.atomic.AtomicInteger

/** Bridges Koog's provider-neutral prompt API to the loaded local llama.cpp model. */
internal class MiniCpmKoogClient(
    private val engine: LlamaEngine,
    private val provider: LLMProvider,
    private val originalTask: String,
    private val maxOutputTokens: Int,
    private val onTrace: suspend (AgentTraceEvent) -> Unit,
) : LLMClient() {

    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Message.Assistant {
        engine.clearContext()
        engine.setSystemPrompt(buildSystemPrompt(prompt, tools))

        val response = StringBuilder()
        engine.sendUserPrompt(buildConversation(prompt), predictLength = maxOutputTokens).collect { token ->
            response.append(token)
        }
        val rawResponse = response.toString().trim()
        onTrace(AgentTraceEvent.LlmText(rawResponse.ifBlank { "(empty LLM output)" }))
        val callableDescriptors = tools.filterNot { it.name == AgentTools.PROTOCOL_ERROR_TOOL }
        val callableTools = callableDescriptors.mapTo(linkedSetOf()) { it.name }
        val requiredArguments = callableDescriptors.associate { descriptor ->
            descriptor.name to descriptor.requiredParameters.mapTo(linkedSetOf()) { it.name }
        }
        return when (
            val parsed = ToolCallProtocol.parse(rawResponse, callableTools, requiredArguments)
        ) {
            is ToolCallProtocol.Result.Valid -> {
                toolCallMessage(parsed.tool, parsed.arguments)
            }

            is ToolCallProtocol.Result.Invalid -> protocolErrorMessage(parsed.feedback)

            ToolCallProtocol.Result.NotAToolCall -> {
                val assistantText = stripThinking(rawResponse)
                val successfulToolResult = latestSuccessfulToolResult(prompt)
                val recoveredCommand = if (
                    ToolUseGuard.requiresExecution(originalTask) && "run_shell" in callableTools
                ) {
                    ToolCallRecovery.bareShellCommand(assistantText)
                } else {
                    null
                }
                if (recoveredCommand != null) {
                    if (successfulToolResult != null) {
                        onTrace(
                            AgentTraceEvent.ProtocolFeedback(
                                "The model repeated a bare command after the tool had already completed; " +
                                    "the verified tool output is returned instead of executing it twice."
                            )
                        )
                        return Message.Assistant(
                            content = "TOOL RESULT (${successfulToolResult.tool}):\n${successfulToolResult.output}",
                            metaInfo = ResponseMetaInfo.Empty,
                            finishReason = "stop",
                        )
                    }
                    onTrace(
                        AgentTraceEvent.ProtocolFeedback(
                            "The model returned a bare shell command instead of a tool call; " +
                                "it was safely repaired and will now be executed."
                        )
                    )
                    return toolCallMessage(
                        "run_shell",
                        buildJsonObject { put("command", recoveredCommand) },
                    )
                }
                val feedback = ToolUseGuard.continuationFeedback(
                    userRequest = originalTask,
                    assistantResponse = assistantText,
                    hasSuccessfulToolResult = successfulToolResult != null,
                )
                if (feedback != null) {
                    protocolErrorMessage(feedback)
                } else {
                    Message.Assistant(
                        content = assistantText,
                        metaInfo = ResponseMetaInfo.Empty,
                        finishReason = "stop",
                    )
                }
            }
        }
    }

    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        ModerationResult(isHarmful = false, categories = emptyMap())

    override fun llmProvider(): LLMProvider = provider

    override fun close() = Unit

    private fun buildSystemPrompt(prompt: Prompt, tools: List<ToolDescriptor>): String {
        val suppliedSystem = prompt.messages
            .filterIsInstance<Message.System>()
            .joinToString("\n") { it.textContent() }
        val toolText = tools
            .filterNot { it.name == AgentTools.PROTOCOL_ERROR_TOOL }
            .joinToString("\n") { tool ->
                val required = tool.requiredParameters.joinToString(", ") { "${it.name}: ${it.type}" }
                val optional = tool.optionalParameters.joinToString(", ") { "${it.name}: ${it.type} (optional)" }
                val parameters = listOf(required, optional).filter { it.isNotBlank() }.joinToString(", ")
                "- ${tool.name}($parameters): ${tool.description}"
            }
        return """
            $suppliedSystem

            You are running locally on Android. The only accessible project area is the private Agent workspace.
            Available tools:
            $toolText

            To call exactly one structured file tool, reply with ONLY one JSON object using this exact shape:
            {"tool":"tool_name","arguments":{"argument":"value"}}
            For Python and shell, prefer a raw tool block so code does not need JSON escaping:
            <tool_call name="run_python">
            print("hello from Python")
            </tool_call>
            <tool_call name="run_shell">
            echo 'hello from shell'
            </tool_call>
            For a non-trivial Python script, write it first without JSON escaping, then execute the saved file:
            <tool_call name="write_file" path="script.py">
            def main():
                print("hello from a saved script")
            main()
            </tool_call>
            <tool_call name="run_python_file">
            script.py
            </tool_call>
            Do not wrap a tool call in Markdown. After receiving a tool result, either call another tool or answer normally.
            A raw block's body is passed verbatim as code or command. Shell remains single-line; Python may be multiline.
            JSON strings must escape double quotes. If a TOOL_CALL_FORMAT_ERROR result is returned, correct the call
            and try the tool immediately. A promise to try a tool is not a tool call and must not be your final answer.
            Never invent tool output. Prefer list_files before assuming a file exists. The complete raw LLM response is
            recorded for the user before parsing, including malformed tool-call data.
        """.trimIndent()
    }

    private suspend fun protocolErrorMessage(feedback: String): Message.Assistant {
        onTrace(AgentTraceEvent.ProtocolFeedback(feedback))
        return Message.Assistant(
            parts = listOf(
                MessagePart.Tool.Call(
                    id = "protocol_error_${toolCallId.incrementAndGet()}",
                    tool = AgentTools.PROTOCOL_ERROR_TOOL,
                    args = buildJsonObject { put("message", feedback) },
                )
            ),
            metaInfo = ResponseMetaInfo.Empty,
            finishReason = "tool_calls",
        )
    }

    private suspend fun toolCallMessage(
        tool: String,
        arguments: kotlinx.serialization.json.JsonObject,
    ): Message.Assistant {
        onTrace(AgentTraceEvent.ToolCall(tool, formatArguments(arguments)))
        return Message.Assistant(
            parts = listOf(
                MessagePart.Tool.Call(
                    id = "local_tool_${toolCallId.incrementAndGet()}",
                    tool = tool,
                    args = arguments,
                )
            ),
            metaInfo = ResponseMetaInfo.Empty,
            finishReason = "tool_calls",
        )
    }

    private fun formatArguments(arguments: Map<String, Any?>): String = arguments.entries
        .joinToString("\n") { (name, value) ->
            val displayValue = (value as? JsonPrimitive)?.contentOrNull ?: value.toString()
            "$name:\n$displayValue"
        }

    private fun latestSuccessfulToolResult(prompt: Prompt): MessagePart.Tool.Result? = prompt.messages
        .asReversed()
        .firstNotNullOfOrNull { message ->
            message.parts.filterIsInstance<MessagePart.Tool.Result>().lastOrNull { result ->
                result.tool != AgentTools.PROTOCOL_ERROR_TOOL && !toolOutputFailed(result.output)
            }
        }

    private fun toolOutputFailed(output: String): Boolean {
        val normalized = output.lowercase()
        return "tool_execution_error" in normalized ||
            "\"status\": \"error\"" in normalized ||
            "\"status\":\"error\"" in normalized ||
            Regex("exit_code=(?!0(?:\\s|$))\\d+").containsMatchIn(normalized)
    }

    private fun stripThinking(response: String): String =
        response.replace(THINKING, "").trim()

    private fun buildConversation(prompt: Prompt): String = buildString {
        prompt.messages.filterNot { it is Message.System }.forEach { message ->
            val role = when (message) {
                is Message.User -> "USER"
                is Message.Assistant -> "ASSISTANT"
                is Message.System -> "SYSTEM"
            }
            append(role).append(":\n")
            message.parts.forEach { part ->
                when (part) {
                    is MessagePart.Text -> append(part.text)
                    is MessagePart.Reasoning -> append(part.content.joinToString("\n"))
                    is MessagePart.Tool.Call -> append(
                        "{\"tool\":\"${part.tool}\",\"arguments\":${part.args}}"
                    )
                    is MessagePart.Tool.Result -> append(
                        "TOOL RESULT (${part.tool}):\n${part.output}"
                    )
                    else -> append(part.toString())
                }
                append('\n')
            }
        }
        append("ASSISTANT:\n")
    }

    private companion object {
        val toolCallId = AtomicInteger(0)
        val THINKING = Regex("(?s)<think>(.*?)</think>\\s*", RegexOption.IGNORE_CASE)
    }
}
