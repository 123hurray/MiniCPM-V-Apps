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
import kotlinx.serialization.json.put
import java.util.concurrent.atomic.AtomicInteger

/** Bridges Koog's provider-neutral prompt API to the loaded local llama.cpp model. */
internal class MiniCpmKoogClient(
    private val engine: LlamaEngine,
    private val provider: LLMProvider,
) : LLMClient() {

    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): Message.Assistant {
        engine.clearContext()
        engine.setSystemPrompt(buildSystemPrompt(prompt, tools))

        val response = StringBuilder()
        engine.sendUserPrompt(buildConversation(prompt), predictLength = 1536).collect { token ->
            response.append(token)
        }
        val rawResponse = response.toString().trim()
        val callableDescriptors = tools.filterNot { it.name == AgentTools.PROTOCOL_ERROR_TOOL }
        val callableTools = callableDescriptors.mapTo(linkedSetOf()) { it.name }
        val requiredArguments = callableDescriptors.associate { descriptor ->
            descriptor.name to descriptor.requiredParameters.mapTo(linkedSetOf()) { it.name }
        }
        return when (
            val parsed = ToolCallProtocol.parse(rawResponse, callableTools, requiredArguments)
        ) {
            is ToolCallProtocol.Result.Valid -> Message.Assistant(
                parts = listOf(
                    MessagePart.Tool.Call(
                        id = "local_tool_${toolCallId.incrementAndGet()}",
                        tool = parsed.tool,
                        args = parsed.arguments,
                    )
                ),
                metaInfo = ResponseMetaInfo.Empty,
                finishReason = "tool_calls",
            )

            is ToolCallProtocol.Result.Invalid -> Message.Assistant(
                parts = listOf(
                    MessagePart.Tool.Call(
                        id = "protocol_error_${toolCallId.incrementAndGet()}",
                        tool = AgentTools.PROTOCOL_ERROR_TOOL,
                        args = buildJsonObject { put("message", parsed.feedback) },
                    )
                ),
                metaInfo = ResponseMetaInfo.Empty,
                finishReason = "tool_calls",
            )

            ToolCallProtocol.Result.NotAToolCall -> Message.Assistant(
                content = rawResponse,
                metaInfo = ResponseMetaInfo.Empty,
                finishReason = "stop",
            )
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

            To call exactly one tool, reply with ONLY one JSON object using this exact shape:
            {"tool":"tool_name","arguments":{"argument":"value"}}
            Valid shell example:
            {"tool":"run_shell","arguments":{"command":"echo 'hello from agent'","timeoutSeconds":15}}
            Do not wrap a tool call in Markdown. After receiving a tool result, either call another tool or answer normally.
            JSON strings must escape double quotes. Never include a newline in a shell command.
            If a TOOL_CALL_FORMAT_ERROR result is returned, correct the JSON and try the tool call again immediately.
            Never invent tool output. Prefer list_files before assuming a file exists.
        """.trimIndent()
    }

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
    }
}
