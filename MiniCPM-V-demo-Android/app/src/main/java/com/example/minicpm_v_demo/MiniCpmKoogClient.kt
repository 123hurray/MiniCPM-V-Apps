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
        return Message.Assistant(
            content = response.toString().trim(),
            metaInfo = ResponseMetaInfo.Empty,
            finishReason = "stop",
        )
    }

    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        ModerationResult(isHarmful = false, categories = emptyMap())

    override fun llmProvider(): LLMProvider = provider

    override fun close() = Unit

    private fun buildSystemPrompt(prompt: Prompt, tools: List<ToolDescriptor>): String {
        val suppliedSystem = prompt.messages
            .filterIsInstance<Message.System>()
            .joinToString("\n") { it.textContent() }
        val toolText = tools.joinToString("\n") { tool ->
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
            Do not wrap a tool call in Markdown. After receiving a tool result, either call another tool or answer normally.
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
}
