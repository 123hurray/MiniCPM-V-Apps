package com.example.minicpm_v_demo

import android.content.Context
import ai.koog.agents.core.agent.AIAgent
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.processor.ManualToolCallFixProcessor

internal data class AgentConversationTurn(
    val user: String,
    val assistant: String,
)

/** A bounded Koog tool agent backed entirely by the selected on-device text model. */
internal class TextModelAgent(
    context: Context,
    private val engine: LlamaEngine,
) {
    private val appContext = context.applicationContext
    private val sandbox = AgentSandbox(context.applicationContext)

    suspend fun run(
        task: String,
        history: List<AgentConversationTurn> = emptyList(),
        onTrace: suspend (AgentTraceEvent) -> Unit,
    ): String {
        val selectedModel = LlamaEngine.getSelectedModel(appContext)
        require(selectedModel.isTextOnly) { "Agent mode only supports text-only models" }
        val contextLength = LlamaEngine.getAgentContextLength(appContext)
        val maxOutputTokens = LlamaEngine.getAgentMaxOutputTokens(appContext)

        val tools = AgentTools(sandbox, onTrace)
        val provider = LLMProvider("minicpm-local", "MiniCPM Local")
        val model = LLModel(
            provider = provider,
            id = selectedModel.id,
            capabilities = listOf(LLMCapability.Tools),
            contextLength = contextLength,
            maxOutputTokens = maxOutputTokens,
        )
        val client = MiniCpmKoogClient(engine, provider, task, maxOutputTokens, onTrace)
        val executor = MultiLLMPromptExecutor(client)
        val agent = AIAgent(
            promptExecutor = executor,
            llmModel = model,
            toolRegistry = tools.registry,
            systemPrompt = SYSTEM_PROMPT,
            temperature = 0.0,
            maxIterations = 16,
            responseProcessor = ManualToolCallFixProcessor(tools.registry),
        )
        return try {
            agent.run(buildTask(task, history)).trim()
        } finally {
            agent.close()
        }
    }

    private fun buildTask(task: String, history: List<AgentConversationTurn>): String = buildString {
        val recentHistory = history.takeLast(MAX_HISTORY_TURNS)
        if (recentHistory.isNotEmpty()) {
            appendLine("Recent conversation context:")
            recentHistory.forEach { turn ->
                append("USER: ").appendLine(turn.user.take(MAX_HISTORY_ITEM_CHARS))
                append("ASSISTANT: ").appendLine(turn.assistant.take(MAX_HISTORY_ITEM_CHARS))
            }
            appendLine()
        }
        appendLine("Current user request:")
        append(task)
    }

    companion object {
        private val SYSTEM_PROMPT = """
            You are a concise local coding and text-task agent. Work step by step and use tools when they are needed.
            You may list/read/write files, execute restricted shell commands, and run Python in the Agent workspace.
            All paths are relative to that workspace. Do not claim to access files outside it.
            When the user asks to test a tool, call that tool directly with a harmless concrete example.
            A tool call is not complete until you receive and inspect its actual TOOL RESULT.
            If a tool call fails or its format is rejected, inspect the feedback, correct it, and continue.
            For non-trivial or multiline Python, write a .py file first and execute it with run_python_file.
            Never end with a promise such as "I will try"; perform the promised tool call in the same run.
            Use at most the calls needed to complete the task. When done, return a clear final answer in the user's language.
        """.trimIndent()

        private const val MAX_HISTORY_TURNS = 4
        private const val MAX_HISTORY_ITEM_CHARS = 1500
    }
}
