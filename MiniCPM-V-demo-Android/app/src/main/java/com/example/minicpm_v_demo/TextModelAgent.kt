package com.example.minicpm_v_demo

import android.content.Context
import ai.koog.agents.core.agent.AIAgent
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.processor.ManualToolCallFixProcessor

/** A bounded Koog tool agent backed entirely by the selected on-device text model. */
internal class TextModelAgent(
    context: Context,
    private val engine: LlamaEngine,
) {
    private val appContext = context.applicationContext
    private val sandbox = AgentSandbox(context.applicationContext)

    suspend fun run(task: String, onToolActivity: suspend (String) -> Unit): String {
        val selectedModel = LlamaEngine.getSelectedModel(appContext)
        require(selectedModel.isTextOnly) { "Agent mode only supports text-only models" }

        val tools = AgentTools(sandbox, onToolActivity)
        val provider = LLMProvider("minicpm-local", "MiniCPM Local")
        val model = LLModel(
            provider = provider,
            id = selectedModel.id,
            capabilities = listOf(LLMCapability.Tools),
            contextLength = 4096,
            maxOutputTokens = 1536,
        )
        val client = MiniCpmKoogClient(engine, provider)
        val executor = MultiLLMPromptExecutor(client)
        val agent = AIAgent(
            promptExecutor = executor,
            llmModel = model,
            toolRegistry = tools.registry,
            systemPrompt = SYSTEM_PROMPT,
            temperature = 0.0,
            maxIterations = 8,
            responseProcessor = ManualToolCallFixProcessor(tools.registry),
        )
        return try {
            agent.run(task).trim()
        } finally {
            agent.close()
        }
    }

    companion object {
        private val SYSTEM_PROMPT = """
            You are a concise local coding and text-task agent. Work step by step and use tools when they are needed.
            You may list/read/write files, execute restricted shell commands, and run Python in the Agent workspace.
            All paths are relative to that workspace. Do not claim to access files outside it.
            Use at most the calls needed to complete the task. When done, return a clear final answer in the user's language.
        """.trimIndent()
    }
}
