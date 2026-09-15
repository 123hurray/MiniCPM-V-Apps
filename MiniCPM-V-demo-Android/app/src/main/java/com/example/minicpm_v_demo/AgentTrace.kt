package com.example.minicpm_v_demo

internal sealed interface AgentTraceEvent {
    data class Progress(val text: String) : AgentTraceEvent
    data class Thought(val text: String) : AgentTraceEvent
    data class ToolCall(val tool: String, val input: String) : AgentTraceEvent
    data class ToolResult(val tool: String, val output: String, val isError: Boolean) : AgentTraceEvent
    data class ProtocolFeedback(val text: String) : AgentTraceEvent
}

/** Renders a bounded, persistent Agent transcript for the chat bubble. */
internal object AgentTraceFormatter {
    fun render(events: List<AgentTraceEvent>, finalAnswer: String? = null): String = buildString {
        appendLine("### Agent \u8fc7\u7a0b")
        events.takeLast(MAX_EVENTS).forEach { event ->
            when (event) {
                is AgentTraceEvent.Progress -> appendLine("- \u72b6\u6001\uff1a${clip(event.text, MAX_TEXT_CHARS)}")
                is AgentTraceEvent.Thought -> {
                    appendLine("- \u601d\u8003\uff1a")
                    appendCode(clip(event.text, MAX_THOUGHT_CHARS))
                }
                is AgentTraceEvent.ToolCall -> {
                    appendLine("- \u5de5\u5177\uff1a`${event.tool}`")
                    appendLine("  \u8f93\u5165\uff1a")
                    appendCode(clip(event.input, MAX_TOOL_CHARS))
                }
                is AgentTraceEvent.ToolResult -> {
                    appendLine("- ${if (event.isError) "\u5de5\u5177\u9519\u8bef" else "\u5de5\u5177\u8f93\u51fa"}\uff1a`${event.tool}`")
                    appendCode(clip(event.output, MAX_TOOL_CHARS))
                }
                is AgentTraceEvent.ProtocolFeedback -> {
                    appendLine("- \u683c\u5f0f\u7ea0\u9519\uff1a${clip(event.text, MAX_TEXT_CHARS)}")
                }
            }
        }
        if (events.size > MAX_EVENTS) {
            appendLine("- \u65e9\u671f ${events.size - MAX_EVENTS} \u6761\u8bb0\u5f55\u5df2\u7701\u7565")
        }
        finalAnswer?.let {
            appendLine()
            appendLine("### \u6700\u7ec8\u56de\u7b54")
            append(it.ifBlank { "(\u65e0\u6587\u672c\u56de\u7b54)" })
        }
    }.trimEnd()

    private fun StringBuilder.appendCode(text: String) {
        appendLine("```")
        appendLine(text.replace("```", "` ` `"))
        appendLine("```")
    }

    private fun clip(text: String, limit: Int): String =
        if (text.length <= limit) text else text.take(limit) + "\n\u2026(\u5df2\u622a\u65ad)"

    private const val MAX_EVENTS = 40
    private const val MAX_TEXT_CHARS = 2_000
    private const val MAX_THOUGHT_CHARS = 4_000
    private const val MAX_TOOL_CHARS = 6_000
}
