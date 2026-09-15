package com.example.minicpm_v_demo

import org.junit.Assert.assertTrue
import org.junit.Test

class AgentTraceFormatterTest {
    @Test
    fun preservesProcessAndFinalAnswer() {
        val rendered = AgentTraceFormatter.render(
            events = listOf(
                AgentTraceEvent.LlmText("Need to execute the example"),
                AgentTraceEvent.ToolCall("run_python", "{\"code\":\"print(1)\"}"),
                AgentTraceEvent.ToolResult("run_python", "{\"status\":\"ok\",\"stdout\":\"1\"}", false),
            ),
            finalAnswer = "\u6267\u884c\u5b8c\u6210",
        )

        assertTrue(rendered.contains("Agent \u8fc7\u7a0b"))
        assertTrue(rendered.contains("LLM \u6587\u672c"))
        assertTrue(rendered.contains("run_python"))
        assertTrue(rendered.contains("stdout"))
        assertTrue(rendered.contains("\u6700\u7ec8\u56de\u7b54"))
        assertTrue(rendered.contains("\u6267\u884c\u5b8c\u6210"))
    }
}
