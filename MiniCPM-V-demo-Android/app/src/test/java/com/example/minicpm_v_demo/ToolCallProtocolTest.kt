package com.example.minicpm_v_demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolCallProtocolTest {
    private val tools = setOf("run_shell", "run_python", "read_file")
    private val required = mapOf(
        "run_shell" to setOf("command"),
        "run_python" to setOf("code"),
        "read_file" to setOf("path"),
    )

    @Test
    fun parsesValidToolCall() {
        val result = ToolCallProtocol.parse(
            """{"tool":"run_shell","arguments":{"command":"echo 'hello'"}}""",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("run_shell", result.tool)
        assertEquals("echo 'hello'", result.arguments["command"].toString().trim('"'))
    }

    @Test
    fun acceptsFencedJsonAndAliases() {
        val result = ToolCallProtocol.parse(
            """
                ```json
                {"name":"read_file","args":{"path":"notes.txt"}}
                ```
            """.trimIndent(),
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        assertEquals("read_file", (result as ToolCallProtocol.Result.Valid).tool)
    }

    @Test
    fun malformedQuotedShellCommandReturnsProtocolError() {
        val result = ToolCallProtocol.parse(
            """{"tool":"run_shell","arguments":{"command":"echo "hello"\n"}}""",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Invalid)
        assertTrue((result as ToolCallProtocol.Result.Invalid).feedback.contains("not valid JSON"))
    }

    @Test
    fun missingRequiredArgumentReturnsProtocolError() {
        val result = ToolCallProtocol.parse(
            """{"tool":"run_python","arguments":{}}""",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Invalid)
        assertTrue((result as ToolCallProtocol.Result.Invalid).feedback.contains("code"))
    }

    @Test
    fun ordinaryAnswerRemainsText() {
        val result = ToolCallProtocol.parse("The command completed successfully.", tools, required)
        assertTrue(result is ToolCallProtocol.Result.NotAToolCall)
    }

    @Test
    fun parsesMultilineRawPythonWithoutJsonEscaping() {
        val result = ToolCallProtocol.parse(
            """
                <tool_call name="run_python">
                def quicksort(values):
                    return sorted(values)

                print(quicksort([3, 1, 2]))
                </tool_call>
            """.trimIndent(),
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("run_python", result.tool)
        assertTrue(result.arguments["code"].toString().contains("def quicksort"))
        assertTrue(result.arguments["code"].toString().contains("print(quicksort"))
    }

    @Test
    fun parsesRawShellCommand() {
        val result = ToolCallProtocol.parse(
            "<tool_call name='run_shell'>echo 1</tool_call>",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("run_shell", result.tool)
        assertEquals("echo 1", result.arguments["command"].toString().trim('"'))
    }

    @Test
    fun incompleteRawBlockReturnsProtocolError() {
        val result = ToolCallProtocol.parse(
            "<tool_call name=\"run_python\">print(1)",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Invalid)
    }
}
