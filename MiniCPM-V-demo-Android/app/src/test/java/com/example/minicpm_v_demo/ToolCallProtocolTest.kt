package com.example.minicpm_v_demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolCallProtocolTest {
    private val tools = setOf("run_shell", "run_python", "run_python_file", "read_file", "write_file")
    private val required = mapOf(
        "run_shell" to setOf("command"),
        "run_python" to setOf("code"),
        "run_python_file" to setOf("path"),
        "read_file" to setOf("path"),
        "write_file" to setOf("path", "content"),
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

    @Test
    fun parsesRawWriteFileForComplexPython() {
        val result = ToolCallProtocol.parse(
            """
                <tool_call name="write_file" path="quicksort.py">
                def quicksort(values):
                    return sorted(values)
                print(quicksort([2, 1]))
                </tool_call>
            """.trimIndent(),
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("write_file", result.tool)
        assertEquals("quicksort.py", result.arguments["path"].toString().trim('"'))
        assertTrue(result.arguments["content"].toString().contains("def quicksort"))
    }

    @Test
    fun parsesRawPythonFileExecution() {
        val result = ToolCallProtocol.parse(
            "<tool_call name=\"run_python_file\">quicksort.py</tool_call>",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("run_python_file", result.tool)
        assertEquals("quicksort.py", result.arguments["path"].toString().trim('"'))
    }

    @Test
    fun parsesMiniCpmNativeFunctionCall() {
        val result = ToolCallProtocol.parse(
            """<function name="run_shell"><param name="command">echo 1</param></function>""",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("run_shell", result.tool)
        assertEquals("echo 1", result.arguments["command"].toString().trim('"'))
    }

    @Test
    fun parsesMiniCpmNativeCdataScriptAfterLlmText() {
        val result = ToolCallProtocol.parse(
            """
                I will save the script first.
                <function name="write_file"><param name="path">quicksort.py</param><param name="content"><![CDATA[
                def quicksort(values):
                    return sorted(values)
                print(quicksort([3, 1, 2]))
                ]]></param></function>
            """.trimIndent(),
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Valid)
        result as ToolCallProtocol.Result.Valid
        assertEquals("write_file", result.tool)
        assertTrue(result.arguments["content"].toString().contains("def quicksort"))
    }

    @Test
    fun incompleteMiniCpmNativeFunctionReturnsProtocolError() {
        val result = ToolCallProtocol.parse(
            "<function name=\"run_shell\"><param name=\"command\">date</param>",
            tools,
            required,
        )

        assertTrue(result is ToolCallProtocol.Result.Invalid)
    }
}
