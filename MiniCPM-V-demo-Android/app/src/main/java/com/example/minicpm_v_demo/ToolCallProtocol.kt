package com.example.minicpm_v_demo

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put

/** Converts the local model's text protocol into a typed Koog tool call. */
internal object ToolCallProtocol {
    sealed interface Result {
        data object NotAToolCall : Result
        data class Valid(val tool: String, val arguments: JsonObject) : Result
        data class Invalid(val feedback: String) : Result
    }

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = false
    }

    fun parse(
        rawResponse: String,
        allowedTools: Set<String>,
        requiredArguments: Map<String, Set<String>> = emptyMap(),
    ): Result {
        val candidate = unwrap(rawResponse)
        parseRawBody(candidate, allowedTools)?.let { return it }
        if (!looksLikeToolCall(candidate, allowedTools)) return Result.NotAToolCall

        val root = runCatching { json.parseToJsonElement(candidate).jsonObject }
            .getOrElse { error ->
                return Result.Invalid(
                    "The tool call was not valid JSON (${error.message.orEmpty().lineSequence().firstOrNull()}). " +
                        "Return exactly one JSON object. Escape JSON double quotes, use single quotes inside " +
                        "shell commands when possible, and do not put a newline in a command."
                )
            }

        val function = root["function"] as? JsonObject
        val callObject = function ?: root
        val tool = firstString(callObject, "tool", "name", "tool_name")
            ?: return Result.Invalid(
                "The tool call has no tool name. Use " +
                    "{\"tool\":\"tool_name\",\"arguments\":{...}}."
            )

        if (tool !in allowedTools) {
            return Result.Invalid(
                "Unknown tool '$tool'. Available tools: ${allowedTools.sorted().joinToString(", ")}."
            )
        }

        val argumentsElement = firstElement(callObject, "arguments", "args", "parameters", "params", "tool_args")
            ?: return Result.Invalid("Tool '$tool' is missing the arguments object.")
        val arguments = when (argumentsElement) {
            is JsonObject -> argumentsElement
            is JsonPrimitive -> argumentsElement.contentOrNull?.let { encoded ->
                runCatching { json.parseToJsonElement(encoded).jsonObject }.getOrNull()
            }
            else -> null
        } ?: return Result.Invalid("The arguments for '$tool' must be one JSON object.")

        val missingArguments = requiredArguments[tool].orEmpty() - arguments.keys
        if (missingArguments.isNotEmpty()) {
            return Result.Invalid(
                "Tool '$tool' is missing required arguments: ${missingArguments.sorted().joinToString(", ")}."
            )
        }

        return Result.Valid(tool, arguments)
    }

    private fun parseRawBody(candidate: String, allowedTools: Set<String>): Result? {
        val match = RAW_TOOL_CALL.matchEntire(candidate) ?: return if (
            candidate.startsWith("<tool_call", ignoreCase = true)
        ) {
            Result.Invalid(
                "The raw tool block is incomplete. Close it with </tool_call>, or return one valid JSON object."
            )
        } else {
            null
        }
        val attributes = match.groupValues[1]
        val tool = NAME_ATTRIBUTE.find(attributes)?.groupValues?.getOrNull(2)
            ?: return Result.Invalid("The raw tool block is missing a quoted name attribute.")
        if (tool !in allowedTools) {
            return Result.Invalid(
                "Unknown tool '$tool'. Available tools: ${allowedTools.sorted().joinToString(", ")}."
            )
        }
        val argumentName = when (tool) {
            "run_python" -> "code"
            "run_shell" -> "command"
            "run_python_file" -> "path"
            "write_file" -> "content"
            else -> return Result.Invalid(
                "Raw tool blocks are supported only for run_python, run_python_file, run_shell, and write_file. " +
                    "Use JSON for '$tool'."
            )
        }
        val body = match.groupValues[2].trim('\n', '\r')
        if (body.isBlank()) return Result.Invalid("Raw tool block for '$tool' is empty.")
        val writePath = if (tool == "write_file") {
            PATH_ATTRIBUTE.find(attributes)?.groupValues?.getOrNull(2)
                ?: return Result.Invalid(
                    "Raw write_file requires a quoted relative path attribute, for example " +
                        "<tool_call name=\"write_file\" path=\"script.py\">..."
                )
        } else {
            null
        }
        val arguments = buildJsonObject {
            if (tool == "write_file") {
                put("path", requireNotNull(writePath))
                put("content", body)
            } else {
                put(argumentName, body)
            }
        }
        return Result.Valid(tool, arguments)
    }

    private fun unwrap(raw: String): String {
        var text = raw.trim()
        text = text.replace(Regex("(?s)^<think>.*?</think>\\s*"), "").trim()
        if (text.startsWith("```") && text.endsWith("```")) {
            text = text.removePrefix("```").removePrefix("json").trim()
            text = text.removeSuffix("```").trim()
        }
        if (text.startsWith("<tool_call>") && text.endsWith("</tool_call>")) {
            text = text.removePrefix("<tool_call>").removeSuffix("</tool_call>").trim()
        }
        return text
    }

    private fun looksLikeToolCall(candidate: String, allowedTools: Set<String>): Boolean {
        val hasProtocolKey = TOOL_KEY.containsMatchIn(candidate) || FUNCTION_KEY.containsMatchIn(candidate)
        val namesKnownTool = candidate.contains('{') && allowedTools.any(candidate::contains)
        return hasProtocolKey || namesKnownTool
    }

    private fun firstString(obj: JsonObject, vararg keys: String): String? =
        keys.firstNotNullOfOrNull { key -> (obj[key] as? JsonPrimitive)?.contentOrNull }

    private fun firstElement(obj: JsonObject, vararg keys: String): JsonElement? =
        keys.firstNotNullOfOrNull(obj::get)

    private val TOOL_KEY = Regex("\"(?:tool|name|tool_name)\"\\s*:")
    private val FUNCTION_KEY = Regex("\"function\"\\s*:")
    private val RAW_TOOL_CALL = Regex(
        "(?s)^\\s*<tool_call\\s+([^>]+)>\\s*(.*?)\\s*</tool_call>\\s*$",
        RegexOption.IGNORE_CASE,
    )
    private val NAME_ATTRIBUTE = Regex("(?:^|\\s)name\\s*=\\s*([\"'])([A-Za-z0-9_-]+)\\1", RegexOption.IGNORE_CASE)
    private val PATH_ATTRIBUTE = Regex("(?:^|\\s)path\\s*=\\s*([\"'])([^\"']+)\\1", RegexOption.IGNORE_CASE)
}
