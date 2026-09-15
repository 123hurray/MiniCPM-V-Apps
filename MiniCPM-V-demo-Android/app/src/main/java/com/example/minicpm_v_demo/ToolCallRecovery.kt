package com.example.minicpm_v_demo

/** Conservative recovery for small models which emit a command without the tool wrapper. */
internal object ToolCallRecovery {
    fun bareShellCommand(response: String): String? {
        var candidate = response.trim()
        if (candidate.startsWith("```") && candidate.endsWith("```")) {
            candidate = candidate.removePrefix("```").removeSuffix("```").trim()
            val firstLine = candidate.lineSequence().firstOrNull().orEmpty().trim()
            if (firstLine in setOf("sh", "shell", "bash")) {
                candidate = candidate.substringAfter('\n', "").trim()
            }
        }
        if (candidate.isBlank() || candidate.length > MAX_COMMAND_CHARS) return null
        if ('\n' in candidate || '\r' in candidate) return null
        val executable = candidate.substringBefore(' ').substringBefore('\t')
        return candidate.takeIf { executable in SAFE_BARE_COMMANDS }
    }

    private val SAFE_BARE_COMMANDS = setOf(
        "pwd", "ls", "find", "cat", "head", "tail", "wc", "sort", "uniq",
        "grep", "cut", "tr", "date", "echo", "printf", "du",
    )
    private const val MAX_COMMAND_CHARS = 500
}
