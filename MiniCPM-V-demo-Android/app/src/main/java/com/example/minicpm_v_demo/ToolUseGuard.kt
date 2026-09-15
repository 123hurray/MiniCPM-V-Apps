package com.example.minicpm_v_demo

/** Prevents an action request from ending on a promise or refusal before a tool succeeds. */
internal object ToolUseGuard {
    fun continuationFeedback(
        userRequest: String,
        assistantResponse: String,
        hasSuccessfulToolResult: Boolean,
    ): String? {
        if (PROMISE_TO_CONTINUE.containsMatchIn(assistantResponse)) {
            return "You described a future action but did not make the tool call. " +
                "Call the appropriate tool now, inspect its result, and then continue reasoning."
        }
        if (!hasSuccessfulToolResult && REQUIRES_EXECUTION.containsMatchIn(userRequest)) {
            return "The current request explicitly requires real execution, but no tool has completed successfully. " +
                "Call run_python or run_shell now and inspect the actual result before answering."
        }
        return null
    }

    private val PROMISE_TO_CONTINUE = Regex(
        "(?:\u8ba9\u6211|\u6211(?:\u5c06|\u4f1a|\u6765)|\u63a5\u4e0b\u6765).{0,16}(?:\u5c1d\u8bd5|\u6267\u884c|\u8fd0\u884c|\u8c03\u7528)|" +
            "(?:let me|i(?:'ll| will)|next i(?:'ll| will)).{0,24}(?:try|run|execute|call)",
        setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )
    private val REQUIRES_EXECUTION = Regex(
        "(?:\u6267\u884c|\u8fd0\u884c|\u8c03\u7528|\u6d4b\u8bd5|\u9a8c\u8bc1).{0,24}(?:python|shell|\u547d\u4ee4|\u811a\u672c)|" +
            "(?:python|shell|\u547d\u4ee4|\u811a\u672c).{0,24}(?:\u6267\u884c|\u8fd0\u884c|\u8c03\u7528|\u6d4b\u8bd5|\u9a8c\u8bc1)|" +
            "(?:\u6267\u884c\\s*echo|\u751f\u6210.{0,8}\u968f\u673a\u6570)|" +
            "(?:run|execute|test|verify).{0,24}(?:python|shell|command|script)|" +
            "(?:python|shell|command|script).{0,24}(?:run|execute|test|verify)",
        setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )
}
