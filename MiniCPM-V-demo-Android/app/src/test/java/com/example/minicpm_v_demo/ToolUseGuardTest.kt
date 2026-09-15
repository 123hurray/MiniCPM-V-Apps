package com.example.minicpm_v_demo

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class ToolUseGuardTest {
    @Test
    fun executionRequestCannotFinishBeforeToolResult() {
        assertNotNull(
            ToolUseGuard.continuationFeedback(
                userRequest = "\u6267\u884c\u4e00\u4e2a Python \u5feb\u6392\u793a\u4f8b",
                assistantResponse = "\u6211\u65e0\u6cd5\u76f4\u63a5\u8fd0\u884c Python\u3002",
                hasSuccessfulToolResult = false,
            )
        )
    }

    @Test
    fun promiseToTryCannotBecomeFinalAnswer() {
        assertNotNull(
            ToolUseGuard.continuationFeedback(
                userRequest = "\u751f\u6210\u4e00\u4e2a\u968f\u673a\u6570",
                assistantResponse = "\u8ba9\u6211\u5c1d\u8bd5\u4f7f\u7528 shell \u547d\u4ee4\u3002",
                hasSuccessfulToolResult = false,
            )
        )
    }

    @Test
    fun finalAnswerAllowedAfterSuccessfulExecution() {
        assertNull(
            ToolUseGuard.continuationFeedback(
                userRequest = "\u6267\u884c echo 1",
                assistantResponse = "\u6267\u884c\u6210\u529f\uff0c\u8f93\u51fa\u662f 1\u3002",
                hasSuccessfulToolResult = true,
            )
        )
    }

    @Test
    fun bareShellAndCurrentTimeRequestsRequireExecution() {
        assertNotNull(
            ToolUseGuard.continuationFeedback("\u8fd0\u884c echo 1", "echo 1", false)
        )
        assertNotNull(
            ToolUseGuard.continuationFeedback("\u8f93\u51fa\u5f53\u524d\u65f6\u95f4", "date", false)
        )
    }
}
