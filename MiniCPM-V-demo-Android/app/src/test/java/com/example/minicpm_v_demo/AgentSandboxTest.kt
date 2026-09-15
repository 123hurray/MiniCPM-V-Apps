package com.example.minicpm_v_demo

import org.junit.Assert.assertEquals
import org.junit.Test

class AgentSandboxTest {
    @Test
    fun removesOnlyTrailingLineBreaksFromShellCommand() {
        assertEquals(
            "echo 'hello'",
            AgentSandbox.normalizeShellCommand("echo 'hello'\n\r\n"),
        )
        assertEquals(
            "printf 'first\\nsecond'",
            AgentSandbox.normalizeShellCommand("printf 'first\\nsecond'"),
        )
    }
}
