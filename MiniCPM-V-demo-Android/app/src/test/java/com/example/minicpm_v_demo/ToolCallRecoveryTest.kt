package com.example.minicpm_v_demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ToolCallRecoveryTest {
    @Test
    fun recoversScreenshotBareCommands() {
        assertEquals("echo 1", ToolCallRecovery.bareShellCommand("echo 1"))
        assertEquals("date", ToolCallRecovery.bareShellCommand("date"))
        assertEquals("pwd", ToolCallRecovery.bareShellCommand("```sh\npwd\n```"))
    }

    @Test
    fun doesNotAutoExecuteDestructiveOrNarrativeText() {
        assertNull(ToolCallRecovery.bareShellCommand("rm notes.txt"))
        assertNull(ToolCallRecovery.bareShellCommand("I would run echo 1"))
    }
}
