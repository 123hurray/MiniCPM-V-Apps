package com.example.minicpm_v_demo

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.ByteArrayOutputStream
import java.io.File

/** Minimal Hugging Face ByteLevel-BPE runtime for minimind-3o/tokenizer.json. */
class MiniMindOTokenizer(tokenizerFile: File) {
    private val tokenToId: Map<String, Int>
    private val idToToken: Map<Int, String>
    private val mergeRanks: Map<Pair<String, String>, Int>
    private val addedTokens: Map<String, Int>
    private val addedIds: Map<Int, String>
    private val byteEncoder: Map<Int, Char>
    private val byteDecoder: Map<Char, Int>

    private val pattern = Regex(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    )

    init {
        val root = Json.parseToJsonElement(tokenizerFile.readText()).jsonObject
        val model = root.getValue("model").jsonObject
        tokenToId = model.getValue("vocab").jsonObject.mapValues { it.value.jsonPrimitive.int }
        idToToken = tokenToId.entries.associate { (token, id) -> id to token }
        mergeRanks = model.getValue("merges").jsonArray.mapIndexed { rank, element ->
            val pair = element.jsonArray
            (pair[0].jsonPrimitive.content to pair[1].jsonPrimitive.content) to rank
        }.toMap()
        addedTokens = root.getValue("added_tokens").jsonArray.associate { element ->
            val item = element.jsonObject
            item.getValue("content").jsonPrimitive.content to item.getValue("id").jsonPrimitive.int
        }
        addedIds = addedTokens.entries.associate { (token, id) -> id to token }
        byteEncoder = buildByteEncoder()
        byteDecoder = byteEncoder.entries.associate { (byte, character) -> character to byte }
    }

    fun encode(text: String): IntArray {
        val result = ArrayList<Int>()
        var cursor = 0
        val specials = addedTokens.keys.sortedByDescending { it.length }
        while (cursor < text.length) {
            var nextIndex = text.length
            var nextSpecial: String? = null
            for (special in specials) {
                val found = text.indexOf(special, cursor)
                if (found >= 0 && found < nextIndex) {
                    nextIndex = found
                    nextSpecial = special
                }
            }
            if (nextIndex > cursor) encodeOrdinary(text.substring(cursor, nextIndex), result)
            if (nextSpecial == null) break
            result += addedTokens.getValue(nextSpecial)
            cursor = nextIndex + nextSpecial.length
        }
        return result.toIntArray()
    }

    fun decode(ids: List<Int>): String {
        val bytes = ByteArrayOutputStream()
        val text = StringBuilder()
        fun flushBytes() {
            if (bytes.size() > 0) {
                text.append(bytes.toByteArray().toString(Charsets.UTF_8))
                bytes.reset()
            }
        }
        for (id in ids) {
            val added = addedIds[id]
            if (added != null) {
                flushBytes()
                text.append(added)
                continue
            }
            val token = idToToken[id] ?: continue
            for (character in token) {
                val value = byteDecoder[character]
                if (value == null) {
                    flushBytes()
                    text.append(character)
                } else {
                    bytes.write(value)
                }
            }
        }
        flushBytes()
        return text.toString()
    }

    private fun encodeOrdinary(text: String, output: MutableList<Int>) {
        for (match in pattern.findAll(text)) {
            val encoded = buildString {
                for (byte in match.value.toByteArray(Charsets.UTF_8)) {
                    append(byteEncoder.getValue(byte.toInt() and 0xff))
                }
            }
            val pieces = encoded.map { it.toString() }.toMutableList()
            while (pieces.size > 1) {
                var bestPair: Pair<String, String>? = null
                var bestRank = Int.MAX_VALUE
                for (index in 0 until pieces.lastIndex) {
                    val pair = pieces[index] to pieces[index + 1]
                    val rank = mergeRanks[pair] ?: continue
                    if (rank < bestRank) {
                        bestRank = rank
                        bestPair = pair
                    }
                }
                val pair = bestPair ?: break
                val merged = ArrayList<String>(pieces.size)
                var index = 0
                while (index < pieces.size) {
                    if (index < pieces.lastIndex &&
                        pieces[index] == pair.first && pieces[index + 1] == pair.second
                    ) {
                        merged += pair.first + pair.second
                        index += 2
                    } else {
                        merged += pieces[index++]
                    }
                }
                pieces.clear()
                pieces.addAll(merged)
            }
            for (piece in pieces) output += tokenToId[piece] ?: 0
        }
    }

    private fun buildByteEncoder(): Map<Int, Char> {
        val bytes = mutableListOf<Int>()
        bytes += 33..126
        bytes += 161..172
        bytes += 174..255
        val chars = bytes.toMutableList()
        var extra = 0
        for (value in 0..255) {
            if (value !in bytes) {
                bytes += value
                chars += 256 + extra++
            }
        }
        return bytes.indices.associate { bytes[it] to chars[it].toChar() }
    }
}
