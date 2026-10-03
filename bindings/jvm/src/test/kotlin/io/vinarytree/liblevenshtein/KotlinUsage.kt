package io.vinarytree.liblevenshtein

import io.vinarytree.interop.DictionaryResource

/** Compile-checked Kotlin usage of the shipped Java API; no Kotlin facade is published. */
object KotlinUsage {
    fun search(dictionary: DictionaryResource, input: String): List<Match> =
        Transducer(dictionary, Algorithm.STANDARD).use { transducer ->
            transducer.query(input, 2L, QueryOrder.DISTANCE_THEN_TERM).use { cursor ->
                cursor.toList()
            }
        }
}
