/**
 * @file liblevenshtein.h
 * @brief C FFI bindings for liblevenshtein
 *
 * This header provides C-compatible functions for:
 * - Levenshtein distance calculations
 * - Damerau-Levenshtein distance calculations
 * - String memory management
 *
 * All strings returned by liblevenshtein functions must be freed using
 * llev_string_free(). All arrays must be freed using their specific free
 * function.
 *
 * @example
 * ```c
 * #include <liblevenshtein.h>
 * #include <stdio.h>
 *
 * int main() {
 *     // Calculate Levenshtein distance
 *     size_t dist = llev_distance("hello", 5, "helo", 4);
 *     printf("Distance: %zu\n", dist);  // Output: 1
 *
 *     // Calculate Damerau-Levenshtein distance
 *     size_t damerau = llev_damerau_distance("ab", 2, "ba", 2);
 *     printf("Damerau: %zu\n", damerau);  // Output: 1 (transposition)
 *
 *     return 0;
 * }
 * ```
 */

#ifndef LIBLEVENSHTEIN_H
#define LIBLEVENSHTEIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Algorithm Types
 * ============================================================================ */

/**
 * Algorithm type for transducers.
 */
typedef enum {
    /** Standard Levenshtein (insert, delete, substitute) */
    LLEV_ALGORITHM_STANDARD = 0,
    /** Damerau-Levenshtein (adds transposition as single edit) */
    LLEV_ALGORITHM_TRANSPOSITION = 1,
    /** Merge and split operations */
    LLEV_ALGORITHM_MERGE_AND_SPLIT = 2,
} LlevAlgorithm;

/* ============================================================================
 * String Management
 * ============================================================================ */

/**
 * Free a string allocated by liblevenshtein.
 *
 * @param ptr String pointer to free. Passing NULL is safe (no-op).
 *
 * @note The pointer must not be used after this call.
 */
void llev_string_free(char* ptr);

/**
 * Free a string array allocated by liblevenshtein.
 *
 * @param arr Array of string pointers
 * @param len Number of elements in the array
 *
 * @note Both the strings and the array itself are freed.
 */
void llev_string_array_free(char** arr, size_t len);

/**
 * Duplicate a string.
 *
 * @param src Source string (null-terminated)
 * @return Heap-allocated copy, or NULL on failure
 *
 * @note The returned string must be freed with llev_string_free().
 */
char* llev_string_dup(const char* src);

/* ============================================================================
 * Distance Functions
 * ============================================================================ */

/**
 * Calculate Levenshtein distance between two strings.
 *
 * The Levenshtein distance counts the minimum number of single-character
 * edits (insertions, deletions, substitutions) required to change one
 * string into another.
 *
 * @param source Source string (null-terminated UTF-8)
 * @param source_len Length of source string in bytes
 * @param target Target string (null-terminated UTF-8)
 * @param target_len Length of target string in bytes
 * @return Edit distance, or SIZE_MAX on error (null pointer or invalid UTF-8)
 *
 * @example
 * ```c
 * size_t dist = llev_distance("kitten", 6, "sitting", 7);
 * // dist == 3 (k->s, e->i, +g)
 * ```
 */
size_t llev_distance(
    const char* source,
    size_t source_len,
    const char* target,
    size_t target_len
);

/**
 * Calculate Levenshtein distance with early termination.
 *
 * If the distance exceeds the threshold, returns SIZE_MAX - 1 instead
 * of computing the exact distance. This can be significantly faster
 * for filtering candidates.
 *
 * @param source Source string (null-terminated UTF-8)
 * @param source_len Length of source string in bytes
 * @param target Target string (null-terminated UTF-8)
 * @param target_len Length of target string in bytes
 * @param threshold Maximum distance to compute
 * @return Edit distance if <= threshold, SIZE_MAX-1 if > threshold,
 *         SIZE_MAX on error
 */
size_t llev_distance_threshold(
    const char* source,
    size_t source_len,
    const char* target,
    size_t target_len,
    size_t threshold
);

/**
 * Calculate Damerau-Levenshtein distance between two strings.
 *
 * Like Levenshtein distance, but also counts transposition of adjacent
 * characters as a single edit. This is often more appropriate for
 * detecting typos.
 *
 * @param source Source string (null-terminated UTF-8)
 * @param source_len Length of source string in bytes
 * @param target Target string (null-terminated UTF-8)
 * @param target_len Length of target string in bytes
 * @return Edit distance, or SIZE_MAX on error
 *
 * @example
 * ```c
 * // Transposition: "ab" -> "ba" is 1 edit (not 2)
 * size_t dist = llev_damerau_distance("ab", 2, "ba", 2);
 * // dist == 1
 * ```
 */
size_t llev_damerau_distance(
    const char* source,
    size_t source_len,
    const char* target,
    size_t target_len
);

/**
 * Calculate Damerau-Levenshtein distance with early termination.
 *
 * @param source Source string (null-terminated UTF-8)
 * @param source_len Length of source string in bytes
 * @param target Target string (null-terminated UTF-8)
 * @param target_len Length of target string in bytes
 * @param threshold Maximum distance to compute
 * @return Edit distance if <= threshold, SIZE_MAX-1 if > threshold,
 *         SIZE_MAX on error
 */
size_t llev_damerau_distance_threshold(
    const char* source,
    size_t source_len,
    const char* target,
    size_t target_len,
    size_t threshold
);

#ifdef __cplusplus
}
#endif

#endif /* LIBLEVENSHTEIN_H */
