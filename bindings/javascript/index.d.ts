import type { DictionaryResource } from "@vinary-tree/vinary-tree-interop";

/** Edit semantics used when compiling a reusable dictionary transducer. */
export type Algorithm =
  /** Insertions, deletions, and substitutions. */
  | "standard"
  /** Standard edits plus one adjacent transposition per source position. */
  | "transposition"
  /** Standard edits plus adjacent-symbol merge and split operations. */
  | "merge-and-split"
  /** Unrestricted Damerau-Levenshtein edits, including repeated transpositions. */
  | "damerau-levenshtein";

/** Ordering contract for a lazy query cursor. */
export type QueryOrder = "traversal" | "distance-then-term";

/**
 * One dictionary term with an explicit unit domain.
 *
 * Strings contain Unicode scalar values, `Uint8Array` preserves arbitrary
 * bytes, and `BigUint64Array` preserves the complete unsigned 64-bit token
 * domain. No representation reserves an input value as a sentinel.
 */
export type Term =
  | {
      /** Select the Unicode-scalar representation. */
      readonly domain: "unicode";
      /** Matched Unicode string. */
      readonly value: string;
    }
  | {
      /** Select the raw-byte representation. */
      readonly domain: "byte";
      /** Matched bytes, including embedded zero values. */
      readonly value: Uint8Array;
    }
  | {
      /** Select the unsigned 64-bit token representation. */
      readonly domain: "u64";
      /** Matched token sequence over the complete unsigned 64-bit domain. */
      readonly value: BigUint64Array;
    };

/** One owned query result that remains valid after its cursor advances. */
export interface Match {
  /** Matched dictionary term and its exact unit domain. */
  readonly term: Term;
  /** Edit distance under the transducer's selected algorithm. */
  readonly distance: number;
  /** Optional dictionary identifier; `null` means the entry has no value. */
  readonly id: bigint | null;
}

/**
 * Zero-copy result descriptor borrowed from one reducer callback.
 *
 * Every accessor throws after the callback returns. Materialize data during
 * the callback when it must escape that lease.
 */
export interface BorrowedMatch {
  /** Edit distance under the transducer's selected algorithm. */
  readonly distance: number;
  /** Optional dictionary identifier; `null` means the entry has no value. */
  readonly id: bigint | null;
  /** Unit domain selecting the valid term accessor. */
  readonly unitDomain: "byte" | "unicode" | "u64";
  /** Borrow the raw byte term; valid only for byte-domain results. */
  bytes(): Uint8Array;
  /** Decode and borrow the string term; valid only for Unicode results. */
  utf8(): string;
  /** Borrow the token term; valid only for unsigned 64-bit results. */
  u64(): BigUint64Array;
}

/** Zero-copy batch whose descriptors share one reducer-callback lease. */
export interface BorrowedBatch extends Iterable<BorrowedMatch> {
  /** Number of descriptors available in this batch. */
  readonly length: number;
}

/**
 * One-shot query over the immutable dictionary revision captured at creation.
 *
 * Iterate or reduce the cursor once and close it deterministically. Iterator
 * completion closes the native cursor; explicit `close()` is idempotent and is
 * required when traversal stops early.
 */
export interface QueryCursor extends IterableIterator<Match> {
  /**
   * Materialize at most `maximum` owned results.
   *
   * @throws `RangeError` when `maximum` violates the bounded-batch contract.
   * @throws `Error` when the native query reports a non-terminal status.
   */
  nextBatch(maximum: number): Match[];
  /**
   * Fold leased batches without allocating one owned `Match` per result.
   * Borrowed descriptors must not escape `reducer`.
   */
  reduceBatches<Accumulator>(
    reducer: (accumulator: Accumulator, batch: readonly Match[]) => Accumulator,
    initial: Accumulator,
  ): Accumulator;
  /** Release the native cursor; safe to call repeatedly. */
  close(): void;
}

/** State and transition counts of a compiled finite-state automaton. */
export interface AutomatonSize {
  /** Number of automaton states. */
  readonly states: number;
  /** Number of labeled transitions. */
  readonly transitions: number;
}

/** Reusable phonetic or LLRE regular-language automaton. */
export interface PhoneticPattern {
  /** Structural size of the compiled automaton. */
  readonly size: AutomatonSize;
  /** Return whether the complete input string belongs to this language. */
  matches(input: string): boolean;
  /** Release the compiled automaton; safe to call repeatedly. */
  close(): void;
}

/** Parsed phonetic rewrite rules that can be reused across inputs. */
export interface PhoneticRuleSet {
  /** Number of enabled rewrite rules. */
  readonly size: number;
  /** Rewrite `input` to the rule set's fixed point. */
  apply(input: string): string;
  /** Release the parsed rule set; safe to call repeatedly. */
  close(): void;
}

/** Reusable edit automaton retaining a dictionary resource in constant time. */
export interface Transducer {
  /** Start a lazy Unicode query and capture the dictionary revision now. */
  query(query: string, maximumDistance: number, order?: QueryOrder): QueryCursor;
  /** Start a lazy raw-byte query. Traversal ordering is allocation-bounded. */
  query(query: Uint8Array, maximumDistance: number, order?: QueryOrder): QueryCursor;
  /** Start a lazy unsigned 64-bit token query. */
  query(query: BigUint64Array, maximumDistance: number, order?: QueryOrder): QueryCursor;
  /** Start a dictionary-language product query using a compiled pattern. */
  query(pattern: PhoneticPattern, maximumDistance: number): QueryCursor;
  /** Release the retained dictionary and native transducer; idempotent. */
  close(): void;
}

/** TinyLFU/SIEVE policy counters and current bounded residency. */
export interface QueryCacheStats {
  readonly requests: bigint;
  readonly hits: bigint;
  readonly misses: bigint;
  readonly admissions: bigint;
  readonly rejections: bigint;
  readonly evictions: bigint;
  readonly residentEntries: number;
  readonly residentWeight: number;
}

/** Hard bounds for each result-order shard of a query cache. */
export interface QueryCacheOptions {
  readonly maximumEntries?: number;
  readonly maximumWeight?: number;
}

/** Exclusive, synchronization-free cache for complete repeated query results. */
export interface QueryCache {
  /** Aggregate policy counters and current residency for both order shards. */
  readonly stats: QueryCacheStats;
  /** Query Unicode text, exact bytes, or exact unsigned 64-bit tokens. */
  query(
    query: string | Uint8Array | BigUint64Array,
    maximumDistance: number,
    order?: QueryOrder,
  ): QueryCursor;
  /** Drop resident results while preserving policy counters. */
  clear(): this;
  /** Reset counters while preserving residency and frequency state. */
  resetStats(): this;
  /** Release resident results and the retained transducer; idempotent. */
  close(): void;
}

/** Complete project facade exported by every supported JavaScript runtime. */
export interface LiblevenshteinNamespace {
  /** Opaque singleton identity used to reject resources from another runtime. */
  readonly runtimeIdentity: symbol | object;
  /** Retain `dictionary` and compile the selected edit semantics. */
  transducer(dictionary: DictionaryResource, algorithm?: Algorithm): Transducer;
  /** Retain a transducer behind a bounded snapshot-aware complete-result cache. */
  queryCache(transducer: Transducer, options?: QueryCacheOptions): QueryCache;
  /** Compile the phonetic regular-expression language. */
  phoneticPattern(source: string): PhoneticPattern;
  /** Compile an import-free LLRE document. */
  llrePattern(source: string): PhoneticPattern;
  /** Parse an import-free LLEV document or load a built-in rule set. */
  phoneticRules(source: string | "english-orthography" | "english-phonetic"): PhoneticRuleSet;
}

/** Opaque singleton identity shared by all facades backed by this runtime. */
export const runtimeIdentity: symbol | object;
/** Retain a dictionary and compile a reusable edit transducer. */
export function transducer(dictionary: DictionaryResource, algorithm?: Algorithm): Transducer;
/** Create an exclusive bounded cache for repeated queries. */
export function queryCache(transducer: Transducer, options?: QueryCacheOptions): QueryCache;
/** Compile the phonetic regular-expression language. */
export function phoneticPattern(source: string): PhoneticPattern;
/** Compile an import-free LLRE document. */
export function llrePattern(source: string): PhoneticPattern;
/** Parse an import-free LLEV document or load a built-in rule set. */
export function phoneticRules(source: string | "english-orthography" | "english-phonetic"): PhoneticRuleSet;
/** Complete liblevenshtein namespace backed by the singleton shared runtime. */
declare const liblevenshtein: LiblevenshteinNamespace;
export default liblevenshtein;
