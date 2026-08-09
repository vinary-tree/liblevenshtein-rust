import type { DictionaryResource } from "@vinary-tree/interop";

export type Algorithm =
  | "standard"
  | "transposition"
  | "merge-and-split"
  | "damerau-levenshtein";
export type QueryOrder = "traversal" | "distance-then-term";

export type Term =
  | { readonly domain: "unicode"; readonly value: string }
  | { readonly domain: "byte"; readonly value: Uint8Array }
  | { readonly domain: "u64"; readonly value: BigUint64Array };

export interface Match {
  readonly term: Term;
  readonly distance: number;
  readonly id: bigint | null;
}

export interface BorrowedMatch {
  readonly distance: number;
  readonly id: bigint | null;
  readonly unitDomain: "byte" | "unicode" | "u64";
  bytes(): Uint8Array;
  utf8(): string;
  u64(): BigUint64Array;
}

export interface BorrowedBatch extends Iterable<BorrowedMatch> {
  readonly length: number;
}

export interface QueryCursor extends IterableIterator<Match> {
  nextBatch(maximum: number): Match[];
  reduceBatches<Accumulator>(
    reducer: (accumulator: Accumulator, batch: readonly Match[]) => Accumulator,
    initial: Accumulator,
  ): Accumulator;
  close(): void;
}

export interface AutomatonSize {
  readonly states: number;
  readonly transitions: number;
}

export interface PhoneticPattern {
  readonly size: AutomatonSize;
  matches(input: string): boolean;
  close(): void;
}

export interface PhoneticRuleSet {
  readonly size: number;
  apply(input: string): string;
  close(): void;
}

export interface Transducer {
  query(query: string, maximumDistance: number, order?: QueryOrder): QueryCursor;
  query(query: Uint8Array, maximumDistance: number): QueryCursor;
  query(query: BigUint64Array, maximumDistance: number): QueryCursor;
  query(pattern: PhoneticPattern, maximumDistance: number): QueryCursor;
  close(): void;
}

export interface LiblevenshteinNamespace {
  readonly runtimeIdentity: symbol | object;
  transducer(dictionary: DictionaryResource, algorithm?: Algorithm): Transducer;
  phoneticPattern(source: string): PhoneticPattern;
  llrePattern(source: string): PhoneticPattern;
  phoneticRules(source: string | "english-orthography" | "english-phonetic"): PhoneticRuleSet;
}

export const runtimeIdentity: symbol | object;
export function transducer(dictionary: DictionaryResource, algorithm?: Algorithm): Transducer;
export function phoneticPattern(source: string): PhoneticPattern;
export function llrePattern(source: string): PhoneticPattern;
export function phoneticRules(source: string | "english-orthography" | "english-phonetic"): PhoneticRuleSet;
declare const liblevenshtein: LiblevenshteinNamespace;
export default liblevenshtein;
