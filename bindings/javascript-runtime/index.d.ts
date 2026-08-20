import type { DictionaryResource, RuntimeIdentity, UnitDomain } from "@vinary-tree/interop";

export type Algorithm = "standard" | "transposition" | "merge-and-split" | "damerau-levenshtein";
export type QueryOrder = "traversal" | "distance-then-term";
export type DictionaryValue = bigint | null;
export type DictionaryKey = string | Uint8Array | BigUint64Array;
export type DictionaryEntry = readonly [DictionaryKey, DictionaryValue];
export interface Lookup { readonly found: boolean; readonly value: DictionaryValue; }
export type Term =
  | { readonly domain: "unicode"; readonly value: string }
  | { readonly domain: "byte"; readonly value: Uint8Array }
  | { readonly domain: "u64"; readonly value: BigUint64Array };
export interface Match { readonly term: Term; readonly distance: number; readonly id: bigint | null; }

export interface DictionaryEntryCursor extends IterableIterator<DictionaryEntry> {
  readonly size: number;
  readonly identity: Readonly<{ producer: bigint; revision: bigint }> | null;
  nextBatch(maximum: number): DictionaryEntry[];
  reduceBatches<A>(
    reducer: (accumulator: A, batch: readonly DictionaryEntry[]) => A,
    initial: A,
    batchSize?: number,
  ): A;
  close(): void;
  [Symbol.dispose](): void;
}
export interface Dictionary extends DictionaryResource, Iterable<DictionaryEntry> {
  readonly size: number;
  put(term: DictionaryKey, value?: DictionaryValue): boolean;
  putU64(term: BigUint64Array, value?: DictionaryValue): boolean;
  set(term: DictionaryKey, value?: DictionaryValue): this;
  remove(term: DictionaryKey): boolean;
  removeU64(term: BigUint64Array): boolean;
  delete(term: DictionaryKey): boolean;
  has(term: DictionaryKey): boolean;
  hasU64(term: BigUint64Array): boolean;
  lookup(term: DictionaryKey): Lookup;
  lookupU64(term: BigUint64Array): Lookup;
  get(term: DictionaryKey): DictionaryValue | undefined;
  getU64(term: BigUint64Array): DictionaryValue | undefined;
  snapshotEntries(): readonly DictionaryEntry[];
  entries(): IterableIterator<DictionaryEntry>;
  keys(): IterableIterator<DictionaryKey>;
  values(): IterableIterator<DictionaryValue>;
  streamEntries(): DictionaryEntryCursor;
  forEach(callback: (value: DictionaryValue, key: DictionaryKey, dictionary: Dictionary) => void, thisArg?: unknown): void;
  toMap(): Map<string, DictionaryValue>;
  clear(): void;
  compact(): number;
  containsSubstring(term: string): boolean;
  substringFrequency(term: string): number;
  close(): void;
  [Symbol.dispose](): void;
}

export interface QueryCursor extends IterableIterator<Match> {
  nextBatch(maximum: number): Match[];
  reduceBatches<A>(reducer: (accumulator: A, batch: readonly Match[]) => A, initial: A, batchSize?: number): A;
  close(): void;
  [Symbol.dispose](): void;
}
export interface AutomatonSize { readonly states: number; readonly transitions: number; }
export interface PhoneticPattern { readonly size: AutomatonSize; matches(input: string): boolean; close(): void; }
export interface PhoneticRuleSet { readonly size: number; apply(input: string): string; close(): void; }
export interface Transducer {
  query(input: string, maximumDistance: number, order?: QueryOrder): QueryCursor;
  query(input: Uint8Array | BigUint64Array | PhoneticPattern, maximumDistance: number): QueryCursor;
  close(): void;
}

export interface LibdictensteinNamespace {
  readonly runtimeIdentity: RuntimeIdentity;
  dynamicDawg(unitDomain?: UnitDomain): Dictionary;
  doubleArrayTrie(entries: readonly { term: string; value?: DictionaryValue }[], unitDomain?: "byte" | "unicode"): Dictionary;
  scdawg(unitDomain?: "byte" | "unicode"): Dictionary;
}
export interface LiblevenshteinNamespace {
  readonly runtimeIdentity: RuntimeIdentity;
  transducer(dictionary: DictionaryResource, algorithm?: Algorithm): Transducer;
  phoneticPattern(source: string): PhoneticPattern;
  llrePattern(source: string): PhoneticPattern;
  phoneticRules(source: string): PhoneticRuleSet;
  levenshteinDistance(source: string, target: string): number;
  levenshteinDistanceThreshold(source: string, target: string, maximum: number): number | undefined;
  damerauDistance(source: string, target: string): number;
  damerauDistanceThreshold(source: string, target: string, maximum: number): number | undefined;
  trueDamerauDistance(source: string, target: string): number;
  trueDamerauDistanceThreshold(source: string, target: string, maximum: number): number | undefined;
}

export type WeightDomain =
  | "tropical-f64" | "log-f64" | "probability-f64" | "arctic-f64"
  | "signed-tropical-f64" | "count-f64" | "boolean-f64";
export interface WfstArc {
  readonly input: string | null;
  readonly output: string | null;
  readonly target: bigint;
  readonly weight: number;
}
export interface WfstState {
  readonly valid: boolean;
  readonly final: boolean;
  readonly finalWeight: number;
  readonly arcs: readonly WfstArc[];
}
export interface Wfst {
  readonly interfaceId: "vt.scalar-wfst.1";
  readonly runtimeIdentity: RuntimeIdentity;
  readonly weightDomain: WeightDomain;
  start(): bigint;
  state(state: bigint): WfstState;
  close(): void;
}
export interface WfstBuilder {
  addState(): number;
  setStart(state: number): void;
  setFinal(state: number, weight: number): void;
  addArc(from: number, input: string | null, output: string | null, to: number, weight: number): void;
  build(): Wfst;
  close(): void;
}
export interface LlingLlangNamespace {
  readonly runtimeIdentity: RuntimeIdentity;
  vectorWfst(): WfstBuilder;
  compose(first: Wfst, second: Wfst): Wfst;
}
export type DuallityWfstKind =
  | "levenshtein"
  | "universal-standard" | "universal-transposition" | "universal-merge-and-split"
  | "generalized-standard" | "generalized-transposition"
  | "generalized-merge-and-split" | "generalized-phonetic" | "fzf";
export interface DuallityNamespace {
  readonly runtimeIdentity: RuntimeIdentity;
  wfst(dictionary: DictionaryResource, query: string, maximumDistance: number,
       algorithm?: Algorithm, kind?: DuallityWfstKind): Wfst;
}

export const runtimeIdentity: RuntimeIdentity;
export const libdictenstein: LibdictensteinNamespace;
export const liblevenshtein: LiblevenshteinNamespace;
export const llingLlang: LlingLlangNamespace;
export const duallity: DuallityNamespace;
declare const runtime: {
  readonly runtimeIdentity: RuntimeIdentity;
  readonly libdictenstein: LibdictensteinNamespace;
  readonly liblevenshtein: LiblevenshteinNamespace;
  readonly llingLlang: LlingLlangNamespace;
  readonly duallity: DuallityNamespace;
};
export default runtime;
