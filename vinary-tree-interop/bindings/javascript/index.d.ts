/** Identity of one native, WASM, or WASI umbrella runtime instance. */
export type RuntimeIdentity = symbol | object;

/** Unit domains understood by vt.dictionary.v1. */
export type UnitDomain = "byte" | "unicode" | "u64";

/** Value domains understood by vt.dictionary.v1. */
export type ValueDomain = "unit" | "optional-u64";

/** Base contract implemented by every retained Vinary Tree resource. */
export interface Resource {
  readonly runtimeIdentity: RuntimeIdentity;
  readonly interfaceId: string;
  close(): void;
}

/** Opaque dictionary resource produced by libdictenstein or a host provider. */
export interface DictionaryResource extends Resource {
  readonly interfaceId: "vt.dictionary.v1";
  readonly unitDomain: UnitDomain;
  readonly valueDomain: ValueDomain;
}

/** Scalar semirings supported by vt.scalar-wfst.1. */
export type WeightDomain =
  | "tropical-f64"
  | "log-f64"
  | "probability-f64"
  | "arctic-f64"
  | "signed-tropical-f64"
  | "count-f64"
  | "boolean-f64";

/** Opaque scalar WFST resource produced by lling-llang or duallity. */
export interface WfstResource extends Resource {
  readonly interfaceId: "vt.scalar-wfst.1";
  readonly weightDomain: WeightDomain;
}

/** Rejects handles created by a different native/WASM/WASI runtime instance. */
export function assertSameRuntime(
  resource: Resource,
  expected: RuntimeIdentity,
): void;

/** Narrows a generic resource to the version-1 dictionary interface. */
export function assertDictionaryResource(resource: Resource): asserts resource is DictionaryResource;

/** Narrows a generic resource to the version-1 scalar WFST interface. */
export function assertWfstResource(resource: Resource): asserts resource is WfstResource;
