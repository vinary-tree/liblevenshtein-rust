import { liblevenshtein } from "@vinary-tree/javascript-runtime/wasi";
import { assertDictionaryResource, assertSameRuntime } from "@vinary-tree/vinary-tree-interop";
export const runtimeIdentity = liblevenshtein.runtimeIdentity;
export function transducer(dictionary, algorithm) {
  assertDictionaryResource(dictionary);
  assertSameRuntime(dictionary, runtimeIdentity);
  return liblevenshtein.transducer(dictionary, algorithm);
}
export const queryCache = liblevenshtein.queryCache.bind(liblevenshtein);
// The wasi build profile (--no-default-features --features wasi) has no
// phonetic capability, so these namespace members may be absent; binding
// them unconditionally crashed this module at import time (defect found by
// the cross-language benchmark harness, 2026-08-13). Capability-gated
// members stay undefined when the runtime does not provide them.
export const phoneticPattern = liblevenshtein.phoneticPattern?.bind(liblevenshtein);
export const llrePattern = liblevenshtein.llrePattern?.bind(liblevenshtein);
export const phoneticRules = liblevenshtein.phoneticRules?.bind(liblevenshtein);
export default { ...liblevenshtein, runtimeIdentity, transducer, queryCache, phoneticPattern, llrePattern, phoneticRules };
