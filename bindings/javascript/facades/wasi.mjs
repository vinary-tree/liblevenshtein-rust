import { liblevenshtein } from "@vinary-tree/vinary-tree/wasi";
import { assertDictionaryResource, assertSameRuntime } from "@vinary-tree/interop";
export const runtimeIdentity = liblevenshtein.runtimeIdentity;
export function transducer(dictionary, algorithm) {
  assertDictionaryResource(dictionary);
  assertSameRuntime(dictionary, runtimeIdentity);
  return liblevenshtein.transducer(dictionary, algorithm);
}
export const phoneticPattern = liblevenshtein.phoneticPattern.bind(liblevenshtein);
export const llrePattern = liblevenshtein.llrePattern.bind(liblevenshtein);
export const phoneticRules = liblevenshtein.phoneticRules.bind(liblevenshtein);
export default { ...liblevenshtein, runtimeIdentity, transducer, phoneticPattern, llrePattern, phoneticRules };
