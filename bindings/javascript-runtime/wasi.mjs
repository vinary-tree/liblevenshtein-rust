import { createWasiRuntime } from "./wasi-runtime.mjs";

const runtime = await createWasiRuntime();
export { createWasiRuntime };
export const runtimeIdentity = runtime.runtimeIdentity;
export const libdictenstein = runtime.libdictenstein;
export const liblevenshtein = runtime.liblevenshtein;
export const llingLlang = runtime.llingLlang;
export const duallity = runtime.duallity;
export default runtime;
