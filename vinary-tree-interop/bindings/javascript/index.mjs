export function assertSameRuntime(resource, expected) {
  if (resource == null || resource.runtimeIdentity !== expected) {
    throw new TypeError("resource belongs to a different Vinary Tree runtime");
  }
}

export function assertDictionaryResource(resource) {
  if (resource == null || resource.interfaceId !== "vt.dictionary.v1") {
    throw new TypeError("resource does not implement vt.dictionary.v1");
  }
}

export function assertWfstResource(resource) {
  if (resource == null || resource.interfaceId !== "vt.scalar-wfst.1") {
    throw new TypeError("resource does not implement vt.scalar-wfst.1");
  }
}
