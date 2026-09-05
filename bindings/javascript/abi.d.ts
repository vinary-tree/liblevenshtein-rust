/** Generated from bindings/api.json; do not edit numeric values manually. */
export const ABI_VERSION: 1;
export const API_REVISION: 4;
export const DEFAULT_MATCH_BATCH: 256;
export enum Status {
  Ok = 0,
  End = 1,
  InvalidArgument = 2,
  InvalidUtf8 = 3,
  NullPointer = 4,
  Panic = 5,
  Unsupported = 6,
  IoError = 7,
  Closed = 8,
  LimitExceeded = 9,
  ProviderError = 10,
  BatchInUse = 11,
  DomainMismatch = 12,
}
export enum Algorithm {
  Standard = 0,
  Transposition = 1,
  MergeAndSplit = 2,
  DamerauLevenshtein = 3,
}
export enum Queryorder {
  Traversal = 0,
  DistanceThenTerm = 1,
}
export enum Phoneticrulesetkind {
  EnglishOrthography = 0,
  EnglishPhonetic = 1,
}
