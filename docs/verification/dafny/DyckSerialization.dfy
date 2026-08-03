// Auto-active obligations for exact Dyck correction and OperationSet binary decoding.

function ReplacementCost(actual: nat, expected: nat): nat
{
  if actual == expected then 0 else 1
}

lemma ReplacementCostZeroIffEqual(actual: nat, expected: nat)
  ensures ReplacementCost(actual, expected) == 0 <==> actual == expected
{
}

predicate TypedPair(kinds: nat, kind: nat, opening: nat, closing: nat)
{
  kind < kinds && opening == kind && closing == kinds + kind
}

lemma DifferentKindsHaveDifferentClosers(kinds: nat, left: nat, right: nat)
  requires left < kinds && right < kinds
  requires left != right
  ensures kinds + left != kinds + right
{
}

function Min4(first: nat, second: nat, third: nat, fourth: nat): nat
{
  var left := if first <= second then first else second;
  var right := if third <= fourth then third else fourth;
  if left <= right then left else right
}

lemma Min4IsNoGreaterThanEveryCandidate(
    first: nat, second: nat, third: nat, fourth: nat)
  ensures Min4(first, second, third, fourth) <= first
  ensures Min4(first, second, third, fourth) <= second
  ensures Min4(first, second, third, fourth) <= third
  ensures Min4(first, second, third, fourth) <= fourth
{
}

function PairFromFirstCost(
    actualOpen: nat, expectedOpen: nat, innerCost: nat,
    actualClose: nat, expectedClose: nat, suffixCost: nat): nat
{
  ReplacementCost(actualOpen, expectedOpen) + innerCost
    + ReplacementCost(actualClose, expectedClose) + suffixCost
}

lemma ZeroCostConsumedPairIsTypedIdentity(
    actualOpen: nat, expectedOpen: nat, innerCost: nat,
    actualClose: nat, expectedClose: nat, suffixCost: nat)
  requires PairFromFirstCost(
    actualOpen, expectedOpen, innerCost,
    actualClose, expectedClose, suffixCost) == 0
  ensures actualOpen == expectedOpen
  ensures actualClose == expectedClose
  ensures innerCost == 0
  ensures suffixCost == 0
{
  ReplacementCostZeroIffEqual(actualOpen, expectedOpen);
  ReplacementCostZeroIffEqual(actualClose, expectedClose);
}

predicate AcceptEnvelope(
    magicOk: bool, version: nat, flags: nat,
    declaredBytes: nat, availableBytes: nat, consumedBytes: nat,
    payloadLimit: nat, operations: nat, operationLimit: nat,
    nameBytes: nat, nameLimit: nat,
    pairs: nat, pairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
{
  magicOk
  && version == 1
  && flags == 0
  && declaredBytes == availableBytes
  && consumedBytes == declaredBytes
  && declaredBytes <= payloadLimit
  && operations <= operationLimit
  && nameBytes <= nameLimit
  && pairs <= pairLimit
  && pairTextBytes <= pairTextLimit
  && semanticValidation
}

lemma AcceptedEnvelopeIsStrictAndBounded(
    magicOk: bool, version: nat, flags: nat,
    declaredBytes: nat, availableBytes: nat, consumedBytes: nat,
    payloadLimit: nat, operations: nat, operationLimit: nat,
    nameBytes: nat, nameLimit: nat,
    pairs: nat, pairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
  requires AcceptEnvelope(
    magicOk, version, flags,
    declaredBytes, availableBytes, consumedBytes,
    payloadLimit, operations, operationLimit,
    nameBytes, nameLimit, pairs, pairLimit,
    pairTextBytes, pairTextLimit, semanticValidation)
  ensures magicOk && version == 1 && flags == 0
  ensures consumedBytes == availableBytes
  ensures declaredBytes <= payloadLimit
  ensures operations <= operationLimit
  ensures nameBytes <= nameLimit
  ensures pairs <= pairLimit
  ensures pairTextBytes <= pairTextLimit
  ensures semanticValidation
{
}

lemma TrailingBytesCannotBeAccepted(
    magicOk: bool, version: nat, flags: nat,
    declaredBytes: nat, availableBytes: nat, consumedBytes: nat,
    payloadLimit: nat, operations: nat, operationLimit: nat,
    nameBytes: nat, nameLimit: nat,
    pairs: nat, pairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
  requires availableBytes > declaredBytes
  ensures !AcceptEnvelope(
    magicOk, version, flags,
    declaredBytes, availableBytes, consumedBytes,
    payloadLimit, operations, operationLimit,
    nameBytes, nameLimit, pairs, pairLimit,
    pairTextBytes, pairTextLimit, semanticValidation)
{
}

lemma OverLimitCountsCannotBeAccepted(
    magicOk: bool, version: nat, flags: nat,
    declaredBytes: nat, availableBytes: nat, consumedBytes: nat,
    payloadLimit: nat, operations: nat, operationLimit: nat,
    nameBytes: nat, nameLimit: nat,
    pairs: nat, pairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
  requires operations > operationLimit || pairs > pairLimit
  ensures !AcceptEnvelope(
    magicOk, version, flags,
    declaredBytes, availableBytes, consumedBytes,
    payloadLimit, operations, operationLimit,
    nameBytes, nameLimit, pairs, pairLimit,
    pairTextBytes, pairTextLimit, semanticValidation)
{
}

predicate AcceptProtobuf(
    wireWellFormed: bool, supportedFormat: bool,
    payloadBytes: nat, payloadLimit: nat,
    operations: nat, operationLimit: nat,
    largestNameBytes: nat, nameLimit: nat,
    largestOperationPairs: nat, perOperationPairLimit: nat,
    totalPairs: nat, totalPairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
{
  wireWellFormed
  && supportedFormat
  && payloadBytes <= payloadLimit
  && operations <= operationLimit
  && largestNameBytes <= nameLimit
  && largestOperationPairs <= perOperationPairLimit
  && totalPairs <= totalPairLimit
  && pairTextBytes <= pairTextLimit
  && semanticValidation
}

lemma AcceptedProtobufIsPreflightBounded(
    wireWellFormed: bool, supportedFormat: bool,
    payloadBytes: nat, payloadLimit: nat,
    operations: nat, operationLimit: nat,
    largestNameBytes: nat, nameLimit: nat,
    largestOperationPairs: nat, perOperationPairLimit: nat,
    totalPairs: nat, totalPairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
  requires AcceptProtobuf(
    wireWellFormed, supportedFormat,
    payloadBytes, payloadLimit,
    operations, operationLimit,
    largestNameBytes, nameLimit,
    largestOperationPairs, perOperationPairLimit,
    totalPairs, totalPairLimit,
    pairTextBytes, pairTextLimit,
    semanticValidation)
  ensures wireWellFormed && supportedFormat
  ensures payloadBytes <= payloadLimit
  ensures operations <= operationLimit
  ensures largestNameBytes <= nameLimit
  ensures largestOperationPairs <= perOperationPairLimit
  ensures totalPairs <= totalPairLimit
  ensures pairTextBytes <= pairTextLimit
  ensures semanticValidation
{
}

lemma OverLimitProtobufCannotReachAllocation(
    wireWellFormed: bool, supportedFormat: bool,
    payloadBytes: nat, payloadLimit: nat,
    operations: nat, operationLimit: nat,
    largestNameBytes: nat, nameLimit: nat,
    largestOperationPairs: nat, perOperationPairLimit: nat,
    totalPairs: nat, totalPairLimit: nat,
    pairTextBytes: nat, pairTextLimit: nat,
    semanticValidation: bool)
  requires operations > operationLimit
    || largestOperationPairs > perOperationPairLimit
    || totalPairs > totalPairLimit
    || pairTextBytes > pairTextLimit
  ensures !AcceptProtobuf(
    wireWellFormed, supportedFormat,
    payloadBytes, payloadLimit,
    operations, operationLimit,
    largestNameBytes, nameLimit,
    largestOperationPairs, perOperationPairLimit,
    totalPairs, totalPairLimit,
    pairTextBytes, pairTextLimit,
    semanticValidation)
{
}

function EncodeWeightBits(bits: bv64): bv64
{
  bits
}

function DecodeWeightBits(bits: bv64): bv64
{
  bits
}

lemma WeightBitsRoundTripExactly(bits: bv64)
  ensures DecodeWeightBits(EncodeWeightBits(bits)) == bits
{
}

function DecodeU16Le(first: nat, second: nat): nat
{
  first + 256 * second
}

lemma VersionOneHeaderBytesDecodeLittleEndian()
  ensures DecodeU16Le(1, 0) == 1
  ensures DecodeU16Le(0, 1) == 256
{
}

lemma WireCursorAdvanceIsExact(start: nat, width: nat, total: nat)
  requires start <= total
  requires width <= total - start
  ensures start + width <= total
  ensures start + width - start == width
{
}

lemma BoundedVarintCursorStaysWithinInput(
    start: nat, consumed: nat, total: nat)
  requires start <= total
  requires 1 <= consumed <= 10
  requires consumed <= total - start
  ensures start < start + consumed <= total
{
}

lemma LengthDelimitedPartitionIsExact(
    prefixBytes: nat, payloadBytes: nat, suffixBytes: nat, totalBytes: nat)
  requires totalBytes == prefixBytes + payloadBytes + suffixBytes
  ensures prefixBytes + payloadBytes <= totalBytes
  ensures totalBytes - (prefixBytes + payloadBytes) == suffixBytes
{
}

predicate AcceptGzip(
    checksumValid: bool,
    compressedBytes: nat, compressedLimit: nat,
    decompressedBytes: nat, decompressedLimit: nat,
    consumedCompressedBytes: nat, suppliedBytes: nat,
    innerAccepted: bool)
{
  checksumValid
  && compressedBytes <= compressedLimit
  && decompressedBytes <= decompressedLimit
  && consumedCompressedBytes == suppliedBytes
  && innerAccepted
}

lemma TrailingOrOversizedGzipCannotBeAccepted(
    checksumValid: bool,
    compressedBytes: nat, compressedLimit: nat,
    decompressedBytes: nat, decompressedLimit: nat,
    consumedCompressedBytes: nat, suppliedBytes: nat,
    innerAccepted: bool)
  requires consumedCompressedBytes < suppliedBytes
    || compressedBytes > compressedLimit
    || decompressedBytes > decompressedLimit
  ensures !AcceptGzip(
    checksumValid,
    compressedBytes, compressedLimit,
    decompressedBytes, decompressedLimit,
    consumedCompressedBytes, suppliedBytes,
    innerAccepted)
{
}
