import CVinaryTreeInterop

/// A retained `vt.dictionary.v1` producer. Implementations borrow their
/// two-word handle for the duration of `body`; consumers retain it in O(1).
public protocol DictionaryResource: AnyObject, Sendable {
    var unitDomain: UnitDomain { get }
    func withVtResource<Result>(
        _ body: (UnsafePointer<VtResource>) throws -> Result
    ) rethrows -> Result
}

public enum UnitDomain: Sendable {
    case byte
    case unicodeScalar
    case u64

    public var cValue: VtUnitDomain {
        switch self {
        case .byte: VT_UNIT_DOMAIN_BYTE
        case .unicodeScalar: VT_UNIT_DOMAIN_UNICODE_SCALAR
        case .u64: VT_UNIT_DOMAIN_U64
        }
    }
}
