// swift-tools-version: 6.0
// Swift harness for the cross-language benchmark program
// (harnesses/common/PROTOCOL.md). Mirrors bindings/swift/Integration:
// path dependencies on the liblevenshtein Swift facade and the sibling
// libdictenstein Swift facade (vinary-tree-interop arrives transitively).
// The native cdylibs resolve at build time via LIBRARY_PATH and at run time
// via LD_LIBRARY_PATH (the runner points both at the RELEASE target dirs).
import PackageDescription

let package = Package(
    name: "bench-cross-swift",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../../../../bindings/swift/liblevenshtein"),
        .package(path: "../../../../../libdictenstein/bindings/swift/libdictenstein"),
    ],
    targets: [
        .executableTarget(
            name: "bench",
            dependencies: [
                .product(name: "Liblevenshtein", package: "liblevenshtein"),
                .product(name: "Libdictenstein", package: "libdictenstein"),
            ]
        ),
    ]
)
