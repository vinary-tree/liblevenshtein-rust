// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VinaryTreeLiblevenshtein",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "VinaryTreeInterop", targets: ["VinaryTreeInterop"]),
        .library(name: "Liblevenshtein", targets: ["Liblevenshtein"]),
    ],
    targets: [
        .systemLibrary(
            name: "CVinaryTreeInterop",
            path: "vinary-tree-interop/bindings/swift/vinary-tree-interop/Sources/CVinaryTreeInterop"
        ),
        .target(
            name: "VinaryTreeInterop",
            dependencies: ["CVinaryTreeInterop"],
            path: "vinary-tree-interop/bindings/swift/vinary-tree-interop/Sources/VinaryTreeInterop"
        ),
        .systemLibrary(
            name: "CLiblevenshtein",
            path: "bindings/swift/liblevenshtein/Sources/CLiblevenshtein"
        ),
        .target(
            name: "Liblevenshtein",
            dependencies: ["CLiblevenshtein", "VinaryTreeInterop"],
            path: "bindings/swift/liblevenshtein/Sources/Liblevenshtein"
        ),
    ]
)
