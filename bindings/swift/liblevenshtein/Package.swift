// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "liblevenshtein",
    platforms: [.macOS(.v13)],
    products: [.library(name: "Liblevenshtein", targets: ["Liblevenshtein"])],
    dependencies: [
        .package(path: "../../../vinary-tree-interop/bindings/swift/vinary-tree-interop"),
    ],
    targets: [
        .systemLibrary(name: "CLiblevenshtein"),
        .target(
            name: "Liblevenshtein",
            dependencies: [
                "CLiblevenshtein",
                .product(name: "VinaryTreeInterop", package: "vinary-tree-interop"),
            ]
        ),
    ]
)
