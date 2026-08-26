// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Liblevenshtein",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "Liblevenshtein", targets: ["Liblevenshtein"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/vinary-tree/vinary-tree-interop.git",
            exact: "4.0.0-rc.5"
        ),
    ],
    targets: [
        .systemLibrary(
            name: "CLiblevenshtein",
            path: "bindings/swift/liblevenshtein/Sources/CLiblevenshtein"
        ),
        .target(
            name: "Liblevenshtein",
            dependencies: [
                "CLiblevenshtein",
                .product(name: "VinaryTreeInterop", package: "vinary-tree-interop"),
            ],
            path: "bindings/swift/liblevenshtein/Sources/Liblevenshtein"
        ),
    ]
)
