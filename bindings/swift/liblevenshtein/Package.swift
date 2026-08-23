// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "liblevenshtein",
    platforms: [.macOS(.v13)],
    products: [.library(name: "Liblevenshtein", targets: ["Liblevenshtein"])],
    dependencies: [
        .package(
            url: "https://github.com/vinary-tree/vinary-tree-interop.git",
            exact: "4.0.0-rc.1"
        ),
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
