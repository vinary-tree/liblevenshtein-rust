// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "vinary-tree-interop",
    platforms: [.macOS(.v13)],
    products: [.library(name: "VinaryTreeInterop", targets: ["VinaryTreeInterop"])],
    targets: [
        .systemLibrary(name: "CVinaryTreeInterop"),
        .target(name: "VinaryTreeInterop", dependencies: ["CVinaryTreeInterop"]),
    ]
)
