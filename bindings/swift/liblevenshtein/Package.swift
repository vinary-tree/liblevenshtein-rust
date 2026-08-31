// swift-tools-version: 6.0
import PackageDescription

let interopDependency: Package.Dependency = if let localRoot = Context.environment["VINARY_TREE_INTEROP_ROOT"] {
    .package(path: localRoot)
} else {
    .package(
        url: "https://github.com/vinary-tree/vinary-tree-interop.git",
        exact: "4.0.0-rc.6"
    )
}

let package = Package(
    name: "liblevenshtein",
    platforms: [.macOS(.v13)],
    products: [.library(name: "Liblevenshtein", targets: ["Liblevenshtein"])],
    dependencies: [interopDependency],
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
