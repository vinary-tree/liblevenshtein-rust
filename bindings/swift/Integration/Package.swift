// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftBindingIntegration",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../liblevenshtein"),
        .package(path: "../../../../libdictenstein/bindings/swift/libdictenstein"),
    ],
    targets: [
        .executableTarget(
            name: "SwiftBindingIntegration",
            dependencies: [
                .product(name: "Liblevenshtein", package: "liblevenshtein"),
                .product(name: "Libdictenstein", package: "libdictenstein"),
            ]
        ),
    ]
)
