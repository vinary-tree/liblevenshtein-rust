// swift-tools-version: 6.0
import PackageDescription

let libdictensteinDependency: Package.Dependency = if let localRoot =
    Context.environment["LIBDICTENSTEIN_ROOT"]
{
    .package(path: "\(localRoot)/bindings/swift/libdictenstein")
} else {
    .package(path: "../../../../libdictenstein/bindings/swift/libdictenstein")
}

let package = Package(
    name: "SwiftBindingIntegration",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(path: "../liblevenshtein"),
        libdictensteinDependency,
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
