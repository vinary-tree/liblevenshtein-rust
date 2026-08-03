fn main() {
    #[cfg(feature = "protobuf")]
    {
        let mut config = prost_build::Config::new();
        // Protobuf enum values share their enclosing scope, so the conventional
        // prefix prevents cross-language name collisions. Keep that portable
        // schema spelling while suppressing the Rust-only generated-code lint.
        config.enum_attribute(
            ".liblevenshtein.operations.OperationApplicabilityV1",
            "#[allow(clippy::enum_variant_names)]",
        );
        config
            .compile_protos(
                &["proto/liblevenshtein.proto", "proto/operation_set.proto"],
                &["proto/"],
            )
            .expect("Failed to compile protobuf definitions");
    }
}
