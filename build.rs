fn main() {
    #[cfg(feature = "protobuf")]
    {
        use std::{env, fs, path::PathBuf, process::Command};

        const GENERATED: [&str; 2] = ["liblevenshtein.proto.rs", "liblevenshtein.operations.rs"];
        let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
        let checked = PathBuf::from("proto/generated");
        println!("cargo:rerun-if-changed=proto/liblevenshtein.proto");
        println!("cargo:rerun-if-changed=proto/operation_set.proto");
        for file in GENERATED {
            println!("cargo:rerun-if-changed={}", checked.join(file).display());
        }

        // Many language and cross-platform jobs consume protobuf formats but
        // do not install protoc. Keep the generated Rust sources in the source
        // archive and verify them against prost-build wherever protoc exists.
        let have_protoc = env::var_os("PROTOC").is_some()
            || Command::new("protoc")
                .arg("--version")
                .output()
                .is_ok_and(|result| result.status.success());
        if !have_protoc {
            for file in GENERATED {
                fs::copy(checked.join(file), output.join(file))
                    .expect("Failed to copy checked-in protobuf bindings");
            }
            return;
        }

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
        for file in GENERATED {
            assert_eq!(
                fs::read(output.join(file)).expect("Generated protobuf bindings are missing"),
                fs::read(checked.join(file)).expect("Checked-in protobuf bindings are missing"),
                "protobuf bindings are stale: regenerate proto/generated/{file}",
            );
        }
    }
}
