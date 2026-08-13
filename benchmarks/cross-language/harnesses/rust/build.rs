use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let output = Command::new(&rustc)
        .arg("--version")
        .output()
        .expect("failed to invoke rustc --version");
    let version = String::from_utf8(output.stdout)
        .expect("rustc --version produced non-UTF-8 output");
    println!("cargo:rustc-env=BENCH_RUSTC_VERSION={}", version.trim());

    // The liblevenshtein package version, read from the repo-root manifest.
    // The [package] section leads the file, so the first standalone
    // `version = "…"` line is the package version, not a dependency's.
    let manifest = std::fs::read_to_string("../../../../Cargo.toml")
        .expect("failed to read repo-root Cargo.toml");
    let version = manifest
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.strip_prefix("version = \"")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .expect("repo-root Cargo.toml has no package version line");
    println!("cargo:rustc-env=BENCH_LIBLEV_VERSION={version}");
    println!("cargo:rerun-if-changed=../../../../Cargo.toml");
}
