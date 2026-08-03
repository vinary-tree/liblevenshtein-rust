#!/usr/bin/env bash
# Reproducible code-generation witness for the monomorphized Standard variant.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root
readonly artifact_dir="$repo_root/target/phase5-codegen"
readonly probe_binary="$repo_root/target/release/examples/position_kind_codegen_probe"
readonly symbol_name="liblevenshtein_phase5_transition_standard_probe"
readonly native_rustflags="-C target-cpu=native -C opt-level=3"

usage() {
    printf '%s\n' \
        'usage:' \
        '  scripts/check-unit-cost-zero-cost.sh capture LABEL' \
        '  scripts/check-unit-cost-zero-cost.sh compare BASELINE_LABEL CANDIDATE_LABEL' \
        '  scripts/check-unit-cost-zero-cost.sh audit LABEL'
}

validate_label() {
    local label="$1"
    if [[ ! "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        printf 'invalid artifact label: %s\n' "$label" >&2
        exit 2
    fi
}

capture_symbol() {
    local label="$1"
    local symbol_address symbol_size symbol_type parsed_name
    local text_address text_offset
    local symbol_address_dec symbol_size_dec text_address_dec text_offset_dec file_offset
    local ir_path candidate
    local -a ir_files

    validate_label "$label"
    mkdir -p "$artifact_dir"

    (
        cd "$repo_root"
        RUSTFLAGS="$native_rustflags" \
            cargo build --release --example position_kind_codegen_probe
    )

    read -r symbol_address symbol_size symbol_type parsed_name < <(
        nm -S --defined-only "$probe_binary" |
            awk -v wanted="$symbol_name" '$4 == wanted { print $1, $2, $3, $4; exit }'
    )
    if [[ "$parsed_name" != "$symbol_name" || "$symbol_type" != "T" ]]; then
        printf 'exported probe symbol not found: %s\n' "$symbol_name" >&2
        exit 1
    fi

    read -r text_address text_offset < <(
        readelf -SW "$probe_binary" |
            awk '$2 == ".text" { print $4, $5; exit }'
    )
    if [[ -z "$text_address" || -z "$text_offset" ]]; then
        printf 'ELF .text section not found in %s\n' "$probe_binary" >&2
        exit 1
    fi

    symbol_address_dec=$((16#$symbol_address))
    symbol_size_dec=$((16#$symbol_size))
    text_address_dec=$((16#$text_address))
    text_offset_dec=$((16#$text_offset))
    file_offset=$((symbol_address_dec - text_address_dec + text_offset_dec))

    dd if="$probe_binary" \
        of="$artifact_dir/$label.bin" \
        bs=1 skip="$file_offset" count="$symbol_size_dec" status=none
    objdump -d --disassemble="$symbol_name" "$probe_binary" \
        > "$artifact_dir/$label.asm"

    # The exact-byte witness deliberately remains separate from the dispatch
    # audit. Optimized LLVM IR retains enough inlining provenance to prove that
    # the constant-Standard probe selected only the Standard leaf, while
    # ignoring intentional Position layout and ownership instructions.
    (
        cd "$repo_root"
        RUSTFLAGS="$native_rustflags" \
            cargo rustc --release --example position_kind_codegen_probe -- \
                --emit=llvm-ir
    )
    mapfile -t ir_files < <(
        find "$repo_root/target/release/examples" -maxdepth 1 -type f \
            -name 'position_kind_codegen_probe-*.ll' -print
    )
    if (( ${#ir_files[@]} == 0 )); then
        printf 'optimized LLVM IR not found for the probe\n' >&2
        exit 1
    fi
    ir_path="${ir_files[0]}"
    for candidate in "${ir_files[@]:1}"; do
        if [[ "$candidate" -nt "$ir_path" ]]; then
            ir_path="$candidate"
        fi
    done
    awk -v symbol="$symbol_name" '
        !inside && index($0, "@" symbol "(") { inside = 1 }
        inside { print }
        inside && /^}$/ { exit }
    ' "$ir_path" > "$artifact_dir/$label.ll"
    if [[ ! -s "$artifact_dir/$label.ll" ]]; then
        printf 'exported probe function not found in optimized LLVM IR: %s\n' \
            "$symbol_name" >&2
        exit 1
    fi

    printf '%s bytes  ' "$symbol_size_dec"
    sha256sum "$artifact_dir/$label.bin"
}

audit_dispatch_specialization() {
    local label="$1"
    local ir="$artifact_dir/$label.ll"

    validate_label "$label"
    if [[ ! -s "$ir" ]]; then
        printf 'missing optimized LLVM IR artifact: %s\n' "$ir" >&2
        exit 1
    fi

    # The probe has no Algorithm parameter. After optimization its inlining
    # provenance must contain the selected Standard leaf, no runtime selector,
    # and no other variant leaf. A surviving LLVM `switch` would independently
    # falsify the closed-selector-erasure claim.
    if ! grep -q 'transition_standard' "$ir"; then
        printf 'FAIL: optimized probe does not contain the Standard leaf provenance.\n' >&2
        exit 1
    fi
    if grep -Eq 'transition_(position|transposition|merge_split)|OsaV|MergeSplitV' "$ir"; then
        printf 'FAIL: optimized probe retains runtime or non-Standard variant provenance.\n' >&2
        exit 1
    fi
    if grep -Eq '^[[:space:]]*switch[[:space:]]' "$ir"; then
        printf 'FAIL: optimized probe retains an LLVM switch.\n' >&2
        exit 1
    fi

    printf '%s\n' \
        'PASS: optimized Standard probe contains only Standard leaf provenance;' \
        '      no runtime selector, non-Standard leaf, or LLVM switch survives.'
}

case "${1:-}" in
    capture)
        if [[ $# -ne 2 ]]; then
            usage >&2
            exit 2
        fi
        capture_symbol "$2"
        ;;
    compare)
        if [[ $# -ne 3 ]]; then
            usage >&2
            exit 2
        fi
        validate_label "$2"
        capture_symbol "$3"
        if cmp -s "$artifact_dir/$2.bin" "$artifact_dir/$3.bin"; then
            printf 'PASS: exact Standard probe bytes are identical (%s == %s).\n' "$2" "$3"
        else
            printf 'FAIL: Standard probe bytes differ (%s != %s).\n' "$2" "$3" >&2
            cmp -l "$artifact_dir/$2.bin" "$artifact_dir/$3.bin" | head -n 20 >&2 || true
            exit 1
        fi
        ;;
    audit)
        if [[ $# -ne 2 ]]; then
            usage >&2
            exit 2
        fi
        capture_symbol "$2"
        audit_dispatch_specialization "$2"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
