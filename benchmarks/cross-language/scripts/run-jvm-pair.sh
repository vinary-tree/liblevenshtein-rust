#!/usr/bin/env bash
# Phase 1 driver: everything JVM. Stages release natives into the binding
# builds (mirroring ci.yml's staging step), generates the VerifyMain
# launchers run-one.sh uses for gate/construct/memory cells, runs the JMH
# query matrix one cell per invocation (JIT isolation via forks), converts
# JMH JSON into unified cells, and post-fills them.
#
# Usage:
#   run-jvm-pair.sh stage                       # natives + classpaths + launchers
#   run-jvm-pair.sh verify-full <results_dir>   # full-query exact twins
#   run-jvm-pair.sh jmh-cell <results_dir> <target> <backend> <algorithm>
#                        <distance> <queryset> <forks> <iterations>
#                                               # one exact twin + one JMH cell
#   run-jvm-pair.sh jmh <results_dir> [--quick] # all missing JMH query cells
#   run-jvm-pair.sh parity <results_dir>        # exact twins + shared 45-cell pair
#   run-jvm-pair.sh hypothesis <results_dir>    # the two H-J1 deciding cells at
#                                               # 3 forks x 17 iterations (>=51
#                                               # samples, pgmcp protocol)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"
LR="$(cd "$XL/../.." && pwd)"
LD_REPO="$(cd "$LR/../libdictenstein" && pwd)"
JVM="$XL/harnesses/jvm"
STAGE="$XL/.stage/jvm"
GRADLE_ARGS=(--no-daemon -PjavaToolchain=26 -PnativePlatforms=linux-x86_64)

log() { printf '[jvm-pair %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

ensure_environment() {
    local results_dir="$1"
    mkdir -p "$results_dir"
    if [ ! -f "$results_dir/environment.json" ]; then
        python3 "$SCRIPT_DIR/env-capture.py" "$results_dir" >&2
    fi
}

require_compiler_quiet() {
    [ "${XL_REQUIRE_COMPILER_QUIET:-0}" = "1" ] || return 0
    local active
    active="$({ pgrep -ax cargo || true; pgrep -ax rustc || true; pgrep -ax cargo-llvm-cov || true; } | sort -u)"
    if [ -n "$active" ]; then
        printf 'run-jvm-pair: compiler-load gate is red:\n%s\n' "$active" >&2
        return 3
    fi
}

cell_is_complete() {
    local cell="$1"
    [ -f "$cell" ] && jq -e '
        .status == "ok"
        and (.run_id | type == "string" and length > 0)
        and (.environment_ref | type == "string" and length > 0)
    ' "$cell" >/dev/null 2>&1
}

stage_natives() {
    # Mirror ci.yml's staging: the jar's verifyNativeResources gate requires
    # the platform natives under build/native-resources (release .so here).
    mkdir -p "$LR/bindings/jvm/build/native-resources/META-INF/native/linux-x86_64"
    cp "$LR/target/release/libliblevenshtein.so" \
       "$LR/bindings/jvm/build/native-resources/META-INF/native/linux-x86_64/"
    mkdir -p "$LD_REPO/bindings/jvm/build/native-resources/META-INF/native/linux-x86_64"
    cp "$LD_REPO/target/release/liblibdictenstein.so" \
       "$LD_REPO/bindings/jvm/build/native-resources/META-INF/native/linux-x86_64/"
}

write_launchers() {
    mkdir -p "$STAGE"
    (cd "$JVM" && ./gradlew "${GRADLE_ARGS[@]}" :vinary:writeClasspath :legacy:writeClasspath >&2)
    local vinary_cp legacy_cp
    vinary_cp="$JVM/vinary/build/runtime-classpath.txt"
    legacy_cp="$JVM/legacy/build/runtime-classpath.txt"
    [ -s "$vinary_cp" ] && [ -s "$legacy_cp" ] || {
        echo "run-jvm-pair: classpath files missing" >&2; exit 1; }

    cat > "$STAGE/vinary-launcher.sh" <<EOF
#!/usr/bin/env bash
exec java --enable-native-access=ALL-UNNAMED \\
    -Djava.library.path=$LR/target/release:$LD_REPO/target/release \\
    -Xms2g -Xmx2g \\
    -cp "\$(cat "$vinary_cp")" bench.VerifyMain "\$@"
EOF
    cat > "$STAGE/legacy-launcher.sh" <<EOF
#!/usr/bin/env bash
exec java -Xms2g -Xmx2g -cp "\$(cat "$legacy_cp")" bench.LegacyVerifyMain "\$@"
EOF
    chmod +x "$STAGE/vinary-launcher.sh" "$STAGE/legacy-launcher.sh"
    log "launchers written under $STAGE"
}

protobuf_containment_smoke() {
    # R2: prove the query path never classloads com.google.protobuf.
    local smoke_dir="$STAGE/protobuf-smoke"
    mkdir -p "$smoke_dir"
    local log_file="$smoke_dir/verbose-class.log"
    java -verbose:class -Xms512m -Xmx512m \
        -cp "$(cat "$JVM/legacy/build/runtime-classpath.txt")" bench.LegacyVerifyMain \
        --mode verify --algorithm transposition --max-distance 2 \
        --dictionary "$XL/workload/dictionary.txt" \
        --queries "$XL/workload/queries/tr-d2.txt" \
        --backend own --gate-limit 50 \
        --out "$smoke_dir/smoke.json" > "$log_file" 2>&1
    if grep -q 'com\.google\.protobuf' "$log_file"; then
        echo "run-jvm-pair: FAIL — protobuf classes loaded on the transduce path" >&2
        grep 'com\.google\.protobuf' "$log_file" | head -5 >&2
        exit 1
    fi
    log "protobuf containment: no com.google.protobuf classload on the transduce path"
}

runtime_version() {
    java --version 2>&1 | head -1
}

# jmh_cell <results_dir> <target> <backend> <algorithm> <distance> <queryset> <forks> <iters> <outdir>
jmh_cell() {
    local results_dir="$1" target="$2" backend="$3" algorithm="$4" distance="$5" queryset="$6"
    local forks="$7" iters="$8" cell_dir="$9"
    ensure_environment "$results_dir"
    local module include libver
    if [ "$target" = "jvm-vinary" ]; then
        module=":vinary:jmh"; include="VinaryBench"; libver="0.10.0"
    else
        module=":legacy:jmh"; include="LegacyBench"; libver="3.0.0"
    fi
    local params="algorithm=${algorithm};distance=${distance};queryset=${queryset}"
    [ "$target" = "jvm-vinary" ] && params="${params};backend=${backend}"
    [ -n "${XL_JMH_EXTRA_PARAMS:-}" ] && params="${params};${XL_JMH_EXTRA_PARAMS}"
    local rff="$results_dir/jmh/${target}__${backend}__${algorithm}__d${distance}__${queryset}.json"
    mkdir -p "$(dirname "$rff")"
    # A compiler-load retry must never let Gradle reuse the previous JMH task's
    # output as UP-TO-DATE. Preserve that raw evidence for audit, but remove it
    # from the declared output path so the timed task necessarily executes.
    if [ -f "$rff" ]; then
        local invalid_dir="$results_dir/jmh/invalid-retries"
        local invalid_stamp
        invalid_stamp="$(date -u +%Y%m%dT%H%M%S).$$"
        mkdir -p "$invalid_dir"
        mv "$rff" "$invalid_dir/$(basename "$rff" .json).${invalid_stamp}.json"
    fi

    local mhz_start mhz_end load
    require_compiler_quiet
    mhz_start="$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)"
    load="$(cut -d' ' -f1 /proc/loadavg)"

    (cd "$JVM" && taskset -c 2-9 ./gradlew "${GRADLE_ARGS[@]}" "$module" \
        -Pjmh.includes="$include" -Pjmh.params="$params" \
        -Pjmh.forks="$forks" -Pjmh.iterations="$iters" \
        -Pjmh.rff="$rff" >&2)
    require_compiler_quiet
    mhz_end="$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)"

    XL_CELL_DIR_UNUSED=1 python3 "$SCRIPT_DIR/jmh_to_result.py" "$results_dir" "$rff" \
        --target "$target" --backend "$backend" \
        --runtime-version "$(runtime_version)" --library-version "$libver"
    local cell_json="$results_dir/cells/${target}__${backend}__query__${algorithm}__d${distance}__${queryset}.json"
    if [ "$cell_dir" != "$results_dir/cells" ]; then
        mkdir -p "$cell_dir"
        mv "$cell_json" "$cell_dir/"
        cell_json="$cell_dir/${target}__${backend}__query__${algorithm}__d${distance}__${queryset}.json"
    fi
    python3 "$SCRIPT_DIR/postfill.py" "$results_dir" "$cell_json" \
        --cpuset "2-9" \
        --mhz-start "$(awk -v k="$mhz_start" 'BEGIN{printf "%.1f", k/1000.0}')" \
        --mhz-end "$(awk -v k="$mhz_end" 'BEGIN{printf "%.1f", k/1000.0}')" \
        --loadavg "$load"
    echo "$cell_json"
}

# Full-coverage verify twins for the JMH cells. The gate's twins run at
# --gate-limit 200, so their triple describes 200 queries while a JMH sample
# times all 1000 — copying that triple into a timed cell mislabels its
# correctness evidence. These twins re-run verify over the FULL query set at
# every timed coordinate, so each converted JMH cell carries a triple and
# checksum for exactly the pass it timed. Untimed: runs on the build cores.
verify_full() {
    local results_dir="$1"
    ensure_environment "$results_dir"
    local out_dir="$results_dir/verify-full"
    mkdir -p "$out_dir/tsv"
    for target in jvm-vinary jvm-legacy; do
        local backends
        backends="$(awk -F'\t' -v t="$target" '!/^#/ && $1 == t { print $4 }' "$XL/targets.tsv")"
        IFS=',' read -r -a backend_list <<< "$backends"
        for backend in "${backend_list[@]}"; do
            local tsv="$out_dir/tsv/${target}__${backend}.tsv"
            : > "$tsv"
            local pending=0
            while IFS=$'\t' read -r _t _b _m algorithm distance queryset; do
                local name="${target}__${backend}__verify__${algorithm}__d${distance}__${queryset}.json"
                cell_is_complete "$out_dir/$name" && continue
                printf '%s\t%s\t%s\t%s\n' "$algorithm" "$distance" \
                    "$XL/workload/queries/${queryset}.txt" "$out_dir/$name" >> "$tsv"
                pending=$((pending + 1))
            done < <(python3 "$SCRIPT_DIR/matrix.py" cells --target "$target" \
                        --backend "$backend" --mode query)
            if [ "$pending" -eq 0 ]; then
                log "verify-full: $target x $backend already complete"
                continue
            fi
            log "verify-full: $target x $backend — $pending cells (full query sets)"
            XL_CELL_DIR="$out_dir" XL_CPUSET_OVERRIDE="16-31" XL_GATE_LIMIT=1000000 \
                "$SCRIPT_DIR/run-one.sh" cells "$results_dir" "$target" "$backend" \
                verify "$tsv" >/dev/null
        done
    done
    log "verify-full complete under $out_dir"
}

# Capture the exact full-query correctness twin required by one JMH cell. This
# is the focused counterpart of verify_full: hypothesis arms can remain small
# without weakening their semantic evidence or timing every JVM coordinate.
verify_cell() {
    local results_dir="$1" target="$2" backend="$3" algorithm="$4" distance="$5" queryset="$6"
    local out_dir="$results_dir/verify-full"
    local name="${target}__${backend}__verify__${algorithm}__d${distance}__${queryset}.json"
    ensure_environment "$results_dir"
    cell_is_complete "$out_dir/$name" && return
    mkdir -p "$out_dir"
    log "verify-full cell: $target $backend $algorithm d$distance $queryset"
    XL_CELL_DIR="$out_dir" XL_CPUSET_OVERRIDE="16-31" XL_GATE_LIMIT=1000000 \
        "$SCRIPT_DIR/run-one.sh" cell "$results_dir" "$target" "$backend" \
        verify "$algorithm" "$distance" "$queryset" >/dev/null
}

# Closing Java-parity matrix. The legacy engine has no unrestricted-Damerau
# surface, so its 45-cell matrix defines the coordinate intersection. Exact
# full-query twins still cover every supported vinary backend and algorithm;
# only managed timing is restricted to the like-for-like pair.
parity_jmh() {
    local results_dir="$1"
    verify_full "$results_dir"

    local target_backend target backend algorithm distance queryset cell_json
    local pair_index=0
    local -a pair_order
    while IFS=$'\t' read -r _t _b _m algorithm distance queryset; do
        # Keep the two arms adjacent in wall-clock time so host drift affects
        # each coordinate pair as symmetrically as the shared machine permits.
        # Alternate which arm runs first to balance any residual order effect.
        if (( pair_index % 2 == 0 )); then
            pair_order=("jvm-vinary dynamic_dawg" "jvm-legacy own")
        else
            pair_order=("jvm-legacy own" "jvm-vinary dynamic_dawg")
        fi
        for target_backend in "${pair_order[@]}"; do
            read -r target backend <<< "$target_backend"
            cell_json="$results_dir/cells/${target}__${backend}__query__${algorithm}__d${distance}__${queryset}.json"
            if cell_is_complete "$cell_json"; then
                continue
            fi
            log "parity JMH cell: $target $backend $algorithm d$distance $queryset"
            jmh_cell "$results_dir" "$target" "$backend" "$algorithm" "$distance" "$queryset" \
                2 10 "$results_dir/cells" >/dev/null
        done
        pair_index=$((pair_index + 1))
    done < <(python3 "$SCRIPT_DIR/matrix.py" cells --target jvm-legacy \
                --backend own --mode query)

    python3 "$SCRIPT_DIR/pgmcp-upload.py" "$results_dir" || \
        log "pgmcp upload FAILED — local cells intact; re-run scripts/pgmcp-upload.py"
}

cmd="${1:-}"; shift || true
case "$cmd" in
verify-full)
    verify_full "$(cd "$1" && pwd)"
    ;;
jmh-cell)
    [ "$#" -eq 8 ] || {
        echo "usage: run-jvm-pair.sh jmh-cell <results_dir> <target> <backend> <algorithm> <distance> <queryset> <forks> <iterations>" >&2
        exit 2
    }
    results_dir="$(cd "$1" && pwd)"
    target="$2"; backend="$3"; algorithm="$4"; distance="$5"; queryset="$6"
    forks="$7"; iterations="$8"
    verify_cell "$results_dir" "$target" "$backend" "$algorithm" "$distance" "$queryset"
    jmh_cell "$results_dir" "$target" "$backend" "$algorithm" "$distance" "$queryset" \
        "$forks" "$iterations" "$results_dir/cells"
    ;;
parity)
    parity_jmh "$(cd "$1" && pwd)"
    ;;
stage)
    stage_natives
    (cd "$JVM" && ./gradlew "${GRADLE_ARGS[@]}" :common:classes :vinary:classes :legacy:classes >&2)
    write_launchers
    protobuf_containment_smoke
    ;;
jmh)
    results_dir="$(cd "$1" && pwd)"; shift
    quick_flag=()
    [ "${1:-}" = "--quick" ] && quick_flag=(--quick)
    for target in jvm-vinary jvm-legacy; do
        backends="$(awk -F'\t' -v t="$target" '!/^#/ && $1 == t { print $4 }' "$XL/targets.tsv")"
        IFS=',' read -r -a backend_list <<< "$backends"
        for backend in "${backend_list[@]}"; do
            while IFS=$'\t' read -r _t _b _m algorithm distance queryset; do
                cell_json="$results_dir/cells/${target}__${backend}__query__${algorithm}__d${distance}__${queryset}.json"
                if cell_is_complete "$cell_json"; then continue; fi
                log "JMH cell: $target $backend $algorithm d$distance $queryset"
                jmh_cell "$results_dir" "$target" "$backend" "$algorithm" "$distance" "$queryset" \
                    2 10 "$results_dir/cells"
            done < <(python3 "$SCRIPT_DIR/matrix.py" cells --target "$target" --backend "$backend" \
                        --mode query "${quick_flag[@]}")
        done
    done
    # Publish to pgmcp (system of record); idempotent + content-addressed.
    python3 "$SCRIPT_DIR/pgmcp-upload.py" "$results_dir" || \
        log "pgmcp upload FAILED — local cells intact; re-run scripts/pgmcp-upload.py"
    ;;
hypothesis)
    results_dir="$(cd "$1" && pwd)"
    # H-J1 deciding cells at >=51 samples: transposition/d2/tr-d2 both sides.
    jmh_cell "$results_dir" "jvm-vinary" "dynamic_dawg" transposition 2 tr-d2 3 17 \
        "$results_dir/hypothesis"
    jmh_cell "$results_dir" "jvm-legacy" "own" transposition 2 tr-d2 3 17 \
        "$results_dir/hypothesis"
    ;;
*)
    echo "usage: run-jvm-pair.sh stage | verify-full <results_dir> | jmh-cell <results_dir> <target> <backend> <algorithm> <distance> <queryset> <forks> <iterations> | jmh <results_dir> [--quick] | parity <results_dir> | hypothesis <results_dir>" >&2
    exit 2
    ;;
esac
