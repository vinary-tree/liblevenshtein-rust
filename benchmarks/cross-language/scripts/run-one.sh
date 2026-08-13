#!/usr/bin/env bash
# Run one benchmark cell (or one batched cells file) for one target×backend.
#
# Usage:
#   run-one.sh cell  <results_dir> <target> <backend> <mode> <algorithm> <distance> <queryset>
#   run-one.sh cells <results_dir> <target> <backend> <mode> <cells_tsv>   # mode: query|verify
#
# mode: query | construct | memory | verify   (memory maps to the harness's
# memory-child entry point and is wrapped in /usr/bin/time -v).
#
# Responsibilities: resolve the target's build/run recipe (single source of
# truth for commands), pin CPUs via taskset (cpuset from targets.tsv), tee
# all output to logs/, snapshot frequency/loadavg around the cell, and merge
# runner-known facts into each produced JSON via postfill.py (which also
# schema-validates). Every failure path leaves a loud artifact under
# failures/ — a target is never silently skipped.
#
# Cell JSON naming: <target>__<backend>__<mode>__<algo>__d<K>__<set>.json
# under $XL_CELL_DIR (default <results_dir>/cells; the gate overrides it).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL="$(cd "$SCRIPT_DIR/.." && pwd)"
LR="$(cd "$XL/../.." && pwd)"
LD_REPO="$(cd "$LR/../libdictenstein" && pwd)"
WORKLOAD="$XL/workload"
DICT="$WORKLOAD/dictionary.txt"

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[ $# -ge 4 ] || usage
SUBCOMMAND="$1"; RESULTS_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"; TARGET="$3"; BACKEND="$4"
shift 4

CELL_DIR="${XL_CELL_DIR:-$RESULTS_DIR/cells}"
LOG_DIR="$RESULTS_DIR/logs"
FAIL_DIR="$RESULTS_DIR/failures"
mkdir -p "$CELL_DIR" "$LOG_DIR" "$FAIL_DIR"

# ---------------------------------------------------------------------------
# Target manifest lookup
# ---------------------------------------------------------------------------

manifest_field() { # target -> named column from targets.tsv
    awk -F'\t' -v target="$TARGET" -v col="$1" '
        !/^#/ && NR > 0 && $1 == "target" { for (i = 1; i <= NF; i++) idx[$i] = i; next }
        !/^#/ && $1 == target { print $(idx[col]); found = 1 }
        END { if (!found) exit 3 }
    ' "$XL/targets.tsv"
}

CPUSET="$(manifest_field cpuset)" || { echo "run-one: unknown target: $TARGET" >&2; exit 2; }
# Untimed runs (gate verifies) may be rerouted off the timing cores so they
# can proceed while a timed run owns the manifest cpuset. Timed cells never
# set this.
if [ -n "${XL_CPUSET_OVERRIDE:-}" ]; then CPUSET="$XL_CPUSET_OVERRIDE"; fi

# ---------------------------------------------------------------------------
# Environment snapshots
# ---------------------------------------------------------------------------

cpu_mhz() {
    local khz
    khz="$(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)"
    awk -v khz="$khz" 'BEGIN { printf "%.1f", khz / 1000.0 }'
}

loadavg_1m() { cut -d' ' -f1 /proc/loadavg; }

# ---------------------------------------------------------------------------
# Per-target build + run recipes (release native libs everywhere)
# ---------------------------------------------------------------------------

export LD_LIBRARY_PATH="$LR/target/release:$LD_REPO/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build_rust() {
    (cd "$XL/harnesses/rust" && RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet)
}
run_rust() { echo "$XL/harnesses/rust/target/release/bench-cross-rust"; }

build_c() { bash "$XL/harnesses/c/build.sh" >/dev/null; }
run_c() { echo "$XL/.stage/c/bench"; }

build_cpp() { bash "$XL/harnesses/cpp/build.sh" >/dev/null; }
run_cpp() { echo "$XL/.stage/cpp/bench"; }

# Third head-to-head baseline: legacy liblevenshtein-cpp. Both sides of this
# pair are native and AOT-compiled, so it isolates automaton/dictionary
# quality with no VM, JIT, GC, or FFI confounds.
build_cpp_legacy() { bash "$XL/harnesses/cpp-legacy/build.sh" >/dev/null; }
run_cpp_legacy() { echo "$XL/.stage/cpp-legacy/bench"; }

# JVM pair: run-jvm-pair.sh stages natives, compiles, and generates these
# launchers; gate/construct/memory cells then flow through the generic path.
build_jvm_vinary() {
    [ -x "$XL/.stage/jvm/vinary-launcher.sh" ] || bash "$SCRIPT_DIR/run-jvm-pair.sh" stage >&2
}
run_jvm_vinary() { echo "$XL/.stage/jvm/vinary-launcher.sh"; }
build_jvm_legacy() {
    [ -x "$XL/.stage/jvm/legacy-launcher.sh" ] || bash "$SCRIPT_DIR/run-jvm-pair.sh" stage >&2
}
run_jvm_legacy() { echo "$XL/.stage/jvm/legacy-launcher.sh"; }

# JavaScript targets: one bench.mjs serves all four; the wrapper pins the
# backend via XL_JS_BACKEND so the PROTOCOL CLI stays uniform. Prerequisite
# artifacts (N-API release addon, wasm, wasi) are built by
# bindings/javascript-runtime's own npm scripts; doctor verifies presence.
build_js_common() {
    bash "$XL/harnesses/javascript/setup-js.sh" >&2
    local addon="$LR/bindings/javascript-runtime/native/build/Release/vinary_tree_native.node"
    local wasm="$LR/bindings/javascript-runtime/generated/wasm/vinary_tree_bg.wasm"
    local wasi="$LR/bindings/javascript-runtime/generated/wasi/vinary_tree.wasm"
    case "$TARGET" in
        js-native) [ -f "$addon" ] || { echo "run-one: N-API addon missing ($addon); run npm run build:native:release" >&2; return 4; } ;;
        js-wasm) [ -f "$wasm" ] || { echo "run-one: wasm artifact missing ($wasm); run npm run build:wasm" >&2; return 4; } ;;
        js-wasi) [ -f "$wasi" ] || { echo "run-one: wasi artifact missing ($wasi); run npm run build:wasi" >&2; return 4; } ;;
    esac
    mkdir -p "$XL/.stage/js"
    local backend="${TARGET#js-}"
    local wrapper="$XL/.stage/js/${backend}.sh"
    cat > "$wrapper" <<WRAPEOF
#!/usr/bin/env bash
exec env XL_JS_BACKEND=$backend node "$XL/harnesses/javascript/bench.mjs" "\$@"
WRAPEOF
    chmod +x "$wrapper"
}
run_js_target() { echo "$XL/.stage/js/${TARGET#js-}.sh"; }

# Builds are pinned away from the timing cores (2-9).
BUILD_CPUSET="16-31"

build_go() {
    mkdir -p "$XL/.stage/go"
    (cd "$XL/harnesses/go" && taskset -c "$BUILD_CPUSET" env \
        CGO_LDFLAGS="-L$LR/target/release -L$LD_REPO/target/release" \
        go build -o "$XL/.stage/go/bench" .) >&2
    cat > "$XL/.stage/go/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
exec env GOMAXPROCS=1 GOGC=100 "$XL/.stage/go/bench" "\$@"
WRAPEOF
    chmod +x "$XL/.stage/go/bench.sh"
}
run_go() { echo "$XL/.stage/go/bench.sh"; }

build_ruby() {
    mkdir -p "$XL/.stage/ruby"
    cat > "$XL/.stage/ruby/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
exec env \\
    RUBYLIB="$LR/bindings/ruby/lib:$LD_REPO/bindings/ruby/lib" \\
    LIBLEVENSHTEIN_LIBRARY="$LR/target/release/libliblevenshtein.so" \\
    LIBDICTENSTEIN_LIBRARY="$LD_REPO/target/release/liblibdictenstein.so" \\
    ruby "$XL/harnesses/ruby/bench.rb" "\$@"
WRAPEOF
    chmod +x "$XL/.stage/ruby/bench.sh"
}
run_ruby() { echo "$XL/.stage/ruby/bench.sh"; }

build_lua() {
    XL_BUILD_CPUSET="$BUILD_CPUSET" bash "$XL/harnesses/lua/build.sh" >&2
    cat > "$XL/.stage/lua/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
exec env \\
    LUA_CPATH="$XL/.stage/lua/?.so;$XL/.stage/lua/?/?.so;;" \\
    lua5.4 "$XL/harnesses/lua/bench.lua" "\$@"
WRAPEOF
    chmod +x "$XL/.stage/lua/bench.sh"
}
run_lua() { echo "$XL/.stage/lua/bench.sh"; }

build_dotnet() {
    (cd "$XL/harnesses/dotnet" && \
     taskset -c "$BUILD_CPUSET" dotnet build -c Release -m:1 -p:NuGetAudit=false >&2)
    mkdir -p "$XL/.stage/dotnet"
    cat > "$XL/.stage/dotnet/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
exec dotnet "$XL/harnesses/dotnet/bin/Release/net10.0/Bench.dll" "\$@"
WRAPEOF
    chmod +x "$XL/.stage/dotnet/bench.sh"
}
run_dotnet() { echo "$XL/.stage/dotnet/bench.sh"; }

build_swift() {
    (cd "$XL/harnesses/swift" && \
     taskset -c "$BUILD_CPUSET" env LIBRARY_PATH="$LR/target/release:$LD_REPO/target/release" \
         swift build --package-path "$XL/harnesses/swift" -c release >&2)
}
run_swift() { echo "$XL/harnesses/swift/.build/release/bench"; }

build_fortran() {
    (cd "$XL/harnesses/fortran" && \
     taskset -c "$BUILD_CPUSET" fpm build --profile release --compiler gfortran \
         --link-flag "-L$LR/target/release -L$LD_REPO/target/release -Wl,-rpath,$LR/target/release -Wl,-rpath,$LD_REPO/target/release" >&2)
    mkdir -p "$XL/.stage/fortran"
    cp "$(find "$XL/harnesses/fortran/build" -type f -name bench -path '*/app/*' | head -1)" \
       "$XL/.stage/fortran/bench"
}
run_fortran() { echo "$XL/.stage/fortran/bench"; }

build_ocaml() {
    (cd "$XL/harnesses/ocaml" && \
     taskset -c "$BUILD_CPUSET" env \
         LIBRARY_PATH="$LR/target/release:$LD_REPO/target/release" \
         opam exec -- dune build) >&2
}
run_ocaml() { echo "$XL/harnesses/ocaml/_build/default/bin/bench.exe"; }

build_haskell() {
    local stage="$XL/.stage/haskell-native"
    mkdir -p "$stage/lib/pkgconfig" "$stage/include"
    cp "$LR/target/release/libliblevenshtein.so" "$LD_REPO/target/release/liblibdictenstein.so" "$stage/lib/"
    cp "$LR/pkgconfig/liblevenshtein.pc" "$LD_REPO/pkgconfig/libdictenstein.pc" "$stage/lib/pkgconfig/"
    cp "$LR/include/liblevenshtein.h" "$LR/include/liblevenshtein_abi.h" \
       "$LR/vinary-tree-interop/include/vinary_tree_interop.h" \
       "$LD_REPO/include/libdictenstein.h" "$stage/include/"
    (cd "$XL/harnesses/haskell" && \
     taskset -c "$BUILD_CPUSET" env PKG_CONFIG_PATH="$stage/lib/pkgconfig" \
         cabal build bench-cross-haskell) >&2
    local binary
    binary="$(cd "$XL/harnesses/haskell" && cabal list-bin bench-cross-haskell)"
    mkdir -p "$XL/.stage/haskell"
    cat > "$XL/.stage/haskell/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
exec env LD_LIBRARY_PATH="$stage/lib" "$binary" "\$@"
WRAPEOF
    chmod +x "$XL/.stage/haskell/bench.sh"
}
run_haskell() { echo "$XL/.stage/haskell/bench.sh"; }

build_clojure() {
    mkdir -p "$XL/.stage/clojure"
    cat > "$XL/.stage/clojure/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
cd "$XL/harnesses/clojure"
exec env JAVA_TOOL_OPTIONS="--enable-native-access=ALL-UNNAMED -Djava.library.path=$LR/target/release:$LD_REPO/target/release -Xms2g -Xmx2g" \\
    clojure -M:bench "\$@"
WRAPEOF
    chmod +x "$XL/.stage/clojure/bench.sh"
}
run_clojure() { echo "$XL/.stage/clojure/bench.sh"; }

build_clojurescript() {
    bash "$XL/harnesses/javascript/setup-js.sh" >&2
    ln -sfn "$XL/harnesses/javascript/node_modules" "$XL/harnesses/clojurescript/node_modules"
    if [ ! -f "$XL/harnesses/clojurescript/out/harness.js" ]; then
        (cd "$XL/harnesses/clojurescript" && taskset -c "$BUILD_CPUSET" clojure -M:compile) >&2
    fi
}
run_clojurescript() { echo "$XL/harnesses/clojurescript/run.sh"; }

build_python() {
    mkdir -p "$XL/.stage/python"
    cat > "$XL/.stage/python/bench.sh" <<WRAPEOF
#!/usr/bin/env bash
exec env \\
    PYTHONPATH="$LD_REPO/bindings/python/src:$LR/bindings/python/src:$LR/vinary-tree-interop/bindings/python/src" \\
    PYTHONHASHSEED=0 \\
    LIBLEVENSHTEIN_LIBRARY="$LR/target/release/libliblevenshtein.so" \\
    LIBDICTENSTEIN_LIBRARY="$LD_REPO/target/release/liblibdictenstein.so" \\
    python3 "$XL/harnesses/python/bench.py" "\$@"
WRAPEOF
    chmod +x "$XL/.stage/python/bench.sh"
}
run_python() { echo "$XL/.stage/python/bench.sh"; }

# Phase 1/2/4 targets register their recipes here as their harnesses land.
# An unimplemented target aborts loudly (recorded under failures/), it is
# never silently skipped.
target_binary() {
    case "$TARGET" in
        rust) build_rust >&2 && run_rust ;;
        c) build_c && run_c ;;
        cpp) build_cpp && run_cpp ;;
        cpp-legacy) build_cpp_legacy && run_cpp_legacy ;;
        jvm-vinary) build_jvm_vinary && run_jvm_vinary ;;
        jvm-legacy) build_jvm_legacy && run_jvm_legacy ;;
        js-native|js-wasm|js-wasi|js-legacy) build_js_common && run_js_target ;;
        python) build_python && run_python ;;
        go) build_go && run_go ;;
        ruby) build_ruby && run_ruby ;;
        lua) build_lua && run_lua ;;
        dotnet) build_dotnet && run_dotnet ;;
        swift) build_swift && run_swift ;;
        fortran) build_fortran && run_fortran ;;
        ocaml) build_ocaml && run_ocaml ;;
        haskell) build_haskell && run_haskell ;;
        clojure) build_clojure && run_clojure ;;
        clojurescript) build_clojurescript && run_clojurescript ;;
        *)
            echo "run-one: target '$TARGET' has no recipe yet (its phase has not landed)" >&2
            return 4
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

postfill() { # cell_json mhz_start mhz_end loadavg [time_v_log child_cmdline]
    local cell_json="$1" mhz_start="$2" mhz_end="$3" load="$4"
    shift 4
    local extra=()
    if [ $# -ge 1 ]; then extra+=(--time-v-log "$1"); fi
    if [ $# -ge 2 ]; then extra+=(--child-cmdline "$2"); fi
    python3 "$SCRIPT_DIR/postfill.py" "$RESULTS_DIR" "$cell_json" \
        --cpuset "$CPUSET" --mhz-start "$mhz_start" --mhz-end "$mhz_end" \
        --loadavg "$load" "${extra[@]}"
}

fail_loudly() { # log_file message
    cp "$1" "$FAIL_DIR/$(basename "$1")" 2>/dev/null || true
    echo "run-one: FAILED: $2 (log: $1)" >&2
    exit 1
}

case "$SUBCOMMAND" in
cell)
    [ $# -eq 4 ] || usage
    MODE="$1"; ALGORITHM="$2"; DISTANCE="$3"; QUERYSET="$4"
    ;;
cells)
    [ $# -eq 2 ] || usage
    MODE="$1"
    CELLS_TSV="$2"
    case "$MODE" in query|verify) ;; *)
        echo "run-one: cells form supports only query|verify modes" >&2; exit 2 ;;
    esac
    ;;
*) usage ;;
esac

BINARY="$(target_binary)" || {
    echo "target=$TARGET status=build_failed" > "$FAIL_DIR/${TARGET}__${BACKEND}.build_failed"
    exit 4
}

run_cell() { # mode algorithm distance queryset
    local mode="$1" algorithm="$2" distance="$3" queryset="$4"
    local queries="$WORKLOAD/queries/${queryset}.txt"
    [ -f "$queries" ] || { echo "run-one: no such query set: $queries" >&2; exit 2; }
    local cell_name="${TARGET}__${BACKEND}__${mode}__${algorithm}__d${distance}__${queryset}"
    local out_json="$CELL_DIR/${cell_name}.json"
    local log_file="$LOG_DIR/${cell_name}.log"
    local harness_mode="$mode"
    [ "$mode" = "memory" ] && harness_mode="memory-child"

    local mhz_start mhz_end load
    mhz_start="$(cpu_mhz)"; load="$(loadavg_1m)"

    local cmd=(taskset -c "$CPUSET" "$BINARY"
        --mode "$harness_mode" --algorithm "$algorithm" --max-distance "$distance"
        --dictionary "$DICT" --queries "$queries" --backend "$BACKEND" --out "$out_json")
    # Orchestrator overrides (run-all/gate compose these; PROTOCOL defaults
    # apply when unset): XL_REPS for construct (e.g. 3 for the slow DAT
    # builds), XL_SAMPLES for --quick, XL_GATE_LIMIT for verify depth.
    if [ -n "${XL_REPS:-}" ]; then cmd+=(--reps "$XL_REPS"); fi
    if [ -n "${XL_SAMPLES:-}" ]; then cmd+=(--samples "$XL_SAMPLES"); fi
    if [ -n "${XL_GATE_LIMIT:-}" ]; then cmd+=(--gate-limit "$XL_GATE_LIMIT"); fi

    local time_v_log=""
    if [ "$mode" = "memory" ]; then
        time_v_log="$LOG_DIR/${cell_name}.time-v"
        cmd=(/usr/bin/time -v -o "$time_v_log" "${cmd[@]}")
    fi

    if ! "${cmd[@]}" >"$log_file" 2>&1; then
        fail_loudly "$log_file" "$cell_name"
    fi
    mhz_end="$(cpu_mhz)"

    if [ "$mode" = "memory" ]; then
        postfill "$out_json" "$mhz_start" "$mhz_end" "$load" "$time_v_log" "${cmd[*]}" \
            || fail_loudly "$log_file" "$cell_name postfill"
    else
        postfill "$out_json" "$mhz_start" "$mhz_end" "$load" \
            || fail_loudly "$log_file" "$cell_name postfill"
    fi
    echo "$out_json"
}

if [ "$SUBCOMMAND" = "cell" ]; then
    [ -n "${QUERYSET}" ] || { echo "run-one: cell form requires a queryset" >&2; exit 2; }
    run_cell "$MODE" "$ALGORITHM" "$DISTANCE" "$QUERYSET"
else
    # Batched cells: one process, one dictionary build, N cells.
    log_file="$LOG_DIR/${TARGET}__${BACKEND}__cells-${MODE}.log"
    mhz_start="$(cpu_mhz)"; load="$(loadavg_1m)"
    batch_cmd=(taskset -c "$CPUSET" "$BINARY"
        --mode "$MODE" --dictionary "$DICT" --backend "$BACKEND"
        --cells "$CELLS_TSV")
    if [ -n "${XL_SAMPLES:-}" ]; then batch_cmd+=(--samples "$XL_SAMPLES"); fi
    if [ -n "${XL_GATE_LIMIT:-}" ]; then batch_cmd+=(--gate-limit "$XL_GATE_LIMIT"); fi
    if ! "${batch_cmd[@]}" >"$log_file" 2>&1; then
        fail_loudly "$log_file" "${TARGET}__${BACKEND} cells batch"
    fi
    mhz_end="$(cpu_mhz)"
    # Post-fill every JSON the batch produced (4th TSV column).
    while IFS=$'\t' read -r _algo _dist _queries out_path; do
        case "$_algo" in ''|'#'*) continue ;; esac
        postfill "$out_path" "$mhz_start" "$mhz_end" "$load" \
            || fail_loudly "$log_file" "postfill $out_path"
        echo "$out_path"
    done < "$CELLS_TSV"
fi
