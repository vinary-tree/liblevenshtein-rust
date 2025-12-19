#!/bin/bash
# Flamegraph Generation Script
# Profiles a benchmark and generates an SVG flamegraph

set -euo pipefail

BENCH_NAME="${1:-phonetic_nfa_benchmarks}"
FEATURES="${2:-phonetic-rules}"
OUTPUT="${3:-flamegraph_${BENCH_NAME}.svg}"
PROFILE_TIME="${4:-30}"

echo "=== Flamegraph Generation ==="
echo "Benchmark: $BENCH_NAME"
echo "Features: $FEATURES"
echo "Output: $OUTPUT"
echo "Profile time: ${PROFILE_TIME}s"
echo ""

# Check for required tools
if ! command -v perf &> /dev/null; then
    echo "ERROR: perf not found. Install linux-tools-common or equivalent."
    exit 1
fi

if ! command -v stackcollapse-perf.pl &> /dev/null; then
    echo "WARNING: stackcollapse-perf.pl not found."
    echo "Install FlameGraph: git clone https://github.com/brendangregg/FlameGraph.git"
    echo "Add to PATH: export PATH=\$PATH:/path/to/FlameGraph"
    echo ""
    echo "Falling back to cargo-flamegraph..."

    if command -v cargo-flamegraph &> /dev/null; then
        RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
            cargo flamegraph \
            --bench "$BENCH_NAME" \
            --features "$FEATURES" \
            --output "$OUTPUT" \
            -- --profile-time "$PROFILE_TIME"
        echo "Flame graph saved to: $OUTPUT"
        exit 0
    else
        echo "ERROR: cargo-flamegraph not found either."
        echo "Install: cargo install flamegraph"
        exit 1
    fi
fi

# Build with frame pointers
echo "Building with frame pointers..."
RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
    cargo build --release --bench "$BENCH_NAME" --features "$FEATURES"

# Record perf data
echo "Recording perf data for ${PROFILE_TIME}s..."
perf record -g --call-graph dwarf -F 99 \
    taskset -c 0 \
    cargo bench --bench "$BENCH_NAME" --features "$FEATURES" \
    -- --profile-time "$PROFILE_TIME"

# Generate flamegraph
echo "Generating flamegraph..."
perf script | stackcollapse-perf.pl | flamegraph.pl > "$OUTPUT"

echo ""
echo "Flame graph saved to: $OUTPUT"

# Optionally generate perf report
REPORT_FILE="${OUTPUT%.svg}_perf_report.txt"
echo "Generating perf report..."
perf report -n --stdio --sort dso,symbol > "$REPORT_FILE"
echo "Perf report saved to: $REPORT_FILE"
