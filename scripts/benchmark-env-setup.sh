#!/bin/bash
# Benchmark Environment Setup Script
# Sets CPU affinity and frequency for consistent benchmark results

set -euo pipefail

echo "=== liblevenshtein-rust Benchmark Environment Setup ==="

# 1. Check prerequisites
command -v cargo >/dev/null || { echo "ERROR: cargo not found"; exit 1; }

# 2. Display hardware info
echo ""
echo "Hardware Configuration:"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Cores: $(nproc)"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"

# 3. Set CPU frequency (requires root)
if [ "$EUID" -eq 0 ]; then
    echo ""
    echo "Setting CPU governor to performance..."
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$cpu" 2>/dev/null || true
    done
    echo "  Done."
else
    echo ""
    echo "WARNING: Not running as root, skipping CPU frequency setup."
    echo "  For best results, run: sudo cpupower frequency-set -g performance"
fi

# 4. Disable turbo boost (optional, requires root)
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    if [ "$EUID" -eq 0 ]; then
        echo "Disabling turbo boost for consistent measurements..."
        echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
    fi
fi

# 5. Set Rust flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
echo ""
echo "RUSTFLAGS: $RUSTFLAGS"

# 6. Build benchmarks
echo ""
echo "Building benchmarks (release mode)..."
cargo build --release --features phonetic-rules

echo ""
echo "=== Environment Ready ==="
echo ""
echo "To run benchmarks with CPU affinity:"
echo "  taskset -c 0 cargo bench --bench <benchmark_name> --features <features>"
echo ""
echo "Example:"
echo "  taskset -c 0 cargo bench --bench phonetic_compilation_benchmarks --features phonetic-rules"
