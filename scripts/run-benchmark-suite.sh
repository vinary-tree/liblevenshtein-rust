#!/bin/bash
# Automated Benchmark Suite Execution
# Runs all phonetic-related benchmarks and saves results

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Export environment
export RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes"

echo "=== liblevenshtein-rust Benchmark Suite ===" | tee "$OUTPUT_DIR/log.txt"
echo "Timestamp: $TIMESTAMP" | tee -a "$OUTPUT_DIR/log.txt"
echo "Output directory: $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/log.txt"
echo "" | tee -a "$OUTPUT_DIR/log.txt"

# Save environment info
cat > "$OUTPUT_DIR/environment.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "hardware": {
    "cpu": "$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)",
    "cores": $(nproc),
    "ram_gb": $(free -g | grep Mem | awk '{print $2}')
  },
  "software": {
    "rust_version": "$(rustc --version | cut -d' ' -f2)",
    "cargo_version": "$(cargo --version | cut -d' ' -f2)",
    "os": "$(uname -sr)",
    "rustflags": "$RUSTFLAGS"
  },
  "git": {
    "commit": "$(git rev-parse HEAD)",
    "branch": "$(git rev-parse --abbrev-ref HEAD)",
    "dirty": $(git diff --quiet && echo false || echo true)
  }
}
EOF

# Define benchmarks
declare -A BENCHMARKS=(
    ["phonetic_compilation_benchmarks"]="phonetic-rules"
    ["phonetic_nfa_benchmarks"]="phonetic-rules"
    ["phonetic_grep_parallel_benchmarks"]="parallel-grep"
    ["phonetic_rules"]="phonetic-rules"
    ["phonetic_position_skip_benchmark"]="phonetic-rules"
)

# Run each benchmark
for bench in "${!BENCHMARKS[@]}"; do
    features="${BENCHMARKS[$bench]}"
    echo "Running: $bench (features: $features)" | tee -a "$OUTPUT_DIR/log.txt"

    if taskset -c 0 cargo bench \
        --bench "$bench" \
        --features "$features" \
        -- --save-baseline "baseline_${TIMESTAMP}" \
        2>&1 | tee "$OUTPUT_DIR/${bench}.txt"; then
        echo "Completed: $bench" | tee -a "$OUTPUT_DIR/log.txt"
    else
        echo "FAILED: $bench" | tee -a "$OUTPUT_DIR/log.txt"
    fi

    echo "" | tee -a "$OUTPUT_DIR/log.txt"
done

# Copy Criterion output if available
if [ -d "target/criterion" ]; then
    cp -r target/criterion "$OUTPUT_DIR/criterion_data"
fi

echo "=== Benchmark Suite Complete ===" | tee -a "$OUTPUT_DIR/log.txt"
echo "Results saved to: $OUTPUT_DIR"
