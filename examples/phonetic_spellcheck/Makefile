# Phonetic Spellcheck Demo Makefile
# ==================================
#
# Build and run the phonetic spellcheck demonstration.
# This example combines phonetic normalization with Damerau-Levenshtein
# distance for robust fuzzy string matching.
#
# This is a self-contained example that depends on the parent liblevenshtein
# crate via a path dependency.
#
# Usage:
#   make run        - Build and run the interactive demo
#   make aot        - Pre-compile rules to binary cache (~50x faster startup)
#   make run-fast   - Run with pre-compiled cache (requires: make aot)
#   make run-stats  - Run with statistics display on exit
#   make build      - Build only (no run)
#   make clean      - Remove build artifacts
#   make help       - Show this help

# Configuration
CARGO := cargo
CACHE_SIZE := 1000

# File paths (relative to example directory)
RULES_SRC := rules/zompist.llev
RULES_BIN := rules/zompist.llev.bin
DICT_SRC := /usr/share/dict/words
DICT_DST := data/english_words.txt
CACHE_DIR := cache

.PHONY: all build run run-fast run-verbose run-stats aot clean setup-data setup-cache help check

# Default target
all: build

# Setup dictionary file (copy if not present)
setup-data:
	@mkdir -p data
	@if [ ! -f "$(DICT_DST)" ]; then \
		if [ -f "$(DICT_SRC)" ]; then \
			echo "Copying dictionary from $(DICT_SRC)..."; \
			cp "$(DICT_SRC)" "$(DICT_DST)"; \
			echo "  Copied $$(wc -l < $(DICT_DST)) words"; \
		else \
			echo "Warning: $(DICT_SRC) not found."; \
			echo "Creating minimal test dictionary..."; \
			echo -e "phone\nphilosophy\nenough\ncircle\ncat\nknight\nghost\nscience" > "$(DICT_DST)"; \
			echo "  Created minimal dictionary with 8 words"; \
		fi \
	else \
		echo "Dictionary already exists: $(DICT_DST)"; \
	fi

# Build the binary
build: setup-data
	@echo "Building phonetic_spellcheck..."
	$(CARGO) build --release
	@echo "Build complete!"
	@echo ""
	@echo "Binary location:"
	@echo "  target/release/phonetic_spellcheck"

# Run the interactive demo
run: setup-data
	@echo ""
	$(CARGO) run --release

# Setup cache directory
setup-cache:
	@mkdir -p $(CACHE_DIR)

# Pre-compile rules to binary cache for fast startup
# This creates cache/rules.bin which can be loaded instantly
aot: setup-data setup-cache build
	@echo ""
	@echo "Pre-compiling phonetic rules to binary cache..."
	$(CARGO) run --release -- --aot
	@echo ""
	@echo "AOT compilation complete!"
	@echo "Cache files created in $(CACHE_DIR)/"
	@echo ""
	@echo "Now run 'make run-fast' for ~50x faster startup."

# Run with pre-compiled cache (requires: make aot first)
run-fast: setup-data
	@if [ ! -d "$(CACHE_DIR)" ] || [ -z "$$(ls -A $(CACHE_DIR) 2>/dev/null)" ]; then \
		echo "Error: No cache found. Run 'make aot' first."; \
		exit 1; \
	fi
	@echo ""
	$(CARGO) run --release -- --use-cache --cache-size $(CACHE_SIZE)

# Run with verbose normalization output
run-verbose: setup-data
	@echo ""
	$(CARGO) run --release -- --verbose

# Run with statistics display on exit
run-stats: setup-data
	@echo ""
	$(CARGO) run --release -- --stats

# Run with all features: cache, memoization, stats, cycle detection
run-full: setup-data
	@if [ ! -d "$(CACHE_DIR)" ] || [ -z "$$(ls -A $(CACHE_DIR) 2>/dev/null)" ]; then \
		echo "Warning: No cache found. Run 'make aot' first for faster startup."; \
	fi
	@echo ""
	$(CARGO) run --release -- --use-cache --stats --detect-cycles --cache-size $(CACHE_SIZE)

# Check that it compiles (faster than full build)
check:
	$(CARGO) check

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	rm -f $(RULES_BIN)
	rm -rf $(CACHE_DIR)
	@echo "Clean complete."

# Show help
help:
	@echo "Phonetic Spellcheck Demo"
	@echo "========================"
	@echo ""
	@echo "An interactive spell checker combining phonetic normalization"
	@echo "with Damerau-Levenshtein distance for robust fuzzy matching."
	@echo ""
	@echo "This is a self-contained example with its own Cargo.toml that"
	@echo "depends on the parent liblevenshtein crate via path dependency."
	@echo ""
	@echo "Targets:"
	@echo "  make build      - Build the binary"
	@echo "  make run        - Build and run interactively"
	@echo "  make aot        - Pre-compile rules to binary cache (~50x faster startup)"
	@echo "  make run-fast   - Run with pre-compiled cache (requires: make aot)"
	@echo "  make run-verbose- Run with verbose normalization output"
	@echo "  make run-stats  - Run with statistics display on exit"
	@echo "  make run-full   - Run with all features enabled"
	@echo "  make setup-data - Copy dictionary file if needed"
	@echo "  make check      - Quick syntax/type check"
	@echo "  make clean      - Remove build artifacts and cache"
	@echo "  make help       - Show this help"
	@echo ""
	@echo "CLI Options (when running directly):"
	@echo "  --aot           Pre-compile rules to binary cache"
	@echo "  --use-cache     Load from pre-compiled cache"
	@echo "  --verbose, -v   Show normalization steps"
	@echo "  --stats         Show statistics on exit"
	@echo "  --detect-cycles Enable cycle detection in rule application"
	@echo "  --cache-size N  Query cache size (default: 1000)"
	@echo ""
	@echo "Files:"
	@echo "  rules/zompist.llev     - 62 Zompist phonetic rules"
	@echo "  data/english_words.txt - Dictionary (auto-copied from /usr/share/dict/words)"
	@echo "  cache/                 - Pre-compiled rule cache (after 'make aot')"
	@echo "  src/main.rs            - Interactive demo source"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make aot      # Pre-compile rules (once)"
	@echo "  2. make run-fast # Fast startup with cache"
	@echo ""
	@echo "Example Session:"
	@echo "  $$ make run"
	@echo "  query> fone"
	@echo "  Matches for \"fone\" (normalized: \"fon\"):"
	@echo "    1. phone (distance: 0)"
	@echo "    2. foe (distance: 1)"
	@echo ""
	@echo "  query> stats"
	@echo "  === Statistics ==="
	@echo "  Queries: 1, Cache hits: 0 (0.0%), Cache misses: 1"
