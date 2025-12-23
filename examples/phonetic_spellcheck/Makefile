# Phonetic Spellcheck Demo Makefile
# ==================================
#
# Build and run the phonetic spellcheck demonstration.
# This example demonstrates the PhoneticNormalizedDictionary API with
# combined English phonetic rules (zompist + homophones + text_speak).
#
# This is a self-contained example that depends on the parent liblevenshtein
# crate via a path dependency.
#
# Usage:
#   make run    - Build and run the demo
#   make build  - Build only (no run)
#   make clean  - Remove build artifacts
#   make help   - Show this help

# Configuration
CARGO := cargo

# File paths
DICT_SRC := /usr/share/dict/words
DICT_DST := data/english_words.txt

.PHONY: all build run setup-data clean check help

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

# Run the demo
run: setup-data
	@echo ""
	$(CARGO) run --release

# Check that it compiles (faster than full build)
check:
	$(CARGO) check

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	@echo "Clean complete."

# Show help
help:
	@echo "Phonetic Spellcheck Demo"
	@echo "========================"
	@echo ""
	@echo "Demonstrates the PhoneticNormalizedDictionary API with"
	@echo "combined English phonetic rules (zompist + homophones + text_speak)."
	@echo ""
	@echo "This is a self-contained example with its own Cargo.toml that"
	@echo "depends on the parent liblevenshtein crate via path dependency."
	@echo ""
	@echo "Features:"
	@echo "  - Dual-index architecture with BK-tree optimization"
	@echo "  - Combined English rules: Zompist (62) + homophones (24) + text speak (31)"
	@echo "  - Fuzzy matching with edit distance queries"
	@echo "  - Regex queries against normalized dictionary forms"
	@echo "  - Phonetic pattern expansion"
	@echo ""
	@echo "Targets:"
	@echo "  make run        - Build and run the demo (DEFAULT)"
	@echo "  make build      - Build the binary"
	@echo "  make setup-data - Copy dictionary file if needed"
	@echo "  make check      - Quick syntax/type check"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make help       - Show this help"
	@echo ""
	@echo "Quick Start:"
	@echo "  $$ make run"
	@echo ""
	@echo "Files:"
	@echo "  data/english_words.txt  - Dictionary (auto-copied from /usr/share/dict/words)"
	@echo "  src/main.rs             - Demo source"
