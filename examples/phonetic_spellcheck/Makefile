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
#   make build      - Build only (no run)
#   make clean      - Remove build artifacts
#   make help       - Show this help

# Configuration
CARGO := cargo

# File paths (relative to example directory)
RULES_SRC := rules/zompist.llev
RULES_BIN := rules/zompist.llev.bin
DICT_SRC := /usr/share/dict/words
DICT_DST := data/english_words.txt

.PHONY: all build run clean setup-data help check

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

# Check that it compiles (faster than full build)
check:
	$(CARGO) check

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	rm -f $(RULES_BIN)
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
	@echo "  make build     - Build the binary"
	@echo "  make run       - Build and run interactively"
	@echo "  make setup-data- Copy dictionary file if needed"
	@echo "  make check     - Quick syntax/type check"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Files:"
	@echo "  rules/zompist.llev     - 62 Zompist phonetic rules"
	@echo "  data/english_words.txt - Dictionary (auto-copied from /usr/share/dict/words)"
	@echo "  src/main.rs            - Interactive demo source"
	@echo ""
	@echo "Example Session:"
	@echo "  $$ make run"
	@echo "  query> fone"
	@echo "  Matches for \"fone\" (normalized: \"fon\"):"
	@echo "    1. phone (distance: 0)"
	@echo "    2. foe (distance: 1)"
