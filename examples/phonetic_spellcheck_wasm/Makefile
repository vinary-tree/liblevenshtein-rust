# Phonetic Spellcheck WASM Build Makefile

.PHONY: build build-release build-debug serve clean test help

# Default target
all: build

# Build for web (release, optimized for size)
build: build-release

build-release:
	wasm-pack build --target web --release --out-dir pkg
	@echo ""
	@echo "Build complete! WASM output in pkg/"
	@ls -lh pkg/*.wasm 2>/dev/null || true

# Build for web (debug, faster compile, larger output)
build-debug:
	wasm-pack build --target web --dev --out-dir pkg
	@echo ""
	@echo "Debug build complete! WASM output in pkg/"

# Optimize the WASM binary further (requires wasm-opt from binaryen)
optimize: build-release
	@if command -v wasm-opt > /dev/null 2>&1; then \
		echo "Optimizing WASM with wasm-opt..."; \
		wasm-opt -Os pkg/phonetic_spellcheck_wasm_bg.wasm -o pkg/phonetic_spellcheck_wasm_bg.wasm; \
		echo "Optimization complete!"; \
		ls -lh pkg/*.wasm; \
	else \
		echo "wasm-opt not found. Install binaryen for further optimization."; \
	fi

# Build for WASI (server-side/edge WASM runtimes)
build-wasi:
	cargo build --target wasm32-wasip1 --release
	@mkdir -p pkg
	cp target/wasm32-wasip1/release/phonetic_spellcheck_wasm.wasm pkg/phonetic_spellcheck_wasi.wasm
	@echo ""
	@echo "WASI build complete!"
	@ls -lh pkg/phonetic_spellcheck_wasi.wasm

# Start a local development server
serve: build
	@echo ""
	@echo "Starting development server at http://localhost:8080"
	@echo "Press Ctrl+C to stop"
	@echo ""
	python3 -m http.server 8080 -d www

# Run tests
test:
	cargo test
	@echo ""
	@echo "Note: For browser tests, run 'wasm-pack test --headless --chrome'"

# Run browser tests (requires Chrome)
test-browser:
	wasm-pack test --headless --chrome

# Clean build artifacts
clean:
	rm -rf pkg target
	@echo "Cleaned build artifacts"

# Show help
help:
	@echo "Phonetic Spellcheck WASM Build System"
	@echo ""
	@echo "Targets:"
	@echo "  build         Build release WASM for browser (default)"
	@echo "  build-debug   Build debug WASM (faster compile)"
	@echo "  build-wasi    Build for WASI runtime"
	@echo "  optimize      Optimize WASM with wasm-opt"
	@echo "  serve         Build and start dev server at :8080"
	@echo "  test          Run Rust unit tests"
	@echo "  test-browser  Run wasm-bindgen browser tests"
	@echo "  clean         Remove build artifacts"
	@echo "  help          Show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - wasm-pack: cargo install wasm-pack"
	@echo "  - Python 3 (for serve target)"
	@echo "  - binaryen (optional, for optimize target)"
