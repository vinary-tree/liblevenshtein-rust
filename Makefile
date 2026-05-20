.PHONY: coverage coverage-quick coverage-ci coverage-persistent coverage-proptest coverage-report coverage-stable clean-coverage

# Full coverage with all features + branch coverage (requires nightly)
coverage:
	cargo llvm-cov clean --workspace
	mkdir -p target/coverage
	cargo +nightly llvm-cov --all-features --branch --lcov --output-path target/coverage/lcov.info
	cargo +nightly llvm-cov report --html --output-dir target/coverage/
	@echo "Report: target/coverage/html/index.html"

# Quick coverage (default features only, requires nightly for branch)
coverage-quick:
	cargo +nightly llvm-cov --branch --html --output-dir target/coverage/quick

# CI with threshold enforcement (requires nightly for branch)
coverage-ci:
	mkdir -p target/coverage
	cargo +nightly llvm-cov --all-features --branch \
		--fail-under-lines 70 \
		--fail-under-branches 60 \
		--fail-under-functions 75 \
		--lcov --output-path target/coverage/lcov.info

# Persistent ARTrie focused coverage
coverage-persistent:
	PROPTEST_CASES=500 cargo +nightly llvm-cov --features persistent-artrie,group-commit,parallel-merge \
		--branch --html --output-dir target/coverage/persistent

# Extended proptest coverage
coverage-proptest:
	PROPTEST_CASES=1000 cargo +nightly llvm-cov --all-features --branch \
		--html --output-dir target/coverage/proptest

# Generate HTML report from existing coverage data
coverage-report:
	cargo llvm-cov report --html --output-dir target/coverage/

# Coverage without branch (works on stable)
coverage-stable:
	cargo llvm-cov clean --workspace
	mkdir -p target/coverage
	cargo llvm-cov --all-features --lcov --output-path target/coverage/lcov.info
	cargo llvm-cov report --html --output-dir target/coverage/
	@echo "Report: target/coverage/html/index.html"

clean-coverage:
	cargo llvm-cov clean --workspace
	rm -rf target/coverage
