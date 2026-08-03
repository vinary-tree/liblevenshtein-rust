# Developer Guide

Documentation for contributing to and understanding liblevenshtein-rust internals.

## Contents

### [Architecture](architecture.md)
Deep dive into the library architecture:
- Core trait system (Dictionary, DictionaryNode)
- Distance computation algorithms
- Memory management strategies
- Backend implementations

### [Building](building.md)
Instructions for building the project:
- Build requirements
- Feature flags and optional dependencies
- Cross-compilation
- Development builds vs. release builds

### [Contributing](contributing.md)
Guidelines for contributing:
- Code style and conventions
- Testing requirements
- Pull request process
- Issue reporting

### [Performance](performance.md)
Performance optimization guide:
- Profiling tools and techniques
- Benchmark interpretation
- Optimization strategies
- Common performance pitfalls

### [Publishing](publishing.md)
Release and publishing procedures:
- Version management
- Release checklist
- Publishing to crates.io
- Documentation updates

### [Cargo workspace layout](workspace-layout.md)
Why the procedural-macro package is verified as an independent workspace and
which commands CI must run for complete coverage.

## Related Documentation

- [Design Documents](../design/README.md) - Algorithm and feature designs
- [Bug Reports](../bug-reports/README.md) - Known issues and fixes
- [Research](../research/README.md) - Optimization research and experiments
