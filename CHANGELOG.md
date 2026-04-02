# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--preflight` flag for benchmark CLI to verify GPU environment before running
- Platform support documentation (`docs/platform-support.md`)
- Published FF validation test harness (`test/integration/test_published_ff_validation.py`)
- Benchmark results saving to `benchmark_results/` by default
- CHANGELOG.md (this file)

### Changed
- Overhauled `AGENTS.md` and `benchmarks/AGENTS.md` for better AI agent guidance
- OpenMM-CUDA-12 platform gate: now excludes only macOS (was Linux-only)
- JAX engine now supports both harmonic and MM3 functional forms

### Fixed
- Hessian unit conversion for Jaguar `.in` files (Hartree/Bohr² → kJ/mol/Å²)
- Golden fixture tolerance tightened from 2e-3 to 5e-4
- UTF-8 output encoding on Windows for benchmark CLI

## [5.0.0a3] - Pre-release

### Added
- Modern Python rewrite of Q2MM with clean architecture
- Format-agnostic data models (`ForceField`, `Q2MMMolecule`)
- OpenMM, JAX, JAX-MD, Tinker, Psi4 backend engines
- Seminario method for Hessian-based force constant estimation
- Scipy-based optimizers (L-BFGS-B, Nelder-Mead, trust-region)
- Batched Hessian evaluation via `jax.vmap`
- Comprehensive test suite (692+ tests)
- MkDocs documentation site

[Unreleased]: https://github.com/ericchansen/q2mm/compare/v5.0.0a3...HEAD
[5.0.0a3]: https://github.com/ericchansen/q2mm/releases/tag/v5.0.0a3
