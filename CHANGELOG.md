# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0,
minor versions may change file formats and interfaces.

## [Unreleased]

## [0.1.0]

First public release.

### Solver

- Floating random walk extraction of totals and couplings for Manhattan (box) geometry,
  with a statistical error (σ) on every value and per-net accuracy targets (presets
  `fast`, `default`, `signoff`, `reference`).
- Results bit-identical for any thread count (per-walk random streams).
- Boundaries: open space, a grounded box, a substrate plane, reflecting sides.
- Dielectrics: uniform; stratified stacks with exact transition cubes crossing an
  interface; dielectric boxes and conformal shells around conductors, with transition cubes
  centred on dielectric edges and first hops adapted to thin layers and cross-sections.
- Ground handling and coupling thresholds as explicit options.

### Input and output

- Structure files (`*.faradice.toml`) and run configurations (`faradice.toml`).
- GDSII input: flattening, Manhattan decomposition, connectivity tracing from labels, with
  technology files (`*.tech.toml`); a SKY130 stack with its conformal sidewall dielectrics
  is included.
- Outputs: raw statistics (`result.json`), reports (JSON, CSV) and SPEF; reports can be
  regenerated with other options without re-solving (`faradice report`).

### Analysis

- `faradice sens`: dC/dp for widths, thicknesses and positions (finite differences with
  path continuation, or shape derivatives), exact deletions and additions of conductors.
- `faradice whatif`: ground, float, merge or swap conductors in an extracted matrix without
  re-solving.
- `faradice map`: maps of where a small grounded or floating object would change a coupling
  most.

### Tools

- Desktop GUI (iced, `gui/`): live convergence, results and a 3D view.
- Experimental GPU walk engine (wgpu/Vulkan, uniform dielectric; CLI feature `gpu`).
- `bench/compare/`: accuracy comparisons with Palace (FEM) and FasterCap (BEM).

### Known limitations

- Walks near conformal shells are several times slower than in stacks without shells.
- SPEF output has lumped capacitances per net only (no `*CONN`, no resistance).
- Only Linux (x86_64) is tested.

[Unreleased]: https://github.com/hayatree/faradice/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hayatree/faradice/releases/tag/v0.1.0
