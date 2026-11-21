# faradice

**faradice** (Farad + dice) is a 3D parasitic capacitance extractor written in Rust, based
on the floating random walk (FRW) method. It is aimed at the open-source IC design flow and
open PDKs; the repository ships a SKY130 technology file.

> **Status: alpha (0.1).** The solver, the CLI and the GUI work and are validated against
> exact solutions and independent field solvers (below). Interfaces and file formats may
> still change before 1.0.

## Features

- **Manhattan (box) geometry** from structure files or GDSII, with connectivity tracing
  from labels to nets.
- **Dielectrics:** a uniform dielectric, stratified stacks, dielectric boxes and
  conformal shells around conductors (such as SKY130's NILD3_C and NILD4_C). Transition
  cubes crossing one interface use analytical tables; cubes at dielectric edges and thin
  cross-sections use numerically converged kernels; elsewhere walks take smaller cubes.
  The solver never averages permittivities inside a transition cube.
- **A statistical error (σ) on every value.** Each net stops on its own once its relative
  error at 3σ meets the target; presets range from quick estimates (2 % total) to reference
  runs (0.1 %).
- **Reproducible:** results are bit-identical for any thread count.
- **Cost proportional to the nets you ask for:** totals and couplings of selected nets,
  with explicit options for ground handling and coupling thresholds.
- **Outputs:** raw statistics (`result.json`), reports (JSON, CSV) and SPEF.
- **Sensitivities and what-if analysis**, each with σ: dC/dp for widths, thicknesses and
  positions; exact deletions and additions of conductors; grounding, floating or merging
  conductors without re-solving; maps of where a small shield or fill would change a
  coupling most.
- **Desktop GUI** (iced) with live convergence, results and a 3D view.
- **Experimental GPU engine** (wgpu/Vulkan, uniform dielectric only).

## Limitations

- Geometry is Manhattan: axis-aligned boxes only.
- SPEF output carries lumped capacitances per net, without pin connectivity (`*CONN`) or
  resistance; it is meant for inspection and back-annotation by net, not for timing
  analysis.
- Walks near conformal shells are several times slower than in a stratified stack without
  shells (on the SKY130 bus with shells, about 4× the time of the same bus without them).
  Work on this is ongoing.
- The GPU engine is a prototype: uniform dielectric, `f32`, and not bit-identical to the
  CPU engine.
- Only Linux (x86_64) is tested.
- No Python bindings or KLayout/OpenROAD integration yet.

## Validation

- **Exact solutions:** parallel plates, the unit cube, image problems, single and
  multiple dielectric interfaces, potentials at dielectric edges. All agree within the
  test tolerances (a few σ).
- **Independent field solvers** (Palace FEM with adaptive meshes; FasterCap BEM), with
  `bench/compare/`: three-wire, crossing-via and SKY130 bus structures in a uniform
  dielectric, in a stratified stack and with conformal shells. Against Palace's finest
  meshes, totals differ by at most 0.43 % and major couplings by 0.52 %; Palace converges
  from above, and against its extrapolated values the differences are at most 0.12 % and
  0.36 %.
- Details: [`docs/validation/`](docs/validation/).

## Documentation

- [User reference](docs/reference.md): the CLI and every input file.
- [Documentation index](docs/README.md): design documents, validation reports and
  related work.
- [Data provenance](docs/data-provenance.md): where the technology files and examples
  come from.

## Repository layout

| Path | Contents |
|---|---|
| `crates/faradice` | The `faradice` CLI |
| `crates/faradice-core` | Units, identifiers, sinks, statistics, errors |
| `crates/faradice-geom` | Conductor boxes, nets, L∞ nearest-distance spatial indices |
| `crates/faradice-frw` | FRW solver: transition tables, walks, estimators, dielectrics |
| `crates/faradice-io` | Structure and configuration TOML, result JSON, reports (JSON, CSV, SPEF) |
| `crates/faradice-gds` | GDSII reader, flattening, Manhattan decomposition, connectivity tracing |
| `crates/faradice-engine` | Run engine shared by the CLI and the GUI: projects, runs, background jobs and events |
| `crates/faradice-sens` | Sensitivities, deletions, additions and what-if edits |
| `crates/faradice-gpu` | Experimental wgpu/Vulkan walk engine |
| `gui` | The desktop GUI (iced), a separate workspace; see [gui/README.md](gui/README.md) |
| `tech` | Technology files (layer maps and stacks) |
| `examples` | Example structures, configurations and a GDS file |
| `bench` | Comparisons with other field solvers and cost studies; see [bench/README.md](bench/README.md) |
| `docs` | User reference, design documents, validation reports; see [docs/README.md](docs/README.md) |

## Quick start

```sh
cargo build --release
./target/release/faradice solve examples/three_wires.faradice.toml -c examples/faradice.toml -o out
# → out/result.json (raw statistics), out/report.{json,csv}, out/three_wires.spef
./target/release/faradice report out/result.json -c examples/faradice.toml   # re-post-process without re-solving
```

GDSII input with connectivity tracing (labels → nets):

```sh
./target/release/faradice gds trace examples/bus.gds --tech tech/sky130_uniform.tech.toml
./target/release/faradice gds trace examples/bus.gds --tech tech/sky130_uniform.tech.toml --net 'bus[1]' --json
./target/release/faradice solve examples/bus.gds --tech tech/sky130_uniform.tech.toml -o out
```

Desktop GUI (iced; live convergence, results, 3D view):

```sh
cd gui && vendor/fetch.sh && cargo run --release -- ../examples/bus.gds --tech ../tech/sky130_uniform.tech.toml
```

Sensitivities and what-if analysis ([D3](docs/design/D3_sensitivity.md)), all with σ:

```sh
# dC/dp for widths, thicknesses and translations, exact deletions ([[delete]]) and exact
# additions such as a shield ([[add]]); see examples/three_wires.sens.toml
./target/release/faradice sens examples/three_wires.faradice.toml -p examples/three_wires.sens.toml -o out
# → out/sens.json, plus the usual result.json and reports
# --method auto: shape derivatives (much sharper) on faces without geometric edges, such as
# plates spanning a domain with reflecting sides; finite differences elsewhere

# Ground, float, merge or swap conductors without re-solving
# (extract with [output] cross_moments = true for exact σ)
./target/release/faradice whatif out/result.json --float cross --merge left,right

# Where would a small shield (grounded) or dummy fill (floating) change C(left, victim) most?
./target/release/faradice map examples/three_wires.faradice.toml --nets left,victim \
    --grid 5,-0.3:1.0:27,1.0:1.9:19 --radius 0.02 -o map.csv
```

Every reported capacitance carries its standard error (σ). Options for ground handling,
target nets, accuracy and coupling thresholds are documented in the
[user reference](docs/reference.md).

## Building

```sh
cargo build --release        # the CLI and libraries (CPU engine)
cargo test                   # tests of the same crates
```

The minimum supported Rust version (MSRV) is 1.85 for the CLI and the libraries. The
experimental GPU engine (`faradice-gpu`, CLI feature `gpu`) needs Rust 1.87, and the
desktop GUI in `gui/` (a separate workspace) needs Rust 1.92; plain `cargo build` and
`cargo test` leave the GPU engine out. Linux (x86_64) is the supported platform; macOS and
Windows may work but are not tested yet.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). faradice is implemented clean-room from published
papers; the design documents cite the papers each method relies on. Please do not
contribute material derived from proprietary tools, manuals or PDKs.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
