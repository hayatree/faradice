# faradice-gui

Desktop GUI of faradice, built with [iced](https://iced.rs) (MIT). It is a thin
client of `faradice-engine`, so a GUI run gives the same numbers as the CLI
(design: [D1](../docs/design/D1_gui.md)).

## Build and run

```sh
cd gui
vendor/fetch.sh            # once: patched winit / wayland-protocols-misc (see below)
cargo run --release -- ../examples/three_wires.faradice.toml
cargo run --release -- ../examples/bus.gds --tech ../tech/sky130_uniform.tech.toml
cargo run --release --features gpu     # also offer the GPU walk engine
```

Requires Rust 1.92+, Linux (Wayland or X11) and a Vulkan or GL capable GPU
driver. Without one, iced falls back to software rendering (tiny-skia) and
the 3D view stays empty.

## What it does

- Open a structure or GDSII layout (toolbar, `Ctrl+O`, or drop files on the
  window; a dropped `*.tech.toml` is used as technology file, any other
  `*.toml` as run configuration).
- Pick target nets (filter by text or glob), accuracy preset, seed, threads and
  walk engine. **Copy CLI command** gives the equivalent `faradice solve` line.
- Run (`Ctrl+Enter`), watch per-net progress and the convergence plot
  (relative error vs walks with the 1/√N guide and the target), cancel (`Esc`)
  with partial results kept.
- Results: totals, ground, σ and couplings per net, live while running.
- Matrix: the capacitance matrix of the target nets (fF), totals on the diagonal, each
  coupling and ground shaded by its share of the row's total, values within the
  confidence band faded.
- 3D view: depth-tested boxes coloured by net; drag to orbit, right/middle
  drag to pan, wheel to zoom; the net selected in Results is highlighted.
- Export (`Ctrl+E`): `result.json`, `report.json`, `report.csv` and SPEF.

## License policy and the patched crates

faradice uses no GPL-family code (GPL, LGPL, AGPL), including files bundled
inside dependencies. Two crates in iced's Wayland graph (winit, Apache-2.0;
wayland-protocols-misc, MIT) depend on or bundle KDE protocol files licensed
LGPL-2.1-or-later. `vendor/fetch.sh`
downloads them from crates.io (pinned checksums) and removes those protocols:

| Crate | Removed | Effect |
|---|---|---|
| winit 0.30.13 | `wayland-protocols-plasma` (KDE blur) | `Window::set_blur` is a no-op on Wayland |
| wayland-protocols-misc 0.3.12 | KDE `server-decoration` protocol (unused) | none |

The patches stay under the upstream licenses; see
[`vendor/NOTICE.md`](vendor/NOTICE.md). `tools/license_audit.py` checks the resolved dependency graph (license
expressions and bundled license texts); CI runs it with `cargo deny`.

## Accessibility

The GUI is keyboard-operable, and every GUI action has a CLI equivalent
(*Copy CLI command*) with JSON/CSV outputs. Screen readers are not supported:
iced has no accessibility tree yet. This is a maintainer decision (design D1,
gate G4); the GUI will adopt AccessKit when iced supports it.

## Self-test and benchmark modes

- `FARADICE_GUI_SHOTS=DIR faradice-gui INPUT`: load, run, save a screenshot of
  each view into `DIR` (PPM), exit.
- `FARADICE_GUI_BENCH=1 faradice-gui INPUT`: spin the 3D view and print frame
  interval percentiles, exit.
