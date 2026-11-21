# faradice D1 Design: Linux-Native GUI

- Date: 2026-11-15
- Status: Partially implemented. The iced GUI MVP (phase 2 of §8) and a 3D box viewport are in
  `gui/` (§2.6), on the shared `faradice-engine`; `faradice-render`, the `faradice-view` debug
  viewer (skipped, §2.5), the headless server and phases 3–5 beyond the box viewport are not
  implemented. Toolkit license policy decided 2026-11-16 (§2.5); open questions in §9
- Target: the remaining phases of §8 (the MVP required GDS input, M3, which is implemented)
- Related: [M1 FRW solver](M1_frw_solver.md), [D2 agent API and MCP](D2_agent_api_mcp.md)

---

## 0. Summary

| Item | Proposal |
|---|---|
| Users | Analog/custom layout engineers and PEX/sign-off engineers working on open PDKs (sky130, gf180mcu, ihp-sg13g2) |
| Toolkit | **iced** (MIT, wgpu-rendered, Elm architecture), with the **wgpu** 3D viewport drawn in the UI frame through iced's `shader` widget; egui for the Phase 0 debug viewer and as the fallback. No GPL-family dependencies (GPL, LGPL, AGPL), by maintainer decision (§2) |
| Architecture | The GUI is a **thin client of the engine service** (`faradice-engine`) that also backs the CLI and the MCP server. No solver logic in the GUI crate (§5) |
| Core UX promise | Pick nets, run, and *watch the error bars shrink*. Every run is reproducible and has an equivalent CLI command (§3, §4) |
| Non-goals | A layout editor (KLayout is the editor), schematic capture, simulation. The GUI inspects and extracts; it does not draw polygons |

The GUI exists because FRW has two properties that a terminal shows poorly: results arrive
*progressively* with a statistical error, and the *spatial* reason for a coupling (which
geometry the walks hit) is what an engineer needs to trust a number. Both are visual.

---

## 1. Users and Jobs-to-be-Done

| Persona | Typical question | What the GUI must make cheap |
|---|---|---|
| Analog layout engineer | "How much does my new shield cut the coupling from `clk` to `vin_p`?" | Run 2-10 nets at high accuracy, compare two layout revisions side by side |
| PEX / sign-off engineer | "Are these 300 critical nets within 1% total and 5% coupling, and is the domain big enough?" | Batch runs from a preset, convergence dashboard, warnings triage, SPEF export |
| Device / methodology engineer | "Is the substrate model or `ground.ignore = physics` changing the answer?" | A/B runs differing in one option, diff of matrices with σ-aware significance |
| Student / newcomer to open PDKs | "What is the capacitance of this structure at all?" | Sensible defaults, explanations of every option in place, example structures |

Design consequence: the **default path must be three actions** (open, pick nets, run), and
the **expert path must expose every option of M1 §4** without leaving the GUI.

---

## 2. Toolkit Choice

### 2.1 Constraints and criteria

**Hard constraint (maintainer decision, 2026-11-16, §2.5):** no GPL-family license (GPL, LGPL,
AGPL) in the GUI's dependencies, linked statically or dynamically. The GUI crate is an ordinary
workspace member under the unchanged permissive allowlist (`deny.toml`), with no exceptions.

Scope, stated precisely because crate metadata alone does not settle it:

- **In scope:** every Rust crate in the resolved dependency graph, *including sources they bundle
  or generate code from* (e.g. Wayland protocol XML files compiled into bindings), and every
  native library the GUI requires specifically (toolkits, webviews, image or font libraries).
- **Out of scope:** the operating system's base runtime that every Linux program links (glibc,
  libgcc_s, the dynamic loader) and the user's graphics driver stack. glibc is LGPL; the policy
  targets toolkit-level dependencies, not the OS. This interpretation was confirmed by the
  maintainer on 2026-11-16 (§2.5).
- **Enforcement:** `cargo deny` checks crate metadata only, so Phase 0 adds a one-time audit of
  the exact resolved graph of the chosen toolkit (bundled protocol and asset files included),
  repeated whenever the toolkit or winit is upgraded.

Consequences:

- Excluded: GTK4/libadwaita (LGPL-2.1+), Qt (LGPLv3/GPL), Slint (GPL-3.0-only or proprietary
  royalty-free/commercial terms), and WebKitGTK-based webviews such as Tauri/wry on Linux (LGPL).
- Also excluded, because MPL-2.0 is not on the allowlist either: libcosmic (COSMIC's widget
  layer on top of iced). Plain iced is MIT.
- Allowed native libraries are permissive ones: Mesa (MIT), the Vulkan loader (Apache-2.0),
  libwayland and libxkbcommon (MIT).
- **Known issue affecting every winit-based toolkit (iced, egui, Xilem, Freya, Dioxus Native):**
  winit's `wayland` feature pulls in `wayland-protocols-plasma`, whose crate metadata is MIT but
  which bundles KDE protocol files such as `blur.xml` under LGPL-2.1-or-later and generates Rust
  bindings from them (verified for winit 0.30.12). Remedy, required before GUI code lands: build
  against a winit without `wayland-protocols-plasma` (the blur effect is cosmetic), through an
  upstream feature flag in a released winit/iced (preferred, to be proposed upstream), or as the
  single exception to "crates.io releases only" (§2.3): a small, audited patch against a
  *released* winit, vendored in the repository and applied with `[patch]`, removed once the
  upstream release exists. The Phase 0 audit verifies the result. X11-only builds are not an acceptable workaround.

Ranked criteria for the remaining candidates:

1. **Performance.** GPU-rendered UI; smooth scrolling through 10⁵-row coupling tables
   (virtualized: only visible rows are built); a 3D viewport with 10⁵-10⁶ boxes; no continuous
   redraw (and near-zero CPU) when nothing changes.
2. **Modern UI.** Custom theming, light and dark following the desktop, animation, Wayland-first
   with correct fractional scaling, X11 still working.
3. **wgpu viewport** drawn inside the UI frame, ideally on the same wgpu major version as
   `faradice-gpu` (wgpu 30), so that the renderer and the walk engine can share code and,
   later, a device.
4. **Architecture fit** with the engine's event stream (`JobEvent`, §5): state changes in one
   place, long work never on the UI thread.
5. **Accessibility** (AT-SPI on Linux, normally via AccessKit).
6. **Maturity**: release cadence, API churn, widget breadth, Flatpak packaging.

### 2.2 Candidates (checked on crates.io and GitHub on 2026-11-16; "?" = re-verify)

| | iced | egui (eframe) | GPUI | Xilem / Masonry | Freya, Dioxus Native |
|---|---|---|---|---|---|
| License | MIT | MIT OR Apache-2.0 | Apache-2.0 (framework only; the Zed editor crates are GPL-3.0, do not copy code from them) | Apache-2.0 | MIT; MIT OR Apache-2.0 |
| Latest release | 0.14.0 (2025-12-07); master is 0.15.0-dev | 0.37.0 (2026-10-15), releases every few months | 0.2.2 on crates.io (2025-10-22); the ecosystem (gpui-component) pins unofficial `gpui-pre` snapshots | 0.4.0 (2025-10-29) | Freya 0.5.2 (2026-11-15); Dioxus Native 0.7.10 |
| Renderer | wgpu (tiny-skia software fallback) | wgpu (glow optional) | blade-graphics (own Vulkan layer), **no wgpu** | vello on wgpu | Skia; vello (Blitz) |
| wgpu version | 27 in 0.14; **30 on master** (MSRV 1.93, pinned winit fork) | **30**, same as `faradice-gpu` today | — | vello's | — / vello's |
| Model | Elm architecture: retained widgets, messages, `Subscription` for streams | Immediate mode | Retained, reactive (entity/view) | Reactive, retained | React-like components |
| Look | Modern, flat, fully themable; COSMIC desktop apps are built on it | Utilitarian; themable but recognizably egui | **Most polished** (Zed) | Early | Modern |
| Accessibility | **Not implemented** (iced issue #552 open) | AccessKit integrated (eframe feature) | Little or none (?) | AccessKit by design | Partial (?) |
| Data-heavy widgets | Scrollable + `lazy`/`responsive`; table widget status to verify (?) | `egui_extras` table with row virtualization, `egui_plot` | gpui-component: virtualized table and list | Thin | Thin |
| Churn | Pre-1.0, breaking changes per release | Pre-1.0, small breaks per release | No stability promise; follows Zed's needs | Alpha | Young |

### 2.3 Recommendation: iced for the GUI, egui for the Phase 0 debug viewer

Rationale, in priority order:

1. **License.** MIT, passes `cargo deny` unchanged, and needs no copyleft native library. The
   winit protocol issue of §2.1 applies equally to every remaining candidate, so it does not
   distinguish them; it must be resolved either way.
2. **Performance with a modern look.** Fully GPU-rendered through wgpu; message-driven updates on
   a retained widget tree, so idle windows do not redraw. COSMIC shows that it carries a whole
   desktop's worth of applications.
3. **Smaller viewport risk.** The `shader` widget gives wgpu access inside the widget tree, so
   the 3D view draws into the UI frame with no offscreen copy and no dmabuf interop (the main
   technical risk of the GTK plan). What remains to prove is depth-tested rendering, clipping and
   performance through the shader API (Phase 0 gate G3). iced master is on wgpu 30, the version
   of `faradice-gpu`; a *released* iced on wgpu 30 is still pending (gate G1).
4. **Architecture fit.** The Elm architecture maps naturally onto the engine: each `JobEvent`
   becomes a `Message` through a `Subscription`, and `update` is the only place state changes.
   Keeping only view state in the GUI (§5) remains a project convention, but this structure
   makes it easy to follow and to review.

Accepted risks and mitigations:

- **Accessibility is the largest gap** and a step back from the GTK plan: keyboard navigation
  and exports do not expose controls, focus, errors or live progress to a screen reader. The
  decision is therefore taken at the end of Phase 0, before the MVP, not deferred (gate G4):
  either iced has usable AccessKit support, or the project funds/contributes that integration,
  or the maintainer explicitly accepts "keyboard-only, no screen reader" for the first releases
  (documented in the README). If none of these holds, egui (AccessKit today) is chosen.
  Independently, the keyboard-first design of §3 and CLI parity are implemented.
  **Resolved 2026-11-16 (§2.5):** the maintainer does not require screen-reader support and
  keeps iced as the more modern toolkit; egui is no longer a fallback for accessibility.
- **Pre-1.0 churn, MSRV and the unreleased version.** Pin an exact iced release per faradice
  release; the GUI is a thin client of `faradice-engine`, so only widget code needs porting.
  Depend on crates.io releases only, never on iced master or its winit fork (the one exception
  is the vendored winit license patch of §2.1). iced 0.15 (wgpu 30)
  needs Rust 1.93: `faradice-gui` gets its own `rust-version` and stays out of the workspace
  MSRV job, like `faradice-gpu`. If no iced release on wgpu 30 exists at the Phase 0 decision
  (gate G1), choose between: iced 0.14 with `faradice-render` on wgpu 27 and no device sharing
  with `faradice-gpu` (two wgpu versions in one binary, accepted temporarily), or egui. In the
  first case `faradice-view` must use an egui release on the same wgpu 27 (egui 0.33 line) so
  that both clients keep sharing `faradice-render`; no version adapter layer is planned.
- **Dense tables.** Build one virtualized table for `faradice-gui` (fixed row height, only
  visible rows built, sortable columns) and measure it against gate G2. If it misses the
  target, egui is the fallback toolkit (below).

Why not the others:

- **egui** is the strongest runner-up: it is on wgpu 30 today and has AccessKit. Its default
  look is more utilitarian (a subjective judgment, but relevant to the "modern UI" goal), and
  complex view state (job queue, compare views, undo of option edits) needs more structure of
  our own in immediate mode. It remains the toolkit of the
  Phase 0 developer viewer (`faradice-view`) and the fallback if the iced table spike fails.
- **GPUI** (Zed's toolkit, widely regarded as very fast and polished; not benchmarked by us)
  renders through blade rather than wgpu (the
  viewport would need a second GPU API or a copy), its crates.io release lags Zed by a year,
  and it has no stability promise. Re-evaluate if it gains a stable release with wgpu or
  external-texture support.
- **Xilem/Masonry, Freya, Dioxus Native**: promising, but too early for a multi-year tool.

**Phase 0 gates** (all measured inside the Flatpak build, on the development machine (Radeon
8060S iGPU, Mesa RADV) and on one low-end Intel iGPU, at 2560×1440 with 150 % fractional scaling
and at 3840×2160, on GNOME and KDE Wayland). The toolkit choice is confirmed only if all pass:

| Gate | Criterion |
|---|---|
| G0 License | Audit of the resolved graph (§2.1) clean, with the winit protocol remedy in place |
| G1 Release | A crates.io iced release on the wgpu major of `faradice-gpu`, or one of the two alternatives above chosen explicitly |
| G2 Tables | 10⁵-row coupling table: scrolling and sorting with p95 frame time ≤ 16.7 ms and p99 ≤ 33 ms; filter keystroke to updated view ≤ 50 ms |
| G3 Viewport | 10⁶ instanced, depth-tested boxes with correct clipping inside the layout: orbiting at p95 ≤ 16.7 ms on the development machine and ≤ 33 ms on the low-end iGPU |
| G4 Accessibility | One of the three outcomes in the accessibility risk above, recorded |
| G5 Idle and live load | Idle window: < 1 % of one core, no redraws; while a job streams progress at 4 Hz into the table, plot and viewport: UI thread < 10 % of one core, input latency ≤ 50 ms |
| G6 Memory | Resident memory with G2 and G3 data loaded: < 1 GiB |
| G7 Platform | Correct fractional scaling and light/dark following the desktop on both GNOME and KDE |


### 2.4 Viewport integration plan

```
 faradice-render (wgpu, no iced dependency)
    │  prepare(): upload instance buffers (boxes, walk polylines, heatmaps) only on change
    │  render():  one render pass into the target view it is given
    ▼
 iced `shader` widget (Primitive::prepare / Primitive::render)
    └── same wgpu Device/Queue as the window; drawn in the UI frame, no copy
 offscreen (thumbnails, run reports): the same faradice-render into a texture, read back to PNG
```

- The viewport renders **on demand** (camera change, new data), not continuously.
- `faradice-render` is reused by `faradice-view` (egui paint callbacks), report thumbnails and
  possibly a WebGPU viewer.
- **Device sharing with the GPU walk engine** is an optimization, not a requirement: two devices
  on one adapter work. Sharing needs `faradice-gpu` to accept an external `Device`/`Queue`
  (e.g. `GpuEngine::with_device`) and both crates on the same wgpu major; then walk evidence
  buffers could be drawn without a CPU round trip.

### 2.5 Decision record

- **2026-11-16, maintainer:** GPL-family licenses (GPL, LGPL, AGPL) are not acceptable for the
  GUI; preferred is a high-performance, modern UI framework. The OS base runtime (glibc etc.) is
  out of scope (§2.1; confirmed by the maintainer the same day). The earlier recommendation (GTK4 +
  libadwaita as a separate application with an LGPL exception) is withdrawn. The GUI is
  under the unchanged allowlist; no license exception is needed. Toolkit: iced (§2.3), with
  egui as the fallback.
- **2026-11-16, maintainer:** implement the GUI with iced directly (the egui Phase 0 viewer
  is skipped).
- **2026-11-16, maintainer (gate G4):** screen-reader support is not required. iced is kept
  because it is the more modern toolkit; the GUI is keyboard-operable, and every GUI action
  has a CLI equivalent with JSON/CSV outputs. If iced gains AccessKit support, the GUI adopts
  it without a separate decision.

### 2.6 Implementation status (2026-11-16)

- `gui/` is a **separate Cargo workspace** (`faradice-gui`, `publish = false`): the license
  patch below needs `[patch.crates-io]`, which must not affect the library workspace, and
  the library CI stays free of the GUI toolchain. It uses `faradice-engine` only (D1 §5).
- **iced 0.14.0** (the latest release, wgpu 27; gate G1 alternative "iced 0.14 with two wgpu
  versions"): with the `gpu` feature, the binary contains wgpu 27 (UI) and wgpu 30
  (`faradice-gpu`); no device sharing until an iced release on wgpu 30.
- **License remedy (G0):** `gui/vendor/fetch.sh` downloads winit 0.30.13 and
  wayland-protocols-misc 0.3.12 (pinned checksums) and removes the LGPL KDE protocol files:
  KDE blur (winit) and KDE server-decoration (found by the audit; unused). The patches never
  contain the LGPL text itself. `gui/tools/license_audit.py` (license expressions and
  bundled GPL-family texts of all 286 crates with all features) and `cargo deny` run in CI.

Gate measurements so far (Radeon 8060S iGPU, Mesa RADV, GNOME Wayland, 1280×820 window,
release build); the remaining ones are open:

| Gate | Result |
|---|---|
| G0 License | Pass: audit and `cargo deny` clean with the two patched crates |
| G1 Release | Alternative chosen: iced 0.14 / wgpu 27 |
| G2 Tables | Implemented (virtualized, fixed row height, only visible rows built); **not yet measured** at 10⁵ rows |
| G3 Viewport | 10⁶ instanced, depth-tested boxes while orbiting: p50 16.67 ms, p95 17.02 ms, p99 18.3 ms (vsync-bound at 60 Hz); 10⁵ boxes: p95 16.9 ms. 4K, low-end iGPU, KDE and Flatpak **not yet measured** |
| G4 Accessibility | Decided: screen reader not required, iced kept (§2.5); keyboard operation and CLI parity |
| G5 Idle and live load | Idle window: 0 % CPU over 10 s, no redraws. Live load **not yet measured** |
| G6 Memory | 1.8 GB peak RSS when loading the 10⁶-box structure (50 MB TOML parse included); steady state **not yet measured** |
| G7 Platform | Light/dark follows the desktop (GNOME checked); KDE **not yet checked** |

---

## 3. UX Principles

1. **Defaults that are right for 80% of runs.** `accuracy.preset = "default"`, ground boundary
   `ground` with `margin = auto`, power nets as ground, ground/power nets excluded from targets.
   The run button works without opening any settings.
2. **Progressive disclosure in three levels.** *Basic* (nets, preset, ground on/off),
   *Advanced* (the M1 §4 tables grouped as Physical model / Computation scope / Postprocessing),
   *Raw* (the TOML itself, with schema validation and completion). The A/B/C layering of M1 §4 is
   shown literally, with a badge: "changes the physics", "changes cost", "no rerun needed".
3. **Never block.** Runs are jobs; the window stays interactive; several runs can be queued.
   Changing a postprocessing (C) option re-renders results instantly from raw statistics.
4. **Every number has its σ.** No capacitance is displayed without its error; values whose
   relative error is above target are visually muted, not hidden.
5. **Explain warnings where they occur.** A boundary-ratio warning is shown next to the net, with
   a one-click "increase margin and rerun" action and a link to the 3D view of boundary hits.
6. **Keyboard first.** A command palette (`Ctrl+K`) reaches every action and option by name;
   all panes have shortcuts; tables support type-ahead filtering.
7. **Reproducible by construction.** Every run stores resolved config, seed, input hashes and
   tool version (§4.6). "Rerun exactly" and "copy CLI command" are always one click away.
8. **CLI parity.** Anything the GUI can do, the CLI can do; the GUI shows the equivalent command
   and config for every run and export.

---

## 4. Workflows

### 4.1 Open a project

A *project* is a directory with `faradice.project.toml` that references inputs and holds `runs/`.

- Inputs: a structure (`.faradice.toml`, `.cap3d`, or from M3 a GDS plus a technology file:
  layer map, stack, and connectivity rules for the selected PDK).
- Drag-and-drop a GDS: the GUI detects known PDKs by layer usage and proposes a tech file
  (sky130A, gf180mcuD, ihp-sg13g2), never silently.
- Recent projects on the start page; opening a single structure file creates an implicit project
  in `$XDG_STATE_HOME/faradice/scratch/` and offers "Save as project".

### 4.2 Pick nets

- Net browser: searchable list with glob filter (`clk*`, `data[0-7]`), columns for role
  (target / environment / ground / power / ignored), box count, bounding box, layers.
- Roles are set with single keys (`t` target, `e` environment, `g` ground, `p` power,
  `x` ignore) and mapped to `extract.nets`, `ground.nets`, `power.nets`, `scope.ignore_nets`.
- For GDS (M3+): "Trace net from label" (`faradice gds trace --net NAME`) runs as a job and
  highlights the traced conductor in 3D; label conflicts and opens are listed as findings.
- Selecting nets in the 3D view (click, box-select) and in the list are the same selection.

### 4.3 Choose a preset and run

- Preset selector: `fast` / `default` / `signoff` / `reference`, each with its targets visible
  (e.g. "total 1%, couplings ≥5% of total to 5%, 3σ").
- Before running, the GUI calls `validate` and `estimate` on the engine and shows
  "≈ 2.1×10⁶ walks, ≈ 40 s on 16 threads (±50%)". Validation errors block the run with fixes.
- `Ctrl+Enter` runs; the run appears in the run list immediately with state `queued`.

### 4.4 Watch convergence live

- Per-net rows with a horizontal **error bar** (current relative error at `confidence_sigma`
  vs target), walks done, estimated time remaining, and convergence state.
- A convergence plot (relative error vs walks, log-log, with the 1/√N guide line) for the
  selected net; tracked couplings appear as thinner lines. Deviation from 1/√N is a hint of a
  heavy-tailed weight distribution and is flagged.
- Partial results are usable: matrix and 3D views update every N batches (throttled to ≤ 4 Hz).
- Actions while running: cancel, and "stop when current batch is done" (both `job.cancel`,
  D2 §2.1). Deferred until the engine has matching operations (`job.pause`, `job.update` with
  lifecycle and reproducibility semantics, to be added to D2): pause/resume, raising
  `max_walks` for one net, changing the job priority.

### 4.5 Inspect results

- **Summary**: totals with σ, ground breakdown (`by_source` always available in the GUI, since
  raw statistics are kept), warnings with remediation.
- **Coupling matrix / heatmap**: rows = targets, columns = all sinks; color = C_ij / C_ii
  (log scale), hatching for "not statistically significant" (C_ij < kσ); hover shows value ± σ,
  walk count, and whether the coupling was tracked. Sorting by magnitude, clustering by net name.
- **Top couplings table**: top-k per net with value, σ, fraction of total, and the effect of the
  current threshold settings (moved to ground or kept).
- **Postprocessing live preview**: editing `coupling.*` thresholds updates the matrix and shows
  "N couplings moved to ground, total conserved", with no rerun.

### 4.6 3D viewer

- Conductors as boxes colored by net role; layers can be toggled; the dielectric stack is a
  translucent side gauge; the domain boundary is a wireframe.
- **Walk visualization**: a sampled subset of walks (e.g. 1 in 10⁴, stored by the engine on
  request with `debug.record_walks`) drawn as polylines of cube centers, with cubes on hover.
  Terminations are drawn as points colored by sink. This makes "why is C(boundary) large" or
  "which part of net B absorbs the coupling" visible in seconds.
- **Coupling contribution map**: the *weighted, signed* walk contributions (the estimator
  weights of M1 §5, antithetic pairs combined per sample) accumulated per conductor box/face, with
  per-region σ, drawn as a heatmap and filterable by source net. Raw termination *counts* are
  shown only as a separately labeled "hit fraction" layer: because weights vary in magnitude and
  sign, hit density is **not** proportional to induced charge and can point at the wrong region.
- Spatial views need recorded evidence (versioned, §4.9). Runs without it, including all runs
  made before evidence recording exists, show "evidence unavailable for this run" with a "rerun
  with evidence" action, never an empty or misleading map.
- Section planes (x/y/z), measure tool, camera bookmarks, screenshot to PNG with metadata.

### 4.7 Compare runs

- Select two runs: a config diff (only changed keys) plus a matrix diff where each cell is
  `z = ΔC / sqrt(σ_a² + σ_b² − 2·Cov(a, b))`; |z| < 2 is drawn neutral, so noise is not mistaken
  for change.
- The covariance term matters: two runs with the **same seed** share random streams and are
  correlated. The compare action therefore defaults to **independent seeds** for A/B runs it
  launches (Cov = 0, the simple denominator is valid). For existing same-seed runs, the covariance
  is unknown and the GUI says so ("runs share seed 1: z assumes independence and may be wrong")
  instead of silently using the independent formula. Paired same-seed runs with an estimated
  covariance (common random numbers, lower variance of Δ) are a later optimization.
- Typical use: layout revision A vs B with the same config; same layout with
  `ground.ignore = none` vs `physics`; `fast` vs `signoff` to justify the preset.

### 4.8 Export

- SPEF (C-only in M1), CSV matrix, JSON report; the export dialog shows the postprocessing that
  will be applied and the exact `faradice report` command that would produce the same file.

### 4.9 Reproducibility record (per run)

```
runs/2026-11-15T10-42-07_7f3a/
  request.json          # what the user asked for (engine API request, §5)
  config.resolved.toml  # every option, defaults filled in, incl. effective domain and eps_stop
  inputs/               # immutable snapshots: structure/GDS, tech file (content-addressed copies)
  geometry.bin          # the exact solved geometry (boxes, nets) after tracing and net removal
  manifest.json         # tool version, git hash, Green-table hash, backend (CPU/GPU, device), seed,
                        # threads, host; sha256 of every file above
  events.ndjson         # progress events as recorded
  result.json           # raw statistics (M1 §6), incl. per-net walks and stop reason
  evidence/             # optional, versioned spatial evidence (§4.6); absent = "unavailable"
  report.json, *.spef   # derived artifacts, each with the postprocessing config hash
  command.sh            # equivalent CLI invocation (runs against inputs/)
```

The directory is **self-contained and replayable**: "Rerun exactly" uses `inputs/` and
`config.resolved.toml`, not the original paths, so edits to the source files after the run cannot
change what is replayed. Runs that were cancelled or hit a time limit record that state, the
per-net walk counts at stop, and are never presented as complete. Large input snapshots may be
deduplicated in a project-level content-addressed store and hard-linked.

---

## 5. Architecture: One Engine, Three Clients

```
             ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐
             │ faradice-gui  │  │ faradice CLI  │  │ faradice mcp     │
             │ (iced)        │  │ (--json)      │  │ (stdio / HTTP)   │
             └──────┬────────┘  └──────┬────────┘  └────────┬─────────┘
                    │ EngineClient trait (in-process or remote)│
             ┌──────┴──────────────────┴─────────────────────┴─────────┐
             │ faradice-engine: projects, job queue, event bus,        │
             │ artifact store, validate/estimate, schemas              │
             └──────┬───────────────────────────────────────────┬──────┘
                    │                                           │
         faradice-io / faradice-frw / faradice-geom       faradice-render (wgpu)
```

- `faradice-engine` owns all state that matters: projects, jobs, artifacts. Clients hold only
  view state (selection, camera, open panes). This is what makes CLI, GUI and MCP agree.
- **`EngineClient`** is one Rust trait with two implementations: in-process (GUI default, CLI)
  and remote (JSON-RPC over a Unix socket or HTTP, see D2). The GUI never calls the solver
  directly.
- **Event stream**: `JobEvent::{Queued, Started, Progress{net, walks, rel_err, eta}, PartialResult,
  Warning, Finished, Failed, Cancelled}`. The GUI subscribes and folds events into its model on
  the UI thread as iced messages (one `Subscription` per engine connection); the solver runs on a
  rayon pool, never on the UI thread.
- **Headless server mode**: `faradice serve --listen unix:/run/user/1000/faradice.sock` or
  `--listen https://0.0.0.0:7420` (token auth). The GUI's "Connect to server" uses the remote
  client, so extraction runs on a big compute host while the GUI (and its 3D view) runs on a
  laptop. Artifacts are fetched lazily (summary first, walk samples only when the 3D view asks).
- **GUI-generated CLI**: the GUI builds an engine `RunRequest`; the engine renders it as both a
  config TOML and a CLI line (`faradice solve layout.gds --tech sky130A.tech.toml -c run.toml
  --nets 'clk*' --preset signoff --seed 1`). The same renderer is used by `faradice --dry-run`.

### 5.1 Crate layout additions

| Crate | Contents | Depends on iced? |
|---|---|---|
| `faradice-engine` | Job queue, events, artifact store, request/response types, JSON Schemas | No |
| `faradice-render` | wgpu scene (boxes, walks, heatmaps), offscreen rendering | No |
| `faradice-gui` | iced app, widgets (virtualized table, convergence plot), `shader` viewport glue | Yes |
| `faradice-view` | Phase 0 egui debug viewer for structures and result JSON | No (egui) |

---

## 6. Wireframes

### 6.1 Main window (results)

```
┌ faradice ─ bus_m1m3 ─────────────────────────────────── [▶ Run ⌃↵] [Ctrl+K] ☰ ─┐
│ ┌ Nets ──────────────┐┌ 3D ─────────────────────────────────┐┌ Run ───────────┐ │
│ │ / clk*             ││                                     ││ Preset signoff │ │
│ │ ● clk     target   ││        ▄▄▄▄▄▄▄▄▄▄  met3             ││ Ground   plane │ │
│ │ ● clk_b   target   ││   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  met2  · · walks ││ Nets  4 target │ │
│ │ ○ vin_p   env      ││        ▀▀▀▀▀▀▀▀▀▀  met1             ││ ≈ 38 s (16 thr)│ │
│ │ ○ VSS     ground   ││  ────────────────── substrate       ││ ▸ Advanced     │ │
│ │ ○ VPWR    power    ││ [layers] [section z=1.2] [walks ✓]  ││ ▸ Raw TOML     │ │
│ └────────────────────┘└─────────────────────────────────────┘└────────────────┘ │
│ ┌ Convergence ─────────────────────────────────────────────────────────────────┐ │
│ │ net    C_total (fF)      rel.err @3σ  target  walks     state                │ │
│ │ clk    12.431 ± 0.019    ▕██▏ 0.46%   0.5%    1.2e6     ✓ converged          │ │
│ │ clk_b  11.902 ± 0.041    ▕████▏ 1.0%  0.5%    0.6e6     ◐ running  ~12 s     │ │
│ │ ! clk_b: boundary 1.8% of total (> 1%)  [Increase margin & rerun] [Show hits]│ │
│ └──────────────────────────────────────────────────────────────────────────────┘ │
│ [Summary] [Matrix] [Top couplings] [Compare] [Log]            run 7f3a · seed 1 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Coupling heatmap

```
┌ Matrix ── color: C_ij/C_ii (log) ── ▢ hatch = not significant (<2σ) ───────────┐
│            clk    clk_b   vin_p   vin_n   VSS(gnd)  <substrate>  <boundary:zmax>           │
│ clk        ■■■    ▓▓      ░       ·       ▓▓▓       ▓        ·                  │
│ clk_b      ▓▓     ■■■     ·       ░       ▓▓▓       ▓        ░                  │
│ hover: clk → vin_p = 0.212 ± 0.009 fF (1.7% of total, tracked, 3.1e4 hits)      │
│ thresholds: abs 0.01 fF, rel 1% (either) → 14 couplings to ground, Σ conserved  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Command palette

```
┌ > bound                                                    ┐
│   Set ground.boundary (all faces)…              Advanced A │
│   Increase domain.margin…                       Advanced A │
│   Show boundary hits in 3D                      View       │
│   Explain warning: boundary ratio               Help       │
└────────────────────────────────────────────────────────────┘
```

### 6.4 Compare

```
┌ Compare  run 7f3a (rev A)  ↔  run 91c0 (rev B, +shield) ────────────────────────┐
│ Config diff: (none)   Inputs diff: layout.gds sha256 1c2e… → 88ab…              │
│ coupling        A (fF)          B (fF)          Δ        z       verdict        │
│ clk → vin_p     0.212 ± 0.009   0.041 ± 0.006   -80.7%   -15.8   significant    │
│ clk → VSS       3.110 ± 0.020   3.402 ± 0.021   +9.4%    +10.1   significant    │
│ clk_b → vin_n   0.018 ± 0.007   0.015 ± 0.007   -16.7%   -0.3    noise          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Integration with the Ecosystem

- **KLayout**: a small KLayout macro (Python) sends the current cell, selected nets and tech
  name to a running engine (`faradice serve`) and opens the result in the GUI; highlighting a
  coupling in the GUI can send a marker back to KLayout (same socket, D2 API).
- **OpenROAD**: GUI is not involved; the flow uses the CLI or Python bindings. The GUI can open
  any run directory produced by a flow for inspection.
- **Files**: `.faradice-run` directories are self-contained (input snapshots and solved
  geometry included, §4.9); opening one does not require the original project.

---

## 8. Roadmap

| Phase | Milestone | Content | Exit criterion |
|---|---|---|---|
| 0 | with M1 | `faradice-view` (egui): open structure + `result.json`, 3D boxes, matrix table. `faradice-render` crate. iced spike: `shader` viewport and a 10⁵-row virtualized table on Mesa and NVIDIA, GNOME and KDE Wayland | Spike report confirms iced or switches to egui (§2.3); developers use the viewer for debugging |
| 1 | with M2 | `faradice-engine` (job queue, events, artifact store); CLI ported onto it; `--json` everywhere | CLI and engine share one request type; reproducibility record written |
| 2 | M3 (GDS) | GUI MVP: projects, net browser, presets, run with live convergence, summary/matrix/top couplings, SPEF export, CLI command copy | A new user extracts a sky130 example in < 2 min without docs |
| 3 | M3+ | 3D viewer in iced, walk and hit-density visualization, compare runs, command palette, postprocessing live preview | Boundary-ratio warning can be diagnosed visually |
| 4 | M4+ | Headless server + remote connect, KLayout bridge, Flathub release, keyboard-operation audit (screen reader if iced adds AccessKit) | Flathub listing; GUI drives a remote 64-core run |
| 5 | M5 | GPU backend selection and monitoring, run sweeps (parameter studies), report generation (HTML) | — |

---

## 9. Open Questions

0. **Toolkit** (decided §2.5): iced, confirmed or replaced by egui at the end of Phase 0
   according to gates G0-G7 (§2.3).
1. **Accessibility outcome** (gate G4): decided 2026-11-16 (§2.5). Screen-reader support is
   not required; iced is kept. The Phase 4 review only audits keyboard operation.
2. **Desktop integration**: follow light/dark and accent color from the XDG settings portal on
   both GNOME and KDE; how far to go beyond that (native file dialogs via the portal, yes).
3. **Evidence and walk recording format**: a versioned schema (`evidence/v1`) for weighted
   per-box/face contributions (always cheap) and sampled walk polylines (off by default, 10⁴ walks
   max, compressed, on request). Requires new `debug.*` config keys, which the current config
   parser rejects as unknown; they will be added with a config `schema_version` bump.
4. **Project file**: do we need one, or is "a directory with `runs/`" enough? Lean towards a
   minimal `faradice.project.toml` for tech/PDK binding.
5. **Tech file authoring**: should the GUI include an editor for layer maps / stacks, or only
   validate them (preference: validate + show stack diagram, edit in text)?
6. **Web viewer**: the engine API plus `faradice-render` on WebGPU could give a browser viewer
   for remote runs. Worth it, or does the remote desktop client cover the need?
7. **Telemetry**: none by default. Should there be an opt-in crash report? (Preference: no.)
