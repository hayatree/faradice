# faradice M1 Design: Single-Dielectric FRW 3D Capacitance Solver

- Date: 2026-11-15
- Status: Historical design (the M1 plan as accepted on 2026-11-15, decisions in §11). The
  solver, the option system and the outputs are implemented; proposals that were not
  implemented are marked *(not implemented)* in place, and the implementation differs in some
  details
- Target (at design time): faradice v0.1.0
- Background: [01_capacitance_extraction_survey](../research/01_capacitance_extraction_survey.md),
  [02_oss_landscape_novelty](../research/02_oss_landscape_novelty.md)

> This document records the design before implementation. The current user reference for the
> CLI, the structure and configuration files and the outputs is [`docs/reference.md`](../reference.md).

---

## 0. Policy

| Item | Decision |
|---|---|
| Target | Fast, accurate 3D capacitance extraction for open-source PDKs (sky130 / gf180mcu / ihp-sg13g2) |
| M1 goal | Complete an FRW solver for **box-shaped (Manhattan) conductors + a uniform dielectric**, and demonstrate its accuracy and reproducibility with numbers |
| M1 focus | Fix the option system for **ground handling** and **extraction scope (how far to look at self / coupling)** from the start, so that the CLI and configuration files remain compatible when multilayer dielectrics and GDS input are added in M2 and later |
| Open-source readiness | Make the repository publishable from the first commit (license, CI, documentation, clearly stated provenance of test data) |

### 0.1 In scope / out of scope for M1

| In scope (M1) | Out of scope (M2 and later) |
|---|---|
| Conductors represented as sets of boxes (each box tagged with a net name) | GDS/OASIS reading, layer operations, automatic net recognition (M3) |
| Uniform dielectric (a single ε_r) | Multilayer and conformal dielectrics (M2) |
| Ground modeling: substrate ground plane, ground nets, domain boundary conditions | Floating conductors (treating dummy fill as floating; M2) |
| Target net / environment net distinction, coupling thresholds and couple-to-ground | 2.5D tables, pattern caching (M4) |
| Statistical stopping criteria (separate tolerances for total and coupling) | GPU acceleration, ML-based acceleration (M5) |
| Output: JSON (with error bars), CSV matrix, SPEF (C only) | Resistance (R) extraction, RC reduction |
| `.cap3d` (RWCap format) input, for benchmark comparisons *(not implemented)* | Process bias, trapezoidal cross-sections |

---

## 1. Repository Design as an Open-Source Project

### 1.1 License

- **Dual license `MIT OR Apache-2.0`** (the common choice in Rust).
  - Can be embedded in OpenROAD (BSD-3), KLayout (GPL-3), and gdsfactory (MIT).
  - Apache-2.0 includes a patent clause. Commercial EDA vendors hold many patents in this field, so this protects users.
- Dependency crates are restricted to MIT/Apache/BSD-style licenses and checked by `cargo deny` in CI. No GPL crates.
- Test data is either self-made or derived from open-source PDKs with clear licenses (sky130 is Apache-2.0).
  Commercial PDK data and commercial tool output are never committed to the repository.
- **Clean-room policy**: nothing is implemented based on commercial tool manuals, binaries, or output.
  Methods are grounded only in published papers, with sources recorded in `docs/references.md`
  *(not created; sources are cited in `docs/research/` and in the design documents)*.
  StarRC option names appear only as a "mapping table for users" (§4.6).

### 1.2 Cargo workspace layout

```
faradice/
├── Cargo.toml                # [workspace]
├── crates/
│   ├── faradice-core/           # Units, ID types, error types, result types (minimal dependencies)
│   ├── faradice-geom/           # Boxes, conductors, spatial index (L∞ distance queries)
│   ├── faradice-tech/           # Layer stack definition (M1: ε_r and z ranges only; extended in M2) (not implemented)
│   ├── faradice-frw/            # FRW solver core. A pure library with no I/O
│   │   ├── green/            #   Cube Green's function tables (generation and sampling)
│   │   ├── gauss/            #   Gaussian surface construction and sampling
│   │   ├── walk/             #   A single walk
│   │   ├── estimator/        #   Statistics accumulation, stopping criteria
│   │   └── boundary/         #   Ground plane, boundary conditions
│   ├── faradice-io/             # Input (faradice TOML/JSON, .cap3d) and output (JSON, CSV, SPEF)
│   └── faradice-cli/            # `faradice` binary
├── docs/                     # mdBook (design, user guide, validation reports)
├── examples/                 # Analytical-solution cases, small sky130 structures
├── benches/                  # criterion benchmarks
├── tests/                    # Integration tests (golden comparisons)
└── .github/workflows/        # CI
```

*Implemented layout (differs from the plan above):* there is no `faradice-tech` crate (the
dielectric stack is part of the structure file and of `faradice-frw`; GDS technology files are
read by `faradice-gds`), the binary crate is `crates/faradice` rather than `faradice-cli`, and
`faradice-engine`, `faradice-gds`, `faradice-sens` and `faradice-gpu` were added later.
`faradice-io` reads the faradice TOML structure format only (no `.cap3d`). `docs/` is plain
Markdown (no mdBook); integration tests and benchmarks live in each crate's `tests/` and
`benches/` directories.

- `faradice-frw` does not depend on I/O or file formats. Python bindings (`faradice-py`, PyO3, planned for M3; not implemented) and
  KLayout/OpenROAD integration are thin layers on top of this library.
- The public API consists of two crates: `faradice-frw` and `faradice-io`. Both follow semver; while at 0.x, breaking changes are allowed in minor releases.

### 1.3 Open-source project essentials (provided from day one)

- `README.md` (purpose, quick start, accuracy table, comparison with other tools), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CHANGELOG.md` (Keep a Changelog format), `LICENSE-MIT`, `LICENSE-APACHE`
- CI (GitHub Actions): `fmt`, `clippy -D warnings`, `test` (Linux/macOS/Windows; the implemented CI tests on Linux), `cargo deny`,
  MSRV check (targeting roughly stable−2, stated in `rust-version`), documentation build, accuracy regression tests
  (analytical-solution cases, judged against statistical tolerance bands)
- `#![forbid(unsafe_code)]` by default. If unsafe becomes necessary (e.g., for SIMD), it is allowed per module with a documented rationale.
- Configuration files and output JSON carry a **`schema_version`**, and JSON Schemas are published under `docs/schema/`
  (the `schema_version` fields are implemented; published JSON Schemas are not).

---

## 2. Units and Coordinates

| Quantity | Internal unit | Notes |
|---|---|---|
| Length | µm (f64) | Input may specify `nm`, `um`, or `m` |
| Capacitance | fF (f64) | ε0 = 8.8541878128e-3 fF/µm |
| Permittivity | Relative permittivity ε_r | |

- Axes: x, y span the layout plane; z is height. The top of the substrate is z = 0 (matching the PDK layer stack).
- Precision: geometry is handled in f64. Arithmetic inside walks is also f64 in M1 (f32 is evaluated in M5).

---

## 3. Input Model

### 3.1 Structure file (`*.faradice.toml`)

```toml
schema_version = 1
units = "um"

[dielectric]
eps_r = 3.9                  # M1 uses a uniform dielectric. Extended to [[dielectric.layer]] in M2

[[conductor]]
net = "A"
boxes = [                    # [xmin, ymin, zmin, xmax, ymax, zmax]
  [0.0, 0.0, 1.0, 10.0, 0.14, 1.36],
  [9.86, 0.0, 1.0, 10.0, 5.0, 1.36],
]
[[conductor]]
net = "B"
boxes = [[0.0, 0.42, 1.0, 10.0, 0.56, 1.36]]
[[conductor]]
net = "VSS"
boxes = [[-2.0, -2.0, 0.0, 12.0, 7.0, 0.1]]
```

- Boxes belonging to the same net are treated as electrically equipotential (M1 assumes the input is correct).
- Overlapping and touching boxes are allowed (they occur at vias and cell boundaries). Overlapping boxes from different nets are an error.
- `faradice-io` can also read `.cap3d` (for comparison with RWCap and others). In M1, only the box and uniform-dielectric subset of cap3d is supported. *(Not implemented.)*

### 3.2 Run configuration (`faradice.toml`)

The configuration is kept separate from the structure file, because the same structure is analyzed many times with different options. A full example is given in §4.7.

---

## 4. Option System for Ground and Extraction Scope (the core of M1)

In FRW, the meaning of the result depends on "where walks terminate" and "what each termination point is counted as".
Therefore, the following three layers are made explicit as **separate settings**.

```
(A) Physical model     … defines the field problem itself (changes the resulting values)
    ├─ ground plane / ground nets / domain boundary conditions
    └─ conductors included in the analysis (environment extent, ignored nets)
(B) Computation scope  … which nets launch walks, which capacitances must converge (changes computational cost)
    ├─ target nets
    └─ couplings subject to convergence checks, and their tolerances
(C) Output postprocessing … how the computed matrix is reported (no recomputation needed)
    ├─ coupling thresholds, couple-to-ground
    └─ symmetrization, net merging
```

Changing only (C) does not trigger recomputation. Raw statistics are stored in the result file, so that
`faradice report` can redo just the postprocessing.

### 4.1 (A-1) Ground modeling

Ground (the 0 V reference) is handled as one of the following three kinds, and **the kind of termination point reached is recorded separately**.

| Kind | Setting | Treatment in FRW |
|---|---|---|
| **Substrate ground plane** | `ground.substrate = "plane"`, `z = 0.0` | The infinite plane z = z_sub is a Dirichlet boundary (0 V). Cube size is also limited by the distance to the plane, and walks reaching the plane are counted as `<substrate>` |
| **Ground net** | `ground.nets = ["VSS", "VGND", "vssd1"]` (glob allowed) | Treated as an ordinary conductor. Walks reaching it are summed into "capacitance to ground". Alternatively, it can be reported per net without summing |
| **Domain boundary** | `ground.boundary.{xmin,xmax,ymin,ymax,zmax} = "ground" \| "reflect" \| "open"` | `ground`: Dirichlet 0 V; termination is `<boundary:FACE>`. `reflect`: Neumann boundary; the walk is mirrored at the face and continues (below). `open`: infinity is 0 V (§4.1.2). *(Implemented per face: `ground` and `reflect`; the open boundary is selected for the whole domain with `domain.kind = "open"`.)* |

- **Sink labels.** `result.json` (schema v2) labels the fixed ground sinks `<substrate>`,
  `<boundary:FACE>` and `<infinity>`, and the per-target ground sum `<GND>`. Net names of the
  form `<…>` are refused when the geometry is built, so a label never collides with a net (a net
  may be called `GND`). Version 1 files (`GND:substrate`, …, sum `GND`) are read and relabelled.
- `ground.report = "merged" | "by_source"`
  - `merged`: all ground components are combined into a single `C_g` (the standard when writing SPEF).
  - `by_source`: `<substrate>`, `<boundary:FACE>`, `VSS`, etc. are reported separately. **A large capacitance to the domain boundary indicates that the domain is too small**, which is important for debugging.
- **The default domain boundary is `ground`, with margin `domain.margin = auto`** (implemented as an omitted `domain.margin`)
  (from the conductors' bounding box, "maximum wire pitch × k" or roughly the full height of the layer stack; the concrete value is determined by validation in §9).
- A warning is issued if the boundary capacitance ratio `C_boundary / C_total` exceeds `warn.boundary_ratio` (default 1%)
  (implemented with a fixed 1 % threshold; there is no `warn.boundary_ratio` key).

#### 4.1.1 Implementation of the `reflect` boundary

Mirroring the domain across an axis-aligned plane (Neumann boundary) keeps the extended solution harmonic.
Moreover, the distance from a point p in the domain to a mirrored conductor is never shorter than the distance to the original conductor.
Hence **the boundary face can be ignored when computing cube size; a sampled point that crosses the boundary simply needs to be mirrored back**.
The implementation cost is nearly zero.
Uses: periodic structures (e.g., the middle of a bus) and comparison tests against the exact parallel-plate solution.

#### 4.1.2 `open` boundary (infinity as ground)

Outside a sphere of radius R enclosing all conductors, the Poisson kernel for the sphere exterior is used as follows:
- With probability 1 − R/r, the walk is considered to escape to infinity and is counted as `<infinity>`.
- Otherwise, it is returned to a point on the sphere surface.
- If a substrate ground plane is present, the sphere is centered on the ground plane and an odd extension (method of images) is used.
  Walks returning to the lower hemisphere have the sign of their weight flipped.
- Use: comparison against the analytical reference value for the capacitance of a cube in free space (C ≈ 0.6606785 · 4πε·a). Real designs use the `ground` boundary.

#### 4.1.3 Option to ignore ground (`ground.ignore`)

"Ignoring ground" can mean either **excluding it from the field computation** or **only excluding it from the output**.
These two give results with completely different meanings, so they are specified as distinct values.

| `ground.ignore` | Physical model | Output | Use |
|---|---|---|---|
| `"none"` (default) | As configured in §4.1 | As configured in §4.1 | Normal extraction |
| `"report"` | Solved with ground included (values unchanged) | **C_g is not output**. Only couplings are written to SPEF. Raw statistics are kept in `result.json`, so `faradice report` can restore them | When ground cap is handled by a separate flow, or when comparing couplings only |
| `"physics"` | **Ground is removed from the field computation**: the substrate ground plane is removed, ground nets are deleted from the geometry, and boundary faces set to `ground` are forced to `reflect` | No ground component exists | Theoretical analysis of mutual capacitance between conductors only; sensitivity analysis of ground effects (look at the difference from `none`) |

Individual settings are also available; `ground.ignore` is treated as a shorthand that sets them together.

- `ground.substrate = "plane" | "none"`
- `ground.nets_treatment = "ground" | "net" | "remove"` (`net`: VSS etc. are treated as ordinary nets and their couplings are reported. `remove`: deleted from the geometry)
- `ground.report = "merged" | "by_source" | "drop"`

**Caveats of `"physics"` (made explicit in documentation and warnings)**
- If all boundaries are `reflect` and there is no ground at all, the system has no potential reference (it is floating as a whole).
  In this case every walk terminates at some conductor, and each row of the capacitance matrix sums to exactly 0 (there is no ground component).
  The problem is mathematically well-posed, but **the result depends on the size of the analysis domain**.
- Walks that wander far away can take many steps, so `walk.max_steps` sets an upper bound.
  The number of walks that hit the bound is reported as a statistic (values are not silently discarded).
- If `open` is specified for a boundary, infinity effectively remains as ground. This is physically unavoidable, so
  even with `"physics"`, capacitance to infinity is reported as `<infinity>` and a warning is issued.

Net patterns (`extract.nets`, `--nets`, `ground.nets`, `scope.ignore_nets`) are globs (`*`, `?`,
`[...]`). A pattern that equals a net name literally also selects that net, so `bus[2]` works
without escaping. A pattern that contains an escape (`[*]`, `[?]`, `[[]`) is a pure glob:
`bus[[]2]` matches only `bus[2]`.

### 4.2 (A-2) Extent of conductors included in the analysis (environment)

| Setting | Value | Meaning |
|---|---|---|
| `scope.environment` *(not implemented)* | `"all"` (default) | All conductors in the structure file are included in the field computation |
| | `"ambit"` | Only conductors within distance `scope.ambit` of target nets are included (to limit memory on large inputs; an approximation) |
| `scope.ambit` *(not implemented)* | length | Distance for `"ambit"` mode. Lateral and vertical can be specified separately |
| `scope.ignore_nets` | list of globs | Nets **deleted from the geometry**. It is documented that this changes the physics, and a warning is issued when used |

- *(Not implemented, with `ambit`.)* If conductors were cut off by `ambit`, walks reaching beyond the range are treated as hitting the "domain boundary".
  They follow the `ground.boundary` setting, which is ground by default. The fraction of walks reaching beyond the range is also reported.

### 4.3 (B-1) Target nets: which nets' self cap to compute

| Setting | Example | Meaning |
|---|---|---|
| `extract.nets` | `["*"]`, `["clk*", "data[0-7]"]` | **Nets that launch walks** (targets). Their self (total) cap and the couplings seen from them are obtained |
| `extract.exclude_nets` | `["VSS", "VDD*"]` | Excluded from targets. They remain as conductors (become environment) |
| `extract.nets_file` *(not implemented)* | path | One net name per line (for large selections) |

Default: ground nets and power nets specified in `power.nets` are not targets. Power nets can be either treated as ground or
kept as independent nets (`power.treat_as = "ground" | "net"`).

**FRW property**: cost is roughly proportional to the number of target nets and almost independent of the number of environment nets
(only the spatial index lookup grows, as O(log n)). This makes FRW very advantageous for "computing only a few important nets with high accuracy".
This is the decisive difference from BEM and FEM, which solve the full-chip matrix.

### 4.4 (B-2) Accuracy and convergence: how far to look at couplings

FRW error decreases as 1/√N in the number of walks N. **The smaller the coupling, the slower its relative error converges.**
As a rule of thumb, obtaining a coupling with C_ij / C_ii = p to relative error ε requires roughly N ∝ 1/(p·ε²) walks
(further multiplied by a factor representing the variance of the weights).

| Example | p (C_ij / C_total) | Target relative error | Required walks (relative) |
|---|---|---|---|
| total cap | 1 | 0.5% | 4×10⁴ |
| Major coupling | 10% | 5% | 4×10³ |
| Small coupling | 1% | 5% | 4×10⁴ |
| Very small coupling | 0.1% | 5% | 4×10⁵ |

Therefore, "which couplings are subject to the convergence check" directly determines run time.
This is the setting that matters most in practice for random-walk extractors (cf. StarRC's 3D solver), so it is made explicitly configurable as follows.

```toml
[accuracy]
total_rel_tol      = 0.005    # Target relative error for self (total) cap
coupling_rel_tol   = 0.05     # Target relative error for tracked couplings
coupling_track     = { rel = 0.05, abs_ff = 0.0, op = "or" }
                              # Couplings subject to the convergence check: C_ij >= 5% × C_total or
                              # C_ij >= abs_ff. Untracked couplings are still output, but their accuracy is not guaranteed
                              # (implemented without `op`)
coupling_track_max = 0        # 0 means unlimited. >0 tracks only the top k couplings by magnitude (not implemented)
confidence_sigma   = 3.0      # How many σ the error is evaluated at (1σ ≈ 68%, 3σ ≈ 99.7%)
min_walks          = 10_000   # Prevents premature stopping (variance estimates are unstable early on)
max_walks          = 50_000_000  # Per-net upper bound. When reached, the net is flagged as "not converged"
batch_walks        = 4_096    # Unit for statistics updates and stopping checks
```

- Stopping condition: stop when `total` is satisfied **and** all tracked couplings are satisfied.
  The set of tracked couplings is updated every batch based on the current estimates.
  Hysteresis is applied to prevent the stopping decision from oscillating when a coupling hovers around the threshold.
- Every output value carries a **standard error σ**. Commercial tools output only values, so this is a differentiator for faradice.
- Presets: `accuracy.preset = "fast" | "default" | "signoff" | "reference"`
  (e.g., total target 2% / 1% / 0.5% / 0.1%, coupling target 10% / 5% / 3% / 1%). Individual settings take precedence.

### 4.5 (C) Output postprocessing: coupling thresholds and couple-to-ground

| Setting | Value | Meaning |
|---|---|---|
| `coupling.mode` | `"keep"` (default) / `"ground"` | `ground`: all couplings are moved to the ground cap of both nets (= couple-to-ground) |
| `coupling.ground_multiplier` | 1.0 | Factor applied when moving to ground (0–2 is used as a Miller-effect approximation) |
| `coupling.abs_threshold_ff` | 0.0 | Couplings smaller than this are moved to ground |
| `coupling.rel_threshold` | 0.0 | Couplings with C_ij < rel × C_total are moved to ground. Which net's total is used is selected by `rel_ref` |
| `coupling.rel_ref` | `"either"` (default) / `"both"` / `"min"` / `"max"` | `either`: moved if below the criterion for either net. `both`: moved only if below the criterion for both nets |
| `coupling.threshold_op` | `"and"` / `"or"` | How the abs and rel thresholds are combined |
| `coupling.significance_sigma` | 0.0 | Couplings with C_ij < kσ (not statistically significant) are moved to ground. An FRW-specific noise countermeasure |
| `coupling.to_non_target` | `"keep"` / `"ground"` | Whether couplings to non-target (environment) nets are kept or moved to ground |
| `coupling.symmetrize` | `"inverse_variance"` (default) / `"average"` / `"max_side"` / `"none"` | When both nets are targets, how the estimates C_ij and C_ji are combined into one |

- **Charge conservation**: when moving couplings to ground by threshold, the moved coupling value is added to C_g of both nets.
  Each net's total cap is unchanged by the move (when the multiplier is 1.0). This is tested as a property.
- Postprocessed results alone can be output under a different name (`faradice report --config other.toml result.json`;
  implemented as `faradice report result.json -c other.toml -o DIR`).

### 4.6 Mapping table for StarRC users (informative)

> StarRC keyword names and behavior are informative only, based on public information and general knowledge; their accuracy is not guaranteed.
> Consult the official manual for exact semantics. The faradice implementation follows the definitions in this document, not this table.

| faradice | Similar StarRC concept / keyword (informative) |
|---|---|
| `coupling.mode = "ground"` | `COUPLE_TO_GROUND: YES` |
| `coupling.abs_threshold_ff` | `COUPLING_ABS_THRESHOLD` |
| `coupling.rel_threshold` | `COUPLING_REL_THRESHOLD` |
| `coupling.threshold_op` | `COUPLING_THRESHOLD_OPERATION` |
| `coupling.ground_multiplier` | Equivalent to `COUPLING_MULTIPLIER` |
| `extract.nets` / `exclude_nets` | `NETS` / net selection options |
| Solving target nets with the 3D solver | Specifying nets extracted by the field solver (Rapid3D net selection) |
| `accuracy.*` | 3D solver accuracy settings (the notion of guaranteeing an error bound per net) |

### 4.7 Full configuration example

*(As designed. The implemented parser rejects unknown keys, so this example does not load as
is: `scope.environment`/`scope.ambit`, `coupling_track.op` and `run.deterministic` are not
implemented, and `domain.margin = "auto"` is written by omitting the key.)*

```toml
schema_version = 1

[ground]
ignore = "none"              # "none" | "report" | "physics" (§4.1.3)
substrate = "plane"          # or "none"
substrate_z = 0.0
nets = ["VSS", "vssd*", "VGND"]
nets_treatment = "ground"    # "ground" | "net" | "remove"
report = "merged"            # "merged" | "by_source" | "drop"
[ground.boundary]
xmin = "ground"; xmax = "ground"; ymin = "ground"; ymax = "ground"; zmax = "ground"

[power]
nets = ["VDD", "vccd*", "VPWR"]
treat_as = "ground"          # or "net"

[domain]
margin = "auto"              # or 5.0 (µm)

[scope]
environment = "all"          # or "ambit"
ambit = 3.0
ignore_nets = []

[extract]
nets = ["*"]
exclude_nets = []

[accuracy]
preset = "default"
total_rel_tol = 0.005
coupling_rel_tol = 0.05
coupling_track = { rel = 0.05, abs_ff = 0.0, op = "or" }
confidence_sigma = 3.0

[coupling]
mode = "keep"
abs_threshold_ff = 0.0
rel_threshold = 0.0
rel_ref = "either"
threshold_op = "or"
significance_sigma = 2.0
to_non_target = "keep"
symmetrize = "inverse_variance"

[run]
threads = 0                  # 0 = all cores
seed = 1
deterministic = true

[output]
formats = ["json", "spef"]
cross_moments = false       # store row cross moments for `faradice whatif` σ (D3 §5.2)
```

---

## 5. Algorithm Details

### 5.1 FRW estimator

With 1 V applied to conductor i and 0 V to all others, the charge is given by the integral over a Gaussian surface G enclosing conductor i:

  Q_i = −∮_G ε ∂φ/∂n_out dS    (n_out: outward normal of G)

Placing a cube S₁ of half-side h centred at a point r on G (the cube is held fixed and φ is
differentiated at its centre) gives ∂φ/∂x_a(r) = ∮_{S₁} K_a(r, r′) φ(r′) dr′, where
K_a = ∂P/∂x_a is the gradient of the cube's surface Green function with respect to the
evaluation point. On the unit cube the kernel is tabulated as K̂; for half-side h,
K = K̂ / (2h) per unit area of the unit cube (density ∝ h⁻², gradient ∝ h⁻³, weight ∝ h⁻¹).

1. Draw a candidate r uniformly from the faces of the dilated boxes (total candidate area A_cand).
   Candidates not on the outer surface of the union, or outside the domain, are rejected and
   count as zero-weight samples (§5.4), so no estimate of A_G is needed.
2. With outward normal s·e_a at r, sample r′ on S₁ with density ∝ |K̂| and weight
   w = −A_cand · ε · s · sign(K̂(r′)) · ‖K̂‖₁ / (2h).
3. From r′ onward, perform a standard FRW. At point x, place "the largest cube that intersects no conductor or boundary",
   and move to the next point on the cube surface according to P(x, ·). Repeat until a conductor, the ground plane, or a boundary is reached.
4. If the termination is k, add w to C_ik. The mean over N samples is the estimate. The self cap is obtained from the same tally
   (C_ii is the contribution of walks returning to conductor i itself; C_ik (k ≠ i) is negative).
5. Antithetic pairs (default): each sample launches two walks from r ± h·u with weights ±w/2
   (K̂ is odd under point reflection through the cube centre). A pair is one sample; its
   contributions are accumulated per sink, and ground aggregates are accumulated per sample so
   that their variance includes covariances.

Sign convention for reporting: off-diagonal entries of the Maxwell matrix are negative, but output couplings are positive values (−C_ik).

### 5.2 Cube surface Green's function tables

- Consider a cube of side 2; compute the probability density P(x′, y′) on one face as seen from the center, and the normal gradient ∂_z P at the center.
  The gradient has different expressions for the top/bottom faces (z = ±1) and the side faces (x = ±1, y = ±1).
  - The solution with a Dirichlet condition on one face is expressed with a double sine series and sinh.
    **The per-cell probabilities can be computed by integrating the sines analytically, so the tables can be built with no numerical-quadrature error.**
  - Resolution is M × M cells per face (default M = 64; 32 / 64 / 128 are compared in §9).
    After a cell is chosen, the point is sampled uniformly within the cell. The error from this approximation depends on M and is evaluated during validation.
- Sampling: choose one of the 6 faces, then choose a cell on that face in O(1) with the alias method.
- Tables are not generated in `build.rs`; instead, they are **generated on first run, or a small pre-generated file embedded with `include_bytes!`** is used.
  The generation code itself is also published, and `faradice gen-green` can regenerate and verify the tables (for reproducibility).
  *(Implemented: tables are built in memory at run time from the analytical cell integrals; there
  is no embedded table file and no `gen-green` subcommand.)*
- In M2, the tables are extended to multilayer-dielectric cubes (cubes straddling dielectric interfaces) (the RWCap approach).
  Even in M1, `GreenTable` is abstracted as a trait *(not implemented as a trait; the layered
  kernels of M2 are separate table types)*.

### 5.3 Maximum cube size (L∞ distance query)

- The half-side of the largest cube centered at x that intersects no conductor is **the L∞ distance from x to the nearest box**.
  The L∞ distance to a box is max(|dx|, |dy|, |dz|), which is very cheap.
- The actual half-side is h = min(d_∞(conductor), d(substrate ground plane), d(ground boundary), d(ambit boundary)).
  `reflect` boundaries are not included (§4.1.1).
- Termination: when h < ε_stop, the walk is considered to have reached the nearest conductor.
  ε_stop = `walk.stop_ratio` × (that conductor's minimum dimension), with a default of about 1e-3 (determined in §9).
  This approximation introduces a small bias, which is quantified during validation.
- **Spatial index**: in M1, a uniform 3D grid registers the IDs of the boxes overlapping each cell, and the search proceeds in shells from nearest outward.
  Additionally, for each grid cell, "a lower bound on the distance from an empty cell to the nearest box" is precomputed,
  so that large cubes can be placed without searching in regions far from conductors (the key to speed).
  Comparison with a BVH is an M1 benchmarking task (abstracted via a `SpatialIndex` trait).
- Box data is laid out as SoA (xmin[], ymin[], …), in preparation for SIMD and GPU acceleration in M5.

### 5.4 Gaussian surface

- The Gaussian surface of net i is the surface of the solid obtained by inflating i's set of boxes by δ (excluding the interiors of other nets' conductors).
- Sampling: choose a face among the 6 faces of each inflated box in proportion to area, then sample a point uniformly on that face.
  If the point lies inside another inflated box of the same net, it is rejected (this yields a uniform distribution on the surface of the union).
  The rejection rate and the actual surface area A_G are estimated by Monte Carlo during preprocessing. They can be replaced by exact computation if needed.
- Choosing δ: `gauss.offset = "auto"` (implemented as `walk.gauss_alpha`). For each face, find the distance g to other conductors, set δ = α·g (α ≈ 0.5),
  and clamp it between upper and lower bounds. Varying δ per face complicates the geometry, so M1 uses **a constant δ per net**
  (the minimum gap around the net × α).
  - Since the first-step cube is centered on a point of the Gaussian surface, its half-side is at most δ (the distance to the net's own conductor).
- Gradient direction of the first step: only axis-aligned faces are considered, so the normal is one of ±x, ±y, ±z. Gradient tables for 3 directions × sign suffice.

### 5.5 Variance reduction (minimal in M1)

- Implemented in M1: area-proportional Gaussian-surface sampling, gradient-proportional importance sampling (§5.1),
  **antithetic sampling** (pairing the first step with respect to the sign of the gradient; reduces variance at almost no extra cost).
- Considered for M5: control variates, adaptive importance sampling for couplings, stratified sampling.

### 5.6 Random numbers, parallelism, and determinism

- RNG: counter-based (Philox4x32 or equivalent) or `rand_xoshiro` with jump;
  **a stream is uniquely determined by (seed, net_id, batch_id)**.
- Parallelism has two levels (rayon):
  - Outer: per target net.
  - Inner: per batch. Effective when there are few nets and many walks.
- Per-batch tallies are summed in batch_id order. **On the CPU backend, results are bit-identical regardless of thread
  count or execution order** for a given build and target (`run.deterministic = true`). To fix the order of floating-point
  additions, the summation order within a batch is fixed as well.
- GPU backends promise **statistical** agreement only: shader compilers may reassociate or fuse floating-point
  operations, so trajectories are not bit-identical across devices or drivers.
- Stopping is defined as "evaluated at each batch boundary, in batch_id order".
  Batches computed speculatively beyond that point are discarded (slightly wasteful, but determinism takes priority).

### 5.7 Key data structures (draft)

*(Pre-implementation sketch; the implemented types and signatures differ, see the crate
documentation.)*

```rust
// faradice-core
pub struct NetId(u32);
pub enum Sink { Net(NetId), GroundPlane, Boundary(Face), Infinity }

// faradice-geom
pub struct Aabb { pub min: [f64; 3], pub max: [f64; 3] }
pub struct Geometry { boxes: BoxSoA, box_net: Vec<NetId>, nets: NetTable }
pub trait SpatialIndex: Sync {
    /// L∞ distance from point p to the nearest box, and that box's ID (None if it exceeds max_r)
    fn nearest_linf(&self, p: [f64; 3], max_r: f64) -> Option<(f64, BoxId)>;
}

// faradice-frw
pub trait GreenTable: Sync {
    fn sample_transition(&self, rng: &mut impl Rng) -> [f64; 3];             // Point on the unit cube surface
    fn sample_gradient(&self, axis: Axis, rng: &mut impl Rng) -> ([f64; 3], f64); // (point, weight)
}
pub struct SolverConfig { ground: GroundModel, scope: Scope, accuracy: Accuracy, run: RunOpts, walk: WalkOpts }
pub struct RawResult {             // Raw statistics before postprocessing. Saved to JSON
    pub per_target: Vec<NetStats>, // Per net: sinks: Map<Sink, (Σw, Σw², n)>, walks, converged or not
    pub meta: RunMeta,             // Version, seed, config hash, Green table hash, run time
}
pub fn solve(geom: &Geometry, cfg: &SolverConfig) -> Result<RawResult>;
pub fn postprocess(raw: &RawResult, cfg: &CouplingConfig) -> CapReport;      // §4.5
```

---

## 6. Output

| Format | Contents |
|---|---|
| `result.json` | Raw statistics (Σw, Σw², n), all capacitance values and σ, ground breakdown, convergence status, walk counts, execution environment, full configuration text and its hash |
| `report.json` / `.csv` | Postprocessed total, ground, and coupling components with σ |
| `.spef` | IEEE 1481 compliant. C-only `*D_NET` with `*CAP` (lumped capacitance as a single node per net; distributed RC in M3) |
| stdout | Progress, per-net convergence status, warnings (boundary capacitance ratio, unconverged nets, use of ignore_nets) |

---

## 7. CLI

```
faradice solve   <structure.faradice.toml|.cap3d> [-c faradice.toml] [-o out/] [--nets 'clk*'] [--preset signoff] [--threads N] [--seed S]
faradice report  <out/result.json> [-c other.toml] [--format spef,csv]     # Redo postprocessing only
faradice check   <structure>                                              # Input checks (overlaps, contact between different nets, bounding box)
faradice gen-green [--res 64] [--verify]                                  # Regenerate and verify Green tables
faradice bench   <case>                                                   # Accuracy and speed benchmarks (also used in CI)
```

*Implemented:* `solve` (faradice TOML structures, or GDSII with `--tech`; no `.cap3d`),
`report` (formats from the configuration's `output.formats`; no `--format` option) and `check`.
`gen-green` and `bench` are not implemented. Later milestones added `gds`, `sens`, `whatif` and
`map`; see [`docs/reference.md`](../reference.md).

---

## 8. Error and Warning Policy

- Overlapping boxes from different nets, conductors outside the domain, zero target nets: **errors** (typed with `thiserror`).
- Not converged (`max_walks` reached), boundary capacitance ratio above threshold, `ignore_nets` used, stop_ratio too coarse: **warnings**
  (also recorded in the result JSON).
- The library (`faradice-frw`) does not panic. The CLI formats errors with `anyhow`.

---

## 9. Validation Plan (M1 completion criteria)

| # | Case | Reference | Pass criterion |
|---|---|---|---|
| V1 | Parallel plates, with side faces `reflect` and the bottom face a ground plane | εA/d (exact) | Agrees within 3σ, and σ is at or below the target accuracy |
| V2 | Cube in free space (`open` boundary) | 0.6606785·4πεa | Bias < 0.3% |
| V3 | Single wire over a ground plane, sweeping the L∞ stop_ratio and table resolution M | Bias model obtained from V1/V2 | Determine defaults for stop_ratio and M |
| V4 | 3 parallel wires + ground plane | FastCap2 / FasterCap (same boxes given as `.qui`) | total < 1%, major couplings < 3% |
| V5 | Crossing wires (2 layers), L-shaped wire with via | FasterCap, RWCap v4 (`.cap3d`) | Same as above |
| V6 | Property tests (proptest) | — | Reciprocity C_ij ≈ C_ji, charge conservation Σ_k C_ik ≈ 0 (statistically), invariance under translation, rotation (90°), and mirroring, determinism (bit-identical results with 1 / 4 / 16 threads) |
| V7 | Postprocessing | — | total is conserved when moving to ground by threshold; behavior of couple-to-ground and multiplier |
| V7b | `ground.ignore` | — | `report` values exactly match `none` (since it only excludes from output). With `physics` (all boundaries reflect), each row sums to 0 (within statistical error). The number of walks reaching `max_steps` is reported |
| V8 | Small sky130 bus structure on met1–met3 (approximated with uniform ε; multilayer handled in M2) | FasterCap | Same as above. PDK data is used only after confirming its license |

Speed target (informative in M1): at V4 scale, on par with or faster than RWCap v4 (at the same accuracy and core count).
criterion benchmarks live in `benches/`, and CI monitors for regressions (performance numbers are judged only on CI self-hosted runners).
*(Not implemented: a criterion benchmark exists for the nearest-box query, `crates/faradice-geom/benches/`, but CI does not track performance.)*

---

## 10. M1 Task Breakdown (planned work units)

1. Workspace skeleton, license, CI, `faradice-core` (units, IDs, error types)
2. `faradice-geom`: boxes, input checks, uniform grid index + L∞ queries (compared against brute-force search with proptest)
3. `faradice-frw/green`: cube Green's function and gradient tables (analytical cell integration), alias sampling, `gen-green` (not implemented)
4. `faradice-frw/boundary`: substrate ground plane, `ground` / `reflect` boundaries → V1
5. Walk core and Gaussian surface, estimator (Σw, Σw²), deterministic parallelism → single-wire version of V4
6. Stopping criteria (total and coupling tracking selection, presets) → V3
7. `faradice-io`: structure TOML, config TOML, `.cap3d` reading (not implemented), JSON / CSV / SPEF output
8. Postprocessing (§4.5) and `faradice report` → V7
9. `open` boundary → V2
10. Validation report (`docs/validation/M1.md`), comparison scripts against FasterCap / RWCap, README accuracy table → v0.1.0 release

---

## 11. Decisions (resolved 2026-11-15)

1. **Structure file format**: TOML/JSON for M1; a binary format for large inputs is deferred to M3.
2. **`coupling.rel_ref` default**: `either` — a coupling is moved to ground if it falls below the
   relative threshold for either of its two nets.
3. **Power nets**: `power.treat_as = "ground"` is the default.
4. **`open` boundary**: included in M1 (needed for the V2 analytical test and for `ground.ignore = "physics"`).
5. **Name**: **faradice** (Farad + dice). Not taken on crates.io, PyPI or GitHub as of 2026-11-15.
   Crates: `faradice` (CLI and umbrella crate), `faradice-core`, `faradice-frw`, ….
   Reserving the names on crates.io/PyPI is a separate, explicit step.
