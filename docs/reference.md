# faradice reference

This is the reference for the `faradice` command-line tool as it is implemented today:
the input files, every subcommand, every configuration key with its default, and the
output files. For the methods behind the solver, see the design documents in
[`docs/design/`](design/).

All examples below use files from the repository (`examples/`, `tech/`) and can be run
from the repository root after `cargo build --release -p faradice` (the binary is
`target/release/faradice`).

## Contents

1. [Files at a glance](#1-files-at-a-glance)
2. [Command line](#2-command-line)
3. [Structure file (`*.faradice.toml`)](#3-structure-file-faradicetoml)
4. [Run configuration (`faradice.toml`)](#4-run-configuration-faradicetoml)
5. [Technology file (`*.tech.toml`)](#5-technology-file-techtoml)
6. [Sensitivity file (`*.sens.toml`)](#6-sensitivity-file-senstoml)
7. [Output files](#7-output-files)
8. [Units](#8-units)
9. [Statistical error (σ)](#9-statistical-error-σ)

## 1. Files at a glance

| File | Role | Used by |
|---|---|---|
| `*.faradice.toml` | **Structure**: conductor boxes per net and the dielectric. | `solve`, `sens`, `map`, `check` |
| `*.gds` (`.gds2`, `.gdsii`) | **Layout** in GDSII; needs a technology file. | `solve`, `sens`, `map`, `gds` |
| `*.tech.toml` | **Technology**: GDS layer map, layer heights, vias, label layers, dielectric stack, conformal shells. | with GDS input |
| `faradice.toml` | **Run configuration**: ground model, domain, target nets, accuracy, coupling post-processing, walk tuning, outputs. Optional; every key has a default. | `solve`, `sens`, `map`, `report`, `check` |
| `*.sens.toml` | **Sensitivity parameters**: face moves, translations, conductor deletions and additions. | `sens` |

GDSII input is recognised by its file extension (`.gds`, `.gds2`, `.gdsii`, any case);
every other file is read as a structure file.

| Output | Written by | Content |
|---|---|---|
| `result.json` | `solve`, `sens` | Raw per-sink statistics of every target net and the configuration used. Reports can be rebuilt from it. |
| `report.json`, `report.csv` | `solve`, `sens`, `report` | Total, ground and coupling capacitances with σ. |
| `<design>.spef` | `solve`, `sens`, `report` | Capacitance-only SPEF, one lumped node per net. |
| `sens.json` | `sens` | Derivatives dC/dp, deletions and additions with σ. |
| `map.csv` (name set by `--out`) | `map` | Potentials and predicted ΔC on a grid. |

## 2. Command line

```text
faradice <COMMAND>

  solve   Solve a structure and write results and reports
  sens    Capacitance sensitivities dC/dp, deletions and additions of a *.sens.toml file
  whatif  Exact what-if edits of the extracted Maxwell matrix without re-solving
  map     Insertion map: potentials and predicted coupling change on a grid (CSV)
  report  Regenerate reports from result.json without re-solving
  gds     GDSII tools: connectivity tracing (trace) and export (export)
  check   Validate a structure (and configuration) without solving
```

`faradice --version` prints the version; `faradice <command> --help` lists the options.
Progress tables and warnings go to standard error; `--json` output goes to standard
output. List-valued options (`--nets`, `--ground`, `--float`) take comma-separated values.

### Options shared by the solving commands

`solve`, `sens` and `map` accept these options; command-line values override the
configuration file.

| Option | Meaning |
|---|---|
| `<STRUCTURE>` | Structure file, or a GDSII file (then `--tech` is required). |
| `-c, --config <FILE>` | Run configuration. Without it, all defaults apply. |
| `--tech <FILE>` | Technology file, required for GDSII input. |
| `--top <CELL>` | Top cell of a GDSII file (default: the unique top cell). |
| `--threads <N>` | Worker threads, `0` = all cores. Overrides `run.threads`. Results do not depend on it. |
| `--seed <N>` | Master seed. Overrides `run.seed`. |

`solve` and `sens` also take:

| Option | Meaning |
|---|---|
| `-o, --out <DIR>` | Output directory (default `faradice_out`, created if missing). |
| `--nets <GLOBS>` | Target net globs, comma separated. Overrides `extract.nets`. |
| `--preset <NAME>` | `fast`, `default`, `signoff` or `reference`. Overrides `accuracy.preset`; explicit tolerance keys in `[accuracy]` still take precedence over the preset. |

Net globs support `*`, `?` and `[...]` classes (with `!` negation and ranges). A pattern
that equals a net name literally also selects it, so `bus[2]` selects the net `bus[2]`.

### `faradice solve`

Extracts the capacitances of the target nets and writes `result.json` and the report
formats of `output.formats` into the output directory. It prints one summary line per
target: total capacitance, its relative σ, ground capacitance, walks used and the largest
coupling.

| Option | Meaning |
|---|---|
| `--json` | Also print `report.json` on standard output. |
| `--engine <cpu\|gpu>` | Walk engine (default `cpu`). `gpu` needs a build with the `gpu` feature and supports only a uniform dielectric. |

```sh
faradice solve examples/three_wires.faradice.toml -c examples/faradice.toml -o out
faradice solve examples/three_wires.faradice.toml --nets victim --preset fast
faradice solve examples/bus.gds --tech tech/sky130_stack.tech.toml --nets 'bus[1]' -o bus_out
```

With GDSII input, every traced component is a net. Components without a label are named
`$0`, `$1`, … and match the default target glob `*`, so they are solved too; use `--nets`
or `extract.nets` to restrict the targets. GDSII input to `solve`, `sens` and `map` uses
the default tracing options (labels of every hierarchy depth, no joining of same-name
components); use `faradice gds export` for the other tracing options.

### `faradice sens`

Runs the nominal extraction (bit-identical to `solve` with the same configuration and
`walk.window_hops = false`; sensitivities always run without window hops, so against a
default `solve` the nominal values agree statistically) and then
the parameters, deletions and additions of a sensitivity file (section 6). It writes the
`solve` outputs plus `sens.json`. CPU engine only. See
[`docs/design/D3_sensitivity.md`](design/D3_sensitivity.md) for the methods.

| Option | Meaning |
|---|---|
| `-p, --params <FILE>` | Sensitivity file (required). |
| `--one-sided` | One-sided finite differences (sets `richardson = false`). |
| `--method <fd\|shape\|auto>` | Face derivative method; overrides the file's `method`. |
| `--shape-samples <N>` | Samples per face of the shape derivative; overrides `shape_samples`. |
| `--json` | Also print `sens.json` on standard output. |

```sh
faradice sens examples/three_wires.faradice.toml -p examples/three_wires.sens.toml \
    --nets victim --preset fast -o sens_out
```

`output.cross_moments` is ignored (with a warning) by `sens`.

### `faradice whatif`

Applies exact edits to the Maxwell capacitance matrix stored in a `result.json` and prints
the edited matrix, without solving again. Every edited conductor must be a target of the
run. Nets that were not targets are treated as grounded conductors, as in the extraction.
Nothing is written to disk.

| Option | Meaning |
|---|---|
| `<RESULT>` | A `result.json`. For exact σ, produce it with `output.cross_moments = true` (CPU engine); otherwise σ ignores within-row covariances and a warning says so. |
| `--ground <NETS>` | Tie these conductors to ground. |
| `--float <NETS>` | Leave these conductors floating (zero net charge, e.g. dummy fill). |
| `--merge <A,B>` | Connect two conductors into one net named `A+B`. Repeatable. |
| `--swap <A,B>` | Exchange the names of two conductors. Repeatable. |
| `--json` | Print the edited matrix as JSON on standard output. |

Edits are applied in the order swap, merge, ground, float, whatever the order on the
command line.

```sh
printf '[accuracy]\npreset = "fast"\n[output]\ncross_moments = true\n' > cross.toml
faradice solve examples/three_wires.faradice.toml -c cross.toml -o out
faradice whatif out/result.json --float cross
faradice whatif out/result.json --merge left,right --ground cross
```

### `faradice map`

Estimates the potentials of two nets on a grid of points and, at each point, the
first-order change of their Maxwell-matrix entry `C_ab` if a small object were placed
there: a grounded sphere (`ΔC_ab = 4πεr·φ_a·φ_b`) or a floating one
(`ΔC_ab = 4πεr³·∇φ_a·∇φ_b`). The off-diagonal Maxwell entry is the negative of the
coupling capacitance, so a positive `ΔC_ab` means the coupling *decreases* by that amount. Writes CSV. It
needs a uniform dielectric (no stratified stack, dielectric boxes or conformal shells).
The `[accuracy]` section is ignored; the seed, threads, ground, domain and walk options
apply.

| Option | Meaning |
|---|---|
| `--nets <A,B>` | The two nets (required). A single name (or `A,A`) maps the total of one net. |
| `--grid <SPEC>` | `x0:x1:nx,y0:y1:ny,z0:z1:nz` in µm (required); `n` points from `x0` to `x1` inclusive. An axis may be a single value. |
| `--samples <N>` | Walks per point (default 4000). |
| `--radius <R>` | Radius of the sphere-equivalent object in µm (default 0.05). |
| `-o, --out <FILE>` | Output CSV (default `map.csv`). |

```sh
faradice map examples/three_wires.faradice.toml --nets victim,cross \
    --grid 0:10:11,0.35,1.5:2.0:3 --samples 2000 -o map.csv
```

### `faradice report`

Rebuilds the reports from a `result.json` without solving, for example to try other
coupling thresholds. It applies the `[coupling]` and `[ground]` reporting options
(`ground.report`, `ground.ignore = "report"`) and `output.formats` of the given
configuration, or of the configuration stored in the result. Options that change the
physics (ground nets, substrate, domain, targets) cannot be changed this way.

| Option | Meaning |
|---|---|
| `<RESULT>` | A `result.json`. |
| `-c, --config <FILE>` | Configuration to apply (default: the one stored in the result). |
| `-o, --out <DIR>` | Output directory (default: the directory of the result). |
| `--json` | Also print `report.json` on standard output. |

```sh
printf '[coupling]\nmode = "ground"\n' > ctg.toml
faradice report out/result.json -c ctg.toml -o out_ctg
```

The SPEF written by `report` is named after the design stored in the result
(`design.spef` for results written by 0.1.0, which do not store it).

### `faradice gds trace` and `faradice gds export`

`trace` reads a GDSII file with a technology file, traces connectivity, names the nets
from text labels and lists them (labelled nets only, or one net with `--net`). `export`
writes the traced nets to a structure file that `solve` and `check` accept; each net's
boxes are grouped per layer and tagged with the layer name, and the technology file's
dielectric stack, conformal shells and substrate height are copied into it.

Options of both:

| Option | Meaning |
|---|---|
| `<GDS>` | GDSII file. |
| `--tech <FILE>` | Technology file (required). |
| `--top <CELL>` | Top cell (default: the unique top cell). |
| `--join-same-name` | Merge disconnected components that carry the same label. Without it they become `name`, `name:2`, … and a warning is printed. |
| `--label-depth <N>` | Only use labels up to this hierarchy depth (0 = top cell only). |

`trace` only: `--net <NAME>` shows one net (unknown names list similar ones), `--json`
prints JSON. `export` only: `-o, --out <FILE>` (required) and `--nets <GLOBS>` (default:
all labelled nets).

```sh
faradice gds trace examples/bus.gds --tech tech/sky130_stack.tech.toml
faradice gds export examples/bus.gds --tech tech/sky130_stack.tech.toml -o bus.faradice.toml
```

Tracing rules: polygons and paths must be Manhattan (other shapes are an error). Each
shape becomes boxes with its layer's z range; a via spans from the top of its bottom layer
to the bottom of its top layer. Two boxes connect if they share a contact of positive area
(edge- or corner-only contact does not connect). A label names the component whose box on
the label's layer contains the label point. Conflicting labels on one component and
labels that touch no shape are reported as warnings.

### `faradice check`

Parses a structure file and a configuration, resolves the targets, ground nets and domain,
and prints them without solving.

```sh
faradice check examples/three_wires.faradice.toml -c examples/faradice.toml
```

```text
nets: 4  boxes: 4  eps_r: 3.9  dielectric layers: 0
bounds: [0.0, -2.0, 1.0] .. [10.0, 2.7, 2.12] µm
targets (4): left, victim, right, cross
ground nets:
boundary: Box { bounds: Aabb { min: [-8.48, -10.48, 0.0], max: [18.48, 11.18, 10.600000000000001] }, faces: [Ground, Ground, Ground, Ground, Ground, Ground] }
```

Limits of `check`: it accepts structure files only (export a GDSII file first), and it
does not run the solver's checks of `[accuracy]` and `[walk]` values. These are reported
by `solve`.

## 3. Structure file (`*.faradice.toml`)

A structure lists conductors as axis-aligned boxes grouped by net, and the dielectric.
Unknown keys are rejected.

```toml
schema_version = 1
units = "um"

[dielectric]
eps_r = 3.9

[[dielectric.box]]          # optional: a box of another dielectric
min = [-1.0, -1.0, 0.0]
max = [11.0, 2.0, 1.0]
eps_r = 7.0

[[dielectric.conformal]]    # optional: shells around the boxes tagged "m1"
layer = "m1"
side = 0.03
top = 0.01
eps_r = 3.0

[[conductor]]
net = "a"
layer = "m1"
boxes = [[0.0, 0.0, 1.0, 10.0, 0.14, 1.36]]

[[conductor]]
net = "b"
layer = "m1"
boxes = [[0.0, 0.28, 1.0, 10.0, 0.42, 1.36]]
```

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | integer | required | Must be `1`. |
| `units` | string | `"um"` | Length unit of every coordinate and thickness in this file: `nm`, `um` (also `µm`, `micron`), `mm`, `m`. |
| `substrate_z` | float | unset | Substrate plane height in µm (not in `units`). If set, it overrides `ground.substrate_z` of the run configuration. Written by `gds export`. |
| `[dielectric]` | table | required | See below. |
| `[[conductor]]` | array of tables | empty | Conductors. |

### `[[conductor]]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `net` | string | required | Net name. Several entries may share a net. Names of the form `<…>` are reserved. |
| `boxes` | array of `[xmin, ymin, zmin, xmax, ymax, zmax]` | required | Boxes of this net, in `units`. Each box needs `min < max` on every axis. |
| `layer` | string | unset | Layer tag. Only used to select the boxes of `[[dielectric.conformal]]` rules. |

Boxes of one net may touch or overlap; boxes of different nets must not touch or overlap
(that is reported as a short). Box indices used by the sensitivity file count the
boxes of a net in file order, across all `[[conductor]]` entries of that net.

### `[dielectric]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `eps_r` | float ≥ 1 | required | Relative permittivity of the whole domain, or of the medium above the last `[[dielectric.layer]]`. |
| `[[dielectric.layer]]` | array | empty | Stratified stack, bottom-up. |
| `[[dielectric.box]]` | array | empty | Dielectric boxes. |
| `[[dielectric.conformal]]` | array | empty | Conformal shells. |

**Stratified stack** (`[[dielectric.layer]]`). Each entry is the medium below `z_top`,
down to the previous entry's `z_top`; the first entry extends down to the bottom of the
domain, and `dielectric.eps_r` fills everything above the last one. A non-uniform stack
needs `domain.kind = "box"` and the CPU engine.

| Key | Type | Meaning |
|---|---|---|
| `z_top` | float | Height of the layer's top interface, in `units`; strictly increasing. |
| `eps_r` | float ≥ 1 | Relative permittivity. |

**Dielectric boxes** (`[[dielectric.box]]`). A box replaces the stack inside it. Boxes may
touch but must not overlap each other. A box wins over conformal shells.

| Key | Type | Meaning |
|---|---|---|
| `min`, `max` | `[x, y, z]` | Corners, in `units`. |
| `eps_r` | float ≥ 1 | Relative permittivity. |

**Conformal shells** (`[[dielectric.conformal]]`). A rule grows every conductor box
tagged with `layer` by `side` in x and y and by `top` upwards (never downwards); the grown
region minus the box is a shell of `eps_r`. Shells are rebuilt from the current geometry,
so they follow edits made by `sens`. Where shells of different rules overlap, the later
rule wins. A rule whose tag no conductor uses has no effect and causes a warning.
See [`docs/design/M2c_conformal_dielectrics.md`](design/M2c_conformal_dielectrics.md).

| Key | Type | Default | Meaning |
|---|---|---|---|
| `layer` | string | required | Conductor layer tag. |
| `side` | float ≥ 0 | required | Lateral thickness, in `units`. |
| `top` | float ≥ 0 | `0` | Thickness above the conductor tops, in `units`. `side` and `top` must not both be 0. |
| `eps_r` | float ≥ 1 | required | Relative permittivity of the shell. |

## 4. Run configuration (`faradice.toml`)

Every section and key is optional; an empty file (or no `--config`) gives the defaults
below. Unknown keys are rejected. The configuration actually used is stored in
`result.json`. A commented example is `examples/faradice.toml`.

```toml
schema_version = 1

[ground]
report = "by_source"

[ground.boundary]
xmin = "reflect"
xmax = "reflect"

[domain]
margin = 10.0

[extract]
nets = ["victim", "left", "right"]

[accuracy]
preset = "signoff"
coupling_track = { rel = 0.02 }

[coupling]
rel_threshold = 0.01
significance_sigma = 2.0

[output]
formats = ["json", "csv"]
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | integer | unset | If given, must be `1`. |

### How nets are classified

Every net in the structure is one of:

- **removed**: dropped from the geometry before solving (it does not exist for the solver);
- **ground net**: a conductor held at 0 V whose capacitance is reported as ground
  capacitance, not as a coupling;
- **target**: a net whose capacitances are extracted;
- **non-target**: any other conductor, held at 0 V; couplings to it are reported
  (see `coupling.to_non_target`).

The substrate plane (if any), the grounded faces of the domain box and, for an open
domain, infinity are fixed ground sinks; capacitance to them is always ground capacitance.

### `[ground]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ignore` | `"none"`, `"report"`, `"physics"` | `"none"` | `report`: solve with ground as configured but drop ground capacitance from the reports. `physics`: remove ground from the model: ground and power-as-ground nets are removed, no substrate, and all domain faces reflect (results then depend on the domain size; an open domain still has infinity at 0 V). |
| `substrate` | `"plane"`, `"none"` | `"plane"` | `plane`: an infinite grounded plane at `substrate_z`. All conductors must lie strictly above it. |
| `substrate_z` | float (µm) | `0.0` | Height of the substrate plane. Overridden by `substrate_z` of the structure file or of the technology file. |
| `nets` | array of globs | `["VSS", "GND", "vss*", "VGND"]` | Ground nets. |
| `nets_treatment` | `"ground"`, `"net"`, `"remove"` | `"ground"` | What nets matching `ground.nets` are: ground nets, ordinary nets, or removed. |
| `report` | `"merged"`, `"by_source"`, `"drop"` | `"merged"` | `merged`: one ground value per net. `by_source`: also each ground sink separately (ground nets, `<substrate>`, `<boundary:FACE>`, `<infinity>`). `drop`: no ground value. |

### `[ground.boundary]`

Boundary condition of each face of a box domain: `"ground"` (0 V, a ground sink) or
`"reflect"` (zero normal field, a mirror). The bottom face has no key: it is the substrate
plane if there is one and is grounded otherwise (it reflects with `ignore = "physics"`).

| Key | Default |
|---|---|
| `xmin`, `xmax`, `ymin`, `ymax`, `zmax` | `"ground"` |

### `[power]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nets` | array of globs | `["VDD", "VCC", "vdd*", "VPWR"]` | Power nets. |
| `treat_as` | `"ground"`, `"net"` | `"ground"` | `ground`: power nets are ground nets (AC ground). `net`: ordinary nets. |

Note that with the defaults, nets named like supply rails (`VDD`, `VPWR`, …) are treated
as ground and are not targets.

### `[domain]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `kind` | `"box"`, `"open"` | `"box"` | `box`: a finite box around the conductors with the faces of `[ground.boundary]`. `open`: unbounded space with infinity at 0 V (uniform dielectric stacks only). |
| `margin` | float (µm) | automatic | Distance from the conductors' bounding box to the domain faces in x, y and at the top. Automatic: `max(0.5·max(extent_x, extent_y), 4·h, 1)`, where `h` is the height of the highest conductor point above the substrate plane (without a plane: the conductors' vertical extent). With a substrate plane the domain bottom is the plane; without one it is `margin` below the lowest conductor. |

A warning is issued when more than 1 % of a net's total capacitance goes to the domain
boundary; increase `margin` (or make faces reflect) if the boundary is not meant to be
physical.

### `[scope]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ignore_nets` | array of globs | `[]` | Nets removed from the geometry before solving. This changes the physics and causes a warning. |

### `[extract]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `nets` | array of globs | `["*"]` | Target nets. Ground nets are never targets. At least one target must match. |
| `exclude_nets` | array of globs | `[]` | Nets excluded from the targets. |

The cost of a run is proportional to the number of targets; non-targets stay in the
geometry.

### `[accuracy]`

Each target net is sampled in batches until its total capacitance and every tracked
coupling meet their relative tolerance at `confidence_sigma` standard errors (section 9),
or until `max_walks`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `preset` | `"fast"`, `"default"`, `"signoff"`, `"reference"` | `"default"` | Sets `total_rel_tol` and `coupling_rel_tol` (table below). |
| `total_rel_tol` | float | from preset | Relative tolerance of the total capacitance. Overrides the preset. |
| `coupling_rel_tol` | float | from preset | Relative tolerance of tracked couplings. Overrides the preset. |
| `coupling_track.rel` | float | `0.05` | Couplings of at least this fraction of the net's total are tracked. `0` disables this rule. |
| `coupling_track.abs_ff` | float (fF) | `0.0` | Couplings of at least this value are tracked. `0` disables this rule. |
| `confidence_sigma` | float > 0 | `3.0` | Number of standard errors at which the tolerances are checked. |
| `min_walks` | integer | `10000` | Walks per net before convergence is tested. |
| `max_walks` | integer > 0 | `50000000` | Sampling budget per net. Walks are merged in whole batches, so the final count can exceed it by up to one batch. A net that stops at the budget without meeting its tolerances is reported as not converged (with a warning). |
| `batch_walks` | integer > 0 | `4096` | Walks per batch (the unit of merging and stopping checks). |

`coupling_track` is written as a table, e.g. `coupling_track = { rel = 0.02 }`. A coupling
is tracked if either enabled rule selects it; untracked couplings are still reported with
their σ but do not hold up convergence.

| Preset | `total_rel_tol` | `coupling_rel_tol` | Total: target 1σ (at 3σ) |
|---|---|---|---|
| `fast` | 2 % | 10 % | 0.67 % |
| `default` | 1 % | 5 % | 0.33 % |
| `signoff` | 0.5 % | 3 % | 0.17 % |
| `reference` | 0.1 % | 1 % | 0.033 % |

### `[coupling]`

Post-processing of couplings in the reports. It does not affect the solve and can be
changed afterwards with `faradice report`. Thresholds set to 0 are disabled. Decisions use
the symmetrized value, so a coupling between two targets is treated the same in both rows.
A coupling *moved to ground* is removed from the coupling list and
`ground_multiplier × value` is added to the net's ground capacitance.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `"keep"`, `"ground"` | `"keep"` | `ground`: move every coupling to ground (couple-to-ground). |
| `ground_multiplier` | float | `1.0` | Factor applied to couplings moved to ground (e.g. 2 for a Miller-style worst case). With 1 the total is unchanged. |
| `abs_threshold_ff` | float (fF) | `0.0` | Move couplings below this value. |
| `rel_threshold` | float | `0.0` | Move couplings below this fraction of a total (see `rel_ref`). |
| `rel_ref` | `"either"`, `"both"`, `"min"`, `"max"` | `"either"` | Which totals `rel_threshold` compares with: below the threshold of either net's total, of both, of the smaller total, of the larger total. For a non-target partner, the target's own total is used. |
| `threshold_op` | `"or"`, `"and"` | `"or"` | Whether any or all enabled thresholds (`abs_threshold_ff`, `rel_threshold`) must hold. |
| `significance_sigma` | float | `0.0` | Move couplings smaller than this many of their own σ. |
| `to_non_target` | `"keep"`, `"ground"` | `"keep"` | `ground`: move couplings to non-target nets to ground. |
| `symmetrize` | `"inverse_variance"`, `"average"`, `"max_side"`, `"none"` | `"inverse_variance"` | How `C_ij` and `C_ji` are combined when both nets are targets: inverse-variance weighted mean, plain mean, the estimate with the smaller σ, or each row keeps its own estimate. |

With symmetrization (any mode except `none`), the ground capacitance of each row is
`total − Σ couplings`, so `total = ground + Σ couplings` holds exactly in every row when
`ground_multiplier = 1` (in general `ground + Σ kept couplings = total + (m − 1) · moved`
before clamping, with the reported total unchanged); a
negative residual from noise is clamped to 0 with a warning. With `none`, rows use their
own sampled ground and couplings.

### `[walk]`

Tuning of the random walks. The defaults are validated; change them only for studies.

| Key | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `stop_dist` | float (µm) | automatic | > 0 | Distance to a conductor at which a walk stops on it. Automatic: `stop_ratio × min(median smallest box extent, smallest target clearance)`. |
| `stop_ratio` | float | `0.001` | > 0 | Relative stop distance used when `stop_dist` is unset. |
| `gauss_alpha` | float | `0.5` | (0, 1) | Offset of the Gaussian surface around each target, as a fraction of the net's clearance. |
| `max_steps` | integer | `100000` | > 0 | Steps after which a walk is aborted. Aborted walks are counted and reported (they bias the result). |
| `green_resolution` | integer | `64` | even, ≥ 2 | Cells per face edge of the uniform-dielectric transition tables. |
| `antithetic` | bool | `true` | | Launch walks in antithetic pairs from each Gaussian-surface sample (variance reduction). |
| `interface_offsets` | integer | `16` | ≥ 8 | Stratified stacks: tabulated interface positions in a cube. |
| `layered_resolution` | integer | `32` | multiple of `2 × interface_offsets` | Stratified stacks: cells per face edge of the layered tables. |
| `band_fraction` | float | `0.5` | multiple of 1/32 in (0, 1/2] | Stratified stacks: largest half-band of lateral first hops, as a fraction of the cube half-side. |
| `landing_alpha` | float | `0.7` | (0, 1] | Stratified stacks and dielectric boxes: cubes smaller than this fraction of the clearance near an interface are replaced by landing steps. |
| `edge_cubes` | bool | `true` | | Dielectric boxes and shells: land on dielectric edge planes and take edge cubes there. |
| `window_hops` | bool | `true` | | Dielectric boxes and shells: window first hops where the surroundings have a 2D cross-section. |

Background: [`M2_multi_dielectric.md`](design/M2_multi_dielectric.md) (stacks),
[`M2b_interface_anchored.md`](design/M2b_interface_anchored.md) (landing),
[`M2d_edge_cubes.md`](design/M2d_edge_cubes.md), [`M2f_window_first_hops.md`](design/M2f_window_first_hops.md).

### `[run]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `seed` | integer | `1` | Master seed. The same seed gives bit-identical results for any thread count (CPU engine). |
| `threads` | integer | `0` | Worker threads; `0` = all cores. |

### `[output]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `formats` | array of `"json"`, `"csv"`, `"spef"` | `["json", "csv", "spef"]` | Reports written next to `result.json`. An unknown format is an error when the configuration is loaded. |
| `cross_moments` | bool | `false` | Store per-target sink cross moments in `result.json`, needed for exact σ in `faradice whatif`. CPU engine only; ignored by `sens`. |

## 5. Technology file (`*.tech.toml`)

Describes how GDSII layers map to 3D conductors, and the dielectric. Lengths are in µm.
Unknown keys are rejected. The repository ships `tech/sky130_stack.tech.toml` (stratified
stack with conformal shells on met1 and met2) and `tech/sky130_uniform.tech.toml`
(uniform dielectric); their values are nominal and approximate, see the file headers.

```toml
schema_version = 1
name = "example"
eps_r = 4.0
substrate_z = 0.0

[[dielectric]]
z_top = 1.3761
eps_r = 4.05

[[layer]]
name = "met1"
gds = [68, 20]
also = [[68, 16]]
zmin = 1.3761
thickness = 0.36
labels = [[68, 5], [68, 16]]
conformal = { side = 0.030, top = 0.0, eps_r = 3.5 }

[[layer]]
name = "met2"
gds = [69, 20]
zmin = 2.0061
thickness = 0.36
labels = [[69, 5]]

[[via]]
name = "via"
gds = [68, 44]
bottom = "met1"
top = "met2"
```

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | integer | required | Must be `1`. |
| `name` | string | `""` | Display name. |
| `eps_r` | float ≥ 1 | required | Uniform dielectric, or the medium above the last `[[dielectric]]` entry. |
| `substrate_z` | float (µm) | `0.0` | Substrate plane height; overrides `ground.substrate_z`. |
| `[[dielectric]]` | array | empty | Stratified stack, bottom-up: `z_top` (strictly increasing) and `eps_r` (≥ 1), as in the structure file. |
| `[[layer]]` | array | required | Conductor layers. |
| `[[via]]` | array | empty | Via (cut) layers. |

### `[[layer]]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Layer name (also the tag used for conformal shells and in `gds export`). |
| `gds` | `[layer, datatype]` | required | Primary GDS layer and datatype. |
| `also` | array of `[layer, datatype]` | `[]` | Further drawing pairs on the same layer (e.g. pin shapes). |
| `zmin` | float (µm) | required | Bottom of the layer. |
| `thickness` | float (µm) > 0 | required | Thickness. |
| `labels` | array of `[layer, texttype]` | `[]` | Text pairs whose labels name nets on this layer. |
| `conformal` | table | unset | Conformal shell around the layer's shapes: `side` (≥ 0), `top` (≥ 0, default 0, not both 0), `eps_r` (≥ 1), as in section 3. |

### `[[via]]`

| Key | Type | Meaning |
|---|---|---|
| `name` | string | Via name. |
| `gds` | `[layer, datatype]` | GDS layer and datatype. |
| `bottom`, `top` | string | Names of the connected conductor layers; `bottom` must end below `top`'s `zmin`. The via spans the gap between them. |

GDS shapes on pairs that are not mapped are ignored.

## 6. Sensitivity file (`*.sens.toml`)

Input of `faradice sens`. It contains parameters (`[[param]]`), conductor deletions
(`[[delete]]`) and additions (`[[add]]`); at least one entry is required. Unknown keys are
rejected. All lengths in this file are in **µm**, whatever the `units` of the structure
file. Example: `examples/three_wires.sens.toml`.

```toml
schema_version = 1

[[param]]
name = "w_victim"                 # symmetric width change
faces = [
  { net = "victim", box = 0, face = "y-", weight = 0.5 },
  { net = "victim", box = 0, face = "y+", weight = 0.5 },
]
sigma = 0.005

[[param]]
name = "y_right"                  # rigid move along y
translate = { net = "right", axis = "y" }

[[delete]]
name = "no_cross"
nets = ["cross"]

[[add]]
name = "shield"
net = "shield"
boxes = [[3.0, 0.2, 1.46, 7.0, 0.5, 1.52]]
```

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | integer | required | Must be `1`. |
| `delta_rel` | float in (0, 0.25) | `0.05` | Default finite-difference step, as a fraction of the smaller of the box extent normal to the face and the gap in front of the face. |
| `richardson` | bool | `true` | Richardson extrapolation over steps δ and 2δ (`--one-sided` sets it to false). |
| `method` | `"fd"`, `"shape"`, `"auto"` | `"fd"` | Face derivatives: finite differences with path continuation (see the restrictions below); shape derivative (every face must be supported, otherwise an error); or shape where supported and finite differences elsewhere. The shape derivative needs a uniform dielectric and faces without geometric edges. `--method` overrides it. |
| `shape_samples` | integer ≥ 2 | `200000` | Samples per face of the shape derivative. `--shape-samples` overrides it. |

Restrictions with conformal shells: a face of a conductor with a shell moves the shell
with it, and is supported only where the shell has a thickness. The `z−` face is never
supported, `z+` needs `top > 0` and lateral faces need `side > 0`; boxes tagged by several
conformal rules are not supported. With Richardson extrapolation the step must satisfy
`2δ ≤` the shell thickness. With the supplied SKY130 stack (sidewall shells, `top = 0`),
thickness derivatives of met1 and met2 are therefore not available.

### `[[param]]`

A parameter `p` moves faces along their outward normals: face `k` moves by `weight_k · dp`.
The result is `dC_ij/dp` in fF/µm.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Unique name. |
| `faces` | array of face handles | `[]` | Explicit faces (below). |
| `translate` | `{ net, axis }` | unset | Rigid translation of `net` along `axis` (`x`, `y`, `z`): every exposed `-` face with weight −1 and every exposed `+` face with weight +1. |
| `sigma` | float ≥ 0 (µm) | unset | 1σ process variation of the parameter; adds `σ_C = |dC/dp|·sigma` (fF) to `sens.json`. |

A parameter needs `faces`, `translate` or both. Face handle keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `net` | string | required | Net name. |
| `box` | integer | — | Index of the box within the net (file order, from 0). Give `box` or `at`, not both. |
| `at` | `[x, y, z]` (µm) | — | A point on the face; selects the net's box whose face contains it (useful for GDS input). |
| `face` | string | required | `x-`, `x+`, `y-`, `y+`, `z-`, `z+` (or `xmin`, `xmax`, …). |
| `weight` | float | `1.0` | Chain-rule weight. |
| `delta` | float > 0 (µm) | from `delta_rel` | Explicit step for this face. |

The face must have exposed area (not covered by its own net, not on a reflecting wall). A
step must be large compared with the walk stopping distance; otherwise `sens` stops with a
message suggesting a larger `delta` or a smaller `walk.stop_dist` / `walk.stop_ratio`.

### `[[delete]]`

Capacitances without some conductors, and the exact paired differences, from the same
walks.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | Unique name. |
| `nets` | array of strings | Nets deleted together. A deleted target's row is dropped from the edited results. Nets that carry a conformal shell cannot be deleted (it would change the medium). |

### `[[add]]`

Capacitances with added conductors, and the exact paired differences, by replaying the
recorded nominal walks.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | required | Unique name. |
| `net` | string | required | An existing non-target net (its boxes are extended) or a new name (a new conductor at 0 V, e.g. a shield). |
| `ground` | bool | `true` | New nets only: count the new conductor in the `<GND>` sums. Not allowed for an existing net. |
| `boxes` | array of `[x0, y0, z0, x1, y1, z1]` (µm) | required | Added boxes. |

## 7. Output files

All capacitances are in fF and every value carries a 1σ standard error (section 9).

### Report (`report.json`, `report.csv`)

`report.json` has `nets` (one entry per target) and `warnings`. Each net entry:

| Field | Meaning |
|---|---|
| `net` | Net name. |
| `total` | `{value, sigma}`: total capacitance `C_ii`. |
| `ground` | `{value, sigma}`: capacitance to ground nets and fixed ground sinks plus couplings moved to ground; `null` with `ground.report = "drop"`. |
| `ground_by_source` | With `ground.report = "by_source"`: list of `{net, c}` per ground sink (ground net names, `<substrate>`, `<boundary:FACE>`, `<infinity>`). |
| `couplings` | List of `{net, c}`, coupling capacitances (positive), largest first. |
| `moved_to_ground` | Sum of couplings moved to ground, before `ground_multiplier`. |
| `walks` | Walks used. |
| `converged` | Whether the accuracy target was met. |
| `aborted_walks` | Walks that hit `walk.max_steps` (biased if non-zero). |

`report.csv` holds the same data in long format:

```text
net,kind,other,value_ff,sigma_ff
left,total,,1.881470324e0,5.320e-3
left,ground,,4.539948966e-1,6.785e-3
left,coupling,victim,1.246258828e0,3.961e-3
```

`kind` is `total`, `ground`, `ground_source` or `coupling`.

### SPEF (`<design>.spef`)

IEEE 1481 SPEF with capacitances only (`*C_UNIT 1 FF`): one `*D_NET` per target with a
single internal node `net:1`, its ground capacitance and its couplings. Each coupling
appears in the sections of both nets, and each `*D_NET` total is the sum of its section.
There is no `*CONN` section and no resistance, because the inputs carry no pin
information; the file is suitable for inspection and back-annotation by net, not for
static timing analysis. `<design>` is the input file name without `.faradice.toml` or
`.gds`; `faradice report` takes it from `result.json`.

### Raw result (`result.json`)

Per target: walks, aborted walks, convergence flag, and for every sink (other nets,
`<substrate>`, `<boundary:FACE>`, `<infinity>`, and the ground sum `<GND>`) the sums `Σw`
and `Σw²` of the walk weights, from which values and σ follow. It also stores the seed,
stopping distance, elapsed time, all net names, the ground nets, warnings, the design
name (`design`) and the configuration used (`config_toml`). With `output.cross_moments` it also stores sink cross
moments. `faradice report` and `faradice whatif` read it.

### `sens.json`

| Field | Meaning |
|---|---|
| `method` | `fd-continuation`, `shape` or `mixed`. |
| `richardson`, `delta_rel`, `eps_stop_um`, `seed` | Settings used. |
| `nets` | Targets and their nominal walks. |
| `faces` | Each differentiated face: `net`, `box`, `face`, `method`, `delta_um`, `exposed_area_um2`, `gap_um`, and `dC` (per unit outward displacement, fF/µm). |
| `params` | Each parameter: `faces` (face index, weight), `dC` (fF/µm), and with `sigma` also `sigma_C` (fF). |
| `deletions`, `additions` | `C` of the edited geometry and the paired `dC = C(edited) − C(nominal)` (fF). |
| `warnings` | Warnings. |

Entries are `{i, j, value, std_err}` for target `i` and sink `j`. Unlike the reports,
`sens.json` uses Maxwell-matrix signs: `j = i` is the total `C_ii` (positive), another net
`j` gives `C_ij` (negative of the coupling), fixed ground sinks are listed with the same
sign convention, and `j = "<GND>"` is the ground capacitance (positive, as in the report).

### Map CSV

One row per grid point (x fastest), columns:

```text
x_um,y_um,z_um,inside,phi_A,phi_A_err,phi_B,phi_B_err,
dx_phi_A,dy_phi_A,dz_phi_A,dx_phi_B,dy_phi_B,dz_phi_B,
dC_grounded_fF,dC_floating_fF,status
```

`phi` are the potentials of nets A and B (each at 1 V, all others at 0 V) and their
standard errors; `d*_phi` their gradients (1/µm). `dC_grounded_fF` and `dC_floating_fF`
are the predicted changes of the Maxwell entry `C_ab` for an object of radius
`--radius`; the coupling capacitance changes by `−ΔC_ab` (for `A = B`, the change of the
total). The CSV carries standard errors for the potentials only, not for the gradients or
the predictions. `status` is `ok`,
`inside` (the point is in a conductor; `inside = 1`), `outside` (outside the domain) or
`aborted:N`; predictions are empty unless the status is `ok`.

## 8. Units

| Quantity | Unit |
|---|---|
| Structure-file coordinates and thicknesses | `units` of that file (default µm) |
| Structure-file `substrate_z` | µm |
| Run configuration lengths (`substrate_z`, `margin`, `stop_dist`) | µm |
| Technology file lengths | µm |
| Sensitivity file (`at`, `delta`, `sigma`, `boxes`) | µm |
| `map --grid`, `--radius` | µm |
| Capacitances, thresholds (`abs_threshold_ff`, `coupling_track.abs_ff`) | fF |
| Derivatives `dC/dp` | fF/µm |
| Permittivities | relative (`eps_r`; ε₀ = 8.8541878128 × 10⁻³ fF/µm) |

Internally everything is converted to µm. Output coordinates are in µm.

## 9. Statistical error (σ)

faradice is a Monte Carlo method: every capacitance is the mean of many random-walk
weights, and every reported value comes with its **standard error σ** (one standard
deviation of the estimate, in the same unit as the value). The summary table prints σ as a
percentage of the value.

- The true value lies within ±σ of the estimate about 68 % of the time, within ±2σ about
  95 %, within ±3σ about 99.7 %. Halving σ takes about four times as many walks.
- A net stops when `confidence_sigma × σ ≤ tol × |C|` for its total and for every tracked
  coupling. With the default preset (1 % at 3σ) the total therefore has σ ≤ 0.33 %.
- Couplings that are not tracked (below `coupling_track`) can have a large relative σ;
  `coupling.significance_sigma` moves couplings that are not significant to ground.
- σ describes the random error only. Model and discretization errors (geometry, the
  stopping distance, a domain that is too small, aborted walks) are not included; the
  warnings point out the latter two.
- Results are reproducible: the same seed and inputs give bit-identical values for any
  thread count on the CPU engine. Different seeds give values that differ by about σ.
- Derived values (ground after symmetrization, `whatif` edits, sensitivities) carry σ
  propagated to first order (delta method). `whatif` uses the sample covariances when the
  result stores cross moments (`[output] cross_moments = true`) and warns that its σ is
  approximate otherwise. The ground σ of a report neglects covariances. `map` reports σ
  for the potentials only.
