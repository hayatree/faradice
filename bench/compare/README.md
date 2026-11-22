# Comparison with external field solvers

Accuracy cross-checks of faradice against two independent open-source field
solvers, plus one other floating-random-walk solver, each run in its own Docker
image so nothing is installed on the host:

| Image | Solver | Method | License | Problem it solves here |
|---|---|---|---|---|
| `faradice-bench/fastercap` | [FasterCap](https://github.com/ediloren/FasterCap) 6.0.7 | BEM (Galerkin, `-g`) | LGPL-2.1 | **plane**: open space above an infinite grounded substrate plane |
| `faradice-bench/palace` | [Palace](https://github.com/awslabs/palace) v0.19.0 + [Gmsh](https://gmsh.info) 4.13.1 | FEM with adaptive mesh refinement | Apache-2.0 (Gmsh: GPL) | **box**: grounded box around the conductors (faradice's default domain) |
| `faradice-bench/rwcap` | [RWCap v4](https://numbda.cs.tsinghua.edu.cn/download/RWCap_v4_en.html) (2023 binary package, SHA-256 pinned) | Floating random walk | None stated (free binary download) | **box**, as for Palace |

The tools are used only as separate processes for benchmarking. They are not
dependencies of faradice: nothing is linked, vendored or shipped, and their
sources (for RWCap, its binary package) are downloaded (pinned) only when an
image is built. Only their documented input/output formats are used.

## Running

Requires Docker, Python ≥ 3.11 (standard library only) and a Rust toolchain.
Every solver container is capped at `FARADICE_BENCH_MEMORY` (default `24g`, no
swap) and has the highest OOM score, so a reference solver that outgrows the
machine is killed inside its container. Palace stops refining at
`--palace-max-dofs` but may overshoot it by one AMR step; if it is killed, the
last completed mesh is used and the table says so. FasterCap runs are capped
by `--fastercap-timeout` (30 min).

`run_all.sh` builds the two images on its first run (FasterCap ≈ 1 min, Palace ≈ 6 min on
32 threads). `compare.py` and `scaling.py` only use existing images, so build them first
if you start with those:

```sh
docker build -t faradice-bench/fastercap bench/compare/docker/fastercap
docker build -t faradice-bench/palace bench/compare/docker/palace
docker build -t faradice-bench/rwcap bench/compare/docker/rwcap
```

Then:

```sh
bench/compare/run_all.sh /tmp/compare            # all cases, both solvers
bench/compare/compare.py bench/compare/cases/v4_three_wires.faradice.toml \
    --solvers palace --out /tmp/compare          # one case, one solver
bench/compare/scaling.py --bits 4,8,16,32,64 --out /tmp/scaling   # bus scaling study
bench/compare/compare.py CASE --solvers palace --out /tmp/compare --reuse   # rerun the reference only
bench/compare/compare.py CASE --solvers rwcap --confidence-sigma 1 --out /tmp/rwcap  # vs RWCap, same σ targets
```

`run_all.sh` runs FasterCap and Palace by default; pass `--solvers fastercap,palace,rwcap`
to include RWCap (`run_all.sh` builds its image like the others).

Each case writes `OUT/<case>/summary.md` and `summary.json`, plus
the inputs and logs of every solver.

## How each setting is made identical

- **plane** (FasterCap): faradice runs with `domain.kind = "open"` and the
  substrate plane at z = 0. FasterCap has no ground plane, so every conductor
  is mirrored below z = 0 and the plane capacitance is recovered exactly from
  the free-space matrix of the doubled system as `C[i][j] - C[i][j']` (odd
  images). Touching boxes of one net are merged into their exterior surface
  before panelling.
- **box** (Palace): faradice runs with `domain.kind = "box"` and an explicit
  `domain.margin`; the Palace mesh is the same box (bottom face on the
  substrate, all faces grounded) with the conductors cut out. Palace refines
  the mesh adaptively; the table also shows the error against a geometric
  extrapolation of the last three AMR meshes.
- **box** (RWCap): the same box, written as a `.cap3d` case with the box as an
  explicit grounded conductor (six plates just outside the domain). A window
  marked `dirichlet` absorbed no charge in RWCap v4 (every row summed to zero),
  so it is not used. The stack and the dielectric boxes become disjoint medium
  blocks filling the window. RWCap extracts one master net per run, so a case
  with n nets takes n runs. Its `-p` and `-c` stop at a relative error that
  behaved like one standard error in our runs, so `--confidence-sigma 1` gives
  faradice the same targets (`reference` preset: 0.1 % total, 1 % couplings).

faradice uses the `reference` preset (0.1 % total, 1 % coupling at 3σ),
`coupling.symmetrize = "none"` (raw rows) and seed 1. "Major couplings" are
those ≥ 5 % of the net's total capacitance; the M1 criteria are total < 1 %
and major couplings < 3 %.

## Cases

| Case | Structure |
|---|---|
| `v4_three_wires` | Three parallel wires 1 µm above the substrate (V4 in the M1 plan) |
| `v5_crossing_via` | Crossing wires on two layers and an L-shaped net with a via (V5) |
| `v8_sky130_bus` | Four met2 bus bits between met1 and met3 cross lines, SKY130 heights, uniform ε (V8) |
| `l5_wires_in_stack` | Two wires on a thin high-k layer and a cross wire straddling an interface, four-layer stack (L5, M2) |
| `l6_sky130_stack_bus` | The V8 bus in a SKY130-like seven-zone stratified stack (L6, M2) |
| `c4_met1_pair_conformal` | Two met1 wires at minimum width and spacing with NILD3_C sidewall shells in the L6 stack (C4, M2c) |
| `c5_sky130_conformal_bus` | The L6 bus with NILD3_C / NILD4_C sidewall shells on met1 / met2 (C5, M2c) |

Structures with a stratified dielectric (`[[dielectric.layer]]`) are compared with Palace only:
the mesh is split into one volume per layer, each with its own permittivity.
Dielectric boxes (`[[dielectric.box]]`) and conformal shells (`[[dielectric.conformal]]`) are
turned into disjoint labelled boxes by `compare.py` with faradice's rules (shells grow tagged
conductor boxes laterally by `side` and upwards by `top`; explicit boxes win over shells, later
rules over earlier ones; conductors mask everything). `palace_mesh.py` fragments the layer
volumes with these boxes, gives each permittivity its own volume attribute, and refines the
initial mesh near the box faces to their thickness. The table also lists faradice's steps per
walk and any aborted walks.

Results are recorded in [`docs/validation/M1.md`](../../docs/validation/M1.md) and
[`docs/validation/M2.md`](../../docs/validation/M2.md).
Wall times against FasterCap and Palace are reported for information only: the
solvers run at different accuracy targets, so they are not a speed benchmark.
Against RWCap they are, with `--confidence-sigma 1`; see
[`docs/validation/external.md`](../../docs/validation/external.md).
