# M2d: Edge cubes for walks near dielectric edges

Status: Implemented (`crates/faradice-frw/src/edge.rs`, `walk.edge_cubes`, default on; CPU engine).
The cost goal of §5 is not met on C5 ([validation](../validation/M2.md#m2d-edge-cubes-near-dielectric-edges));
the follow-ups are [M2e](M2e_layered_first_hops.md), [M2f](M2f_window_first_hops.md) and
[M2g](M2g_shell_walk_cost.md). Follows [M2c](M2c_conformal_dielectrics.md); results that motivate it are in
[docs/validation/M2.md](../validation/M2.md#m2c-dielectric-boxes-and-conformal-shells).

## 1. Problem

M2c is accurate on SKY130 sidewall shells (C4/C5 agree with extrapolated Palace values within
0.07 % for totals), but slow:

| | without shells | with shells |
|---|---|---|
| C4 CPU time (4 M walks per net) | 30 s | 705 s (23×) |
| C4 steps per walk | 23.9 | 115.0 |
| C5 wall time vs L6 | 91 s | 3140 s (35×) |
| C5 steps per walk, bus / met1 / met3 | 18–22 / 35 / 30 | 114–116 / 126 / 47 |

The time per step is also 4.8× higher in C4, and the variance per walk 2.1×.

**Why.** A dielectric edge is a line where two interface planes meet (normal to axes `b` and
`c`). M2c never builds a cube containing an edge, so a walk at distance `ρ` from an edge takes a
cube of size `O(ρ)`. The function `log ρ` is harmonic in 3D: it does not depend on the axial
coordinate and is harmonic in the cross-section. Every step that is symmetric about the walker
therefore leaves `E[log ρ]` unchanged: near an edge, `log ρ` is a driftless random walk. Walks make
long excursions towards the edge before leaving it or reaching the 10⁻⁸·h snap and its edge ball
(M2c §4.6). Measured on C4, the ratio of cube size to clearance is spread evenly on a log scale
from 0.5 down to 10⁻⁸, and about 60 % of the steps have cubes below 0.01·h.

Any exact fix needs a step from a point near an edge into a domain **containing** the edge, at a
scale much larger than `ρ`.

## 2. Prior work (public papers only)

- RWCap (Yu et al., IEEE TCAD 2013) tabulates two-dielectric transition cubes by finite
  differences. M2a/M2b use analytic series instead.
- Precharacterized cubes for selected 3- and 4-dielectric configurations of one process (Zhang,
  Yu & Zhang, IJNMDF 2016).
- The M2c survey (§2 there): published conformal-dielectric FRW methods homogenize or discretize
  the cube that cuts a shell.

The new element here is a construction that makes the configuration of a cube containing an
edge **discrete**, so a finite set of tables suffices without interpolating the walker's
position.

## 3. Method

### 3.1 Edge frame

At a walker `x` with clearance `h`, let the nearest interface face of each normal (L∞ distance,
from the existing face indices; for `z` also the stack planes) be found. The two axes `b`, `c`
with the nearest faces define the planes `P: x_c = c₀` and `Q: x_b = b₀`, with normal distances
`d_c = |x_c − c₀|` and `d_b = |x_b − b₀|`; `a` is the third axis.

A domain is **quadrant-constant** if no interface face meets it except faces lying in `P` or
`Q`, and no stack plane other than `P` or `Q` crosses it. Then ε is constant in each of the four
open quadrants around the line `P ∩ Q` within the domain: a face in `P` that ends inside the
domain needs a face normal to `a` or `b` where it ends (the medium is made of boxes), which would
be a face off the two planes, or `Q` itself. This is the argument of the edge balls (M2c §4.6).
Within a cube, each open quadrant is connected and contains no jump surface. Reflecting faces use
the mirrored medium within the reach; ground faces lie beyond the clearance. The test uses
the L∞ queries of the edge ball: faces normal to `a`, faces normal to `b` other than at `b₀`,
faces normal to `c` other than at `c₀`, other stack planes, the clearance and the reflection
reach. The quadrant permittivities are read with `eps_at` at the quadrant centres inside the
domain.

### 3.2 Landing on an edge plane

If the walker is on neither plane, it first lands exactly on the nearer one, say `P`
(`d_c ≤ d_b`; otherwise swap the roles). It uses the M2b landing box (M2b §3.3), rotated so that
its axis is `c`:

- Its bottom face lies on `P`, and the walker sits at the fraction `τ ∈ {⅛, ¼, ⅜, ½}` of its
  height `D = d_c/τ` above that face. The smallest `τ` that fits is used.
- It must satisfy `d_c ≤ R`, `D − d_c ≤ R` and `L·D/2 ≤ min(R, d_b)`, where `R` is the room of
  §3.1 (at most the clearance). `L = 2^{r/2}` is the largest ladder ratio that fits. The box
  therefore never crosses `Q`, and it contains no face off the two planes.
- The box lies in one quadrant, so it is homogeneous and its tables are exact.
- A walk that exits through the bottom face is placed exactly on `P` (`x_c = c₀`, the stored
  plane coordinate, as in M2b §3.4). Where `P` is not an interface on the walker's side (a face in
  `P` that exists only beyond `Q`), the landing is still valid; it just places the walker on a
  plane of the edge frame.

### 3.3 Edge cube

A walker exactly on `P` at distance `d_b > 0` from the line takes a cube **centred on itself**
with half-side `r = d_b·K/k`, `k = ⌈K·d_b/R⌉ ∈ [1, K−1]`, where `R` is the room of §3.1 and
`K = walk.interface_offsets`. In the canonical frame `[−1, 1]³`:

- the walker is at the origin;
- `P` is `c = 0`;
- `Q` is `b = t`, `t = k/K` (mirrored in `b` if `Q` lies on the negative side);
- ε is constant in the four quadrants (sign of `b − t`, sign of `c`).

The configuration is the offset `k` and the four quadrant permittivities, up to a common scale.
All of them are discrete, so the tables are exact up to their own discretization (§3.4), with no
interpolation, as for the M2a offsets.

For `k = 1` the cube is `K` times the distance to the edge, so a step from near an edge jumps
outwards by a factor of order `K` and the drift of §1 is broken. A walker exactly on the line
(`d_b = 0`) keeps the edge ball.

### 3.4 Tables

The exit probabilities of a cell are the fluxes `−ε ∂G/∂n` through it of the Green function of
`−∇·(ε∇G) = δ` in the cube, with `G = 0` on its boundary and the source at the origin.

- **Discretization.** A node-based finite-volume scheme on `N` subdivisions per axis of
  `[−1, 1]³` (`N + 1` nodes, the origin a node), with ε per cell. The flux coefficient of a dual
  face is the mean ε of the cells it lies between, and the source is the origin node. `N` is a
  multiple of `2K` and of `M`, so `c = 0`, `b = t` and every table cell boundary are grid planes:
  interfaces are cell boundaries.
- **Exact decoupling.** ε does not depend on `a`, so the discrete sine transform along `a` (DST-I
  of the `N − 1` interior node layers) diagonalizes the `a` part of the 3D system exactly. The 3D
  discrete solution is the sum of `N − 1` 2D systems
  `(−∇_h·ε∇_h + ε λ_j^h) u_j = s_j`, with `λ_j^h = (2/h)² sin²(jπ/2N)`, `h = 2/N`.
  - Only odd `j` contribute, because the source is in the middle layer.
  - Every face flux of the 3D discrete solution is a finite sum over `j`. No series is truncated,
    and the discrete fluxes sum to the unit source (discrete conservation).
  - The total before normalization therefore measures the solver tolerance; it must be 1 within
    10⁻⁹. Every cell must also be non-negative (beyond 10⁻¹² of rounding).
- **Solver.** Conjugate gradients on each symmetric positive definite 2D system, preconditioned by
  the same operator with the mean ε, solved exactly by a 2D DST. The DST is a pair of dense matrix
  products (`O(N³)` per transform, fast enough up to `N = 128`), so there is no new dependency.
  The grid levels are powers of two. The number of iterations grows like `√(ε_max/ε_min)`: about
  10 for the SKY130 contrasts (≤ 1.3). The relative residual tolerance is 10⁻¹². The
  permittivities are scaled so that the largest is 1.
- **Accuracy contract.** The error measure is the L1 distance between cell-probability vectors.
  - The tables are built at `N₁ = max(lcm(2K, M), 32)`, `2N₁` and `4N₁`, with `4N₁ ≤ 128`.
  - Let `P₁, P₂, P₄` be the cell vectors of the three levels, and let `Rᵢ = (4P₂ᵢ − Pᵢ)/3` be the
    Richardson extrapolations `R₁` (levels 1, 2) and `R₂` (levels 2, 4).
  - The observed order `log₂(‖P₂ − P₁‖/‖P₄ − P₂‖)` is computed.
  - If it is within 0.2 of 2, the table is `R₂` and the error estimate is `‖R₂ − R₁‖`. This bounds
    the error of `R₁` and so, conservatively, that of `R₂`.
  - If it is between 1 and 1.8, the table is `P₄` and the estimate is `‖P₄ − P₂‖`, which is
    conservative for convergence of at least first order. Below first order the configuration is
    rejected.
  - A configuration whose estimate exceeds 10⁻⁵ is **not used**. Its walks keep the M2c steps,
    which are exact but slow, and the count of such configurations is reported. This guards
    against tuples with strong edge singularities (high-contrast checkerboards), where
    second-order convergence fails.
  - Richardson can make cells negative; negative cells also reject the configuration.
  - The default `K = 16`, `M = 32` gives levels 32, 64 and 128. Other `K`, `M` for which
    `lcm(2K, M)` is not a power of two, or whose finest level would exceed 128, disable edge cubes
    (reported).
- **Measured** (`edge.rs`):
  - In the degenerate tuples the levels converge at order 2.0. Richardson is within 3.8·10⁻⁷ (L1)
    of the analytic M2a tables.
  - SKY130 tuples (contrast ≤ 1.3): order 2.0, estimate 5.6·10⁻⁶ at `k = 8`. At `k = 1` and
    `k = 15` the estimate is 1.8–2.3·10⁻⁵, so those offsets are rejected.
  - Contrast 3–6 (non-product): order 1.2–1.9; most offsets are rejected.
  - Checkerboard of contrast 1000: order 0.4, rejected.
  - The edge exponent `ν` sets the order (about `2ν`). Graded meshes around the edge line would
    extend the range; that is left for later.
- **Independent checks** (unit tests, §5): the degenerate tuples against the analytic M2a tables,
  and exactly ε-harmonic functions in product media. The checks use the L1 norm and nonlinear
  harmonic functions, not only sums.
- **Storage and builds.** One table set per quadrant tuple and `k`: 6 faces of `M × M` cells,
  folded by the `a → −a` symmetry, with alias tables.
  - Tables are built lazily, one `OnceLock` slot per `(tuple, k)`, owned by `BoxSteps`, which fixes
    `K` and `M`. The tuple is a quadruple of indices into the distinct permittivities of the
    medium, so the slots are a flat array (no lock on lookups). Only three offsets are tabulated
    (§3.5), and a tuple and its mirror in `c` share a table (the smaller index quadruple is
    canonical; the walk flips the sampled `c`).
  - Builds are sequential and deterministic (no parallel reductions), so results do not depend on
    which thread builds a table.
  - The workspace is a few 2D vectors, about 1 MB at `N = 128`.
  - A build takes about 0.3 s per `(tuple, k)` single-threaded, mostly the 64 odd modes at
    `N = 128`. Each 1D transform of the dense DST is split by the symmetry
    `s_j(N − i) = (−1)^{j+1} s_j(i)` into an odd-`j` and an even-`j` half, which halves the
    work, and the inner loops use plain multiply-adds (`f64::mul_add` is a library call without
    the FMA target feature and stops vectorization).
    Different `(tuple, k)` build concurrently on the walk threads, and the odd modes of one
    build are solved on scoped threads (not the walk's rayon pool, whose work stealing could
    re-enter the table's `OnceLock`) and summed in mode order, so a table is bit-identical to a
    sequential build.
  - The number of tables, their build time and the rejected configurations are reported with the
    other table statistics.
- **Sampling.** A face is drawn, then a cell, then a uniform point in the cell, as for every other
  table. Cell boundaries include `c = 0` and `b = t`, so a cell never straddles an interface.

### 3.5 Step selection

In `BoxSteps::step`, after the existing candidates (stack step, stack step capped by the box
distance, best axis cube):

1. If the best of them is at least `α·h` (`α = walk.landing_alpha`, default 0.7), take it, as
   today.
2. Otherwise, find the edge frame of §3.1:
   - on both planes: edge ball (M2c §4.6);
   - on one plane: edge cube (§3.3) with the smallest **tabulated and accepted** offset `k' ≥ k`
     whose half-side exceeds the best candidate. A larger offset gives a smaller cube
     (`r = d_b·K/k'`), which is still valid. Only `k' ∈ {⌈K/8⌉, ⌈K/4⌉, ⌈K/2⌉}` (2, 4, 8 for
     `K = 16`) are tabulated: a full set of `K − 1` offsets per tuple cost more in builds than
     it saved in steps (C4: 150 tables for 10 tuples), while these three keep the jump factor
     between 2 and 8. Every candidate cube has `r ≥ 2·d_b` (`K ≥ 8`), so the quadrant
     permittivities are read once, at `(b, c) = (−½·d_b, ±½·d_b)` and `(1½·d_b, ±½·d_b)`
     from the walker (canonical `b` towards `Q`), all inside the smallest candidate;
   - on neither: landing box (§3.2) if one fits.
   - The room queries of §3.1 are capped at the largest domain that can be used,
     `d_b·max(K/k'_min, 8)` (the largest edge cube, or a landing box with `τ = ⅛`), which
     shortens the face searches; before them, a walker on a plane whose largest edge cube is no
     larger than the best candidate, or whose smallest one exceeds the clearance, skips the edge
     step.
3. Otherwise take the best candidate (today's step, including the edge ball).

### 3.6 Time per step

These changes leave every random draw unchanged, so the results stay bit-identical:

- Compute the box distance `v` first. If `v > EDGE_SNAP·h`, no face is within the snap tolerance:
  faces lie on box boundaries, and an image is never nearer than its original (`dielectric.rs`
  module docs). The three snap queries are then skipped.
- Query the nearest face of each normal once per step and share the result between the axis-cube
  candidates and the edge frame. The queries are repeated whenever the snap has moved the walker,
  because a move can change which face is nearest.
- `stack_planes` returns a fixed-size array instead of a `Vec`, with the same filter and the same
  order (ties included).
- Edge cubes and landing boxes are walk steps only; first hops are unchanged (they need gradient
  kernels).
- The dielectric face and volume indices use cells 12× the median smallest box size
  (`faradice_geom::RECT_CELL_FACTOR`) instead of the conductor default 3×. Shell faces and boxes
  are 0.03 µm thin, while queries reach about the clearance, so with the default a ring search
  on C5 visited about 130 mostly empty cells. A ring search also stops before its first ring when
  the cell clearance exceeds the query radius. Queries are exact, so results stay
  bit-identical; C5 CPU time falls by 38 %, C4 by 8 %.
- Each thread keeps local lists of the faces of each normal and of the material boxes within
  4× the last query radius around a walker (`faradice_geom::LocalBoxes`). A query at `p` with
  radius `r` uses the list when `‖p − centre‖∞ + r ≤ R`; then every box within `r` of `p` is
  listed (L∞ triangle inequality, with a 1e-12 relative margin), so the answer, ties
  included, is the grid's. One pass over a list gives the nearest face per axis and the nearest
  one on another plane, which is what the axis plans (second plane) and the edge room (faces
  off the edge planes) need. Lists are keyed by a per-`BoxSteps` id. C4 −22 %, C5 −9 … −12 %
  at the default accuracy, bit-identical.

### 3.7 What stays approximate

The edge cubes are exact up to their tables (§3.4). The walk keeps the M2c approximations:

- the 10⁻⁸·h moves onto planes and edges (M2c §4.6), whose error is `O((10⁻⁸)^ν)` with the
  smallest edge exponent `ν`. That is small for moderate contrasts but not for extreme
  checkerboards (`ν ≈ 0.04` at contrast 1000);
- aborts at vertices and at `max_steps`, which are reported.

Edge cubes make both rarer: walks leave edges instead of approaching them. The validation reports
the snap and abort counts.

## 4. Expected effect

A prototype (not included in the repository) implemented the landing step exactly and replaced the edge-cube kernel by the
homogeneous cube; the walk geometry was real, so the step counts are realistic. It gave:

| | M2c | prototype | without shells |
|---|---|---|---|
| C4 steps per walk | 114.5 | 40.7 | 23.9 |
| C5 bus / met1 / met3 | 113–116 / 126 / 47 | 36–39 / 52 / 33 | 18–22 / 35 / 30 |

On C4, landing succeeds in 46 % of the attempts, and walks take 2.6 edge cubes each. The
remaining excess is consistent with facing shells (§6).

## 5. Validation

- **Tables** (unit tests):
  - the two degenerate configurations against the analytic M2a tables (§3.4);
  - probabilities sum to 1;
  - exactly ε-harmonic functions are reproduced, with the expectation taken over the cell averages
    of the function on the exit cells (what cell-uniform sampling gives), within 10⁻⁴ of its range
    on the cube. Cell-uniform sampling itself biases curved functions by a few 10⁻⁵ at `M = 32`
    (as in M2a/M2b); the table error is bounded separately by the L1 checks:
    - for **every** quadrant tuple: `a² − c²` and `a² − (b − t)²`. They are harmonic in each
      quadrant, and their normal derivatives vanish on both planes, so the flux is continuous;
    - in product media (`ε = A(b)·B(c)`): the piecewise affine `U(b)` with `(A U′)′ = 0` and `V(c)`
      with `(B V′)′ = 0`;
  - convergence from `N` to `2N` for a non-product tuple.
- **Walks:**
  - the four-quadrant product medium of `edge_balls_reproduce_a_quadrant_potential` with walkers
    starting off the edge (exact φ);
  - C1–C3, C6 and C7, and the bias check (8 seeds), unchanged at 0.02 %;
  - the replay agreement test;
  - L-series tests without boxes bit-identical.
- **Bench:** C4 and C5 again against Palace (same criteria). Steps per walk, time against
  L6/no-shell runs, edge-cube and landing counts.
  - Goal: steps per walk within 1.7× of the no-shell runs, and CPU time within 3× of L6.

## 6. Open questions

1. **Facing shells.** Between two facing strips (an 0.08 µm NILD3 gap in SKY130) one-interface
   cubes are limited by the second plane. Multi-interface axis cubes (the M2b multi-layer cube
   rotated) would remove this.
2. **First-hop variance.** With shells the Gaussian offset must clear the strips, which leaves the
   first hop about 0.04 µm from the strips instead of 0.07 µm. Multi-interface first-hop kernels
   would recover most of the 2.1× variance.
3. **Vertices** (three planes) still have no step; one C5 walk in 36.7 M aborted. A vertex cube
   would need a 3D solve per configuration.

## 7. Task breakdown (work units)

1. This document.
2. Time per step (§3.6), bit-identical.
3. `edge.rs`: FFT/DST, the finite-volume solver and the edge-cube tables, with the table tests of
   §5.
4. Walk integration (§3.2, §3.3, §3.5), walk tests, C4/C5 rerun and `docs/validation/M2.md`.
