# M2b: interface-anchored walks for thin layers

Status: Implemented (`crates/faradice-frw/src/multilayer.rs`, `medium.rs`). Follows [M2_multi_dielectric.md](M2_multi_dielectric.md) (M2a); validation is
recorded in [`docs/validation/M2.md`](../validation/M2.md).

## 1. Problem

M2a transition cubes contain at most one dielectric interface. A cube is therefore limited by
the second-nearest interface, whatever the geometric clearance. On the SKY130-like stack of L6
(seven zones, 0.1 µm LINT and 0.34 µm NILD2 under met1) this dominates the cost:

| L6, net m1_1 (4 M walks) | steps | share | mean cube / clearance |
|---|---|---|---|
| homogeneous cube (clearance below the nearest interface) | 60 M | 17 % | 1 |
| layered cube, limited by the clearance | 42 M | 12 % | 0.87 |
| layered or touching cube, **limited by the second interface** | 241 M | 70 % | 0.21 |

A walk takes 87 steps against 23.5 in the homogeneous V8 run. The cubes limited by the second
interface are about a fifth of what the geometry allows, so in that region the walk advances like
a 1-D walk with small steps.

## 2. Prior work (public papers)

- RWCap (Yu et al., TCAD 2013) tabulates cubes with **two** dielectrics by FDM, and shrinks the
  cube until the interface sits on a grid height. Tables for three layers were rejected
  because of their size.
- Zhang, Yu, Zhang (IJNMDF 2016) pre-characterize selected 3- and 4-dielectric cubes for one
  process.
- Rapid3D-style homogenization maps every cube onto a template of four equal layers with
  averaged permittivities. The tables are process-independent, but the result is biased.
- Huang & Yu (DAC 2024; known only from citing papers) compute the layered surface Green
  function online and cache it by configuration.

In all of these, the walker is at the centre of its cube. A cube whose centre lies at an
arbitrary height inside a stack has a continuous configuration: the position of the walker within
its layer. Exact tables therefore exist only for one interface, which can be snapped by shrinking
the cube. Any table for more interfaces must quantize that position, which introduces a bias.

## 3. Method: anchor the walker on interfaces

The configuration of a centred cube becomes **discrete** when the centre lies *on* an interface.
For a centre on interface `i` and a half-side `s`, every other interface inside the cube is at a
fixed position `(z_j − z_i)/s` given by the stack. If `s` is taken from a fixed geometric ladder,
there are finitely many configurations `(i, level)`. Each has an exact table with all of its
interfaces at their true heights.

Walks are therefore steered onto interfaces, and large multi-layer cubes are used from there.

### 3.1 Steps

Let `z` be the walker height, `h` the geometric clearance, and `d₁ ≤ d₂` the distances to the
nearest and second-nearest interface. Each step chooses as follows:

1. **On an interface** (`d₁ = 0`, exactly; see §3.4), with `ladder(h) > d₂`: use a centred
   multi-layer cube of half-side `s = ladder(h)`, the largest ladder value `≤ h`. The ladder is
   `s_l = 2^{l/4}` length units, `l ∈ [−64, 48]`. The table is `T(i, l)` (§3.2).
   - A cube that reaches no second interface is the M2a cube through the centre, of size
     `min(h, d₂)`, and needs no new table.
2. **Within the clearance** (`h ≤ d₁`): use a homogeneous cube of half-side `h` (as in M2a).
3. **Off an interface, `h > d₁`.** If the M2a one-interface cube has a half-side of at least
   `α·h` (`α = 0.7`), use it.
   - Otherwise use a **landing box** (§3.3). Its bottom face lies on the nearest interface, and
     the walker is at the fraction `τ` of its height `D = d₁/τ` above that face, with `τ` the
     smallest of `{⅛, ¼, ⅜, ½}` for which the box stays within the layer and the clearance
     (`D − d₁ ≤ min(h, d_far)`, where `d_far` is the distance to the next interface on the other
     side). Its lateral half-width is `L·D/2 ≤ h`, with `L` the largest ladder ratio that fits.
   - A walk that exits through the bottom face is placed exactly on the interface
     (`z = z_i`), so the next step is case 1. For flat boxes this happens with probability
     `≈ 1 − τ`.

All three kinds of domain are conductor-free and centred on the walker. They need no
interpolation, so the estimator is exact up to the cell-uniform position sampling already used
in M1 and M2a.

A prototype with idealized kernels (step counts only; not included in the repository) estimated about 36 steps per walk for
m1_1, against 87 for M2a and 23.5 for the homogeneous case. The implementation takes 38.8.

A carried clearance lower bound (M2a §8) replaces the geometry query only off interfaces and
beyond the second-nearest interface. Any lower bound is a valid clearance, so the domains stay
conductor-free.

### 3.2 Multi-layer cube with the centre on an interface

Canonical frame: the unit cube `[0,1]³` with the centre at `z = ½` on an interface. `ε(z)` is
piecewise constant, with interfaces at `ζ_1 < … < ζ_n` in `(0, 1)`. The series of M2a §3.2 carry
over layer by layer:

- **Top / bottom faces.** For each `(x, y)` mode with `γ = π√(m² + n²)`, the profile solves
  `Z″ = γ² Z` in every layer, with `Z` and `ε Z′` continuous. It is propagated from the far face
  through all layers by 2×2 transfer steps, then evaluated at `½` and divided by its value on the
  data face.
- **Side faces.** The eigenfunctions of `−(εψ′)′ = μ² ε ψ` with `ψ(0) = ψ(1) = 0` are
  propagated through the layers in the same way. The Prüfer angle `θ(1; μ)` is strictly
  increasing, so the root `θ = kπ` is found by bisection on `[0, (k + n + 1)π]`. The norms
  `∫ ε ψ²` and the cell integrals `∫ ε ψ` are closed-form per layer.
- **Side-face cells** in `z` are `4M` uniform cells, split at every interface. A cell then never
  straddles an interface, and sampling is uniform within each piece. Top and bottom faces keep
  the uniform `M × M` grid.
  - With every interface on a uniform grid line, cell-uniform sampling reproduces the layered
    linear potential exactly, as in M2a.
  - Off the grid it does not. The error is about `1e-5` of the cube's potential range with `M`
    cells, falls like `Δz³`, and is below `1e-6` with `4M` cells (unit test).
- **Storage.** The tables are folded into the symmetry classes `x ↔ y`, `x → −x`, `y → −y`, as
  in M2a. A table is built lazily on first use and kept per `(i, l)` in a `OnceLock`, so results
  do not depend on which thread builds it.

### 3.3 Landing box

The landing box is a homogeneous box with half-sizes `(b, b, D/2)`, `L = 2b/D ≥ 1`, laterally
centred on the walker, which sits at the fraction `τ ∈ (0, ½]` of its height above the bottom
face. With the lateral coordinates scaled to `[0, 1]` (height `1/L`), it has the double sine
series of the homogeneous box:

- **Top / bottom faces:** cell probability `Σ 4 σ_s σ_t A_s(i) A_t(j) · sinh(γτ)/sinh γ` (top)
  or `sinh(γ(1 − τ))/sinh γ` (bottom), with `γ = π√(s² + t²)/L` and odd `s, t` up to
  `NTERMS·L/(2τ)`. The ratio is evaluated in exponential form, which does not overflow.
- **Side faces:** cell probability `Σ 4 σ_t sin(kπτ) A_t(i) B_k(j) / (2 cosh(π√(t² + k²L²)/2))`,
  over all `k` (the start point is not at mid-height).
- **Cells.** The lateral cells are graded towards the walker (boundaries `½ ± ½ sinh(cs)/sinh c`),
  so the central cells are at most `τ·D/2` wide. The density on the near face is concentrated
  within `≈ τD` of the walker's foot point. With uniform cells, flat boxes put it into one cell.
  The quadratic harmonic `x² − z²` then had an error of 0.17 `(D/2)²` at `L = 16`, `τ = ½`;
  graded cells reduce it to 0.05. The unit test bounds it by `10⁻³ L²` for all `L` and `τ`.

`L = 1`, `τ = ½` reproduces the homogeneous cube tables, which is a unit test. The tables are
process-independent and built lazily per `τ` and ladder ratio `L = 2^{r/2}`, `r = 0 … 8`
(`L_max = 16`). As `L` grows, the probability of the bottom face tends to `1 − τ` (unit test).

### 3.4 Exact interface positions

The landing step writes `z = z_i` (the stored interface height) instead of adding a
displacement. A walker on an interface is recognized by exact equality, and nothing else
produces such points except the folding at reflecting faces, whose images are part of the
effective stack.

### 3.5 First hop

The first hop sets the variance per walk (its weight is `∝ ε/h` for a first cube of half-side
`h`), so a thin layer that limits the first cube costs walks rather than steps. Measured on L5
(wire on a 50 nm layer, 8 M samples), the second moment of a sample is about 5.4 for a
homogeneous medium, against 72 for lateral Gaussian points inside the layer and 12 on the bottom
face. Two changes address this.

**Lateral faces: half-bands of anchored cubes.** The M2a half-band estimator (M2a §4) averages
the tangential gradient kernel of a cube centred on the interface point `p = (x₀, y, z_i)` over
the band, and its cube is limited by `d₂`. If the anchored cube of half-side
`s = ladder(h(p))` reaches a second interface (`s > d₂`), the band uses that multi-layer cube
instead:

- The band runs from the interface into the layer on the walker's side, with the full length
  `ℓ = min(η_max·s, t/2)`, where `t` is the thickness of that layer. Bands of adjacent
  interfaces therefore meet in the middle of a thin layer, and no point of the layer is left to
  a small centred cube.
- If the accepted Gaussian segment is shorter than the full length (the face ends within the
  band), the band is cut to the largest tabulated fraction `j·s/32` of the run. The rest of the
  run uses centred cubes, which have `d₁ ≥ ℓ` there. A run shorter than `s/32` keeps the M2a
  band.
- The kernel (`multilayer::BandGradient`) is the mean of `∂ₓG` over the segment. Per mode, it
  integrates the layered top/bottom profile (closed form per layer) and the side eigenfunctions
  over the segment. With one interface through the centre it reproduces the M2a half-band kernel
  to 1e-11 (unit test).
- The tables are built lazily per `(interface, ladder level, side, fraction)`.
- Everything that decides the band (`s`, `t`, the run) depends only on the face and `y`, so
  every point of a band uses the same estimator and the average stays exact.

**Horizontal faces: paired normal kernel.** The layered normal kernel has no reflection
symmetry, so M2a launched one walk per sample. Since `∫K_z = 0`, the positive and negative
parts each carry `½‖K_z‖₁`. A sample now launches one walk from each part, with weights `±½`
(as the homogeneous antithetic pair). A mirrored cube swaps the parts. On L5 the second moment
of the bottom face falls from 12.3 to 4.8.

## 4. Cost

- **Table builds.** An anchored table takes about 4 ms (`M = 32`, a few interfaces) and is
  built only for the ladder levels that reach a second interface (about a hundred on the L6
  stack, about 2 s of CPU per solve in total). Landing boxes take up to 30 ms (`L = 16`),
  9 tables.
- **Memory.** An anchored table holds about 2.6 k alias entries.

Known limitation: for extreme stacks (tens of interfaces inside one cube with contrasts of
100), truncation can leave a table weight slightly below zero, and the build stops with a
panic instead of falling back to an M2a cube.

## 5. Validation

- **Tables:**
  - The cell probabilities sum to 1.
  - The band kernel with one interface equals the M2a half-band kernel. It reproduces the
    lateral linear potential `φ = x`, and its sampling matches the cells (chi-square).
  - They reproduce the exact layered linear potential (piecewise linear `φ` with `ε φ′`
    constant).
  - Unit contrast reduces to the homogeneous cube.
  - A single interface through the centre matches the M2a `t = 0` table.
  - The landing box with `L = 1` matches the homogeneous cube.
  - Sampling matches the per-cell distribution (chi-square).
- **Solver:** L1–L3 and L7 unchanged within their tolerances. L5 and L6 against Palace. L6
  steps per walk and wall time against M2a and the homogeneous V8.
