# faradice M2 Design: Stratified Dielectrics

- Date: 2026-11-17
- Status: Implemented (stage M2a; `crates/faradice-frw/src/stack.rs`, `layered.rs`, `medium.rs`; validation in
  [`docs/validation/M2.md`](../validation/M2.md)). Thin layers are handled by
  [M2b](M2b_interface_anchored.md), conformal and box dielectrics by [M2c](M2c_conformal_dielectrics.md)
- Target (at design time): faradice v0.2.0
- Background: [M1 design](M1_frw_solver.md) §5, [M1 validation](../validation/M1.md),
  [survey](../research/01_capacitance_extraction_survey.md) §3

---

## 0. Scope

| In scope (M2a, this document) | Later (M2b / M2c) |
|---|---|
| **Stratified** dielectric stack: planar interfaces `z = const`, one ε_r per layer | Conformal / non-stratified dielectrics (boxes of dielectric around conductors): M2c ([design](M2c_conformal_dielectrics.md)) |
| Exact transition and first-hop kernels for a cube **crossing one interface** | Cubes crossing two or more interfaces (thin layers): M2b, transfer matrix per mode |
| Structure file and technology file syntax for a stack | Floating conductors (dummy fill) |
| `domain.kind = "box"` (grounded or reflecting faces) | `domain.kind = "open"` with a stack (§6) |
| CPU engine | GPU engine (the WGSL walk rejects a stack until ported) |
| Validation against exact cases and Palace (multi-material mesh) | |

Real SKY130 / GF180 / IHP stacks have 10–20 dielectric layers, several of them thin
(etch stops, nitride caps of 30–100 nm) and some conformal. M2a makes the stratified
approximation exact and fast; M2b removes the step-size penalty of thin layers; M2c adds the
conformal layers. Each stage is validated against Palace before the next one starts.

## 1. Physics

With piecewise-constant ε(z), the potential is harmonic inside each layer, and across an
interface both φ and the normal displacement ε ∂φ/∂z are continuous. The capacitance estimator
of M1 §5.1 is unchanged except that the charge is the flux of D = −ε∇φ:

    Q_i = −∮_G ε(r) ∂φ/∂n dS,

so the first-hop weight uses ε at the Gaussian-surface point (not a global ε). The surface G may
cross interfaces; only the point value ε(r) matters.

## 2. Prior work (public papers only)

Sources were read as abstracts, open texts or citing papers; where only a citing paper or a
slide summary was available this is stated. No source code of any solver was consulted.

| Approach | Source | Accuracy / cost | Use in faradice |
|---|---|---|---|
| Interface jump rules: at an interface step a distance h into side i with probability ε_i/(ε_i+ε_e) | Mascagni & Simonov, walk-on-spheres for molecular electrostatics (ICCS 2003); stochastic finite differences for stratified domains (Maire & Nguyen, Math. Comput. Simul. 121, 2016); Bossy, Champagnat, Maire & Talay (ESAIM M2AN 44(5), 2010) | Local error O(h²), O(1/h) interface visits → **O(h) bias** for the simple rule | Rejected (bias) |
| Pre-characterized two-layer cube Green functions by FDM; cubes shrunk to match a table entry | Zhuang, Yu, Hu & Ye, ASICON 2011 (slides); Yu et al., "RWCap", IEEE TCAD 32(3), 2013 (per citing papers); table reduction: Int. J. Numer. Model. 29(1), 2016; IEEE TCAD 41(5), 2022 | Bias = FDM discretization error; tables per stack; homogenization of thin layers to find a match | Same *shrink-to-match* idea (§3.1), but our kernels are analytic |
| Analytic surface Green functions of cubes with any number of planar layers, computed online with a hash-table cache | Huang & Yu, DAC 2024 (method known only from citing papers) | Exact for stratified cubes | Confirms the direction; M2a derives the one-interface case independently (§3.2), M2b the multi-layer case (transfer matrix per mode) |
| Non-stratified cubes: eight homogeneous octants by volume-weighted averaging | Song, Yang & Yu, DATE 2020; Yang & Yu, IEEE TCAD 39(12), 2020 | Approximation; errors at high contrast | M2c candidate (with care) |
| Non-stratified cubes: stochastic finite differences sampling exactly the FDM Green function ("MicroWalk") | Huang & Yu, arXiv:2507.09730 (IEEE TCAD 2025) | Bias = FDM grid; not used for the first hop because of variance | M2c candidate |
| Learned kernels | Visvardis, Liaskovitis & Efstathiou, IEEE TCAD 42(8), 2023; DeepRWCap, arXiv:2511.06831 | Approximate | Out of scope |

Observations used below:

- Published FRW extractors bound the problem by an extraction window (Neumann sides, e.g.
  DeepRWCap); no far-field treatment for layers extending to infinity was found (§6).
- Thin and numerous layers are common in real stacks (e.g. arXiv:2504.20586 uses transition
  domains with five layers), so one-interface cubes are a first step, not the end state (M2b).
- The first hop is where variance matters most; layered first hops need care (§4).

## 3. Transition cubes in a stratified medium

### 3.1 Choice of the cube

Notation at a walk point x:

- h_g: geometric clearance (conductors, Dirichlet boundaries; M1 §5.3). Absorption is decided
  on h_g only (h_g < ε_stop), never on an interface-limited size.
- d₁ ≤ d₂: distances to the nearest and second-nearest interface of the **effective stack**
  (clipped to the domain, with reflections, §3.5).
- h_max = min(h_g, d₂): the largest cube that contains at most one interface.
- h: the half-side actually used.

Kernels are tabulated for interface offsets t ∈ T = {0} ∪ {k/K : k = 1 … K−1}, where t is the
interface's distance from the centre in units of h; K ∈ {8, 16, 32} (default 16).

1. h_g ≤ d₁: homogeneous M1 cube, h = h_g.
2. d₁ = 0 (the point lies on an interface; measure zero but possible): t = 0 table,
   h = h_max.
3. d₁/h_max > (K−1)/K: homogeneous cube with h = d₁ (the interface touches its boundary).
4. Otherwise choose the smallest t_k ≥ d₁/h_max and **h = d₁/t_k ≤ h_max**
   (if d₁/h_max < 1/K: t = 1/K, h = K·d₁).

Position-dependent cube sizes are legitimate: given the cube, the exact exit kernel preserves
the conditional mean of φ, so shrinking to a tabulated offset introduces **no interpolation and
no bias**; the only approximations remain those of M1 (uniform sampling within a cell, stopping
distance).

Cost: in case 4 with k ≥ 1, h ≥ h_max·k/(k+1) ≥ h_max/2; in the sub-case d₁ < h_max/K the cube
has h = K·d₁, which has no lower bound relative to h_max but grows geometrically: the next point
lies at distance ≈ K·d₁·|u_z ± t| from the interface. Small K can give a negative log-drift
(K = 2: E[log|1 + 2U_z|] < 0 in the homogeneous limit), hence K ≥ 8; near-interface step
counts and `max_steps` aborts are measured in §7 (L4).

### 3.2 Canonical layered cube and its kernels

Canonical cube [0,1]³ with centre c = (½, ½, ½) and interface at z = ζ = ½ + t/2 (above the
centre). An interface below the centre uses the same tables under z ↦ 1 − z, which swaps the
top and bottom faces and flips the sign of the z-gradient kernel. The centre's layer has ε_c,
the other ε_o; the kernels depend only on κ = ε_o/ε_c and t. The cube is symmetric in x ↔ y,
x ↦ 1 − x and y ↦ 1 − y, so three face kernels are distinct: bottom (z = 0, centre layer),
top (z = 1, other layer) and side (four lateral faces, identical up to symmetry). With t = 0 the
centre lies on the interface; ε_c is then the medium on the side the walk came from (only the
transition density is used at t = 0, which is single-valued).

Every 1D profile is built by **propagation from one end** (continuity of the function and of
ε × its derivative at ζ), which avoids removable 0/0 cases. With s_m(u) = sin(mπu):

**Top face (data on z = 1).** φ = Σ A_mn s_m(x) s_n(y) Z(z)/Z(1), γ = π√(m² + n²),
A_mn = 4∫∫ f s_m s_n, and

    Z(z) = sinh(γz)                                            (z ≤ ζ)
    Z(z) = sinh(γζ) cosh(γ(z−ζ)) + κ⁻¹ cosh(γζ) sinh(γ(z−ζ))    (z ≥ ζ)

**Bottom face (data on z = 0).** The profile is propagated from z = 1:
W(z) = sinh(γ(1−z)) above ζ, continued below ζ with continuity of W and εW′, and every factor
is normalised by W(0) (W(½)/W(0), W′(½)/W(0)); the propagated derivative already carries the
sign of the normal.

**Side face (data f(y, z) on x = 1).** Weighted Sturm–Liouville problem −(εψ′)′ = λεψ,
ψ(0) = ψ(1) = 0; in each layer ψ is a sine of the same frequency μ = √λ:

    ψ(z) = sin(μz)                                             (z ≤ ζ)
    ψ(z) = sin(μζ) cos(μ(z−ζ)) + κ⁻¹ cos(μζ) sin(μ(z−ζ))        (z ≥ ζ)

and the eigenvalues μ_k > 0 are the roots of the entire function F(μ) = ψ(1; μ) (μ = 0 is
excluded). Roots are counted with the Prüfer angle of the propagated solution (monotone in μ),
which brackets each root exactly once — fixed π-wide brackets can contain two roots at high
contrast (e.g. κ = 10, t = 1/16) — and refined by safeguarded Newton. The ψ_k are orthogonal
with weight ε, so

    φ = Σ_{n,k} B_nk s_n(y) ψ_k(z) sinh(β_nk x)/sinh β_nk,   β_nk² = (nπ)² + μ_k²,
    B_nk = 2 ∫∫ f s_n(y) ε(z) ψ_k(z) dy dz / N_k,   N_k = ∫ ε ψ_k² dz.

**Kernels at the centre** (unit cube; for half-side h multiply gradients by 1/(2h)). Each face
cell gets, per mode, the cell integral of the coefficient functional times an evaluation factor:

| Face | Coefficient on a cell | Transition density | ∂_x kernel (tangential) | ∂_z kernel (normal) |
|---|---|---|---|---|
| top z = 1 | 4 ∫ s_m(x) s_n(y) | s_m(½) s_n(½) Z(½)/Z(1) | mπ cos(mπ/2) s_n(½) Z(½)/Z(1) | s_m(½) s_n(½) Z′(½)/Z(1) |
| bottom z = 0 | same | s_m(½) s_n(½) W(½)/W(0) | mπ cos(mπ/2) s_n(½) W(½)/W(0) | s_m(½) s_n(½) W′(½)/W(0) |
| side x = 1 | 2 ∫ s_n(y) ε(z) ψ_k(z) / N_k | s_n(½) ψ_k(½) / (2 cosh(β/2)) | s_n(½) ψ_k(½) β / (2 sinh(β/2)) | s_n(½) ψ_k′(½) / (2 cosh(β/2)) |
| side y = 1 | 2 ∫ s_n(x) ε(z) ψ_k(z) / N_k | as side x = 1 | nπ cos(nπ/2) ψ_k(½) / (2 cosh(β/2)) | as side x = 1 |

The faces x = 0 and y = 0 follow by reflection (sign change of the ∂_x resp. tangential factor
where the reflected coordinate is differentiated). Modes
with ψ_k(½) = 0 still contribute to the ∂_z kernel through ψ_k′(½). The gradients are taken
with the cube and the interface **fixed** (not through the snapping map).

**Cell integrals stay analytic.** The face grid is M × M cells. With M a multiple of 2K the
interface lies on a cell boundary for every t ∈ T, so each z-cell lies in one layer and the cell
integrals of s_n, ψ_k, Z and W are integrals of sines and hyperbolic functions in closed form,
as in M1. Only the series is truncated (terms decay like e^{−β/2} at the centre).

### 3.3 Table storage

Per (κ, t): one alias table over the 6·M² face cells for the transition (0.30 MB at M = 64 with
the current layout). Gradient kernels are needed only for first hops (§4) and are built only for
(κ, t) actually used there (two axes, ≈ 1 MB). A stack with L interfaces has up to 2L values of
κ × K offsets; for L = 10, K = 16 the transition tables alone could reach ≈ 95 MB. Tables are
built **lazily** (per-slot `OnceLock`, deterministic content, so results do not depend on build
order or thread timing) under a configurable memory budget; layered tables may use a smaller M
or `f32` cell weights if the bias study (L4) allows. Cold-build time, peak memory and table
counts are reported.

### 3.4 Checks of the tables (unit tests)

Exact identities of the continuous kernels (checked on the series, 1e-10):

- κ = 1 reproduces the M1 homogeneous tables cell by cell.
- Transition probabilities are non-negative and sum to 1; ∫K = 0 for each gradient kernel.
- The layered linear potential (φ piecewise linear in z, εφ′ continuous) and x-linear potentials
  are reproduced by the transition and gradient kernels.
- Orientation: mirroring z with κ unchanged (κ is other/centre) maps the interface-above tables
  to interface-below; κ ↔ 1/κ is an identity only at t = 0.
- Eigenvalue solver: root counts against a dense sign scan with refinement; orthogonality of ψ_k.

Sampled-table checks (statistical tolerances, as in M1): Monte Carlo of the cell sampler against
the cell probabilities; uniform within-cell sampling reproduces linear moments only up to the
cell discretization.

### 3.5 Boundaries and the effective stack

The **effective stack** is built once per solve:

1. Merge adjacent layers with equal ε_r (so a κ = 1 stack is the homogeneous problem, L7).
2. Clip to the domain in z: below the first `z_top` the first layer extends to the domain
   bottom; interfaces at or below the substrate plane or outside the domain are dropped.
3. `reflect` faces in z mirror the clipped medium (even extension). An interface that coincides
   with a reflecting face disappears (both sides hold the interior medium); mirrored interfaces
   are added with their labels, then step 1 is applied again.

Grounded faces and the substrate stay Dirichlet. `reflect` faces in x or y need nothing (planes
z = const are invariant).

### 3.6 Code changes

- `faradice-frw`: a `Stack` (interface heights, ε per layer, lookup by binary search) and a
  `LayeredTables` cache, carried in `Scene` and `Walker`, used by `CpuEngine`.
- `Walker::run`: after `clearance(x)` apply §3.1; the far-field shortcut (`far_step`, clearance
  lower bound) is capped by d₁/d₂ the same way.
- First hops are currently implemented twice (`plan_batch` and `run_batch` in `solver.rs`, both
  with a global ε and homogeneous tables). Both call one shared launch function implementing §4.
- With no stack the code path is unchanged apart from one branch on `Option<&Stack>`; the
  homogeneous speed is checked with the existing benchmarks.
- GPU engine: a non-trivial effective stack is a typed error (after normalisation, so a κ = 1
  stack still runs on the GPU).

## 4. First hop from the Gaussian surface

The M1 first hop places a homogeneous cube of half-side h = h_g(r) at the Gaussian point r and
launches the antithetic pair r ± h·u (the homogeneous kernel is odd under point reflection). The
weight is ∝ ε(r)/h, so the size of the first cube controls the variance; a rule that lets h → 0
on a set of positive area (e.g. h ∝ |z − z_i| near an interface line) makes the variance
infinite (∫ dz/z²).

**Horizontal Gaussian faces** (normal ±z) lie at one height, so d₁ is constant over a face.

- The per-net offset δ is chosen so that every horizontal Gaussian face keeps d₁ ≥ δ/2 from the
  interfaces, searching δ ∈ [δ_min, δ₀] (δ₀ = M1 offset, δ_min = δ₀/4) subject to the existing
  clearance rule. The Gaussian surface (boxes, rejection grid, candidate area) is built after the
  choice.
- If no such δ exists, take the δ in that range that maximises the smallest d₁ over the net's
  horizontal faces; it is strictly positive (finitely many interface heights), and such a face
  uses the §3.1 selector (cases 3/4) for a cube centred at r. The weight is then bounded by
  ∝ 1/h ≤ 1/d₁ (= 1/(K·d₁) only in the smallest-offset sub-case) — constant over the face, so
  the variance is finite. The number of such faces and their smallest d₁ are reported.
- If h_g ≤ d₁: unchanged M1 hop with ε(r).

**Lateral Gaussian faces** (normal ±x or ±y) can cross an interface line. For a face with normal
x, write r = (x₀, y, z) and p = (x₀, y, z_i) for the projection onto the nearest interface.

1. *Half-bands.* Compute at p: H = min(h_g(p), d₂(p)). On each side σ ∈ {−, +} of the
   interface let ℓ_σ be the length of the accepted segment {z : (x₀, y, z) on the accepted
   Gaussian surface and inside the domain} adjacent to z_i, capped at η_max·H (default
   η_max = ½). The half-band on side σ is z_i … z_i ± ℓ_σ, estimated with a cube centred at p
   (t = 0) of half-side H_σ = ℓ_σ/η_j, where η_j is the smallest tabulated fraction
   ≥ ℓ_σ/H (the same shrink-to-match rule as §3.1, so H_σ ≤ H). A point r in a half-band uses
   the **half-band-averaged kernel**

       (1/ℓ_σ) ∫_{half-band σ} ε(z) ∂_x G(p + (z − z_i) e_z, r′) dz,

   tabulated per (κ, η_j, σ) in closed form (the z-integrals of Z, W, ψ_k are analytic).
   Because ℓ_σ and H_σ depend only on (face, y), not on z, replacing the integrand by its
   half-band average keeps the expectation over uniformly sampled r unchanged: the estimator is
   **exact, not an approximation**. Its weight is ∝ 1/H_σ = η_j/ℓ_σ; the
   contribution to the second moment is ∝ ℓ_σ/H_σ² ≈ η²/ℓ_σ, finite for every positive ℓ_σ.
   A segment that ends exactly at the interface on one side simply has no half-band there
   (ℓ_σ = 0: no surface points on that side). Degenerate geometries where ℓ_σ(y) → 0 over a
   range of y are counted and reported.
2. *Outside the half-bands* (d₁ ≥ ℓ_σ on r's side) with h_g(r) > d₁: layered cube centred at r
   with the §3.1 selector using h_max(r). Here d₁ ≥ ℓ_σ, so h ≥ min(d₁, h_max/2) is bounded
   below on every face where ℓ_σ = η_max·H (the generic case).
3. Otherwise (h_g ≤ d₁): the M1 hop with ε(r).

**Antithetic pairs.** The layered tangential kernel is odd under reflection in its own axis
(x ↦ 1 − x keeps the medium), so lateral first hops keep a reflected pair (u_x ↦ −u_x). The
normal kernel of a layered cube (horizontal faces in case 4) has no such symmetry: one launch,
or positive/negative parts sampled separately with balanced weights (∫K_z = 0); the better of
the two in variance per CPU time is chosen in L4. Per-sample covariance accounting (M1 §5.1)
is unchanged.

## 5. Input syntax

### 5.1 Structure file

The existing `[dielectric] eps_r` keeps its meaning when no layers are given. A stack is a list
of layers from the bottom up; `dielectric.eps_r` is the medium above the last layer:

```toml
[dielectric]
eps_r = 1.0                  # above the stack (air / passivation); the whole domain if no layers

[[dielectric.layer]]         # bottom-up; the first layer extends downwards to the domain bottom
z_top = 0.9361
eps_r = 3.9

[[dielectric.layer]]
z_top = 1.3761
eps_r = 4.5
```

Validation: `z_top` strictly increasing, all ε_r ≥ 1 and finite. `schema_version` stays 1 (the
change is additive).

### 5.2 Technology file (GDS flow)

`faradice-gds` gains the same list (`[[dielectric]] z_top, eps_r`), copied into the generated
structure file.

### 5.3 Run configuration

- `walk.interface_offsets = 16` (K ≥ 8; smaller K can drift towards an interface, §3.1).
- `walk.layered_resolution = 32` (M, a multiple of 2K).
- `walk.band_fraction = 0.5` (η_max, a multiple of 1/32 up to ½).
- `walk.landing_alpha = 0.7` (M2b).

The defaults come from L4 (`docs/validation/M2.md`). Tables are small after L4 (46 MB peak on
L6), so there is no cache budget option.

## 6. Open domain

The exterior-sphere step of the `open` boundary assumes a homogeneous medium outside the sphere.
Interfaces that extend to infinity break that assumption, so in M2a a non-trivial effective
stack requires `domain.kind = "box"` and the solver returns a typed error otherwise. Published
FRW extractors also bound the problem by a window (§2). Later options: a stack homogeneous above
some height with a hemisphere jump, or the half-space layered Green function.

## 7. Validation plan (M2a completion criteria)

| # | Case | Reference | Pass criterion |
|---|---|---|---|
| L1 | Parallel plates (reflecting sides), interface parallel to the plates; κ ∈ {0.1, 0.5, 2, 10}; several interface heights | ε₀A / Σ(d_i/ε_i) (exact) | Within 3σ, σ at the `signoff` target |
| L2 | Vertical plates (normal x) spanning two layers, reflecting y and z faces | ε₀ W Σ(ε_i h_i)/d (exact) | Within 3σ |
| L3 | Cube centred on an interface in a grounded box symmetric about it; also a two-conductor symmetric case for couplings | (ε₁+ε₂)/2 × the homogeneous matrix (exact identity; homogeneous run with an independent seed) | Within 3σ of the combined error |
| L4 | Parameter and bias study: K ∈ {8, 16, 32} with valid M (multiple of 2K, M ≤ 64), band fraction η_max, antithetic variants; near-interface step counts and `max_steps` aborts; stopping distance with a stack | L1–L3, and a wire crossing an interface against a Palace run with uncertainty < 0.05 % (fine AMR + extrapolation) | Defaults with no detectable bias at 0.05 % |
| L5 | Two and three wires in a 3–5 layer stack: faces on, near and away from interfaces; a thin layer; a `reflect` z face; box-margin sensitivity | Palace (one volume per layer, per-attribute permittivity) | total < 1 %, major couplings < 3 % |
| L6 | V8 bus with a published SKY130-like stratified stack (license checked) | Palace | Same; time and memory vs the homogeneous V8 run |
| L7 | Determinism and thread invariance with a stack; a κ = 1 stack bit-identical to the homogeneous run | — | Exact |

The bench tooling (`bench/compare/`) gets a layered mode: `palace_mesh.py` fragments the domain
box at the interface heights and tags one volume per layer; `compare.py` writes per-attribute
permittivities. FasterCap is not used for M2 (its dielectric handling needs extra panels and it
was already inaccurate on V5/V8).

## 8. Performance expectations

The homogeneous path is unchanged. In a stack, the extra cost per step is the interface lookup
and smaller cubes near interfaces (≥ h_max/2 after snapping except within h_max/K of an
interface, and limited by d₂ in thin layers). Reported per case: steps per walk (mean and tail),
time per walk, variance per CPU time, table cold-build time and peak memory. Thin layers are the
main expected slowdown; L6 decides the priority of M2b (cubes with several interfaces, transfer
matrix per mode).

## 9. Task breakdown (work units)

1. This design document.
2. `faradice-io` / `faradice-gds`: stack syntax and validation; effective stack (§3.5); typed
   errors for `open` + stack and for the GPU engine.
3. `faradice-frw/green`: layered kernels (eigenvalue solver, analytic cell integrals, band
   kernels, lazy cache) with the tests of §3.4.
4. Walk and first hop (§3.1, §4) through one shared launch function; ε(r) in the weight →
   L1–L3, L7.
5. Bench: layered Palace mode → L5; parameter study L4; defaults.
6. SKY130 stack case → L6; `docs/validation/M2.md`; release notes.

## 10. Open questions

1. Table budget: is a smaller layered M or `f32` storage accurate enough (L4)? Answered by
   L4: M = 32 is as accurate as M = 64 and cheaper; the tables need about 40 MB on L6.
2. Horizontal faces without a feasible δ: frequent enough in real layouts to need a better
   estimator?
3. Thin layers: how much slower is L6 than the homogeneous V8, i.e. when is M2b needed?
4. Should `domain.kind = "open"` with a stack fall back to a box automatically (with a warning)
   instead of an error?
