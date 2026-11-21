# faradice M2c Design: Conformal and Box Dielectrics

- Date: 2026-11-18
- Status: Implemented (`crates/faradice-frw/src/regions.rs`, `axis.rs`;
  `[[dielectric.box]]` and `[[dielectric.conformal]]` in the structure file, `conformal` in the
  technology file). The walk cost near dielectric edges is addressed by [M2d](M2d_edge_cubes.md)
- Follows: [M2 (stratified)](M2_multi_dielectric.md), [M2b (interface-anchored)](M2b_interface_anchored.md);
  validation goes to [`docs/validation/M2.md`](../validation/M2.md)

---

## 0. Scope

| In scope (M2c) | Later |
|---|---|
| Dielectric **boxes** (axis-aligned, one ε_r each) that override the stratified stack | Non-Manhattan dielectrics (sloped sidewalls) |
| **Conformal rules**: shells of given side/top thickness around the conductors of a tagged layer, generated in the solver | Lattice (FDM) sub-walks inside non-stratified cubes (§11, only if §9 shows a need) |
| Exact transitions through cubes stratified along **any** axis (x, y or z), reusing the M2a one-interface tables by rotation | Multi-interface cubes and landing domains along x or y |
| First hops with the Gaussian surface outside every box | Half-bands for Gaussian faces crossing box faces |
| Technology file syntax for conformal layers (GDS flow), SKY130 NILD3_C / NILD4_C | Passivation over the top metal (TOPOX / TOPNIT / PI1) |
| CPU engine | GPU engine (typed error, as for stacks) |

## 1. Problem

The public SKY130 process stack diagram (Apache-2.0; `metal_stack.svg` linked from
`rules/assumptions.html`) labels two conformal dielectrics, **NILD3_C** and **NILD4_C**, both
K = 3.5. Read from the drawing (the page text gives no geometry): each is a strip on the
**sidewalls** of met1 resp. met2, 0.030 µm wide (dimension line next to the strip, not labelled
with the layer name), as tall as the metal (met1 1.3761–1.7361 µm, met2 2.0061–2.3661 µm), with
no layer on the metal top. The surrounding dielectrics are NILD3 (K = 4.5) and NILD4 (K = 4.2);
below the metal bottoms are NILD2 (4.05) and NILD3. These values must be checked against the PDK
version in use; the technology file says so (§6.3).

The L6 case leaves the strips out. A parallel-plate estimate for the minimum met1 spacing
(0.14 µm) shows the size of the effect: 0.03 + 0.03 µm of K = 3.5 and 0.08 µm of K = 4.5 in
series give an effective K of 4.01 instead of 4.5, so the lateral plate part of a met1–met1
coupling falls by about 11 %. Fringe parts change less. The goal accuracy of M1/M2 (1 % total,
3 % couplings) therefore needs the strips.

A strip is not stratified: next to a sidewall, ε depends on x (strip | NILD3), and at the strip's
bottom edge three media meet (NILD2 below, strip and NILD3 above). M2a/M2b cubes are stratified
in z only and cannot contain such a point.

## 2. Prior work (public papers only)

No patents and no solver source code were read. Some entries are known only from abstracts or
citing papers, as marked.

| Approach | Source | Exact? | Reported accuracy / cost |
|---|---|---|---|
| Eight-octant cubes: a cube touching a non-stratified dielectric is replaced by eight homogeneous octants with volume-averaged ε, sampled on the fly | Song, Yang & Yu, DATE 2020 (full text); Yang & Yu, IEEE TCAD 39(12), 2020 (abstract) | No (homogenization; the octant sampling rule is checked empirically) | ±1.2 % vs Raphael at moderate contrast (7 vs 3.9, 5–40 nm shells); 3.5–10 % on industrial and FinFET cases per a later paper |
| Analytic stratified cubes, non-stratified cubes homogenized | Huang & Yu, DAC 2024 (citing papers only) | No for non-stratified cubes | 20–24 % on high-k FinFET cases (per the MicroWalk paper) |
| MicroWalk: inside a non-stratified cube, a stochastic finite-difference walk samples the 7-point FDM Green function on a 24³ voxel lattice | Huang & Yu, arXiv:2507.09730 (IEEE TCAD 2025) | Unbiased for the lattice; bias = lattice, not analysed for layers thinner than a voxel | 0.5–2 % vs Raphael/QuickCap; ≈ 0.34 N² hops per transition; no tables. First cube still special |
| Learned kernels for non-stratified cubes (3D CNN on 23³ voxels) | Rodriguez, Huang & Yu, arXiv:2511.06831 (AAAI-26) | No | 1.24 % mean; GPU |
| Interface jump rules, skew Brownian motion, walk on interfaces | Mascagni & Simonov 2004; Bossy, Champagnat, Maire & Talay 2010; Lejay & Maire 2013; Maire & Nguyen 2016; Ding & Stinchcombe, arXiv:2508.16767 (abstracts) | No (O(h) or O(h²) bias near the interface); smooth interfaces assumed, edges and corners not treated | — |

Observations:

- All published FRW treatments of conformal dielectrics either **homogenize** a cube that cuts a
  shell or **discretize** it on a lattice. No unbiased continuum estimator for dielectric edges
  or corners was found.
- faradice's tables are analytic and exact for one interface at any contrast (M2a). The design
  below never builds a cube that contains a dielectric edge; near edges it uses smaller cubes.
  This is exact up to the M1 approximations (cell-uniform sampling, stopping distance). The price
  is extra steps near dielectric edges, measured in §9 (C5, C6).

## 3. Model

### 3.1 Medium

The material definition does not depend on which conductors exist. Conductors only mask it,
because walks never enter them.

- ε(r) is that of the **box** containing r, if any. Otherwise it is the stratified stack at z
  (M2a).
- Boxes are axis-aligned and may touch. They must not overlap one another. Explicit boxes and
  conformal shells are combined into disjoint boxes before solving (§6.2).
- Boxes are **not** clipped by conductors. Deleting a conductor (D3) therefore exposes the
  material that was defined there, not the stack.

Reflecting faces of the domain mirror the whole medium, boxes included, before anything else is
derived. This follows M2a §3.5. A box that touches a reflecting face continues across it, so its
face there is not an interface.

### 3.2 Interface faces

The **interface faces** are the parts of box faces across which ε jumps. They are built once per
solve, on the mirrored medium:

1. Collect the face rectangles of all boxes.
2. Split each rectangle along the edges of the other boxes' faces on the same plane.
3. Split vertical rectangles additionally at every stack height. Above and below a stack height
   the outside medium differs.
4. Evaluate ε on both sides at the centre of each piece. Keep the pieces with a jump, then merge
   neighbours.

The stack planes are the remaining z-interfaces. Where a box covers a stack plane, the plane is
not a physical interface.

Three spatial indices hold the box-derived faces, one per normal axis. They are rectangle
indices, because `Aabb` rejects degenerate boxes; §10 budgets them. A fourth index holds the box
volumes.

> **Lemma.** Let Q be an open, conductor-free cube that contains no interface (box-derived face or
> stack plane) normal to b, for both axes b ≠ a. Then on every a-plane, the union of the interface
> pieces covers either all of Q's cross-section or none of it. Hence ε restricted to Q depends on
> the a-coordinate only.

Proof: inside Q the jump set of ε is closed. Suppose an a-plane carried a jump on part of Q's
cross-section only. Then the boundary of that part would be an edge inside Q. Along that edge the
jump must continue on a face normal to some b ≠ a, and that face would meet Q.

The pieces of one plane may be several rectangles (step 2). The lemma is about their union, so
seams between pieces do not matter. Stack planes are counted in the hypothesis even where a box
overrides them. This only shrinks the admissible cubes, and the contrast across such a plane is
measured, not assumed (§4.2).

## 4. Transition domains

### 4.1 Distances

At a walk point p:

- h_g is the conductor clearance. It is unchanged, and absorption is decided on h_g only.
- δ_a(p) is the L∞ distance to the nearest interface normal to a: box-derived a-faces, and for
  a = z also every stack plane, counted as a full plane.
- r_a = min(h_g, δ_b, δ_c) for {a, b, c} = {x, y, z}. Within r_a the lemma applies.
- d₁ᵃ ≤ d₂ᵃ are the distances to the nearest and second-nearest **distinct planes** of a-normal
  interfaces within r_a.
- v(p) is the L∞ distance to the nearest box **volume**; it is 0 inside a box.

Without boxes, v = ∞, and the step is exactly the M2b step. That is one `Option` branch, as for
the stack in M2a.

### 4.2 Candidates

1. **Stack cube (M2b).** The full M2a/M2b selection (anchored cubes, landing boxes,
   one-interface cubes), with the clearance h_g replaced by min(h_g, v).
   - Every domain it builds is centred on p or contains p, and its half-extent from p is at most
     that clearance. The M2b landing box is sized from the clearance in the same way. Such a
     domain is disjoint from every box volume, so the stack profile and the M2b tables are the
     true medium.
   - Inside a box, or touching one, v = 0, and this candidate does not exist.
2. **Axis cube along a ∈ {x, y, z}.** The M2a selector (M2a §3.1) with h_max = min(r_a, d₂ᵃ) and
   the nearest a-plane at d₁ᵃ.
   - It gives a homogeneous cube (h ≤ d₁ᵃ), a touching cube, or a one-interface cube snapped to a
     tabulated offset t_k.
   - The contrast is κ = ε(other side)/ε(centre side), measured by two point lookups inside the
     cube (the lemma makes them representative). An overridden stack plane gives κ = 1, and the
     κ = 1 table equals the homogeneous one (M2a §3.4).
   - The kernel is the M2a table for (κ, t_k), rotated so that the canonical z axis becomes a
     (§4.3).

The step takes the **largest** candidate cube. Ties go first to the stack candidate, then to x,
y, z. Every candidate is conductor-free and stratified with exact kernels, so the estimator stays
exact, as in M2a §3.1.

**Progress.**

- Near a single interface the axis cube is the M2a cube. Its geometric growth away from the
  plane needs K ≥ 8 (M2a §3.1), which the solver enforces.
- On an edge line (a point with r_a = 0 for every a, and v = 0 or h_g = 0), no candidate exists.
  The continuous process reaches such a line with probability zero, but the walk does not: near
  an edge every cube is limited by the distance r to the line, so log r performs a driftless
  random walk and comes within 10⁻⁹ h_g of the line in a few percent of the walks (9–13 % with
  a strip box in the tests of §9, 1.5 % with shells). The first implementation aborted those
  walks, which biased the result by up to 15 %. Edges therefore get their own domain (§4.6).
  - At a vertex (three planes within the tolerance) no domain exists; the walk aborts, and a
    potential-map point there gets a typed error. Walks drift away from points in 3D, so this
    is rare.
- Within EDGE_SNAP · h_g (10⁻⁸ h_g) of an interface plane, the walk moves onto it. Near a plane
  the offset cube shrinks with the distance (K·d₁), while the cube centred on the plane does
  not. φ is Lipschitz near a smooth part of an interface, so the move changes φ by
  O(10⁻⁸) relative. This replaces the rounding-sized snap.

### 4.3 Rotated tables

The M2a tables are defined in the canonical cube, with the interface normal to z, and depend
only on (κ, t). Axis a maps by the cyclic permutation that sends z to a. This permutation is a
rotation, so the cube's symmetries and the alias tables carry over. Only the face labels and the
direction of the sampled displacement change.

A cache keyed by the bit patterns of (ε_c, ε_o) holds the tables for box contrasts. A solve has
few distinct ε values, so the cache stays small. Stack interfaces keep their existing
per-interface slots.

### 4.4 Exits on interface planes

M2b writes `z = z_i` exactly when a landing box exits through its bottom face (M2b §3.4). M2c
extends this by **provenance**. When the selector builds a cube, it records which cube face lies
on which interface plane, by construction: the touching cube of case 3, or the offset cube with
t_k = 1. An exit through that face sets the coordinate to the stored plane value, mirrored if
needed.

Beyond provenance, the only moves onto planes are the EDGE_SNAP moves of §4.2 and §4.6. They are
bounded by 10⁻⁸ h_g and never cross a nearer stack interface; a nearest-face distance that comes
from a tangential gap moves the point within its own material.

The existing tolerance projection onto stack interfaces (`medium.rs`, about 10⁻¹²) is left
unchanged here. It is listed in §11.

### 4.5 Cost near edges

Near an edge line the cube is limited by the distance s to the edge, while the conductor
clearance can be about w + s. The ratio of the two tends to 0 as s → 0, so edges do cost steps.
This document does not claim the cost is small. §9 measures it:

- the steps per walk by candidate type;
- the tail of the step distribution;
- the aborts.

The expected main contributors are:

- **Facing shells.** Between two met1 wires at minimum spacing, two parallel x-faces lie within
  one cube. The axis cube is one-interface, so h ≤ d₂ˣ: 0.04 µm instead of the 0.07 µm clearance
  in the middle of the gap.
- **The three-media bottom edge**, where a strip meets the stack plane at the metal bottom.

If they are expensive, the next steps are multi-interface axis cubes (the M2b propagation applies
along any axis) and landing on vertical faces.

### 4.6 Edge balls

Let a walk point lie within EDGE_SNAP · h_g of two interface planes normal to axes b and c, and
let a be the third axis. The walk moves onto their intersection line, by at most
EDGE_SNAP · h_g, and takes a ball of radius R around the new centre. R is capped by:

- the clearance left after the move;
- the reflection reach;
- the L∞ distance to every interface face that does not lie in one of the two planes, and to
  every stack plane other than them.

**ε inside the ball.** Within such a ball, ε is constant on each of the four quadrants around the
line, so ε depends only on the azimuth θ around it. An interface face in one of the planes can
end only along another interface face, and no such face lies within R.

**Exit distribution.** Write the equation in cylindrical coordinates (ρ, θ, a) around the line.
Integrate ∇·(ε∇φ) = 0 over a period in θ. The θ-flux term (1/ρ²) ∂_θ(ε ∂_θ φ) integrates to
zero, because ε ∂_θ φ is continuous and periodic. So the weighted average
φ̄(ρ, a) = ∫ ε φ dθ / ∫ ε dθ satisfies the homogeneous axisymmetric Laplace equation. Its boundary
data are the ε-weighted azimuthal averages of the data of φ. On the axis, φ = φ̄.

The value at the centre is therefore the homogeneous ball average of the ε-weighted data. A
uniform point on the sphere whose azimuth is redrawn with density ∝ ε(θ) samples it: draw the
quadrant with probability ∝ its ε (read at its centre in the ball), then θ uniformly within it.

The kernel is exact. The only approximation is the move onto the line. φ is Hölder continuous
there, with exponent α > 0 set by the contrasts. The bias is therefore of order
P(reach) · (10⁻⁸)^α, which is negligible for realistic contrasts (§9 C6).

## 5. First hop

The first-hop weight is ∝ ε(r)/h (M2a §4). The variance is finite if, on each Gaussian face, the
first domain has a positive lower bound.

**Gaussian surface outside the boxes.** δ₀ is computed from the conductor gaps, as in M1. δ is
chosen in [δ₀/4, δ₀] to maximise the smallest first-cube half-side over the net's surface, with
v included in that size (M2a §4, as revised in L5), under one hard constraint:

- **The surface must not touch any interface face** (checked geometrically on the exposed faces
  of the dilated net boxes).

The constraint excludes the case where a box face crosses a Gaussian face transversally. That
case would need half-bands, which are out of scope. The surface may lie inside a box. There the
first cube is the larger of two:

- the material piece's interior clearance (stack planes are overridden);
- the distance to the interfaces and stack planes.

The score of an offset also includes the distance from the Gaussian faces to the interfaces,
sampled on a 3 × 3 grid per face. A candidate whose smallest first cube is zero (a Gaussian face
on a stack plane) is rejected.

- **Shells.** For SKY130 met1 at minimum spacing, δ₀ = 0.5 × 0.14 = 0.07 µm. The surface then
  lies 0.04 µm from the own shell and 0.04 µm from the neighbour's.
- **No δ in range keeps the surface off the interfaces.** This happens with explicit boxes, or
  when a gap is narrower than its two shells and they merge. The solve then returns a typed
  error that names the net.

**Selector.** The first hop uses its own selector:

- **Lateral faces crossing a stack interface.** The M2a/M2b half-band rules are used, with
  every clearance replaced by min(h_g, v). This covers H, the anchored level `ladder(h(p))` and
  the band lengths ℓ.
  - All of these are evaluated at the band's anchor point p = (x₀, y, z_i). They therefore depend
    only on the face and y, which is what keeps the band average exact (M2a §4). The current
    `band_hop` sizes its domain from h_g and stack distances only, so it must take v as well.
  - Each band domain is disjoint from the boxes because its half-side is at most v(p).
- **Elsewhere.** The first hop needs gradient kernels, which exist only for centred cubes.
  Landing boxes (M2b §3.3) provide transition sampling only. So the first hop chooses the
  larger of two cubes:
  - the existing centred first-hop cube of M2a/M2b, with its clearance capped by v;
  - the centred axis cubes of §4.2.

  It never uses a landing box. Horizontal faces keep M2a's rule of staying off stack interfaces.
  The gradient along the face normal n comes from:
  - the M2a normal kernel (± pair, M2b §3.5), if the chosen cube is an axis cube along n;
  - the rotated M2a tangential kernel, otherwise;
  - the existing stack kernels, for a stack cube.
- **Variance.** Every first domain has half-side ≥ min(stack lower bound, inf v) > 0 on the
  surface, so the variance is finite.

## 6. Input

### 6.1 Structure file

Additive (schema 1 stays valid):

```toml
[[conductor]]
net = "a"
layer = "met1"                # optional tag; used by conformal rules and kept through D3 edits
boxes = [[0.0, 0.0, 1.3761, 4.0, 0.14, 1.7361]]

[[dielectric.box]]            # explicit box, overrides the stack
min = [0.0, 0.0, 1.3761]
max = [4.0, 0.5, 1.7361]
eps_r = 3.5

[[dielectric.conformal]]      # shells around the conductor boxes tagged with this layer
layer = "met1"
side = 0.030                  # lateral thickness
top = 0.0                     # thickness above the conductor top (0: none)
eps_r = 3.5
```

Validation:

- ε_r ≥ 1 and finite; `min` < `max`.
- `side` and `top` ≥ 0, not both 0.
- Explicit boxes must not overlap each other; touching is allowed.
- A rule whose layer tags no conductor box gives a warning.

### 6.2 Shells from conformal rules (in the solver)

Shells are generated from the current conductor geometry and the layer tags. Edits that move or
split a tagged box keep its tag, so the shell follows the edited box, including a changed
vertical extent.

For a rule with tagged boxes B₁ … Bₙ:

1. The **grown region** G = ⋃ᵢ Bᵢ ⊕ ([−side, side]² × [0, top]). Each box grows laterally by
   `side` and upward by `top`; it does not grow downward.
2. The shell is G ∖ ⋃ᵢ Bᵢ. Only the rule's own boxes are subtracted. Other conductors overlapping
   the shell just mask it (§3.1).
3. All shells and explicit boxes are combined into disjoint, material-labelled boxes:
   - explicit boxes win over shells;
   - a later rule wins over an earlier one;
   - each overlap is reported as a warning with its volume.

This is a 3D rectilinear Boolean with labels. The 2D Manhattan code of `faradice-gds` (integer
polygons) does not provide it, so §10 budgets a slab-sweep implementation in `faradice-geom`:

- split in z at all box heights;
- run a 2D rectangle union/difference per slab;
- merge equal neighbouring slabs.

L∞ growth gives square outer corners. The real PDK corners are rounded, which is below the
accuracy goal and not modelled.

### 6.3 Technology file (GDS flow)

```toml
[[layer]]
name = "met1"
# ...
conformal = { side = 0.030, top = 0.0, eps_r = 3.5 }   # NILD3_C
```

The GDS writer tags every conductor box with its layer name. It emits one
`[[dielectric.conformal]]` per layer that has `conformal`.

`tech/sky130_stack.tech.toml` gets NILD3_C on met1 and NILD4_C on met2. A comment gives the source
and how the 0.030 µm dimension line was read (§1).

## 7. Interactions

- **Open domain:** allowed with boxes and no stack if the exterior sphere encloses every box
  (the radius is computed from conductors ∪ boxes). With a stack, still an error (M2a §6).
- **GPU engine:** boxes are a typed error, like a non-trivial stack.
- **D3 sensitivities:**
  - **A0 (finite differences)** reuses the nominal walks and continues only those absorbed by
    the moved box (`sens.rs`, `continue_end`). That is valid only if the medium does not change.
    - A moved face of a box tagged by a conformal rule moves its shell, and with it the
      transition probabilities everywhere. D3 §4.4c splits the derivative into a continuation
      term (in an intermediate medium) and a shell-interface term. Lateral faces and `z+` faces
      with `top > 0` are supported this way. Other faces of tagged boxes are refused with a
      typed error.
    - Faces of untagged conductors, and explicit boxes, are fine. The medium is defined without
      the conductors (§3.1), so it is unchanged.
  - **A (shape derivative):** already refuses multi-dielectric inputs.
  - **D-del / D-add replay:** valid only if the medium is unchanged. Deleting or adding a box
    tagged by a conformal rule changes a shell, so the replay is refused with a typed error.
    Deleting other conductors exposes the defined material (§3.1) and stays valid.
- **Potential map (`faradice map`):** uses the walk. Points on dielectric edges take edge balls
  (§4.6); points on a vertex get a typed error (§4.2). Gradients are refused with boxes for now
  (the interface kernels of D3 §4.4c cover points on interfaces only).
- **Reports:** the number of boxes and faces, the steps per walk by candidate type, aborts, and
  the chosen Gaussian offsets.

## 8. Cost expectations

- Without boxes the path is unchanged. This is checked with the existing benchmarks and by
  bit-identical results (L7).
- With boxes, each step adds one box-volume query. Steps near boxes add three face queries.
- Tables: a few (κ, t) sets per distinct ε pair, about 1 MB each at M = 32.
- Building faces and shells is output-sensitive. Crossing tagged bars can give Θ(m²) shell boxes
  in one slab. The counts of generated boxes and faces and their memory are reported with the
  table build time. A configurable limit returns a typed error instead of exhausting memory.

## 9. Validation plan

| # | Case | Reference | Pass criterion |
|---|---|---|---|
| C1 | Vertical parallel plates (normal x), reflecting y/z faces, one or two box slabs between them (adjacent to a plate, as a shell; and in the middle); κ ∈ {0.1, 0.78, 10} | Series formula (exact) | Within 3σ at 0.05 % |
| C2 | Same as L1 (horizontal plates) with the layer given as a box spanning the domain instead of a stack layer | L1 exact value | Within 3σ (exercises z axis cubes from boxes) |
| C3 | A box with the stack's ε inside one layer; a κ = 1 box | Run without boxes | No faces (§3.2); results within 3σ (the volume index still caps stack cubes) |
| C4 | Two met1 wires at minimum spacing with NILD3_C strips in the SKY130 stack (reflecting ends) | Palace, one volume per box and layer | total < 1 %, coupling < 3 %; Palace uncertainty stated |
| C5 | L6 bus with NILD3_C / NILD4_C strips | Palace | Same; steps per walk by candidate type and their tail; time vs L6; change of each capacitance vs L6 |
| C6 | Edges: a plate next to a three-media edge (strip on a stack plane) and a box corner in an otherwise exact configuration (C1 with a slab ending inside the domain), stopping distance ×3 and ÷3, `max_steps` and zero-progress aborts | Palace; C1/C2 | No detectable bias at 0.05 % where exact; step tails and aborts reported |
| C7 | Determinism and thread invariance with boxes; no boxes bit-identical to M2b | — | Exact |

Palace meshing: `bench/compare/palace_mesh.py` fragments the stack volumes with the box volumes
and tags each box with its own attribute. 30 nm strips need local mesh refinement. The mesh size
and AMR settings are recorded with the results.

## 10. Task breakdown (work units)

1. This document.
2. `faradice-geom`: rectangle index for degenerate faces (L∞ nearest query), box-volume index,
   and the labelled 3D rectilinear Boolean (slab sweep, §6.2) with unit tests.
3. `faradice-io` / `faradice-gds`: conductor `layer` tags, `[[dielectric.box]]`,
   `[[dielectric.conformal]]`, tech `conformal`; validation; writer.
4. `faradice-frw`: medium with boxes, mirroring, shells, interface faces (§3.2). Unit tests:
   - the lemma on random box sets with a stack (brute-force ε sampling inside chosen cubes);
   - face construction, including stack-height splits and reflection.
5. Walk: axis cubes with rotated tables, the selection of §4.2, provenance snapping (§4.4),
   K ≥ 8 → C1–C3, C7.
6. First hop (§5), including the v-capped `band_hop`; D3 refusals (§7) → C1–C2 with Gaussian
   faces near boxes; C6.
7. Bench: Palace box mode → C4, C5; `docs/validation/M2.md`; SKY130 tech file; CHANGELOG.

## 11. Open questions

1. Are facing shells and three-media edges (§4.5) expensive enough to need multi-interface axis
   cubes or vertical landing (C5, C6)?
   **Answered by C4/C5** ([validation](../validation/M2.md#m2c-dielectric-boxes-and-conformal-shells)):
   yes. SKY130 sidewall shells take 4.8× the steps per walk and 23–35× the time. Walks spend most
   steps approaching dielectric edges with cubes limited by the distance to the edge.
2. Do real PDK conformal layers need top shells or rounded corners? SKY130 shows neither; GF180
   and IHP to be checked.
3. If edge regions turn out costly at high contrast (FinFET-like shells), would a lattice
   sub-walk conforming to the faces (MicroWalk-style, with no voxel partially filled) be worth its
   lattice bias?
4. Passivation over the top metal: same mechanism with `top > 0`; worth modelling for top-metal
   capacitance?
5. The existing 10⁻¹² tolerance projection onto stack interfaces can remove a noticeable part of
   a layer thinner than about 10⁻¹⁰ (relative units). Replace it by provenance snapping (§4.4)?
6. A0 with conformal shells: a replay that detects changed dielectric regions, instead of
   refusing?
