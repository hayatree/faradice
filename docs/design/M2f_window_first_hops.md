# M2f: Window first hops for 2D cross-sections

Status: Implemented (`crates/faradice-frw/src/windows.rs`, `section.rs`; `walk.window_hops`, default
on; CPU solve paths, off for replays and sensitivities, §3.4). Follows M2e §5.

## 1. Problem

After M2e, conformal shells still need more walks than the same stack without them (C5, default
accuracy, relative to L6): 1.8× for the bus bits and 1.4× for the met1 lines. The first-hop
oracle of M2e §1 puts all of this excess on the first cube being smaller than the clearance `h`.

What remains are Gaussian points whose cube meets planes of two normals. On a lateral face of a
bus bit, these are:

- the shell's face, normal to the face's normal `u`;
- the stack plane and the shell's ends, normal to the face's other tangential axis `v`.

Neither M2c (one plane at a tabulated offset) nor M2e (planes of one normal) covers that cube. In
C5, such planes lie within `h` of 56 % of the lateral face.

These cubes have one thing in common: along the wire axis `w` nothing changes. Every interface in
them spans the cube along `w`, so the cross-section in `(u, v)` is a 2D rectilinear partition.

## 2. Why not a point kernel

A point kernel would need the gradient of the Green function of the 2D-partitioned cube at the
Gaussian point. The plane offsets relative to the point are continuous, because points are drawn
uniformly over the face. Tables would then need interpolation in those offsets, which biases the
result, or one build per point, which is far too slow.

## 3. Method

### 3.1 Window fluxes

The charge is `Q = ∫ q dA` with `q = ε E·n`. Each face is covered by windows `ω_i` that add up to
one: `Σ ω_i ≡ 1` on the face. Then `Q = Σ_i ∫ ω_i q dA`.

The estimator works as follows:

1. Draw a point `p` uniformly from the candidate area `A`, as now.
2. Pick window `i` with probability `ω_i(p)`.
3. Weight the sample by `A · X_i / ∫ω_i`, where `X_i` is an unbiased estimate of `∫ ω_i q dA`.

Then `E = Σ_i (∫ω_i/A) · A · ∫ω_i q / ∫ω_i = Q`.

The point `p` only selects the window. A window that does not qualify (§3.3) leaves `p` to the
point estimators of M2c/M2e. This keeps the estimate exact, because qualification depends only on
the window, not on `p`. Mixing the two estimators is then the identity
`Q = Σ_qualified ∫ω_i q + ∫ (Σ_other ω_i) q`.

The windows have spacing `P` along each tangential axis, with centres at `lo + iP`. Each is
`ω(x) = S(1 − |x − c|/P)`, where `S(t) = f(t)/(f(t) + f(1 − t))` and `f(t) = e^{−1/t}`.
These windows are C∞, add up to one, and have `∫ω = P` per axis.

Windows cut by the face's edges are not tabulated: they always go to the point estimators.

The half-band estimator of M2a §4 is not a point estimator: for a lateral point near a stack
interface it estimates the average of `q` over the band (the run of the face along `z` from
the interface, at the point's other coordinates). It is unbiased only if every point of the band
uses it with the same probability. After windows, a band is used only if, for each of the two
windows covering the point along the other tangential axis, all windows meeting the band along
`z` qualify alike. Since the windows along `z` add up to one, the fallback probability is then
constant along the band. Otherwise the point takes the centred cube. Turning bands off after
every fallback instead is unbiased, but its tiny cubes next to the interface blow up the
variance (met1 1.3×, one C5 net 10⁴×).

### 3.2 Kernels

The box around window `i` is the window's support plus one half-support on each side along `w` and
`v`, and `±h` along `u` (capped by the offset). Its kernel `F` on the box surface satisfies
`∫_box-surface F φ = ∫ ω ε (−∂φ/∂u) dA` over the window's plane, for every `φ` harmonic in the
box.

`F` is computed in `faradice_frw::section` as follows:

- **Discretization.** A node-based finite-volume scheme on a tensor grid. In `(u, v)` it has nodes
  on every plane, the window plane and the walls, and is non-uniform. Along `w` it is uniform.
- **Functional.** The window flux is a linear functional of the interior nodes: the flux across
  the node line of the window plane, interpolated from the two adjacent dual faces and weighted by
  `ω` at the nodes. `F` is the boundary coupling of the adjoint solution.
- **Splitting along `w`.** The discrete sine transform along `w` splits the system exactly into
  2D systems, one per odd mode, since the window is even.
- **2D solves.** Each 2D system is solved by conjugate gradients. The preconditioner is the
  constant-coefficient operator on the same non-uniform grid, inverted exactly by the fast
  diagonalization method (1D generalized eigenproblems by cyclic Jacobi). About 4 iterations per
  mode are needed.
- **Accuracy gate.** Three levels with 1, 2 and 4 grid cells per table cell, and Richardson
  extrapolation as in M2d. Tables whose estimated relative L1 error exceeds the limit are
  rejected.
- **Table cells.** The table cells are the coarsest level's grid cells. Within a cell, a point is
  drawn from the finest level's 4×4 sub-cells where they all share the cell's sign, and
  uniformly otherwise. This removes most of the bias of cell-uniform sampling (§4.3).

The kernel is signed with `∫F = 0`. It is sampled either from `|F|` with its sign, or as an
antithetic pair from its positive and negative parts, each carrying `½‖F‖₁`. The hop weight is
`‖F‖₁ · scale / (P_w P_v)`.

Kernels are cached by the bits of their canonical configuration: positions relative to the box,
rounded to 2⁻³⁶, and ε. Window decisions are cached per (surface, face, window).

### 3.3 Qualification

A window qualifies when all of the following hold:

- its support lies on the outer Gaussian surface (no other dilated box of the net meets it);
- the box is conductor-free (the L∞ cube of its largest half-extent at the centre);
- the box lies inside the domain and within the faces' reflection reach;
- every interface face in the open box is normal to `u` or `v` and spans the box along `w`, and
  no stack plane crosses it along `w`;
- there are at most six planes per axis;
- the window plane is not an interface;
- the box contains at least one plane normal to `v`. Where the cross-section is layered or
  homogeneous, the point cubes of M2e already reach `h`, and the window estimator has the higher
  variance there (§4.1).

The wire axis `w` is the face's longer tangential axis.

### 3.4 Where window hops are used

Window hops are used in `solve` and `solve_with` (and the cross-moment solve). They are off in
two cases:

- **Recorded samples.** Replays (D3 §8) and sensitivities (D3) re-run samples under edited
  geometry and must reproduce the nominal sample exactly. Window decisions are cached per medium
  and do not see overlays.
- **Walkers with an overlay or added conductors.**

The trace of a window hop is the box, as an L∞ cube around its centre.

Kernels are built on first use, by the walker that needs them. Their modes are solved on extra
threads taken from a process-wide budget of four times the available parallelism, so
concurrent builds cannot multiply the thread count. A failed spawn leaves its modes to the
calling thread. Other walkers
needing the same kernel wait for it. A pre-build of all windows of a net before its walks was
tried. It saved CPU (C5 −3 %), but it delayed the walks and cost wall time on C4 (+8 %), so it
was dropped.

## 4. Results

### 4.1 Variance

C5, 50 k walks per net. Variance relative to M2e (point first hops):

| variant | bus bits | met1 lines | met3 lines |
|---|---|---|---|
| box-car patches, every qualifying box | 0.78–0.81 | 1.10 | 1.56 |
| box-car patches, 2D only, box = 2× patch | 0.68–0.71 | 0.90 | 1.10 |
| box-car, 2D only, box = 4× patch | 0.59–0.63 | 0.81–0.83 | 1.00 |
| box-car, 2D only, box = 8× patch | 0.57–0.61 | 0.78 | 0.99 |
| windows, 2D only (§3) | 0.62–0.66 | 0.83–0.84 | 1.01 |

C10 (`c10_window_first_hops_agree_with_the_point_hops`: two wires with sidewall shells on a
stack plane, 200 k walks) has a variance ratio of 0.68 and agrees with the point hops within
0.6σ.

Walks at the default accuracy, relative to L6: 1.42 (M2e) → 1.14. Steps per walk also drop:
bus bits 37–42 → 33–38.

### 4.2 Accuracy

C5 with 1 M walks per net, two seeds, agrees with M2e: 18 entries, mean −0.7σ. C4 agrees at
100 M walks per net, pooled over two seeds, within 1.8σ (−0.05 … +0.02 %). The `reference`
runs against Palace match M2e (`docs/validation/M2.md`).

Kernel checks:

- the linear potential `φ = −u` gives `ε ∫ω` to 1e-6, both homogeneous and with planes normal to
  `v`;
- the layered linear potential across the shells gives it to 1e-4;
- `∫F = 0` holds to 1e-9 of `‖F‖₁`.

### 4.3 Kernel accuracy and cost

The true error was measured against a reference at twice the resolution (C5-like box with shells
and a stack plane, `∫Fφ` for four test functions):

| base cells (u, v) / w | estimate | worst true error / ‖F‖₁ | time (1 core) |
|---|---|---|---|
| 8 / 8 | 2.1e-4 | 8.9e-5 | 39 ms |
| 12 / 8 | 1.2e-4 | 3.4e-5 | 111 ms |
| 16 / 16 | 1.9e-5 | 1.4e-5 | 384 ms |

The error depends on the `(u, v)` resolution and barely on `w`. Uniform sampling within table
cells contributes a bias: halving the table cells, to the coarsest level's grid, cut the true
error by 4×. Sampling sub-cells within the cells cut it again: for the cubic harmonic
`4[(u−½)³ − 3(u−½)(v−½)²]` in a unit box with a plane across the window, from 5.1e-4 to 3.0e-5
of ‖F‖₁. faradice uses 12 / 8 with a gate of 3e-4. `tables_agree_with_a_finer_reference` and
`sampling_within_cells_is_unbiased` check that the true error stays below 1e-4.

C5 builds 40 kernels and 45 k window decisions (about 50 µs each).

### 4.4 Time

16 threads, default accuracy. The implementation was measured in A/B against the preceding
code, three rounds:

| | M2e | M2f | L6 |
|---|---|---|---|
| C5 | 6.37–6.69 s | 5.70–5.76 s | 1.23 s |
| C4 | 1.47–1.57 s | 1.35–1.44 s | |

At the `reference` accuracy the fixed costs matter less: C5 takes 686 s → 490 s wall
(CPU 12 617 s → 7 025 s), and C4 195 s → 80 s (`docs/validation/M2.md`).

Kernel builds and window decisions cost about 0.8 s of the C5 gain.

## 5. Rejected variants

- **Box-car patches.** Their edges make the functional singular. The `w` resolution then
  dominates the error, and coarser `w` grids fail the gate.
- **Exact sine modes along `w`** (no `w` grid). A compactly supported smooth window has slowly
  decaying sine coefficients, and C5 needed 318 modes for 1e-12. The discrete transform sums all
  modes of its grid exactly instead.
- **Windows everywhere.** Where the point cube already reaches `h`, the window estimator has more
  variance than the point estimator: met3 1.56×, met1 1.10× with box-car patches.

## 6. Open

- **Remaining gap to the C5 goal (≤ 3× L6).** C5 is now 4.8× L6. Steps per walk are 1.5× those
  of L6, and the time per step 3.4× (box steps near shells, M2d). Those are the larger levers
  now.
- **Window hops in replays and sensitivities.** These would need window decisions that see
  overlays.
- **A persistent kernel cache.** Kernels depend only on canonical configurations, so they could
  be reused across runs of the same process stack.
