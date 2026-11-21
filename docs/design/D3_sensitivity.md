# faradice D3 Design: Capacitance Sensitivities and Layout What-If Analysis

- Date: 2026-11-17
- Status: Partially implemented (stages in §11, open questions there). Implemented: A0 (library, parameter files, `faradice sens`; validated on S1–S3, S9, S12 and S14, with conformal shells on S15 and S16); C (`faradice whatif`; S5, S10, S13); D deletion (S6) and D addition by replay (library and `[[add]]` in `faradice sens`, checked against a re-solve); B maps (`faradice map`; S11, grounded objects); A on edge-free faces (S1, S2) and remedy 1 of §6.4 at edges (S8, partial). Not implemented: remedies 2–4 for edges, the GUI overlay of B, Python bindings
- Target: the remaining stages of §11 (edge remedies for A, Python bindings) are unscheduled
- Related: [M1 FRW solver](M1_frw_solver.md), [D1 GUI](D1_gui.md), [D2 agent API](D2_agent_api_mcp.md)

---

## 0. Summary

Analog layout work keeps asking the same questions: *which edge should I move to cut this
coupling?*, *how much does C change per nm of width?*, *is it worth adding a shield here?*,
*what if I swap these two tracks, ground this one, leave that one floating?* Today the answer
is "edit and re-extract". D3 adds an optional capability that answers them from one or a few
extractions:

| Pillar | Question | Method | Cost |
|---|---|---|---|
| **A0** | dC/dp for a geometry parameter | Finite difference with *path continuation* (shrink-only perturbation, nominal walks reused, §4) | ≈ 1 nominal solve + continued walks |
| **C** | Swap / ground / float / merge tracks | Exact algebra on the conductor-level Maxwell matrix (§5) | µs per candidate |
| **A** | dC/dp for many parameters, whole matrix at once | Hadamard shape derivative estimated with a *face normal-derivative kernel* (§6). Edge variance is an open problem (§6.4) | Scales with faces, not nets |
| **B** | Where would a small grounded/floating object change C_ij most? | Topological derivative, point potentials (§7) | Walks per map point |
| **D** | Exact effect of adding or deleting a wire | Deletion = path continuation; addition = replay of affected walks (§8) | Affected walks only |

Everything is behind cargo features and separate entry points. **The plain `solve` path does
not change**, and neither its results nor its speed depend on whether the features are built.

Naive automatic differentiation of the walk does not work: a walk ends when it gets within
`ε_stop` of a conductor, and moving a face changes that hit/miss decision discontinuously.
Per-sample autodiff (e.g. Enzyme) misses this boundary term and returns zero or biased
gradients. This is the visibility-discontinuity problem of differentiable rendering. D3
therefore uses estimators that are built for the discontinuity: shared paths (A0, D), a
boundary-integral formula (A) and an asymptotic expansion (B).

---

## 1. Goals and Non-Goals

Goals:

1. Sensitivities dC_ij/dp, each with a standard error, for continuous Manhattan geometry
   parameters: one face of one box (width, spacing, length, thickness) and box translation
   (position). Any parameter that is a linear map of face displacements (§3) works through the
   chain rule.
2. Process-variation estimates σ_C² ≈ Σ_p (dC/dp)² σ_p² (first order, independent parameters).
   Noise inflates it, since E[D̂²] = D² + se², so the report subtracts se² per term (floored at
   0) and states the size of the correction.
3. Exact and instant evaluation of discrete *electrical* edits (C) and fast evaluation of
   discrete *geometric* edits (D).
4. Maps that a designer or an agent can read: per-face "push here" density (A) and an
   insertion map (B), as GUI overlays (D1) and agent tools (D2).

Non-goals (this design):

- Multi-dielectric media. The formulas in §6–§7 assume one homogeneous dielectric (M1).
  A0, C and D do not depend on this and keep working when M2 lands. §6.6 lists what M2 adds.
- Non-Manhattan geometry.
- A built-in optimizer. D3 produces gradients and fast what-if evaluations. The search loop
  (gradient descent, annealing, enumeration, an agent) sits outside, in Python or D2.
- Differentiating with respect to solver settings (ε_stop, Green resolution) or ε_r. The
  last one is trivial for one dielectric: C ∝ ε_r.

---

## 2. Prior Work (public literature only)

Clean-room rule: only public papers were read; no source code of any extractor or solver.

| Work | What it does | Relation to D3 |
|---|---|---|
| El-Moselhy, Elfadel, Daniel, *A capacitance solver for incremental variation-aware extraction*, ICCAD 2008 (doi:10.1109/ICCAD.2008.4681647) | FRW finite-difference sensitivity with **random path sharing**: perturb conductors only by *shrinking* them, so every nominal transition cube stays conductor-free. A nominal walk that ends on the shrunk part is *continued* in the perturbed geometry. Reports the sensitivity matrix at under 2× the nominal cost and ~10× faster than BEM adjoint or finite differences. Also handles large perturbations incrementally, using the sparse dependence of each path on a few conductors | A0 follows this scheme (§4). D's deletion case is the same idea (§8) |
| El-Moselhy, Elfadel, Daniel, *A hierarchical floating random walk algorithm for fabric-aware 3D capacitance extraction*, ICCAD 2009 | Many topologically different configurations built from the same motifs, using Markov transition matrices between motifs. Reports 120 configurations at the cost of 2 | Relevant to pitch and finger-count enumeration (§9). Not planned now |
| El-Moselhy, PhD thesis, MIT 2010, *Field solver technologies for variation-aware interconnect parasitic extraction* | Extended version of the above | Background |
| Maffezzoni, Levantino et al., TCAD 2018 | FRW with generalized polynomial chaos for capacitor variability | A different route to σ_C (stochastic collocation). D3's first-order σ_C is cheaper and coarser |
| Miller, Sawhney, Crane, Gkioulekas, *Differential walk on spheres*, ACM TOG 43(6), 2024 (arXiv:2405.12964) | Monte Carlo derivatives of PDE solutions with respect to boundary shape and data, for screened Poisson. The shape derivative becomes a boundary-value problem with data from the normal derivatives of the primal solution | Confirms the boundary-integral route of §6 and the reasons naive autodiff fails. D3 needs only a *functional* (C_ij) of the solution, not the derivative field, so it does not have to solve the derivative PDE: the self-adjoint formula of §6.1 is enough |
| Yilmazer et al., *Solving inverse PDE problems using grid-free Monte Carlo estimators*, 2022 (arXiv:2208.02114) | Gradient estimators for WoS in inverse problems | Background |
| Hadamard (1908); Pólya & Szegő, *Isoperimetric Inequalities in Mathematical Physics*, 1951; Henrot & Pierre, *Shape Variation and Optimization*, EMS 2018 | Hadamard variational formula for the Dirichlet energy and capacity | §6.1 formula |
| Ammari & Kang, *Polarization and Moment Tensors*, Springer 2007; Novotny & Sokołowski, *Topological Derivatives in Shape Optimization*, Springer 2013 | Small-inclusion asymptotics and polarization tensors | §7 formulas |

**Open-source landscape** (see [02_oss_landscape_novelty.md](../research/02_oss_landscape_novelty.md)):
no open-source capacitance extractor we know of reports sensitivities. FasterCap, Palace,
Magic, OpenRCX and KLayout-PEX can only be differenced by re-solving. The "path continuation"
finite difference (A0) is published prior art. The face-kernel shape-derivative estimator (A)
and the combined what-if toolbox (A0 + C + D) exposed through CLI, GUI and agent API are, as
far as this survey found, new.

---

## 3. Parameters and Face Velocities

A **face handle** is `(net, box, face)`: one face (`x-`, `x+`, …, `z+`) of one box of a net's
`boxes` list, in the order of the structure file (GDS inputs: the box order of the Manhattan
decomposition, which is deterministic). A parameter p is a list of face handles with weights:

```text
dp moves face f by w_f · dp along its outward normal
```

| Parameter | Faces and weights |
|---|---|
| Width of a y-running wire, symmetric | `y-`: 0.5, `y+`: 0.5 |
| Width, one-sided | `y+`: 1 |
| Spacing between wires A (below) and B (above), moving B | B `y-`: −1, B `y+`: +1 (translation of B) |
| Length | `x+`: 1 |
| Thickness | `z+`: 1 |
| Position along y | `y-`: −1, `y+`: +1 |

Sensitivities are computed **per face** (`dC_ij/dn_f`, the derivative with respect to a pure
outward displacement of face f) and combined by the chain rule:
`dC_ij/dp = Σ_f w_f · dC_ij/dn_f`. This holds because the first-order variation is linear in
the normal velocity (§6.1). Per-face results are the natural output for maps and for the
Python VJP (§10).

Only the **exposed** part of a face matters: the part not covered by another box of the same
net, not touching another net, and not lying on a reflecting domain face (mirror symmetry: no
dielectric there). A face edge that lies on a reflecting domain face is not a geometric edge
either. The mirror image continues the conductor, so §6.2 and §6.4 ignore it for `r_edge`. Touching nets are rejected by the solver anyway
(`SolveError::NoClearance`).

**Exposure rule (implemented, `sens::covered_parts`).** Another box C of the same net *covers*
the part of face f it overlaps (positive area) when C contains the face plane and extends
beyond it, outward. A0 shrinks only the exposed part: the strip of depth δ under each covered
part is filled back in with an extra box of the net, so no slit opens inside the conductor and
the perturbed conductor is still a subset of the nominal one. A face with no exposed area is an
error. A box of the same net that is **flush** with f (ends in its plane on the inner side) and
overlaps it is also an error: the union's surface there belongs to both faces, and moving
either one alone does nothing while moving both moves it, so per-face derivatives would not add
up linearly. Vias from technology files span exactly between their metals and are never flush;
overlapping coplanar shapes must be merged first.

Parameter file (`*.sens.toml`):

```toml
schema_version = 1

[[param]]
name = "w_victim"            # unique
faces = [
  { net = "victim", box = 0, face = "y-", weight = 0.5 },
  { net = "victim", box = 0, face = "y+", weight = 0.5 },
]
sigma = 0.005                # µm, optional: 1σ process variation, for σ_C (§1 goal 2)

[[param]]
name = "x_right"
translate = { net = "right", axis = "y" }   # shorthand: all exposed faces of the net, ±1
```

---

## 4. Pillar A0 — Finite Differences with Path Continuation

### 4.1 Idea

Per El-Moselhy et al. (§2): choose the finite-difference step so that every perturbed
conductor is a **subset** of its nominal shape (a face moves *inward* by δ). Then:

1. Every transition cube of every nominal walk is still conductor-free, so the nominal path is
   a valid FRW path in the perturbed geometry. FRW only needs each cube to be free of
   conductors, not maximal. This includes the `far_step` lower-bound steps, folds at reflecting
   walls and exterior-sphere jumps.
2. The Gaussian surface of each target net still encloses its net and still does not touch any
   other conductor, so the launch phase is unchanged.
3. Only the **termination** can differ. At the end point x_end, the solver re-evaluates the
   clearance in the perturbed geometry, using a per-face perturbed `Geometry` and
   `UniformGrid`, because the nominal index under-estimates distances to the shrunk box. If the
   clearance is still below ε_stop, the walk ends there (the sink may change if the nearest
   conductor changed). Otherwise the walk **continues** in the perturbed geometry until it
   terminates. Only walks whose nominal sink is the moved box's net and whose end lies within
   ε_stop of the moved box are examined. Every other termination is provably unchanged.

**Geometry-derived settings are pinned to the nominal run**: ε_stop, the Gaussian surfaces and
their offsets, and the boundary model (including the open-boundary escape sphere, which still
encloses the shrunk conductors). Recomputing them for the perturbed geometry would change the
discretization between the two sides of the difference. Re-solve references used in
validation set `walk.stop_dist` explicitly for the same reason.

So each sample yields a nominal contribution and a perturbed contribution from the *same*
path. The finite difference is the mean of exact paired differences:

```text
D_f(δ) = (C_ij(p) − C_ij(p − δ)) / δ  ≈ dC_ij/dn_f      (n_f: outward displacement of face f)
```

Its standard error comes from the per-sample paired differences (§4.3), not from two
independent σ's. The per-sample difference is formed **after** the M1 sample bookkeeping
(`record`), as record(perturbed pair) − record(nominal pair). With the odd image (open boundary
+ substrate), a continued walk can keep its sink but flip its sign, and the ground-plane entry
is reconstructed from the pair weights. The difference is therefore not only "sink changed".
Continuations carry the walk's accumulated sign.

### 4.2 Several faces and several step sizes

- **Granularity: one face per perturbed geometry.** Each requested face handle gets its own
  shrunk geometry and spatial index (memory: K geometries and indexes, 2K with Richardson; each
  is rebuilt and re-validated, O(boxes) per level, which is fine for analog cells and worth
  replacing by an index patch for large layouts). At the end of each nominal walk,
  continue it only for faces whose perturbed geometry changes its termination. In practice that
  is at most one or two faces, because the walk ended near one face.
- **Parameters by the chain rule.** `dC/dp = Σ_f w_f · D_f`, with D_f ≈ dC/dn_f (outward) for
  every face, whatever the sign of w_f. A translation (one face out, the opposite face in)
  therefore needs no outward perturbation. For an exact σ of the combination, the face–face
  covariance is needed: the parameter driver accumulates the moments of the per-sample
  combination `Σ_f w_f · Y_f` directly (per Gaussian sample, after `record`), not per face.
  The library does this: `solve_sensitivity(…, faces, params, …)` takes `Param`s (lists of
  `(face index, weight)`) and accumulates each parameter's per-sample combination inside the
  hook. Faces of the same sign that never interact, such as the two sides of a symmetric width,
  may later share one shrunk geometry as an optimization.
- **Richardson extrapolation.** Continuation nests: the geometry shrunk by 2δ is a subset of the
  one shrunk by δ. A single walk can therefore be continued to δ, and that continuation
  continued to 2δ. `2·D(δ) − D(2δ)` cancels the O(δ) bias, at about twice the continuation cost
  and roughly 2.5–4× the variance of D(δ) (measured: σ ×1.6, §4.3). It is the library default,
  and the one-sided estimate is always reported alongside.

### 4.3 Estimator, variance and step size

Per sample and sink, `Y = −Δw/δ` (one-sided) or `Y = −(2Δw(δ) − Δw(2δ)/2)/δ` (Richardson), with
Δw = perturbed − nominal contribution. The moments of Y are accumulated per (target net, sink,
face), with implicit zeros for unchanged samples, exactly like the M1 sink moments.

**Cost.** Every walk that ends within ε_stop of a shrunk face is *re-examined*, and because
δ ≫ ε_stop nearly all of them are *continued*. Only the sink *change* is O(δ). The continuation
cost is therefore about (fraction of walks absorbed by the face) × (one walk length). It does
not scale with δ.

**Variance ∝ 1/δ.** A continued walk changes sink with probability O(δ/gap) and then Δw = O(w),
so Var(Y) = O(1/δ). Bias is O(δ/gap), or O((δ/gap)²) with Richardson. The relevant length is
the **gap** in front of the face, not the box size: shrinking changes gap g into g + δ. The
default step is therefore

```text
δ = min(0.05 · extent of the box along the normal, 0.05 · gap in front of the face)
```

with the gap taken along the face's normal (`faradice_sens::params::front_gap`): the nearest box
(any net, not touching the face) whose projection overlaps the face, a grounded domain face, the
substrate plane (bottom faces), or twice the distance to a reflecting domain face (the mirror
image). With nothing in front (open space), only the extent counts. The factor is `delta_rel`
in the parameter file, and `delta` overrides it per face. The library enforces δ ≥ 20 · ε_stop
(`MIN_STEP_OVER_EPS_STOP`) and returns an error otherwise. Below that, the stopping rule masks
the perturbation and the derivative would silently come out too small.

**Measured (V1 plate, bottom face, 4·10⁵ walks, exact dC/dn = εA/d₁²):**

| δ / t | one-sided err | one-sided σ | Richardson err | Richardson σ |
|---|---|---|---|---|
| 0.01 | −12.9 % | 17.4 % | −21.6 % | 27.3 % |
| 0.02 | −6.2 % | 12.0 % | −4.5 % | 18.9 % |
| 0.05 | −0.2 % | 7.6 % | +1.3 % | 12.0 % |
| 0.10 | +4.4 % | 5.4 % | +5.8 % | 8.5 % |
| 0.20 | −0.8 % | 3.7 % | +2.6 % | 5.9 % |

(t = 0.2 µm plate thickness, d₁ = 1 µm, so all δ ≪ gap.) σ ∝ δ^{−1/2} as predicted, and all
errors are within 2σ. About 78 % of walks end on this face and are continued, at ~0.9× the
nominal step count. Percent-level sensitivities need ~10⁷ walks per target on this case. A0 is
therefore a **reference and baseline**, not the production estimator for many parameters. That
is the motivation for A (§6).

Faces too thin for the step (box extent along the normal ≤ 2δ, or 4δ with Richardson) and
faces lying on a reflecting domain face are errors, not clamped. A continuation that exceeds
`max_steps` leaves that walk of the sample **unchanged at every level** (so Δw = 0 for it). It is
counted in `aborted_continuations`, and `FaceResult::converged()` becomes false.

**Validation runs (library, `tests/sensitivity.rs`, δ = 0.2 t on the plate, 0.1 a on the cube,
Richardson):**

| Case | Walks | Error | σ |
|---|---|---|---|
| S1 plate bottom face, dC_PP/dn = εA/d₁² | 1.6·10⁶ | −0.9 % | 3.0 % |
| S2 plate top face, dC_PP/dn = εA/d₂² | 1.6·10⁶ | −0.3 % | 1.2 % |
| S3 cube, each face, C/(3a) | 8·10⁵ | 0.1–2.8 % | 3.8 % |
| S3 cube, uniform growth as one parameter, 2C/a | 8·10⁵ | 0.04 % | 1.6 % |

**S12 (L-shaped net, partially covered top face, one-sided δ = 0.04 µm = 13 % of the gap,
pinned `stop_dist`):** A0 against independent re-solves of the explicitly shrunk geometry at the
same δ, 4·10⁶ walks each: dC_LL/dn 0.545 ± 0.003 vs 0.550 ± 0.015, dC_LN/dn −0.519 ± 0.003 vs
−0.535 ± 0.012 fF/µm (both within 1.3σ).

**S9 (two wires 0.5 µm above the substrate, open boundary with odd image, one-sided δ = 0.045 µm,
4·10⁶ walks each, fF/µm):**

| Face | Entry | A0 | Re-solves at the same δ |
|---|---|---|---|
| toward B (`y+`, 15 % of the gap) | dC_AA/dn | 0.2631 ± 0.0011 | 0.2656 ± 0.0042 |
| | dC_AB/dn | −0.1806 ± 0.0009 | −0.1798 ± 0.0033 |
| toward the substrate (`z-`, 9 % of the gap) | dC_AA/dn | 0.2167 ± 0.0011 | 0.2172 ± 0.0043 |
| | dC_AB/dn | −0.0572 ± 0.0005 | −0.0569 ± 0.0033 |

Continuations that cross the substrate image flip sign exactly like nominal walks.

### 4.4 Determinism and nominal identity

The CPU batch path (`run_batch`) draws launches and transitions from one batch stream and has
no per-walk key. Continuations therefore draw from their own streams, keyed by splitmix chaining
over (seed, net, batch, sample index, antithetic slot, box id, face, Richardson level). The key
contains the face *handle*, never its position in the request. The sensitivity driver reuses
`run_net`/`run_batch` through a zero-cost per-sample hook (the `()` hook compiles away), so it
has the same waves, merge order and stopping checks. Therefore:

- the nominal statistics are **bit-identical** to CPU `solve` with the same seed and
  configuration, including adaptive stopping, which only looks at nominal statistics. They are
  not identical to `solve_with` (engine/GPU path, per-launch keys), which agrees only
  statistically;
- face results do not depend on thread count, on face order, or on which other faces are
  requested. Tests cover all three.

A later `accuracy.sens_rel_tol` (stop when the derivatives converge) would change the walk
count, so it would break identity with a plain solve by design.

### 4.4a Stratified dielectrics

A0 and deletion work in stratified stacks (M2a) without changes to the estimator. Continuations
run the same layered walker from the walk ends. The driver uses the medium-aware Gaussian
surface and the layered launch weights exactly as `solve` does, so the nominal statistics stay
bit-identical. Validated on the V1 plate inside a three-layer stack: the closed forms
dC/dn = ε₀A/(ε_adj (Σ t_k/ε_k)²) of both faces are met within 0.3σ and 0.9σ (σ 2.7 % and 1.4 %
at 1.6·10⁷ walks). A deletion of a cube that straddles an interface matches a layered re-solve
(+1.1σ). Layered first hops are noisier, so a given σ needs about 10× the walks of the
homogeneous case. Pillar A and potential gradients (B) remain homogeneous-only.

### 4.4b Cost with many faces

A perturbed level is an *overlay* (`walk::Overlay`), not a copy of the geometry: the nominal
index stays valid as a lower bound, because the edit only removes conductor from the moved box.
When the nominal nearest box is the moved one, the walker checks the replacement boxes, the
boxes within a reach R of the original box, and the boundary. Any other box is at least
R − d_b away. The step max(d_b, min(…, R − d_b)) is valid, and it is exact whenever it is
below ε_stop (R ≥ 4 ε_stop). Per sample, only faces whose box lies within ε_stop of a walk end
can change, and a spatial hash of the faces finds them. Memory is O(faces), and the per-sample
cost is independent of the number of faces. Translating a 2000-box comb (3000 faces) against a
victim: 0.19 s and 29 MB (it was 4.4 s and 2 GB with a geometry copy per level).

### 4.4c Conformal shells (M2c)

A conformal shell (M2c §6.2) follows its conductor. Moving a face f of a tagged box B by dn
therefore moves two surfaces: the conductor face, and the part Σ_f of the shell's outer surface
that is swept by the face of the grown box `grown(B)`. Continuation (§4.1) cannot carry the
second move. Every walk that ends on B crosses the shell, usually through a one-interface cube
that straddles Σ_f or a cube capped at it, and each such cube changes its kernel. The fraction of
paths that changes is therefore O(1), not O(δ). This is why M2c §7 refuses these faces today.

**Decomposition.** Let M₀ be the nominal medium and M₂ the medium of the edited geometry B'
(shells regenerated from B', M2c §6.2). Define the intermediate medium

```text
M₁ = M₂ inside the nominal box B,   M₀ outside B
```

i.e. the conductor shrinks and the vacated slab B ∖ B' takes the material it has in the edited
geometry, while everything outside B is left as it was. Then

```text
dC_ij/dn_f = T_ij^cond + T_ij^Σ
T^cond: M₀ → M₁ (conductor face only)    T^Σ: M₁ → M₂ (medium outside B only)
```

- **T^cond is exact A0.** M₀ and M₁ agree outside B, and nominal cubes lie outside B. Every
  nominal path is still valid, and continuations run in M₁. For a side face with
  `δ ≤ side`, the slab becomes shell material. For a `z+` face with `top = 0` it becomes the
  stack material above, because the edited shell no longer reaches the old top. Nothing in §4.1
  to §4.4b changes except the medium given to the continuation walker: one `BoxSteps` per face
  and level, sharing the M2a tables of the nominal run.
- **T^Σ is a dielectric-interface shape derivative.** As dn → 0, M₁ → M₂ moves the interface
  Σ_f between the shell (ε_s) and the outside material (ε_o) by dn along the face normal. For
  an interface moving into ε_o, the Hadamard formula for transmission problems gives

  ```text
  T_ij^Σ = (ε_s − ε_o) ∫_{Σ_f} [ E_t^i · E_t^j + D_n^i D_n^j / (ε_s ε_o) ] dS
  ```

  E_t and D_n are continuous across Σ, so the integrand is well defined. Check: plates at gap g
  with a shell of thickness t on plate 1 have C = A / (t/ε_s + (g − t)/ε_o). Growing the shell
  gives dC/dt = (ε_s − ε_o) A σ² / (ε_s ε_o) with σ = C/A, which matches.

  Σ_f is the exposed part of the grown face. It excludes parts inside the grown region of
  another tagged box, parts overridden by explicit boxes or later rules (M2c §6.2), parts masked
  by conductors, and parts on reflecting faces. It is built with the same rectangle machinery as
  the interface faces (M2c §3.2).

**Size of T^Σ.** In the one-dimensional gap model above, a width change moves both surfaces,
and `T^Σ / (dC/dn) = 1 − ε_o/ε_s`. For SKY130 (NILD3_C 3.5 inside NILD3 4.5) that is −29 %, and
T^cond overshoots by +29 %. The interface term can therefore not be dropped.

**Estimating T^Σ.** Products need independent walks (§6.3). For each sample point on Σ_f, run
two independent gradient estimates of every φ_k and form the U-statistic, as in pillar B. Two
pieces are new:

1. **Gradient kernels at a point on an interface.** Today `BoxSteps` has none: a cube centred
   on the interface has no gradient kernel (`axis.rs`), and `faradice map` refuses gradients
   with boxes or a stack. The integrand needs E_t (the tangential kernel of an axis cube with
   its interface through the centre) and D_n (the flux through the centre plane). Both follow
   from the M2a one-interface Green function at offset 0. If that turns out to be
   ill-conditioned, the fallback is to evaluate the fields at ±η off Σ and extrapolate η → 0.
   This kernel also lifts the gradient refusal of `faradice map`.
2. **Variance near the edges of Σ_f.** On side faces Σ_f lies `side` (30 nm on SKY130) from the
   conductor, so the conductor does not shrink the field cube. The edges of Σ_f do. On SKY130
   they meet a stack plane (the wire bottom), the shell's top, and the shell's ends. At distance
   s from such an edge the cube's half-side is at most s. A single walk scores ±1/s with
   probability O(1), not a small difference: φ is O(1) there, as in §6.4. The product of two
   independent scores therefore has a second moment ∝ s⁻⁴, and its integral diverges. (An
   earlier draft argued from the field exponent λ and wrongly expected a finite variance.)
   Measured on C4: single points contributed up to 10⁶ times the mean.

   **Remedy: edge margin.** Points are still drawn uniformly on Σ_f, but the fields are taken
   at the nearest point of Σ_f shrunk by a margin m·g along its edges (g the shell thickness,
   `SensOptions::shell_edge_margin` = m, default 1). Edges on reflecting walls are not real
   edges and keep no margin. This is a zeroth-order extrapolation over the edge strips. Near a
   three-media edge with λ ≈ 1 the integrand is nearly constant, so the bias is
   O((m·g)²)/|Σ_f|. On C4, m ∈ {0.25 … 3} moves T^Σ by at most 1.5 % of itself (1–2σ), while m =
   1 cuts σ 4× against m = 0.25 ([validation](../validation/D3.md)). For `z±` faces with
   `top = 0`, Σ_f is the ring of the side shell's top or bottom, and it touches the conductor
   edge. There the field itself is singular, the margin argument does not hold, and §6.4 applies.

**Fallback: same-δ re-solves.** `method = "resolve"` solves the edited geometry and differences
it against the nominal run with a matched large δ and a pinned `stop_dist`, as the references
of §9 do. Its σ ≈ √2·σ_C/δ, so it is a reference and a last resort, not a production method.

**Stages.**

1. T^cond (continuation in M₁). Report it only inside the total, never as dC/dn on its own.
   **Done**: `BoxMedium::edited_inside` builds M₁ per face and level, and
   `BoxSteps::with_medium` shares the nominal tables. `SensOptions::shell_conductor_term` returns
   T^cond for validation. Richardson continues level-1 walks at level 2, so both levels must
   agree on the slab vacated at level 1: lateral faces need 2δ ≤ `side`, and `z+` faces need
   2δ ≤ `top` (or `top = 0`). S15 conductor term passes ([validation](../validation/D3.md)).
2. T^Σ on side faces (interface gradient kernels, the Σ_f construction, U-statistics). Side
   faces then become supported, and width and spacing parameters work on SKY130.
   **Done**: `LayeredGradients::on_interface` (tangential field and lower-trace flux at a
   centre on the interface), `BoxSteps::interface_hops`, the shell module (Σ_f clipped to the
   domain, material probes, edge margin, U-statistics, odd image by partition of unity), and
   `FaceResult::shell` / `ParamResult::shell`, included in `dc`. `faradice sens` accepts lateral
   faces with `side > 0` and `z+` faces with `top > 0`. S15 and S16 pass.
3. `z±` faces of tagged boxes with `top = 0` (the ring at the conductor edge), and lateral
   faces with `side = 0`. These need an edge remedy (§6.4) or the re-solve fallback, and are
   refused until then. Boxes tagged by several rules are refused too.

Validation (S15, S16 in §9): a closed form with vertical plates and a shell on the moved plate,
then SKY130 met1 pairs against same-δ re-solves.

### 4.5 What A0 cannot do

- It needs Gaussian-surface walks *per target net*, like extraction: the cost grows with the
  number of nets whose rows are wanted, and with K only through continuations.
- With variance ∝ 1/δ, very small δ is noisy. Pillar A has no δ.

---

## 5. Pillar C — Exact Algebra on the Maxwell Matrix

### 5.1 Operations

Extract with every *track* as its own net, and with every candidate net in `extract.nets` (each
row needs that net's walks). The result is the conductor-level Maxwell matrix C (n × n, ground
eliminated; diagonal = total, off-diagonal = −coupling). For voltages V and charges Q, Q = C V.

| Edit | Exact result | Why |
|---|---|---|
| Swap the net assignment of tracks (permutation P) | C' = P C Pᵀ | Relabelling the conductors |
| Ground track k | Delete row and column k. Its couplings to the others become their ground capacitance | V_k = 0 is already the extraction condition |
| Float track k (dummy fill) | Schur complement: C' = C_aa − C_ak C_kk⁻¹ C_ka | Q_k = 0 ⇒ V_k = −C_kk⁻¹ C_ka V_a |
| Merge tracks k, l into one net | C' = Mᵀ C M with M merging the columns | V_k = V_l, Q = Q_k + Q_l |
| Several at once | Compose (float a set: block Schur complement) | Associative |

These are exact for the extracted geometry, not first-order. They cannot remove or add a
conductor; that needs D (§8).

### 5.2 Statistics

Each row i of C comes from net i's walks. Rows are independent, but entries *within* a row are
correlated (the same samples hit several sinks). The variance of a non-linear function (the
Schur complement) uses the delta method with the full per-row covariance. M1 stores only
per-sink moments, so D3 adds **per-row sink cross-moments**: each sample hits at most two sinks
(antithetic pair), so at most a few products per sample. These are stored sparsely in
`result.json` (schema bump, optional field). The matrix is symmetrized as
`(C_ij + C_ji)/2`, with variance `(σ_ij² + σ_ji²)/4`. Rows are independent, so there is no
cross term. The symmetrization reported in `report.json` (`coupling.symmetrize`) is not used:
D3 reads the raw `result.json`.

Requirements on the source run, all checked with a clear error: no coupling thresholds or
`moved_to_ground` post-processing (raw `result.json` only); every net involved in an edit is a
target; seed and walk counts recorded.

### 5.3 Implementation and validation

`faradice-frw::cross::solve_with_cross_moments` accumulates Σ c_k c_l per target through the
per-sample hook (`solve` is unchanged; the `Solution` is bit-identical). With
`[output] cross_moments = true` the engine stores them as an optional `cross` list per target in
`result.json` (schema version unchanged: the field is optional and omitted otherwise; CPU engine
only). `faradice_sens::whatif::apply` builds the symmetrized matrix of the targets, applies the
edits in the fixed order swap → merge → ground → float (ground and float commute), and computes
σ by the delta method with an exact Jacobian over the n(n+1)/2 symmetric entries, propagated in
forward mode through the edits (all linear except float, whose tangent is
dC − (dc cᵀ + c dcᵀ)/c_kk + c cᵀ dc_kk/c_kk²; no step size, so disparate scales cannot break it).
The cost is O(n⁵) in total: fine for tens of conductors; buses need a reverse-mode or blocked
formulation.
Without stored cross moments, σ ignores within-row covariances and a warning says so.

The reported ground capacitance of a conductor is the row sum Σ_j C'_ij of the edited matrix
(ground plus non-target nets). It is exact but noisier than the direct per-sample ground sum of
a plain report (≈1–3 % vs ≈0.3 % on the three-wire example), because it is a difference of large
correlated entries.

| Case | Walks | Result |
|---|---|---|
| S10 floating middle plate, C'_12 = c₁c₂/(c₁+c₂), C'_11 = c₀ + series | 4·10⁵ | +0.7σ, +0.3σ, −1.4σ (σ 0.2–0.3 %) |
| S5 ground the crossing wire vs re-solve with it as a non-target (three wires) | 3·10⁵ each | all totals, grounds and couplings within 1.7σ |
| S13 (C) 1.645σ coverage over 50 seed pairs, 9 quantities each, ground (vs re-solve) and float (vs the mean of the other runs) | 4·10⁴ | 0.880 and 0.896 (nominal 0.90) |

### 5.4 Search

With µs per candidate, exhaustive enumeration covers small problems (8 tracks: 8! = 40320
permutations in well under a second). Larger ones use annealing or a user objective in Python.
D3 ships `faradice whatif` for single edits and a Python API for search; the objective
(e.g. minimize |C_victim,aggressor| subject to C_victim,total < X) is the user's.

---

## 6. Pillar A — Shape Derivative with a Face Normal-Derivative Kernel

### 6.1 Formula

Let φ_k be the potential with net k at 1 V and all other conductors and ground at 0 V, in one
homogeneous dielectric ε. The Maxwell matrix is the energy bilinear form
`C_ij = ε ∫_Ω ∇φ_i · ∇φ_j dV`. Move the conductor boundary with normal velocity V_n, positive
when the conductor **grows** into the dielectric. The Hadamard formula for Dirichlet problems
with piecewise-constant boundary data gives

```text
dC_ij = ε ∮_{conductor surfaces} (∂_n φ_i)(∂_n φ_j) V_n dS,       n = outward normal of the conductor
      = (1/ε) ∮ σ_i σ_j V_n dS,                                   σ_k = surface charge density for φ_k
```

On a conductor surface φ_i is constant, so ∇φ_i is normal and only the normal derivative
appears.

Sign checks (all must hold, and all do):

| Case | Expected | Formula |
|---|---|---|
| Parallel plates, plate 1 grows into the gap (gap d shrinks) | dC_11/dd = −εA/d² | ∂_nφ_1 = −1/d on plate 1 ⇒ dC_11 = εA/d² per unit growth ✓ |
| Same, mutual term | C_12 = −εA/d becomes more negative | ∂_nφ_1·∂_nφ_2 = (−1/d)(+1/d) ⇒ dC_12 = −εA/d² ✓ |
| A grounded conductor k between i and j grows | coupling \|C_ij\| drops (shielding) | φ_i, φ_j both rise away from k ⇒ product > 0 ⇒ C_ij (negative) rises toward 0 ✓ |

The formula is self-adjoint: no separate adjoint solve is needed. **One sampling pass over the
faces of interest gives dC_ij for every pair (i, j) at once**, because a walk's sink tells which
φ_k it samples. Unlike A0 and extraction, the cost scales with the number and area of the faces
studied, not with the number of nets.

### 6.2 Estimating ∂_nφ at a conductor face point

The M1 gradient estimator (first-hop kernel at a cube *centre*, §M1) applied at a point offset
from the face is biased by the offset, and its variance ∝ 1/offset² is useless as the offset
goes to 0. Instead D3 uses a cube that **rests on the face**:

- x₀ on the exposed face of conductor k, outward normal n. Cube Q of half-side h with its bottom
  face B centred at x₀ and lying in the conductor face (B ⊂ exposed face); the rest of Q is
  conductor-free.
- φ_i is harmonic in Q, and φ_i = V_k on B, where V_k = 1 if k = i and 0 otherwise. Since the
  constant V_k is harmonic with zero normal derivative,

  ```text
  ∂_n φ(x₀) = (1/h) ∫_{∂Q∖B} K_B(ŷ) · (φ(y) − V_k) dŷ
  ```

  with K_B the normal derivative, at the centre of the bottom face, of the Poisson kernel for
  data on the other five faces. The cube convention follows `green.rs`: the sine series live on
  [0,1]³ and the walk uses half-side h, so the scale factor (1/h vs 1/(2h)) is fixed by the
  unit tests below, not by hand. **K_B is bounded**: every point of ∂Q∖B is at least h away from
  x₀. Both pieces have closed-form double sine series like the M1 tables. For data on the top
  face, the derivative at (½, ½, 0) is `Σ A_mn sin(mπ/2) sin(nπ/2) γ_mn / sinh γ_mn`. Data on a
  side face (x = 1) needs its own series, `Σ A_mn sin(mπ/2) · nπ · sinh(γ_mn/2) / sinh γ_mn`
  (only the four side faces are related by symmetry). Per-cell integrals are analytic, so a new
  `FaceTables` has no quadrature error, exactly like `GreenTables`.
- Sample a cell ∝ |K_B| and a point y in it, walk from y, and score
  `g_i = ‖K_B‖₁/h · sign(K_B) · (s_walk · 1[sink = i] − δ_ik)` for every conductor i at once.
  s_walk is the odd-image sign of the walk. It multiplies the indicator only, not the subtracted
  constant, matching the ground-plane reconstruction in `record`. The pair (k, ·) is
  accumulated for every sample, even when no walk reached k.

**The subtraction of V_k is a built-in control variate.** Near the conductor φ − V_k is small,
so walks that come straight back to k score zero. This is what keeps the variance finite at the
edges (§6.4).

h is the largest half-side with B inside the exposed face and Q free of other conductors:
`h = min(r_edge(x₀), h_free)`. Here r_edge is the L∞ distance from x₀ to the boundary of the
exposed face. h_free is computed directly, not by a fixed-point iteration (which can collapse
to 0). For each nearby box b (from the spatial index) and each Dirichlet wall or substrate,
`h_b = max(tangential L∞ gap, normal gap / 2)`, and h_free = min_b h_b. Reflecting walls are
handled by folding.

### 6.3 Products need independent walks

`dC_ij` needs `E[∂_nφ_i · ∂_nφ_j]` at each x₀, and `E[X]E[Y] ≠ E[XY]` for one walk. Each surface
sample therefore draws **two independent** kernel samples and walks, and scores the symmetrized
product `½(g_i⁽¹⁾ g_j⁽²⁾ + g_j⁽¹⁾ g_i⁽²⁾)`. This is unbiased for ∂_nφ_i ∂_nφ_j at x₀. The face
integral is the area times the mean over uniformly sampled x₀ (or with importance weights,
§6.4). Every (i, j) pair seen by the two walks is accumulated in one pass.

### 6.4 Edges and corners — open problem

At a convex right-angle edge of a box the exterior angle is 3π/2, so φ ∼ r^{2/3} and
σ ∼ r^{−1/3}. The integrand σ_iσ_j ∼ r^{−2/3} is integrable but **concentrated at the edges**:
fringe fields carry a large share of dC for thin wires.

**The plain estimator has infinite variance there.** Near an edge the kernel cube shrinks
(h ≤ r). The score of one factor takes only the values 0 and ±‖K_B‖₁/h. The control variate
makes non-zero scores *rare*: a walk from ∂Q ends off conductor k with probability
E|φ − V_k| = O(h^{2/3}). It does not make them small. So E[g²] = O(h^{−2}·h^{2/3}) = O(h^{−4/3}),
and the product of two independent factors has E[Y²] = O(r^{−8/3}). With x₀ drawn with density
∝ r^{−α}, the second moment is ∝ ∫ r^{−8/3+α} dr. That is finite only for α > 5/3, but the
density is normalizable only for α < 1. No power-law importance density works, and corners
(vertex exponent ≈ 0.45) are worse. (An earlier draft of this section used the mean O(h^{−1/3})
instead of the second moment and reached the wrong conclusion.)

Candidate remedies, to be evaluated before A becomes a production method:

1. **Edge strip by A0.** Use A in the interior r > r_c and cover the strip r < r_c with the
   continuation finite difference, shrinking only the strip's part of the face. This needs a
   box split, which is fine for Manhattan geometry. A is unbiased. The A0 part has O(δ) bias
   (O(δ²) with Richardson), and near the r^{2/3} singularity it needs δ ≪ r_c, so its 1/δ
   variance makes the strip expensive. S8 compares the remedies by cost per σ².
2. **Singular-coefficient fit.** In the strip, model σ_i ≈ a_i r^{−1/3} (plus higher terms),
   estimate a_i from interior samples at r ∈ [r_c, 3r_c], and integrate analytically.
   Truncation bias is O(r_c^{1/3}) relative to the strip content and must be controlled.
3. **Splitting.** Draw M(r) kernel samples per factor near edges. This trades cost for variance.
   A quick scaling estimate suggests it is only marginally sufficient (borderline log
   divergence), so it is a fallback.
4. **A different first hop at edges.** Use transition domains that wrap the edge (an L-shaped
   region with a precomputed kernel), so the score does not scale like 1/r. This is the most
   work, and the M2 transition-domain machinery may make it cheaper.

Until one of these is validated (S8), A is limited to faces whose edges carry a small share of
the derivative, or to the interior part of a face. A0 remains the reference.

**Remedy 1 measured (S8, [validation](../validation/D3.md)).** `FaceStep::keep` lets A0 move only
the strip. A (interior) + A0 (strip) matches the exact cube face and whole-face A0 when
δ/r ≲ 0.1. At δ/r = 0.35 the step left at the strip's inner boundary biases the strip by −4 %.
The variance gain over whole-face A0 is only 1.2–1.9×, because the strip carries most of the
derivative on thin wires. Remedy 1 is therefore a correct fallback, not the production answer.
Remedies 2 and 4 are next.

### 6.4a Implementation (edge-free faces and interiors)

`faradice_frw::shape` (feature `sensitivity`): `FaceTables` holds the kernel K_B per cell,
with cell integrals that are exact for the truncated series. For data on the top face, the
coefficient is 4γ/sinh γ over odd m, n. For data on a side face it is 2/cosh(γ/2) · lπ over
odd n and all l. On the unit cube, Σ_cells K·f reproduces ∂_z f(½,½,0) for the harmonic
polynomials z, xz, xyz and z³ − 3zx², and for data constant on B, to the midpoint error of the
cells. `solve_shape_derivative` samples x₀ uniformly on the face and computes
h = min(r_edge, h_free). The obstacles for h_free are the other boxes and the mirror images of
all boxes across reflecting walls. The face's own mirrors across tangential walls are excluded,
because they continue the face. Dirichlet planes and the substrate also bound h. Two independent
hops per sample score ε·A·½(g_i¹g_j² + g_j¹g_i²) for every pair of conductor nets seen, plus the
row sums (the ground-capacitance derivative). Batches run on keyed streams and are merged in
order, so results do not depend on the thread count.

Faces must be uncovered and must not lie on a reflecting wall. A face with geometric edges is
refused unless the caller gives an edge margin r, and then only the interior (at least r from
the edges) is integrated. On the V1 plate faces and on two plates (mutual term), 2·10⁶ samples
match the closed forms within 0.6σ at σ = 0.16 %, which takes 0.07 s for 2·10⁵ samples. A0
needs 1.6·10⁶ walks for σ = 1–3 % on the same faces.

**CLI.** `faradice sens --method fd|shape|auto` (or `method` in the parameter file) selects the
face estimator; `fd` is the default. `shape` takes A for every face and refuses the run, listing
the faces and reasons, when a face is not supported (geometric edges, partly covered, on a
reflecting wall, stratified dielectric). `auto` takes A where supported and A0 elsewhere. The
two parts of a parameter are independent (A runs on its own keyed streams), so their values add
and their standard errors add in quadrature. A has no per-sink split of the fixed ground sinks
(substrate, walls, infinity), so a parameter with an A face reports them only in its `<GND>` sum.
Each face in `sens.json` carries `method` (`fd` or `shape`, with `samples`), and the file's
`method` is `fd-continuation`, `shape` or `mixed`. `shape_samples` (default 2·10⁵) sets the
samples per face. Edge margins (§6.4, remedy 1) stay library-only. In practice few faces of
routed wires qualify: every edge of the face must lie on a reflecting wall, which holds for
plates and planes that span the domain but not for a wire, whose faces have edges along its
length.

Validation (`faradice-sens/tests/sens_run.rs`): the top face of a plate between the substrate
and a second plate, both spanning a reflecting box. With `auto`, the face takes A and matches
the closed form εA/d² (2·10⁵ samples: σ 0.5 %, +0.2σ). A0 at 2·10⁵ walks agrees (σ 7 %). A
`shape` run with an edged face is refused with that face named.

### 6.5 Output

Per face: `dC_ij/dn_f` ± σ for all pairs seen. Optionally, a per-face density map
`(ε ∂_nφ_i ∂_nφ_j)(x)` on a coarse grid: a "push here to reduce C_ij" overlay for the GUI
(D1 3D view) and the agent API. Parameters combine faces by §3.

### 6.6 Limits and M2

Single homogeneous dielectric only. With layered or conformal dielectrics, faces touching a
dielectric interface and the interfaces themselves add terms, and the cube-on-face kernel
needs the M2 transition domains. Until then the `sensitivity` estimator for A refuses
multi-dielectric inputs, and A0 is the method to use.

---

## 7. Pillar B — Insertion Map (Topological Derivative)

Leading-order effect of a small object at x₀, far from other conductors relative to its size
(Maxwell-matrix signs: C_ij for i ≠ j is negative):

| Object | ΔC_ij to leading order | Notes |
|---|---|---|
| Small **grounded** conductor of free-space capacity c | **+c · φ_i(x₀) φ_j(x₀)** | For i ≠ j the coupling \|C_ij\| *drops* by c φ_iφ_j (shielding). For i = j, C_ii *rises* by c φ_i² |
| Small **floating** conductor with polarizability tensor M (sphere of radius a: M = 4πε a³ I) | **∇φ_i(x₀) · M ∇φ_j(x₀)** | Uncharged, so dipole order (∝ a³ instead of ∝ a). Parallel plates: ∇φ_1·∇φ_2 < 0, so the coupling grows, as expected for floating metal in a gap |

Derivation of the grounded case: with net i at 1 V, the grounded object picks up charge
q = −c φ_i(x₀). By reciprocity, a point charge q at x₀ with all conductors grounded induces
−q φ_j(x₀) on conductor j, so ΔQ_j = c φ_i(x₀) φ_j(x₀).

In the map, φ_k(x₀) for all k comes from walks started at x₀ (a walk's sink identifies k), and
∇φ_k(x₀) from the M1 centre gradient kernel. Both are FRW-native: no global re-solve, unlike
FEM or BEM. The map φ_i φ_j (or ∇φ_i·∇φ_j) **ranks** candidate locations. For real shields (long
wires, not point-like) the size condition fails, and the predicted magnitude is only
indicative. The candidate is then evaluated exactly with D.

**Implemented.** `faradice_frw::map::potential_map` (feature `sensitivity`) runs, per point, one
walk for φ_k and one antithetic gradient-kernel pair per axis for ∇φ_k (cube half-side = the
point's clearance, as for a Gaussian launch), on a per-point stream, in parallel over points.
With the odd substrate image the walk's sign weights conductor potentials, as in extraction.
Points within ε_stop of a conductor report φ = 1 for it. Gradients in stratified media are not
supported yet. `faradice map S --nets a,b --grid x0:x1:nx,y0:y1:ny,z0:z1:nz [--radius r]` writes
a CSV with both potentials, gradients and the predictions for a sphere of radius r (exact
constants c = 4πεr, M = 4πεr³ I), and prints the best location for each.

Products φ_iφ_j and ∇φ_i·∇φ_j for the predictions are U-statistics over distinct samples, which
are unbiased for any N ≥ 2. The product of the two means from the same samples is not: for
φ_1φ_2 at a midpoint it is biased by the factor (1 − 1/N). Points outside the domain (outside
the box or below the substrate) are flagged and skipped. The CSV has a `status` column (`ok`,
`inside`, `outside`, `aborted:N`), and predictions are empty unless the status is `ok`.

The gradient estimator is discrete-valued (fixed |kernel factor|, binary walk outcomes), so two
independent points occasionally report the same mean; their σ differ.

S11 (grounded cube of edge a at the centre of a 2 µm gap between full-width plates, ΔC from
deleting the cube with D-del, 10⁶ walks): ΔC_12/(c φ₁φ₂) = 1.16, 1.06, 0.97, 0.99 ± 0.04 for
a = 0.4, 0.2, 0.1, 0.05 µm (c = 0.66068·4πεa), log–log slope 0.98 between the two smallest.
Potentials and gradients between and below the plates match the linear field within σ. The
floating case (slope 3) is not checked yet: its ΔC is ∝ a³ and needs Pillar C at a precision
the current runs do not reach.

---

## 8. Pillar D — Incremental FRW for Added and Deleted Wires

- **Deletion is free of replay.** Removing a conductor is a shrink to nothing, so §4.1 applies
  unchanged. Nominal paths stay valid, and only walks that terminated on the deleted conductor
  are continued. If the deleted conductor is a target, its row disappears.
- **Superset extraction.** To compare many candidate insertions (shields, dummies), extract once
  with **all candidates present**, each as its own net. Then:
  - grounding, floating or merging any subset is C (§5);
  - removing any subset is continuation (above);
  - every subset is evaluated exactly, without a single replayed walk.

  The cost is a smaller Gaussian offset for targets next to candidates, plus continuations.
  Each removed subset needs its own continuation chain, because a walk can hop from one removed
  candidate to another. The cost is therefore per subset, though still far below a re-solve.
  Floating candidates (Schur complement) need the candidates' own rows, so every candidate is a
  target net.
- **Addition to an existing run** (no superset) needs a replay. A walk is affected only if the
  new box intersects one of its transition cubes, or its first-hop cube. If each walk stores the
  AABB of the union of its cubes plus its launch key, an edit replays only the walks whose AABB
  intersects the new box, with the **same streams**. All other walks are kept, so the result is
  statistically identical to a full re-solve. The engine path already keys each walk by its own
  stream (`Launch::key`); the batch path (`run_batch`) does not, and would need per-sample keys,
  as in `plan_batch`. The AABB must also cover the first-hop cube, the final sub-ε_stop
  neighbourhood and the Gaussian launch point, and ε_stop, the Gaussian offsets and the escape
  sphere must stay pinned to the original run. Storage is ~32 B per walk (key + f32 AABB), e.g. 320 MB for 10⁷ walks, so
  this is opt-in. Coarser alternatives (per-walk cell bitmasks) are open (§11).

**Implemented (deletion).** `sens::solve_sensitivity_with` takes `Deletion { nets }` sets
alongside faces and parameters; `faradice sens` reads them as `[[delete]] name, nets` and writes
`deletions[]` to `sens.json` with the edited `C_ij` and the paired `ΔC_ij`. The edited geometry
keeps the other nets in order, and continuation sinks are mapped back to the nominal net ids.
A deleted net that is also a target has no row in the edited results; the other rows are
bit-identical to a run without it as a target. Walks absorbed by a deleted net are re-checked
against ε_stop first (another conductor may be that close), then continued on their own stream keyed by (seed, deleted set, net, batch, sample,
slot). The escape sphere of an open domain stays that of the nominal geometry, which is still a
valid sphere for the smaller conductor set.

S6 (delete the crossing wire of the three-wire example, 6·10⁵ walks per target, pinned
`stop_dist`): edited totals and couplings match re-solves without the wire (five seed pairs:
|z| ≤ 2.8 over 25 comparisons, mixed signs; two pairs at 6·10⁶ walks: |z| ≤ 1.9). The paired
ΔC_tt has σ ≈ 0.0003–0.0006 fF against 0.006 fF for the difference of two solves (10–20×), with
~3 % of walks continued. Nominal statistics are bit-identical to `solve`.

Superset extraction plus continuation covers the motivating "delete / add / swap a wire" cases
before any replay infrastructure exists. Replay is therefore the last item in the staging.

**Implemented (addition by replay).** `replay::solve_recorded` is `solve` with a recording hook
(`SampleHook::RECORD`). Its statistics are bit-identical. Per Gaussian sample it keeps the batch
stream at the sample's start (32 B) and the bounding box of all clearance cubes of the sample,
including the first hop, the band cube of a layered lateral first hop, and the final sub-ε_stop
cube (24 B, f32 rounded outward, clamped at the f32 range). The batch
keys do not change: the stream is copied, not re-keyed. The batch loop body is one function,
`solver::sample_once`, shared by `solve` and the replays.

`replay::replay_addition` replays the samples whose box meets an added box. The edited walker is
the nominal walker plus the added boxes, and an added box replaces the nominal step only when it
is **strictly** nearer. Where no added box cuts a nominal cube, the step, the stream and the path
are therefore identical, so skipped samples are exactly unchanged. Each replayed sample runs
twice from the same stream: nominal, then edited. The difference is the exact paired ΔC, and the
edited moments are the nominal moments plus the replayed terms
(Σ(o+d)² = Σo² + Σ(2od + d²)). Added boxes may form a new net (at 0 V, reported as id
`num_nets`, optionally counted as ground) or extend a non-target net. Extending a target is
refused, because its Gaussian surface would not enclose the new boxes. Boxes are also refused if
they overlap or touch another net, leave the domain or the escape sphere, or cut a target's
Gaussian surface. A recording meant for additions
near targets should use a smaller `walk.gauss_alpha`. A fingerprint of the geometry, targets and the
whole solver configuration, plus the walks and sink sums of the recorded solution, guards
against pairing the records with another run.

Validation (`tests/replay.rs`): a grounded shield added between two wires, with 4·10⁵ walks per
target. The paired ΔC of all six entries matches minus the paired deletion ΔC of the superset
run, an independent exact estimator (|z| ≤ 1.5). The edited matrix matches a re-solve with the
shield (|z| ≤ 1.8). Replaying every record in the nominal geometry reproduces the nominal sums
(unit test). In a stratified box domain, a shield at the interface matches the
paired deletion (|z| ≤ 2.3), and a new net counted as ground matches a re-solve with it in
`ground_nets`. Boxes hidden inside a conductor of their own net change nothing, bit for bit,
although samples are replayed. Recording costs about 20 % in time and 56 B per sample
(224 MB for 4·10⁶ samples). A central shield replays 35 % of the samples; a far edit replays
fewer.

**CLI.** A parameter file lists additions as `[[add]] name, net, boxes = [[x0, y0, z0, x1, y1,
z1], …]` (µm), with `ground` (default `true`) for a new net. `net` names an existing non-target
net to extend it, or a new name for a new conductor at 0 V. `faradice sens` records the nominal
extraction once and replays every addition against it, each relative to the nominal geometry.
When nothing needs the finite-difference pass (no `[[delete]]` and no face taking finite
differences), the recorded run is the nominal one. Otherwise that pass runs as well; its
nominal statistics must be bit-identical to the recorded ones, which the run checks. `sens.json` gains `additions`: per addition, the edited `C` and the
paired `dC` of every target (sinks include the added net), `replayed_samples` and
`recorded_samples`. A warning flags an addition that replays more than half of the samples,
where a re-solve costs about the same. Recordings are not persisted between runs.

Validation (`faradice-sens/tests/sens_run.rs`): a grounded shield over the victim of the three
wires, 3·10⁵ walks. The edited row matches a re-solve with the shield (|z| ≤ 2.2 over five
entries), and the nominal `result.json` equals `faradice solve`'s.

---

## 9. Validation Plan

| ID | Case | Reference | Pillars | Pass criterion |
|---|---|---|---|---|
| S1 | Parallel plate, reflecting sides (V1 setup), bottom face of the plate moved | dC/dd₁ = −εA/d₁², closed form | A0, A | A0 passes (§4.3) |
| S2 | Same plate, top face moved (gap d₂ to the grounded top shrinks or grows) | dC/dd₂ = −εA/d₂², closed form | A0, A | A0 passes (§4.3) |
| S3 | Cube in free space (V2), each face displaced outward, and uniform growth as a parameter | dC/dn_f = C/(3a) per face, Σ = 2C/a (growing all faces by dn grows the side by 2dn, and C ∝ a) | A0, A | A0 passes (§4.3). Exercises edges and corners (§6.4) |
| S4 | Three wires (`examples/three_wires`): width, spacing and thickness of the victim | Palace finite differences (`bench/compare`, two structures at a large step) | A0, A | A0 and A agree within 4σ. Palace within 2 % + its own FD error |
| S5 | Grounding a track (three wires + crossing) | Re-solve with that track in `ground.nets` | C | Within 3σ |
| S6 | Deleting the crossing wire | Re-solve without it | D (continuation) | Within 3σ; nominal identical to `solve` bit for bit |
| S7 | A0 bias–variance: δ sweep (0.5 %–10 %) on S4 | A | A0 | Bias ∝ δ (one-sided) or δ² (Richardson). Variance ∝ 1/δ |
| S8 | A at edges (§6.4) | S3 and A0 | A | σ·√N stable for N from 10⁴ to 10⁷, and the tail of \|Y\| checked. A finite sample σ alone does not prove finite variance |
| S9 | Open boundary + substrate (odd image): two wires above the substrate | One-sided A0 D(δ) at a large δ (10–20 % of the gap) vs the finite difference of re-solves at the *same* δ, pinned `stop_dist`, walks sized for a stated σ budget | A0, A | Within 4σ |
| S10 | Floating middle plate between two plates (series capacitors, closed form) | Closed form | C | Within 3σ |
| S11 | Small grounded and floating cube between plates, a → 0 | ΔC vs c·φ_iφ_j and ∇φ·M∇φ; log–log slopes 1 (grounded) and 3 (floating) | B | Slopes within 10 %, prefactor within 20 % at the smallest a |
| S12 | L-shaped two-box net (partially exposed faces, concave edges) | As S9 (same-δ re-solves) | A0, A | Within 4σ |
| S13 | σ coverage: 50 seeds of S1 and S5, δ-bias ≪ σ | z-scores | A0, C | 90 % ± 13 % within 1.645σ (3 binomial sd at n = 50) |
| S14 | Determinism: thread count, face order, face subset | Bit equality | A0 | Implemented (`tests/sensitivity.rs`) |
| S15 | Vertical plates (normal x, reflecting y/z faces), the moved plate tagged by a conformal rule (shell of thickness t, ε_s), outer material ε_o; κ = ε_o/ε_s ∈ {0.5, 1.29, 2} | Closed forms of T^cond, T^Σ and their sum (§4.4c) | A0 (§4.4c) | Each term and the sum within 4σ; κ = 1 matches an untagged run |
| S16 | SKY130 met1 pair with NILD3_C (C4 geometry): width and spacing of one wire | Same-δ re-solves (as S9) | A0 (§4.4c) | Within 4σ |

Common pass criterion for closed-form cases: error within 4σ plus a stated bias allowance,
**and** σ small enough for the check to mean something (≤ 5 % of the exact value, 10 % for
single cube faces). A finite-difference reference from independent re-solves has
σ ≈ √2·σ_C/δ, which is useless at small δ, so such references always use a large, matched δ.

Unit tests: the `FaceTables` kernel reproduces ∂_nφ for polynomial harmonic data on the cube
(e.g. φ = z, φ = x² − z²) to series precision. Exposed-face detection. The parameter-file parser.
Pillar C identities on random SPD matrices (Schur complement of a merged/floated net, permutation
invariants).

Results are recorded in `docs/validation/D3.md`.

---

## 10. API, Features and Code Layout

**Cargo features (all off by default in the library crates):**

| Crate | Feature | Contents |
|---|---|---|
| `faradice-frw` | `sensitivity` | `solve_sensitivity` (A0 continuation, D deletion), `FaceTables` and `solve_shape_derivative` (A), `potential_map` (B). Needs crate-internal walker access, hence in this crate |
| `faradice-sens` (new) | — | Parameter files, face handles, chain rule, σ_C, Pillar C algebra on `result.json`, JSON output |
| `faradice` (CLI) | `sens` (default on) | `faradice sens`, `faradice whatif`, `faradice map` |
| `faradice-io` | — | Optional per-row cross-moments in `result.json` (§5.2), written only when requested |

The existing `solve` / `solve_with` code paths and their outputs are unchanged. `run_batch` gains
no branches; continuation runs in a separate driver that reuses the launch and walk functions.

**CLI:**

```sh
faradice sens STRUCTURE -c CONFIG -p p.sens.toml [--one-sided] [--method fd|shape|auto] -o OUT
#   → OUT/sens.json: per param, dC_ij/dp ± σ (and per face), method, δ, walk counts; σ_C if sigmas
#     given; per [[delete]] and [[add]], the edited C and the paired ΔC
faradice whatif OUT/result.json --ground t3 --float t5 --swap a,b --merge c,d [--json]
#   → modified matrix ± σ (Pillar C)
```

`sens.json` (schema v2, as written by `faradice sens`; `j == i` is the total C_ii, `"<GND>"` the
sum over ground-like sinks in the positive convention, other `j` the Maxwell entry C_ij):

```json
{ "schema_version": 2, "method": "fd-continuation", "richardson": true, "delta_rel": 0.05,
  "eps_stop_um": 0.00014, "seed": 1, "nets": [ { "net": "victim", "walks": 512000 } ],
  "faces": [ { "net": "victim", "box": 0, "face": "y-", "method": "fd", "delta_um": 0.007,
               "exposed_area_um2": 3.6,
               "gap_um": 0.14, "continued_walks": 349498, "aborted_continuations": 0,
               "dC": [ { "i": "victim", "j": "left", "value": -7.98, "std_err": 0.22 } ] } ],
  "params": [ { "name": "w_victim", "unit": "fF/um", "sigma_um": 0.005, "faces": [[0, 0.5], [1, 0.5]],
                "dC": [ { "i": "victim", "j": "left", "value": -4.1, "std_err": 0.1 } ],
                "sigma_C": [ { "i": "victim", "j": "left", "value": 0.02, "std_err": 0.0005 } ] } ],
  "deletions": [ { "name": "no_cross", "nets": ["cross"], "continued_walks": 9100,
                   "aborted_continuations": 0, "C": [ … ], "dC": [ … ] } ],
  "additions": [ { "name": "shield", "net": "shield", "new_net": true, "ground": true,
                   "boxes": [[3.0, 0.2, 1.46, 7.0, 0.5, 1.52]], "replayed_samples": 120000,
                   "recorded_samples": 2000000, "C": [ … ], "dC": [ … ] } ],
  "warnings": [] }
```

D2 operations to add later: `sensitivity`, `what_if`, `insertion_map`.

**Python (later):** PyO3 bindings with a JAX `custom_vjp` / PyTorch `autograd.Function`. The
forward pass returns C, and the backward pass is the per-face derivatives of A (or A0)
contracted with the upstream gradient: `p̄ = Σ_ij C̄_ij dC_ij/dp`.

**Repository:** D3 lives in this repository. A and D need solver internals (walker, Green
tables, RNG streams) that are not, and should not become, public API. Keeping it in-tree also
gives one CI, one clean-room record and direct GUI/agent integration.

---

## 11. Staging and Open Questions

| Stage | Content | Exit |
|---|---|---|
| A0 | Continuation solver, exposure rule, parameter file, δ default, `faradice sens` (done: S1–S4, S7, S9, S12–S14) | S1–S3, S7, S9, S12–S14 pass |
| C | Cross-moments, `faradice whatif` (done: S5, S10, S13) | S5 passes |
| A | `FaceTables`, interior estimator (done: S1, S2; `sens --method shape\|auto`), then an edge remedy (§6.4; remedy 1 measured, S8 partial) | Interior: S1, S2 (no edges with reflecting walls). Full: S3, S4, S8 |
| D-del | Deletion and superset workflow (done: S6) | S6 passes |
| B | Potential map, GUI overlay (map and `faradice map` done: S11 grounded) | Visual check + agreement with D on S6 |
| Py | Bindings and VJP | Gradient check vs A0 |
| D-add | Replay with per-sample stream + cube box (done: library, shield test; `[[add]]` in `faradice sens`) | Matches re-solve |
| A0-shell | Faces of conductors with conformal shells (§4.4c): T^cond by continuation in M₁, T^Σ by interface gradients with an edge margin (done: S15, S16; lateral faces and `z+` with `top > 0`), then `z±` rings and `method = "resolve"` | S15, S16 |

Open questions:

1. Default δ for A0: S7 ([validation](../validation/D3.md)) shows a linear one-sided bias and a
   Richardson plateau from 0.5 % to 4 % on a tightly coupled side face, so the 5 % default is kept.
   S4 matches Palace at a fixed step; other geometries and an independent δ → 0 reference
   (pillar A) are still to check.
2. Which edge remedy for A (§6.4)? This is the main research risk of D3.
3. ~~Face handles by point~~ Done: a face may give `at = [x, y, z]` instead of `box`. It selects
   the net's box whose face contains the point; a point on a shared edge of abutting boxes is
   ambiguous and rejected.
4. Cross-moment storage size for nets that hit thousands of sinks (bus-like structures): cap or
   sparsify by threshold?
5. D-add storage: per-sample stream + f32 cube box (56 B) is implemented. Open: storing the
   nominal per-sample contributions too (+≤64 B) would halve the replay cost. `faradice sens`
   records and replays within one run; persisting recordings across runs (a file format, its
   versioning and size) is open.
6. Floating conductors in the solver itself (for S5 references). Currently only Pillar C can
   float a track. Is a solver-side floating net (charge-neutral constraint) wanted?
