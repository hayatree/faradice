# M2e: Layered first hops between facing interfaces

Status: Partially implemented (the first stage, §3: layered first hops between facing interfaces,
in `crates/faradice-frw/src/axis.rs` and `multilayer.rs`). The later stages of §5 are open;
[M2f](M2f_window_first_hops.md) covers part of them with window first hops. Follows M2c §5 and M2d.

## 1. Problem

With conformal shells the walks need more samples for the same σ (M2c validation, C4/C5): on C5
the bus bits need 2.8× and the met1 lines 1.45× the walks of L6 at the default accuracy.

An oracle run (an ad-hoc code variant not included in the repository), in which every first hop takes the homogeneous cube of half-side `h` (the
conductor clearance) and ignores the interfaces, has the variance of L6 exactly (C5, 200 k walks
per net; the values are then biased). So the excess variance comes entirely from the first cube
being smaller than `h`.

At SKY130 met1 minimum spacing (0.14 µm) the Gaussian surface lies at δ₀ = 0.07 µm. That is
0.04 µm beyond the conductor's own 0.03 µm shell and 0.04 µm short of the neighbour's shell. The
M2c first cubes (homogeneous, or one interface at a tabulated offset) are therefore limited to
about 0.04 µm, against 0.07 µm without shells. The first-hop weight scales like `1/s`, so the
second moment rises by about `(0.07/0.04)² ≈ 3`.

## 2. Prior work (public papers only)

The first-hop estimator is the one of M2a §4 (gradient kernels of the transition cube at the
Gaussian point). Layered cubes with interfaces at their true positions are those of M2b §3.2
(series per layer with transfer of `Z` and `ε Z′`). Nothing new is needed beyond differentiating
that series at the centre.

## 3. Method (implemented)

### 3.1 Normal kernel of a layered cube

`multilayer::NormalGradient` is the derivative along the layering axis, at the centre of the unit
cube, of the M2b harmonic measure, for any profile whose centre lies inside a layer.

- Top and bottom faces use `Z′(½)/Z(1)` in place of `Z(½)/Z(1)`, where `Z′ = (εZ′)/ε` in the
  centre layer; the bottom face is the mirrored profile with the sign reversed.
- Side faces use `ψ′(½)` of each side eigenmode in place of `ψ(½)`.
- The side cells are split at the interfaces, as in M2b.
- The kernel is signed. It is sampled from `|K|`, or as a ± pair from its two halves, which each
  carry `½‖K‖₁` because `∫K = 0`. This is the same estimator as the M2b paired normal kernel.
- A build takes about 2 ms at `M = 32`.

Checks:

- With one interface the kernel equals the M2a layered normal kernel to 1e-11 for κ ∈ {1, ¼, 3,
  10} and three offsets.
- It reproduces the exact layered linear potential (`ε φ′` constant) to 1e-6. This was checked
  for two facing interfaces and for three interfaces off the grid.
- `∫K = 0`.
- Its sampling matches the cells (χ², and the signs of the pair).

### 3.2 Selection

At a Gaussian point `p` on a face with normal `n` (clearance `h`):

- **Room.** `R` is `h` capped by the reflection reach, by the L∞ distance to the nearest face of
  each other normal, and by the nearest stack plane unless `n = z`.
- **Half-side.** `s` is the largest value `2^{j/8} ≤ R` (`j` an integer). The ladder keeps the
  profiles few.
- **Planes inside the cube.** These are the faces normal to `n` with L∞ distance below `s` (and,
  for `n = z`, the stack planes within `s`). Such a face spans the cube's cross-section: it could
  only end at a face of another normal, and none lies within `s`. The search allows at most eight
  planes.
- **Profile.** The positions of the planes in `[0, 1]`, and ε read at the layer midpoints on the
  normal line through `p`, relative to ε at `p`.
- **When it is used.** `s` must exceed the best M2c candidate (the stack cube or the best axis
  cube), the cube must contain at least one plane, and `p` must lie inside a layer.

The hop's factor is `ε_r(p)·‖K‖₁/(2s)` with the kernel's sign, as for the M2c normal kernels.

### 3.3 Tables

- Kernels are cached per profile: the bits of the positions and of the relative ε, in a shared
  `HashMap` that `BoxSteps::with_medium` (D3) shares too.
- At most 1024 are cached (about 0.5 MB each; C5 uses 63, C4 27). Beyond that a kernel is
  built for each use (about 2 ms), so whether a point uses a layered hop never depends on which
  profiles other threads met first.
- A kernel depends only on its key, so results do not depend on which thread builds it.

## 4. Results

C5 with 200 k walks per net (variance relative to L6):

| | M2c | M2e | oracle |
|---|---|---|---|
| bus bits | 2.25–2.81 | 1.80–1.88 | 1.00 |
| met1 lines | 1.43–1.46 | 1.41–1.43 | 1.00 |
| met3 lines | 1.00 | 1.00 | 1.00 |

The values agree with M2c within 1.8σ. See `docs/validation/M2.md` (M2e) for C4/C5 against
Palace.

## 5. What remains (later stages)

The rest of the excess comes from Gaussian points whose cube also meets a plane of another
normal:

- lateral faces near the shell's ends in z, and lateral faces that cross a stack plane (the
  met1 lines);
- top faces near the shells' side faces.

Those cubes contain an edge: two perpendicular planes with quadrant-constant ε. A first-hop
kernel there needs the gradient of the edge-cube Green function (M2d §3.4) at a source off both
planes. The finite-volume tables of M2d give exit probabilities for a source at a node. A
gradient kernel would need a dipole source and its convergence study. That is left open.
