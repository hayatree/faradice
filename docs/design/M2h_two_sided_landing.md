# M2h: Two-sided landing near dielectric edges

Status: Design (issue #4). Follows [M2g](M2g_shell_walk_cost.md) §5 and the ceiling measured in
M2g §7. Not implemented.

## 1. Goal

Near a dielectric edge (two interface planes `P` and `Q`, M2d), a walker that is off both planes
lands on the nearer plane with a tabulated landing box (M2d §3.2). That box may not cross `Q`, so
its lateral half-width is at most the distance `d_Q`, and it lands on `P` in only 45 % of the
landing steps of C5 (M2g §1). The walker then needs further steps before it reaches a plane and
can leave the edge with an edge cube.

This design replaces that box, where it fits, by a homogeneous box with one face on `P` and one
on `Q`. The walker is not at the box's centre. Its exit is sampled exactly at the walker's actual
position, without tables, because the coordinates of Brownian motion in a box are independent
(§3). The estimator stays exact: the box is conductor-free and homogeneous, and its exit law is
the box's harmonic measure from the walker.

M2g §7 bounds the gain at 13.9 % (C5) and 16.6 % (C4) of the walking time, and puts the realistic
gain at 5–10 % if a landing succeeds in 76–94 % of the cases. §2 measures that rate.

## 2. Domain

Take the edge frame of M2d §3.1:
- `P` and `Q` are the two nearest interface planes;
- `d_P ≤ d_Q` are the walker's distances to them;
- `a` is the remaining axis.

Let `s` be the edge room at the full clearance cap: the L∞ distance to the nearest interface face
off `P` and `Q`, to other stack planes, and to conductors (`edge_room` with `cap = d.cap`).

If `d_Q < s`, the box is

- along the normal of `P`: from `P` to `s` beyond the walker;
- along the normal of `Q`: from `Q` to `s` beyond the walker;
- along `a`: `s` on both sides of the walker.

It lies inside the L∞ cube of half-side `s` around the walker, so it holds no conductor. No
interface face lies inside it either: a face off `P` and `Q` would be nearer than `s`, and the
faces on `P` and `Q` lie only on its boundary. The medium is therefore homogeneous, the walker's
own. When `d_Q ≥ s`, the walk keeps the one-sided landing box of M2d. The choice depends only on
the position, so the estimator stays exact.

Exits through the faces on `P` and `Q` are moved onto the plane exactly, as in M2c §4.4. That
holds whether or not the plane is an interface along the whole face.

**Measured applicability.** A temporary probe (not included in the repository) recorded the
offsets at every off-plane entry of the edge branch in the default-accuracy runs. A sample of
1 in 64 was then run through a Python prototype of the sampler (§3).

| | C5 | C4 |
|---|---|---|
| off-plane entries (one-sided landing attempts) | 24.9 M | 5.0 M |
| on-plane entries (edge cubes) | 8.3 M | 1.6 M |
| the box fits (`d_Q < s`) | 94 % | 95 % |
| median `d_P / s`, `d_Q / s` | 0.21, 0.53 | 0.21, 0.52 |
| lands on `P` or `Q`, where it fits | 83 % | 83 % |
| lands, of all off-plane entries | 78 % | 78 % |

The one-sided box lands in 45 % of the cases. The two-sided rate of 78 % is at the low end of
the 76–94 % assumed in M2g §7, so the expected gain is about 5–7 % of the walking time.

## 3. Exit sampling

Let `B = Π [0, ℓᵢ]` be the box, the walker at `x ∈ B`, and `W` standard Brownian motion. The
harmonic measure of `B` from `x` is the law of `W` at its exit time. The three coordinates are
independent one-dimensional Brownian motions. Let `Tᵢ` be the time at which coordinate `i` leaves
its interval. The exit time is `T = min Tᵢ`, and the exit leaves through a face normal to axis
`j = argmin Tᵢ`.

The joint law factorises:

`P(j, Tⱼ ∈ dt, side, Wᵢ(t) ∈ dyᵢ for i ≠ j) = f_{j,side}(t) dt · Π_{i≠j} q_i(t, xᵢ, yᵢ) dyᵢ`

Here `f_{j,side}` is the density of the exit time of coordinate `j` through that side, and `qᵢ`
is the transition density of coordinate `i` killed at the interval's ends. So:

1. For each axis, draw `(Tᵢ, sideᵢ)` from its own exit law.
2. Take `j = argmin Tᵢ`, `t = Tⱼ`, and put coordinate `j` on its exit side.
3. For each `i ≠ j`, draw `yᵢ` from `qᵢ(t, xᵢ, ·) / Sᵢ(t)`, where `Sᵢ` is the survival
   probability. The drawn `Tᵢ` (`i ≠ j`) are discarded.

Step 3 uses the other `Tᵢ` only through the event `Tᵢ > t`. Multiplying the probability of step 2,
`f_{j,side}(t) Π Sᵢ(t)`, by the densities of step 3 gives the joint law above.

The same construction (exit time and position from a parallelepiped, started at any interior
point) underlies the random walk on rectangles of Deaconu and Lejay (Methodol. Comput. Appl.
Probab. 8, 2006). Its full text was not available here, so the samplers below are derived
directly. M2c §2 found no FRW capacitance solver that uses such a box.

### 3.1 One-dimensional laws

Scale each axis to the unit interval: `u = xᵢ / ℓᵢ`, and time by `ℓᵢ²`.

- **Exit density through 0.** The image series (short times) and the eigenfunction series (long
  times) are
  - `f₀(t) = Σ_{k∈ℤ} (u + 2k) (2π t³)^{-1/2} e^{-(u+2k)²/2t}`
  - `f₀(t) = π Σ_{n≥1} n sin(nπu) e^{-n²π²t/2}`
- **Exit through 1.** The same series with `u → 1 − u`. `P(side 0) = 1 − u` exactly.
- **Killed density.** The image series and the eigenfunction series are
  - `q(t, u, y) = (2πt)^{-1/2} Σ_k (e^{-(y−u+2k)²/2t} − e^{-(y+u+2k)²/2t})`
  - `q(t, u, y) = 2 Σ_{n≥1} sin(nπu) sin(nπy) e^{-n²π²t/2}`

A switch time `t*` (about 0.3) selects the series. Below it the image series is used, above it
the eigenfunction series, and either needs two to three terms.

### 3.2 Samplers

Every sampler is a rejection sampler with an exact proposal. Its acceptance test `U·g < f` is
decided by the series method (Devroye, *Non-Uniform Random Variate Generation*, Springer 1986,
§IV.5). Partial sums of `f` are refined until a certified tail bound separates them from `U·g`.
Tail bounds decide; they never truncate. The samplers are exact up to floating-point rounding.

- **Exit time and side.**
  1. Draw the side with probability `1 − u` (side 0) or `u` (side 1). Let `d` be the distance
     to that end (`u` or `1 − u`).
  2. Propose `t = d² / Z²` with `Z` normal: this is the Lévy density `g(t)` of the first hit of
     that end, ignoring the other end. `f_side ≤ g`, because exiting through that end implies
     hitting it.
  3. Accept with probability `f_side(t) / g(t)`. The acceptance rate is `P(side)`, so the
     expected number of proposals per axis is 2.
- **Position given survival, `t ≤ t*`.** Propose `y ~ N(u, t)`. Reject if `y ∉ (0, 1)`, otherwise
  accept with the ratio of the killed to the free kernel. That ratio is the probability that the
  Brownian bridge from `u` to `y` stays in `(0, 1)`, an image series.
- **Position given survival, `t > t*`.** Propose `y = arccos(1 − 2U) / π`, which has density
  `(π/2) sin(πy)`. Accept with `q(t, u, y) / (M sin(πy))`. Here `M = 2 Σ n |sin(nπu)|
  e^{-n²π²t/2}`, which bounds `q / sin(πy)` because `|sin(nπy)| ≤ n sin(πy)`.

**Tail bounds.**
- Image series: for `|k| ≥ K` the exponents grow like `2k²/t`, so the tail is dominated by a
  geometric series in `e^{-2K/t}`.
- Eigenfunction series: `|sin| ≤ 1`, which gives a geometric series in `e^{-π²(2N+1)t/2}`.
- Both are derived in the code next to the series and unit-tested against long partial sums.

**Prototype (Python).** The box `[0,1] × [0,1.3] × [0,2]` was sampled from `(0.1, 0.25, 0.9)`,
20 000 exits. The six face probabilities agree with the separation-of-variables solution within
1.6σ. There were 6.0 time proposals and 2.7 position proposals per exit, as expected: two per
axis for the times, and the position proposals are needed only for the two axes that do not
exit.

## 4. Cost

A two-sided landing costs about 9 proposals. Each proposal is one or two normal or uniform
variates, one exponential and two to three series terms, so the landing should take about
0.3–0.4 µs, without any table build. M2g §5 sets the limit at twice a table landing, about
1.3 µs (M2g §7). The fixed part gains nothing, because the one-sided landing tables are still
needed where the box does not fit.

## 5. Plan

| # | Work | Decision measurement |
|---|---|---|
| 1 | `crates/faradice-frw/src/box_exit.rs`: the 1-D samplers and the box exit, with tests (§6) | all tests pass |
| 2 | Microbenchmark of one box exit at the measured offset distribution | below 1.3 µs; else stop |
| 3 | `landing_step` uses the two-sided box where it fits (§2) | C5 user-space instructions per walk at least 4 % lower, no other case more than 0.5 % higher; else not adopted |
| 4 | End-to-end validation (§6), M2g §7 and M2.md updated | as in §6 |

Steps per walk and instructions per walk are compared on the same seeds. Wall times are not
compared (M2g §7: they scatter by ±15 % on the shared machine).

## 6. Validation

- **1-D samplers.** Kolmogorov–Smirnov tests against the analytic CDFs:
  - the exit time for `u ∈ {0.01, 0.1, 0.3, 0.5}`;
  - the killed position for times on both sides of `t*`.
- **Face probabilities.** Separation-of-variables solutions for off-centre points in boxes of
  several aspect ratios, as in the prototype.
- **Harmonic invariance.** For harmonic `φ` (`x`, `x² − y²`, `xyz`, `x² + y² − 2z²`) the mean of
  `φ` at the exit equals `φ(x)` within 4σ. This tests the whole sampler, including the
  conditional positions.
- **End to end.**
  - C4 and C5 with four seeds per variant: the matrix entries differ from the current walk by
    an rms of about 1σ, as in M2g step 4.
  - C4 totals within the combined σ of the extrapolated Palace values (external.md).
  - All other compare cases are bit-identical, since they have no dielectric edges and never
    reach the edge branch.

## 7. Risks

- **The gain is near its floor.** The landing rate of 78 % gives about 5–7 % of the walking
  time, near the 4 % gate. If the gate fails, the sampler is still a general off-centre
  homogeneous box. It could replace other tabulated landing boxes, but that is not part of
  this design.
- **Floating-point ties.** Decisions within rounding of the threshold have measure of order one
  ulp, as for the other samplers.
- **Rare long times.** Lévy proposals have heavy tails. Times beyond a few `ℓ²` are accepted with
  probability below `e^{-π²t/2}`, so they cost one cheap rejection each.
