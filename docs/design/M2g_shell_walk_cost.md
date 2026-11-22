# M2g: Cost of walks near conformal shells

Status: Partially implemented (a performance study, results in §7). Implemented: the
bit-identical cheaper table builds and the plane rejection in the nearest-face queries. Steps 1
and 5 of §2 and a carried clearance bound (§7) were measured and not adopted; steps 2, 3, 4 and 6 are not implemented. The goal of
§1 is not yet met. Follows [M2d](M2d_edge_cubes.md) and [M2f](M2f_window_first_hops.md).

## 1. Goal and accounting

Goal (M2d §5): C5 within 3× the wall time of L6 at the default accuracy, cold (no table cache
from an earlier run). After M2f and the per-step query work (local face lists, M2d §3.6; compare-based L∞ distances
and per-call stamps in the local lists), C5 takes 5.10 s and L6 1.15 s
(16 threads): 4.4×. 3× is 3.45 s.

The time splits into a fixed part and a walking part. The fixed part is measured with one wave of
small batches (`batch_walks = 64`, 64 walks per net; a wave is `current_num_threads()` batches,
§6):

| | fixed (wall / CPU) | walking (wall) |
|---|---|---|
| C5 | 0.90 s / 20 CPU-s | ~4.2 s |
| L6 | 0.12 s / 0.6 CPU-s | ~1.03 s |

With no fixed cost at all C5 would still be 3.75× L6. Both parts must shrink.

**Fixed part.** One cold C5 run builds 36 edge-cube tables (M2d), 40 window kernels (M2f), 18
landing boxes and 64 normal-gradient kernels, and makes window decisions (`windows::plan_of`).
Nets run concurrently (`solve_hooked`); kernels are built lazily by the first walker that needs
them, while others wait. (A first reading of 3.8 and 4.0 CPU-s for the tables and kernels summed
wall times of internally parallel builds; §7 has the CPU.)

**Walking part.** Walks: 1.14× L6. Steps per walk: bus bits 1.7–1.8×, met1 1.53×, met3 1.11×.
CPU per step: about 2×. Of the C5 steps:

| step | share |
|---|---|
| stack step, box distance ≥ clearance | 40 % |
| stack step capped at the box distance | 21 % |
| axis cube | 22 % |
| M2d edge branch (best cube < 0.7 h) | 21 % |
| — landing (0.2–0.75 h; lands on the plane in 45 % of cases) | 13.6 % |
| — edge cube | 4 % |
| — nothing fits | 3.6 % |

A step-location heat map (ad-hoc instrumentation, not included in the repository) puts the excess steps (C5 minus L6) on the conductor's top and bottom
planes, where the shell ends flush with the conductor and its outer corner lies 0.03 µm from the
conductor corner. An oracle (likewise not included) that replaces every edge-branch step by a homogeneous cube of
half-side `h` (wrong physics, only the cost counts) cuts the walking time by 23 %.

A gdb sampling profile of the walking part, after that work: nearest box and face queries on
the local lists about 30 %, conductor clearance 19 %, axis plans 9.5 %, stack steps 9.5 %.

## 2. Plan

Each step has a measurement that decides whether it goes in. Expected times are planning
estimates (central / optimistic, cumulative, C5 wall).

| # | Work | Exact? | Decision measurement | C5 |
|---|---|---|---|---|
| 1 | Landing boxes that cross a supporting plane where it is no interface inside the box (§3) | yes | edge-branch steps, walk time | 4.9 / 4.8 s |
| 2 | Build scheduling: one builder budget for M2d and M2f, dependency-aware queue | bit-identical | dependency waits, cold wall | 4.8 / 4.6 s |
| 3 | Window decisions: indexed slots, cached band summaries | bit-identical | hits vs. unique drafts, cold wall | 4.65 / 4.45 s |
| 4 | Edge tables without the finest level where a finer reference validates it | no (tables change within the error contract) | per-configuration error vs. reference | 4.4 / 4.2 s |
| 5 | Certified local environments (§4) | bit-identical | shadow test, then walking CPU (≥ 8 %) | 3.9 / 3.6 s |
| 6 | Analytic two-sided landing (§5), only if it pays | yes | CPU from corner entry to exit | 3.8 / 3.3 s |

The central outcome is about 3.3× L6, the optimistic one about 2.9×. 3× stays the goal, but it
needs most steps at their optimistic values. L6 is re-measured after shared changes, since a
faster L6 tightens the goal.

Step 1 is first because it is small, exact, and its effect shows directly in the step counts.

## 3. Landing across a terminated plane

M2d §3.2 lands an off-plane walker on the nearer edge plane `P` with a landing box whose lateral
half-width is at most `min(R, d_Q)`, so the box never crosses `Q`. That is needed only where `Q`
is an interface inside the box. At the top outer corner of a shell (`P` the shell's top plane,
`Q` its outer face) the shell face ends at `P`: a box above `P` meets no face in `Q`'s plane, and
a box beside the shell (gap side of `Q`) meets no face in `P`'s plane.

So:

- Try both orientations (land on `P`, land on `Q`).
- Bound the box by the faces actually inside it (the room query of M2d §3.1 with the box's own
  extents), not by the other supporting plane.
- The box stays homogeneous, so the tables and the finite configuration set (`τ`, ratio) are
  unchanged. No new offset enters: a plane with no interface in the box is not a parameter of
  its Green function.

A flat box lands with probability about `1 − τ`, against 45 % now. If half to three quarters of
the landings qualify, the steps fall by roughly 3–5 % of all steps.

A landing box above the conductor top may, if it extends beyond the clearance, have part of its
bottom face on the conductor; an exit there would be absorbed. That needs a clearance
certificate for the box shape and gains little (a few percent of the exits near the corner), so
it is left out of step 1.

## 4. Certified local environments

The local lists (M2d §3.6) cache every box within `4r` of a walker, but each query still scans them.
Instead, keep a small active set per list and a lower bound `L` on the distance of everything else
from an anchor `p₀`. At `p`, every excluded object is at least `L − ‖p − p₀‖∞` away (L∞ triangle
inequality, with a rounding margin). A query that evaluates only the active set is exact when its
answer beats this bound:

- box distance `v`: the active minimum, or `v = 0` from an active containing box;
- `v ≥ h`: every active box and the excluded bound are at least `h`;
- nearest face per axis, and nearest on another plane: the winner (and the second) against the
  excluded bound of the axis;
- conductor clearance: the active conductor minimum against the excluded conductors, then the
  domain boundary as now.

Ties keep the order of the full query (distance, then id), and anything ambiguous falls back to
it. A one-winner cache fails near a shell corner, where competitors are 0.03 µm apart and steps
are 0.03–0.06 µm; the active set must hold the corner's faces. The expected saving is 8–15 % of
the walking CPU, after refreshes. It is shadow-tested on recorded walker positions first: every
certified answer against the full query.

## 5. Two-sided landing, analytic

A homogeneous rectangular box with faces on both `P` and `Q` and the walker inside lands on one
of the edge planes with high probability (a bound gives > 76 % for sides 4× the offsets, > 94 %
for 8×). Tables cannot serve it: the walker's two fractional offsets fix the aspect ratio, which
is continuous. Its exit can instead be sampled exactly at the actual source, because the three
coordinates of Brownian motion in the box are independent: draw each coordinate's interval exit
time from the interval heat kernel (image series for short times, sine series for long ones,
with certified tail bounds), take the earliest, then draw the other coordinates conditionally.

This is worth it only if a two-sided landing costs less than about twice a table landing. It is
step 6, after step 1 shows how much landing coverage the cheaper construction leaves.

## 6. Rejected

- **Recentring on fixed points.** In a homogeneous ball, the Poisson kernel is a mixture of the
  uniform measure (weight `α(q) = (1 − q)/(1 + q)²`) and a residual, so a walker can be moved to
  the ball's centre with probability `α`, exactly. Fixed centres would allow absorbing
  cross-section kernels tabulated at a fixed source. Near the shell termination, though, balls
  are limited by the 0.03 µm shell and its top face, and walkers a few nm from the termination
  get `α` of a few percent. Expected saving 0–5 %, with new builds.
- **Off-centre edge cubes or landing boxes straddling `Q` at a tabulated offset.** Moving the
  centre cancels out: the source and interface coordinates fix `d_P/d_Q` to a finite set, so
  generic walkers would need snapping.
- **Larger `α`.** Does not enlarge any domain; no gain on its own.
- **Interface scattering series at weak contrast.** Exact in principle, but singular quadrature
  at triple junctions and signed-weight variance make it a research project.
- **Faster scans of the local lists.** Structure-of-arrays lists (no gain in a microbenchmark:
  20 boxes scan in 35 ns), list reach 2/3/6 (4 stays best), lists sorted by distance from the
  centre with early exit (3 % slower), one fused query per step (about 5 % by an A/B with
  duplicated queries).

The measurements of the fixed part corrected two earlier readings: nets are not solved one
after another, and `run_net` computes a full wave of batches before it checks convergence, so a
"one batch per net" run computes `current_num_threads()` batches per net.

## 7. Results

**Step 1 (§3), not adopted as built.** Landing boxes that cross a terminated plane raise the
share of landings on the plane from 45 % to 52 % and cut C5's steps by 2–4 %. A wider variant,
boxes above the conductor top bounded only by what reaches into the half-space beyond `P`
(71 % of off-plane landings were limited by the conductor clearance), cuts 5–7 %. Neither is
faster: the extra queries of the wider box (five local-list queries, 10.8 M calls, 180 ns when
it fails and 760 ns when it fits) cost about what the saved steps do (≈ 4.2 vs ≈ 4.3 CPU-s),
and the narrower one saves too little. They wait for step 5, which makes such queries cheap.
Neither variant is included in the repository; their statistics were not validated.

**Fixed part, measured per build** (one walker thread, so builds do not overlap; process CPU):

| | CPU-s | wall per build, alone |
|---|---|---|
| 36 edge tables | 10.8 | 23 ms |
| 29 window kernels (40 with 16 walkers) | 4.6 | 33 ms, of which 31 ms is the serial Jacobi eigen decomposition |
| window decisions without kernels | < 0.3 | — |
| 18 landing boxes | 0.2 | 10 ms |

With 16 walkers the blocked walker time (building or waiting for another thread's build) was
4.7 thread-s on edge tables, 6.1 on window kernels, 1.0 on landing boxes and 0.5 on decisions:
12.3 of 83 thread-s. Steps 2 and 3 as planned target the wrong parts: decisions are cheap, and
the builds are CPU-bound rather than badly scheduled.

**Cheaper builds (bit-identical).** About 95 % of a table's flops are the dense products of the
fast-diagonalization preconditioners. Tiling them (4 × 4 outputs in registers, every sum in its
original order) cuts the edge tables to 7.6 CPU-s. The window kernels' Jacobi rotations update
the eigenvectors in contiguous rows, kernels of one medium share the decompositions of equal
node spacings (108 distinct of 240), and the two axes are decomposed concurrently. Cold C5,
fixed part: 0.94 s / 20.5 CPU-s → 0.69 s / 15.3 CPU-s; blocked walker time 12.3 → 8.9
thread-s. Default accuracy (5 rounds, loaded machine): C5 5.46 → 5.13 s, C4 1.39 → 1.29 s, L6
unchanged; all compare cases bit-identical.

**Walking part, measured with `perf`** (frame pointers, C5 at the default accuracy, after the
build work above). The earlier gdb-sampling profile overstated what a change could remove; the
shares below are cycle samples of the whole process:

| | share |
|---|---|
| nearest face per axis and the one on another plane (`axis_deltas`, three lists of 11, 19 and 30 faces) | 21 % |
| conductor clearance (`base_clearance`, grid query with near-cell lists) | 16.5 % |
| box distance (`box_distance`, a list of 18 boxes) | 10–13 % |
| window-kernel and edge-table builds (`dense::product` and the rest of the builds) | 11–13 % |
| everything else in a step (plans, stack, tables, sampling) | the rest |

The face scans run about 26 instructions per listed face, with a serial chain of maxima. Of the
listed faces, 18 % have their plane within the query radius and 6 % are within it.

**Step 5 (§4), not adopted.** Certified active subsets of the local lists (the boxes within
twice the query radius of an anchor, and a lower bound on the others) prove 92 % of the answers
and halve the scanned entries (8 G → 3.8 G per C5 run). They were bit-identical but not faster:
C5 CPU +2.5 % on their own, because the refreshes and certificates cost about what the shorter
scans save. Conductor clearances from per-thread local lists (98 % certified, 3.8 boxes) changed
C5 by nothing, V5 by −20 % and V8 by +11 %: the grid's near-cell lists are already short, and
each uncertified query costs a ring search and a rebuild. Neither variant is included in the
repository.

**Plane rejection (bit-identical).** A face normal to `a` is at least `|x_a − c|` away, the exact
`a` component of its L∞ distance. Testing that first (one loop per axis, so `c` is a fixed
field) skips most faces: C5 CPU −2 %, other cases unchanged. Sorting each list by plane, so that
only the faces within the radius are visited, costs more in sorting than it saves.

**Carried clearance bound, not adopted.** The conductor clearance is 1-Lipschitz in the max
norm, so after a move of L∞ length `m` from a point with clearance `h`, `h − m` (with a rounding
margin) is a valid conductor-free half-side at the new point. The stratified walk already uses
this bound where the second interface limits the cube (M2a §8); the shell walk queries every
step. Decision measurement: at least 5 % of the walking CPU on C5.

Instrumented C5 run (default accuracy, 16 threads, 188 M clearance queries), the carried bound
`b` against the exact clearance `h`:

| `b / h` | < 0.1 | 0.1–0.5 | 0.5–0.9 | ≥ 0.9 |
|---|---|---|---|---|
| queries | 54 % | 32 % | 8.5 % | 5.5 % |

Most steps move nearly their full clearance, so the bound is gone by the next step. Only 16 % of
the steps move no farther than the bound they started with. Even if each of those queries were
skipped for free with the step unchanged, the saving would be about 16 % × 16.5 % ≈ 2.6 % of
the CPU.

A skip rule that uses the bound in place of the query when `b ≥ (1 − q)·h_prev` (and
`b ≥ ε_stop`) keeps the walks exact but shrinks the steps it serves. User-space instructions per
walk on C5 against the query on every step (cycle counts varied by ±15 % on the shared machine;
instruction counts by 0.003 %):

| `q` | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|
| steps per walk (48.06 without) | 48.14 | 48.38 | 48.50 | 49.31 | 50.45 | 52.39 | 77.45 |
| instructions per walk | −0.4 % | −1.2 % | −0.9 % | −2.4 % | −1.6 % | +0.9 % | +17 % |

The best setting saves about 2.5 % of the instructions (−1.1 % on 8 pinned threads), half the
threshold. The variant is not included in the repository.
