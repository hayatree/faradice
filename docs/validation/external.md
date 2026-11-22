# External baselines (issue #7)

[M1](M1.md) and [M2](M2.md) check faradice against two open-source field solvers:
FasterCap (BEM) and Palace (FEM). This report adds RWCap v4, the floating-random-walk solver
published by the Tsinghua group, as a third reference. It compares accuracy, and also speed,
because RWCap uses the same method. It also records why CapBench is not used yet.

Machine: AMD Ryzen AI MAX+ 395 (32 threads, shared). Both solvers run with 16 threads. Wall
and CPU times on this machine vary by about ±15 % between repeats.

## RWCap v4

### Setup

- **Image.** `bench/compare/docker/rwcap` builds the image from the 2023 binary package.
  - The package is downloaded from the RWCap v4 page and pinned by SHA-256.
  - It contains only the executables, their precomputed tables and example cases. No source
    is included, and none was read.
  - The package and its web page state no license or terms of use. The binary is run as a
    separate process inside the container, and it is not redistributed.
- **Run.** `compare.py CASE --solvers rwcap --confidence-sigma 1` runs both solvers.
  - **Domain:** the same grounded box as Palace: the conductors' bounding box plus a 2 µm
    margin, with its bottom face on the substrate plane.
  - **Input:** a `.cap3d` file. The stack and the dielectric boxes become disjoint medium
    blocks, and the box itself is an explicit grounded conductor made of six plates just
    outside the domain.
  - **Runs:** RWCap extracts one master net per run, so a case with n nets takes n runs.
- **The window boundary.** A window marked `dirichlet` absorbed no charge in RWCap v4: every
  row summed to zero, and V4 differed by up to 58 %. Writing the box as a conductor fixed
  this, so the `dirichlet` window is not used.
- **What `-p` means.** RWCap reports no standard error, and its result is deterministic. It
  does not change with the thread count or with small shifts of the domain margin.
  - **Independent samples:** splitting the master's box in two at different points changes
    the walks. In V4, eight such variants spread by 0.94 × `-p` (one standard deviation).
    The same check on seven C4 variants gave 0.95 × `-p`.
  - **Matching faradice:** `-p 0.001` therefore behaves like a 1σ target of about 0.09 %.
    faradice gets the same targets from `confidence_sigma = 1` with the `reference`
    tolerances: 0.1 % for totals and 1 % for couplings, each at 1σ.
  - **Couplings:** `-c 0.01` did not change RWCap's walk counts in these cases, and the
    coupling targets did not change faradice's. Both solvers therefore stopped on the total.

### Accuracy

The total capacitance is the largest relative difference over all nets of a case. The
runs use matched targets: faradice σ = 0.10 %, RWCap about 0.09 %. Their difference
therefore has σ ≈ 0.13 %.

| Case | Dielectric | RWCap − faradice, totals | RWCap − faradice, major couplings |
|---|---|---|---|
| V4 three wires | uniform | 0.41 % | 1.29 % |
| V5 crossing + via | uniform | 0.26 % | 1.35 % |
| V8 SKY130 bus | uniform | 0.23 % | 1.01 % |
| L5 wires in stack | 4 layers, 50 nm high-k | 0.70 % | 1.45 % |
| L6 SKY130 stack bus | 7 layers | 0.31 % | 1.45 % |
| C4 met1 pair + shells | 7 layers + conformal | 0.66 % | 0.50 % |
| C5 SKY130 bus + shells | 7 layers + conformal | 0.65 % | 1.70 % |

**Uniform dielectric and the plain stack (V4, V5, V8, L6).**
- The total differences are within 3σ of the combined error (largest: V4, 3.0σ; the
  other 23 nets within 2.3σ).
- The coupling differences are about 1 %, the same size as the 1 % coupling targets.

**Thin layers and conformal shells (L5, C4, C5).**
- The total differences reach 5σ.
- Longer runs show that the difference is a systematic error of RWCap, not noise. They
  are compared below with Palace, extrapolated from its last three adaptive meshes.
- faradice used the `reference` preset at 3σ (σ ≈ 0.034 %), and RWCap ran with
  `-p 0.0003` (σ ≈ 0.03 %).

| Net | Palace extrapolated (fF) | faradice | RWCap |
|---|---|---|---|
| L5 a, total | 1.09629 | −0.02 % | −0.58 % |
| C4 a, total | 1.20904 | −0.04 % | +0.44 % |
| C4 a, total, master box split in two | 1.20904 | (same geometry) | −0.29 % (`-p 0.0005`) |

The last row is the same C4 geometry with the master's box written as two blocks. RWCap's
converged result moved by 0.7 % with this change. In V4, with a uniform dielectric, the
split variants agree with the single-box result within their noise. faradice does not depend
on how a net is divided into boxes.

### Speed at matched targets

The targets are those above: about 0.1 % at 1σ for totals. The faradice column is wall time
from the `faradice solve` call. The RWCap column is the sum of the elapsed times it reports
for each master net, which excludes the container start-up. Walk counts are the sums over
all nets.

| Case | faradice | RWCap | RWCap / faradice | Walks, faradice / RWCap | Steps per walk, faradice / RWCap |
|---|---|---|---|---|---|
| V4 | 1.5 s | 18.4 s | 12.0 | 7.1 M / 14.2 M | 16.7 / 8.4 |
| V5 | 4.0 s | 58.8 s | 14.6 | 16.0 M / 33.4 M | 22.5 / 11.8 |
| V8 | 6.3 s | 101.5 s | 16.2 | 29.7 M / 61.1 M | 21.4 / 10.8 |
| L5 | 5.2 s | 40.4 s | 7.7 | 12.8 M / 25.5 M | 25.4 / 9.6 |
| L6 | 14.5 s | 123.7 s | 8.5 | 30.4 M / 66.4 M | 29.9 / 11.0 |
| C4 | 11.6 s | 47.6 s | 4.1 | 7.4 M / 24.2 M | 40.3 / 8.5 |
| C5 | 62.7 s | 246.6 s | 3.9 | 34.7 M / 97.6 M | 43.9 / 10.7 |

**CPU time.** The CPU-time ratios are the same: both solvers keep 11–14 of the 16 threads
busy. For example, C5 took 868 CPU-s in faradice and 3395 CPU-s in RWCap.

**The ratios slightly favour faradice.** RWCap's σ at a given `-p` is about 0.94 × `-p`, so RWCap
needs about 0.9 × its time to reach exactly 0.1 %. That correction brings the ratios to
3.4–14.

**Where the difference comes from.**
- **Walks:** faradice needs about half as many walks for the same σ (antithetic first
  hops). With conformal shells it needs a third as many.
- **Steps:** faradice takes 2–5 times as many steps per walk. RWCap's steps cover more
  space. According to its readme, its transition cubes either hold four dielectric layers
  of equal thickness (Yang, Yu, Song and Xu, IEEE TCAD 41(5), 2022) or are eight-octant
  cubes sampled on the fly for conformal dielectrics (Yang and Yu, IEEE TCAD 39(12), 2020).
  A cube of four equal layers cannot match a real stack exactly. That is a plausible source
  of the L5 and C4 errors above, but it was not verified.
- **Time per walk:** faradice is 6–8 times faster per walk with a uniform dielectric, about
  4 times faster in the plain stack (L5, L6) and 1.3–1.4 times faster with shells.

**The shells.** The cases with shells are where faradice's lead is smallest, at about
4 times. They are the open performance work of M2g ([M2g](../design/M2g_shell_walk_cost.md),
#4).

## CapBench

[CapBench](https://arxiv.org/abs/2604.11202) (DAC 2026) contains 61,855 3-D windows of
ASAP7, NanGate45 and Sky130HD layouts. It is not used yet, for three reasons:

- **Terms.** Neither the repository nor the dataset page states a license for the code or
  the data.
- **Label accuracy.** The labels come from RWCap, which stops at a 0.5 % variance target for
  the total. The paper compares the labels with Raphael and reports errors of 0.4–0.6 % for
  totals and about 1.8 % for couplings. That is looser than the faradice–Palace agreement
  (≤ 0.1 % for totals in M2). As the RWCap section above shows, RWCap itself can be off by
  0.5 % with thin or conformal dielectrics.
- **Format.** The windows are cut from GDS polygons. Their storage format and dielectric
  stacks are not described in the paper.

CapBench would be useful as a test of breadth, with many realistic windows, rather than as
an accuracy reference. Using it needs a stated license and a `.cap3d` reader with polygon
support.

## FasterCap and Palace on the M2 cases

- **Palace:** every M2 case with a multi-layer dielectric (L5, L6, C4 and C5) is already
  compared with Palace in [M2](M2.md). The L5 and C4 runs above repeat those comparisons
  with the current code:
  - L5: totals within 0.18 % of the finest mesh and 0.04 % of the extrapolation.
  - C4: totals within 0.10 % and 0.04 %.
- **FasterCap:** it is not extended. The comparison models the substrate plane with odd
  images, which is exact only in a uniform dielectric. FasterCap also disagreed with both
  Palace and faradice on V5 and V8 in M1.

## Reproducing

```sh
docker build -t faradice-bench/rwcap bench/compare/docker/rwcap
for c in bench/compare/cases/{v4,v5,v8,l5,l6,c4,c5}_*.faradice.toml; do
    bench/compare/compare.py "$c" --solvers rwcap --threads 16 --confidence-sigma 1 --out /tmp/rwcap
done
```
