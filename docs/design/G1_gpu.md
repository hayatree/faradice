# G1 — GPU walk engine (experimental)

Status: Implemented as an experimental prototype (crate `faradice-gpu`, CLI feature `gpu`,
`--engine gpu`; homogeneous dielectric only, stacks and dielectric boxes need the CPU engine).
The next steps of §6 are proposed.

## 1. Split of work

Every FRW sample has two phases:

| Phase | Where | Precision | Why |
|---|---|---|---|
| Launch: Gaussian-surface sample, first hop (gradient kernel), weight | CPU (rayon) | `f64` | weights carry the estimator's sign and scale; exactness matters, cost is one hop per walk |
| Transition: hops until a sink is within `eps_stop` | `WalkEngine` (CPU or GPU) | `f64` / `f32` | where almost all hops are spent |

`faradice_frw::solve_with(geom, targets, cfg, make_engine)` runs the same
estimator, stopping rules and batch order as `solve`, but delegates the
transition phase to any `WalkEngine`. `CpuEngine` is the reference
implementation; `faradice_gpu::GpuEngine` runs the walk in a WGSL compute
shader via wgpu (Vulkan).

### Reproducibility

Each launch carries a 64-bit key drawn from the batch's launch stream. Engines
derive an independent per-walk random stream from it, so a walk's outcome does
not depend on how launches are grouped into calls or dispatches. Batches are
merged strictly in batch order, so results are bit-identical for a given seed
*and engine* regardless of wave size (tested), and agree with `solve`
statistically. The GPU is deterministic on a given device/driver; across
devices only statistical agreement is promised (WGSL allows reassociation and
fused multiply-add).

## 2. GPU kernel

`crates/faradice-gpu/src/walk.wgsl` ports `Walker::run` and the uniform-grid
queries (`clearance_lower_bound`, `nearest_linf` with the near-cell cache and
ring search, `block_reach`), the boundary model (substrate plane, grounded and
reflecting box faces, open boundary with the exterior-sphere jump and odd image
across the substrate), and the transition sampler (alias table over face cells,
uniform within a cell). The RNG is xoshiro128++ seeded by hashing the launch key.

Buffers: one uniform block of parameters, one `vec4<f32>` box buffer, one `u32`
buffer holding every index array (grid CSR, clearance transform, near-cell
lists, alias table, box→net map), the persistent walk state, compact launch
input (20 B/walk) and compact outcomes (8 B/walk). All buffers are reused
across calls.

Dispatches are bounded (`steps_per_dispatch`, default 256 hops): after each
dispatch the host reads an atomic count of running walks, so no GPU job is long
enough to trigger a driver watchdog, and the loop terminates after
`max_steps / steps_per_dispatch + 2` rounds at the latest (walks reaching
`max_steps` are reported as aborted, like on the CPU).

### Precision

Coordinates are rebased to the centre of the conductors' bounding box and
converted to `f32`. Positions and distances are then accurate to about one ulp
of the scene scale `L`, which perturbs hop sizes negligibly as long as
`eps_stop ≫ ulp(L)`. `GpuEngine::new` refuses scenes with
`eps_stop < 32 · L · 2⁻²³` (`GpuError::Precision`); the CPU engine must be used
there. For the default `stop_ratio = 1e-3` and 0.14 µm wires this allows
scenes up to roughly 70 µm across. Tiled local frames with `f64` continuation
near conductors are the next step for full-chip use (§6, item 3).

## 3. Pipelining

`run_net_engine` overlaps three things per wave of ~2²⁰ walks: the engine runs
wave *k* on a scoped thread while the CPU plans wave *k + 1* and computes the
statistics of wave *k − 1*. Convergence is checked when wave *k − 1* is merged,
so a converged run discards the wave in flight (deterministic, a few ms of
wasted work per net).

## 4. Results (AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU, RADV)

`reference` preset, wall time of `faradice solve` (best of a few runs; noisy):

| Case | Walks | CPU, 32 threads | GPU engine | GPU CPU-time |
|---|---|---|---|---|
| `examples/bus.gds` (4 nets) | 78 M | 6.3 s (171 s CPU) | 4.1 s | 29 s |
| 3-layer 48+48+24 wire grid (3 nets) | 34 M | 2.6 s (66 s CPU) | 2.0 s | 13 s |

Many small nets are the opposite case: all 120 nets of the grid at the
`default` preset take 1.1 s on the CPU but 3.5 s on the GPU, because the engine
path solves targets one after another and each net needs only a few short
waves (waves start at `min_walks` and double up to ~2²⁰ launches, so small nets
no longer pay for a full wave, but per-wave latency remains). Use the GPU for
few nets at high accuracy until multi-target dispatch (§6, item 4) lands.

Per wave of 1 M launches: kernel ≈ 9 ms, input packing ≈ 4 ms, upload/readback
≈ 2 ms, launch planning ≈ 7 ms (overlapped). Walks here average only ~15 hops,
so the CPU launch phase and transfers are a large share; the GPU advantage
grows with longer walks. Agreement with the CPU is checked statistically
(`crates/faradice-gpu/tests/agreement.rs`: parallel plate vs exact, three wires
and an open boundary with substrate vs CPU within 4.5σ, steps/walk within 5 %).

## 5. SIMD (CPU) findings

Explicit data-layout changes did not pay off on the CPU path (see
`docs/validation/M1.md`, performance section): an SoA layout of per-registration
coordinates was 0.83× (slower), and `-C target-cpu=native` (AVX-512) gave
< 5 %. The hot loop is dominated by indirect, branchy grid lookups rather than
arithmetic; the near-cell candidate cache of the uniform grid gave the large win instead.
Batched walk-front SIMD (one lane per walk, as on the GPU) is the remaining
CPU option and would reuse the GPU kernel's structure.

## 6. Next steps

1. Launch phase on the GPU (Gaussian surface sampling and first hop) to remove
   the per-walk CPU cost and most transfers.
2. Walk compaction for the long tail (currently finished lanes idle until the
   round ends; rounds are short, so the effect is small).
3. Tiled local coordinates + `f64` CPU continuation for large layouts.
4. Multiple targets per dispatch, so small nets fill the device.
5. `run.engine` in the config file and the engine name in `result.json`.
