# Benchmarks and studies

| Path | What it measures | How to run |
|---|---|---|
| [`compare/`](compare/README.md) | Accuracy against two independent open-source field solvers (FasterCap BEM, Palace FEM), run in Docker | see [`compare/README.md`](compare/README.md) |
| `l4/run_cli.sh` | Cost of the stratified-dielectric walk on real stacks (cases L5 and L6) for several tuning parameters: fixed samples per net, one CLI run per variant | `cargo build --release && bench/l4/run_cli.sh OUTDIR [WALKS]` |
| `l4/summarize.py` | Summary of the bias and efficiency study in `crates/faradice-frw/tests/l4_study.rs` (cases with exact answers) | `cargo test --release -p faradice-frw --test l4_study -- --ignored --nocapture > l4.log`, then `bench/l4/summarize.py l4.log` |
| `crates/faradice-geom/benches/` | Criterion microbenchmarks of the spatial queries | `cargo bench -p faradice-geom` |

The results and their interpretation are in [`docs/validation/`](../docs/validation/). The
L4 study takes hours at its default sample counts; see the header of `l4_study.rs` for the
environment variables that shorten it.
