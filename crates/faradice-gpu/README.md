# faradice-gpu

Experimental GPU walk engine for [faradice](https://github.com/hayatree/faradice),
built on [wgpu](https://wgpu.rs/) compute shaders (Vulkan on Linux).

The CPU keeps the launch phase (Gaussian surface sampling, first hop and
weights, in `f64`); the GPU runs the transition walks in `f32` with
coordinates relative to the scene centre. Results agree with the CPU solver
statistically, not bit for bit. See `docs/design/G1_gpu.md`.

## Status and limits

- **Prototype.** Uniform dielectric only: structures with stratified stacks, dielectric
  boxes or conformal shells are rejected with an error (use the CPU engine).
- Needs Rust 1.87 or newer (wgpu) and a Vulkan adapter. Without one, `GpuEngine::new`
  returns an error, and the tests skip themselves with a message.
- Results are statistically equivalent to the CPU engine, not bit-identical.

## Use

```sh
cargo build --release -p faradice --features gpu
./target/release/faradice solve examples/three_wires.faradice.toml --engine gpu -o out
cargo test -p faradice-gpu          # agreement tests against the CPU engine
```
