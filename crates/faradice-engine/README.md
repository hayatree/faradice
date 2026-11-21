# faradice-engine

The run engine of [faradice](https://github.com/hayatree/faradice), shared by the CLI and
the desktop GUI.

- `Project`: a loaded input (a structure file, or a GDSII file traced with a technology
  file) that can be solved many times with different configurations.
- `run`: one blocking extraction with progress reports and cancellation.
- `Job`: the same run on a background thread, reporting events to a callback and
  cancellable at any time.

Feature `gpu` enables the experimental GPU walk engine (`faradice-gpu`, Rust 1.87+).
See the crate documentation (`cargo doc -p faradice-engine --open`).
