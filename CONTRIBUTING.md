# Contributing to faradice

Thank you for your interest in faradice!

## Ground rules

- **Be respectful and constructive** in issues, reviews and discussions.
- **Clean-room implementation.** Algorithms must come from published papers or your own
  work. Do not contribute code, data or wording derived from proprietary EDA tools, their
  manuals or outputs, or from non-redistributable PDKs. Cite the literature you rely on in
  the code and in `docs/`.
- **Test data** must be self-made or derived from openly licensed sources (e.g. sky130 is
  Apache-2.0). State the source and license of any data file you add.
- **Dependencies** must use permissive licenses; CI enforces this with `cargo deny`
  (see `deny.toml`).
- **No `unsafe`.** Every crate forbids it (`unsafe_code = "forbid"` in the workspace
  lints). If a change truly needs it, open an issue first to discuss changing that policy.

## Workflow

1. Open an issue for anything larger than a small fix, so the design can be discussed first.
2. Keep pull requests focused; one logical change per PR.
3. Before pushing, run:

   ```sh
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
   cargo test --workspace --all-features --locked
   RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
   cargo deny check                                       # needs cargo-deny
   cargo +1.85 check --workspace --exclude faradice-gpu --locked   # MSRV
   ```

   `--workspace` and `--all-features` include the GPU engine (Rust 1.87). Its tests skip
   themselves when no Vulkan adapter is found. A few long statistical tests are
   `#[ignore]`d; run them with `cargo test --release -- --ignored` when you change the
   solver's numerics.

   If you touch the desktop GUI, also run its checks (a separate workspace in `gui/`):

   ```sh
   cd gui
   vendor/fetch.sh                  # fetches and patches two Wayland crates
   python3 tools/license_audit.py
   cargo fmt --all --check
   cargo clippy --all-targets --all-features --locked -- -D warnings
   cargo test --all-features --locked
   cargo deny --all-features check
   ```

   On a pull request, CI checks what the change can affect: the tests of the changed
   crates and of the crates that depend on them, without the long statistical validation
   tests of `faradice-frw`, and the GUI jobs only when the GUI or a crate it uses changed.
   `main` and a weekly run check everything. Add the label `full-validation` to a pull
   request to run everything on it.

4. **Golden tests** (`crates/faradice-engine/tests/golden.rs`) compare short fixed-seed
   runs with stored results, bit for bit. A change that keeps every walk the same (a
   refactor, most optimizations) passes them unchanged. If your change is meant to alter
   the numerics, run the full tests (`cargo test --workspace --all-features`, or label the
   pull request `full-validation`), then refresh the stored results with
   `FARADICE_BLESS=1 cargo test -p faradice-engine --test golden` and commit them with the
   change.

5. Add or update tests. Numerical code should be checked against analytical solutions or
   independent references, with tolerances justified by the statistical error.
6. Update `CHANGELOG.md` under "Unreleased" for user-visible changes.

## Licensing of contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in faradice by you, as defined in the Apache-2.0 license, shall be dual licensed
under MIT OR Apache-2.0, without any additional terms or conditions.
