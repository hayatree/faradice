//! Golden results: small fixed-sample runs whose statistics must match the
//! stored files exactly.
//!
//! faradice is bit-reproducible for a given seed, so any change to the
//! numerics (sampling, tables, walk rules, estimators) changes these values,
//! while refactors and optimizations that keep every walk the same do not.
//! The runs are far too short to say anything about accuracy; that is the job
//! of the statistical validation tests (`faradice-frw/tests/`, run in full on
//! `main`).
//!
//! When a change is *meant* to alter the numerics, run the full validation
//! (`cargo test --workspace --all-features`), then refresh the files with
//! `FARADICE_BLESS=1 cargo test -p faradice-engine --test golden` and commit
//! them together with the change.

#![allow(missing_docs)]

use std::path::PathBuf;

use faradice_engine::{Project, RunRequest, run};
use faradice_io::config::Config;

/// Walks per net: enough to visit every walk rule of the cases, small enough
/// for a few seconds per case on four cores.
const WALKS: u64 = 8192;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn config() -> Config {
    let text = format!(
        "schema_version = 1\n\
         [accuracy]\n\
         total_rel_tol = 1e-9\n\
         coupling_rel_tol = 1e-9\n\
         min_walks = {WALKS}\n\
         max_walks = {WALKS}\n\
         batch_walks = 1024\n\
         [run]\n\
         seed = 7\n"
    );
    toml::from_str(&text).unwrap()
}

/// Solves `case` and compares its per-target statistics with
/// `tests/golden/<name>.json`.
fn check(name: &str, case: &str) {
    let project = Project::load(&root().join(case), None, None).unwrap();
    let out = run(
        &project,
        &RunRequest {
            config: config(),
            ..RunRequest::default()
        },
        &(),
    )
    .unwrap();
    // `serde_json` writes the shortest representation that reads back to the
    // same `f64`, so equal text means bit-identical values.
    let got = serde_json::to_string_pretty(&serde_json::json!({
        "eps_stop": out.result.eps_stop,
        "targets": out.result.targets,
    }))
    .unwrap()
        + "\n";
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/golden/{name}.json"));
    if std::env::var_os("FARADICE_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}; create it with FARADICE_BLESS=1", path.display()));
    if got != want {
        let diff = got
            .lines()
            .zip(want.lines())
            .enumerate()
            .find(|(_, (g, w))| g != w)
            .map_or_else(String::new, |(i, (g, w))| {
                format!(
                    "first difference at line {}:\n  got:  {g}\n  want: {w}\n",
                    i + 1
                )
            });
        panic!(
            "{name}: the statistics differ from {}.\n{diff}\
             If the numerics were meant to change, run the full validation and refresh the \
             golden files (see the header of tests/golden.rs).",
            path.display()
        );
    }
}

/// Uniform dielectric: three wires and a crossing wire over a substrate.
#[test]
fn uniform_dielectric() {
    check(
        "v4_three_wires",
        "bench/compare/cases/v4_three_wires.faradice.toml",
    );
}

/// Stratified stack: wires inside a layered dielectric.
#[test]
fn stratified_stack() {
    check(
        "l5_wires_in_stack",
        "bench/compare/cases/l5_wires_in_stack.faradice.toml",
    );
}

/// Conformal shells in a SKY130-like stack: dielectric boxes, edge cubes and
/// window first hops.
#[test]
fn conformal_shells() {
    check(
        "c4_met1_pair_conformal",
        "bench/compare/cases/c4_met1_pair_conformal.faradice.toml",
    );
}
