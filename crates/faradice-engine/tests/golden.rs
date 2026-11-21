//! Golden results: small fixed-sample runs whose statistics must match the
//! stored files.
//!
//! faradice is reproducible for a given seed, so any change to the numerics
//! (sampling, tables, walk rules, estimators) changes these values, while
//! refactors and optimizations that keep every walk the same do not. Counts
//! must match exactly and sums to a relative 1e-9: the math library of another
//! machine can round a transcendental function differently in the last bit,
//! which moves a sum by about 1e-16, while a single changed walk among the
//! 8192 of a net moves it by about 1e-4.
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
use serde_json::Value;

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
    let got: Value = serde_json::from_str(&got).unwrap();
    let want: Value = serde_json::from_str(&want).unwrap();
    if let Some(diff) = first_difference(&got, &want, String::new()) {
        panic!(
            "{name}: the statistics differ from {}.\n{diff}\n\
             If the numerics were meant to change, run the full validation and refresh the \
             golden files (see the header of tests/golden.rs).",
            path.display()
        );
    }
}

/// Relative tolerance of floating-point values (see the module header).
const REL_TOL: f64 = 1e-9;

/// The first place where `got` and `want` differ: integers, strings and
/// structure exactly, other numbers to [`REL_TOL`].
fn first_difference(got: &Value, want: &Value, at: String) -> Option<String> {
    let differ = || Some(format!("at {at}: got {got}, want {want}"));
    match (got, want) {
        (Value::Number(g), Value::Number(w)) if g.is_f64() || w.is_f64() => {
            let (g, w) = (g.as_f64().unwrap(), w.as_f64().unwrap());
            if (g - w).abs() <= REL_TOL * g.abs().max(w.abs()) {
                None
            } else {
                differ()
            }
        }
        (Value::Array(g), Value::Array(w)) if g.len() == w.len() => g
            .iter()
            .zip(w)
            .enumerate()
            .find_map(|(i, (g, w))| first_difference(g, w, format!("{at}[{i}]"))),
        (Value::Object(g), Value::Object(w))
            if g.len() == w.len() && g.keys().all(|k| w.contains_key(k)) =>
        {
            g.iter()
                .find_map(|(k, v)| first_difference(v, &w[k], format!("{at}.{k}")))
        }
        _ if got == want => None,
        _ => differ(),
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
