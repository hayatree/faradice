//! L4 (design M2a §7): parameter and bias study of the stratified-dielectric
//! walk on cases with exact answers. Long; run with
//! `cargo test --release --test l4_study -- --ignored --nocapture`.
//!
//! Every run uses a fixed number of samples (`FARADICE_L4_WALKS`, default
//! 20 M) and the seeds of `FARADICE_L4_SEEDS` (default "1 2 3 4");
//! `FARADICE_L4_VARIANTS` selects variants by name (default all). Each run
//! prints one `L4|variant|case|seed|value|std_err|exact|walks|steps|seconds`
//! line for `bench/l4/summarize.py`.

#![allow(missing_docs)]

use std::time::Instant;

use faradice_core::{Axis, EPS0_FF_PER_UM, Face, NetId, Side};
use faradice_frw::{
    Accuracy, Boundary, DielectricLayer, FaceKind, SolverConfig, StackOptions, WalkOptions, solve,
};
use faradice_geom::{Aabb, GeometryBuilder};

fn layer(z_top: f64, eps_r: f64) -> DielectricLayer {
    DielectricLayer { z_top, eps_r }
}

struct Variant {
    name: &'static str,
    stack: StackOptions,
    antithetic: bool,
}

fn variants() -> Vec<Variant> {
    let d = StackOptions::default();
    let v = |name, stack| Variant {
        name,
        stack,
        antithetic: true,
    };
    let mut out = vec![
        v("default", d.clone()),
        v(
            "K8",
            StackOptions {
                interface_offsets: 8,
                ..d.clone()
            },
        ),
        v(
            "K32_M64",
            StackOptions {
                interface_offsets: 32,
                layered_resolution: 64,
                ..d.clone()
            },
        ),
        v(
            "M64",
            StackOptions {
                layered_resolution: 64,
                ..d.clone()
            },
        ),
        v(
            "eta_quarter",
            StackOptions {
                band_fraction: 0.25,
                ..d.clone()
            },
        ),
        v(
            "alpha_0.5",
            StackOptions {
                landing_alpha: 0.5,
                ..d.clone()
            },
        ),
        v(
            "alpha_0.85",
            StackOptions {
                landing_alpha: 0.85,
                ..d.clone()
            },
        ),
    ];
    out.push(Variant {
        name: "no_antithetic",
        stack: d,
        antithetic: false,
    });
    if let Ok(sel) = std::env::var("FARADICE_L4_VARIANTS") {
        let names: Vec<&str> = sel.split_whitespace().collect();
        out.retain(|v| names.contains(&v.name));
    }
    out
}

fn env_list(name: &str, default: &str) -> Vec<u64> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

/// One exact case: geometry, boundary, substrate, stack, top permittivity
/// and the exact total capacitance of net 0.
struct Case {
    name: &'static str,
    boxes: [f64; 6],
    bounds: [f64; 6],
    grounded: Axis,
    substrate: Option<f64>,
    layers: Vec<DielectricLayer>,
    eps_top: f64,
    exact: f64,
}

fn cases() -> Vec<Case> {
    let a = 16.0 * EPS0_FF_PER_UM; // 4 × 4 plates
    let series = |d: &[(f64, f64)]| 1.0 / d.iter().map(|(t, e)| t / e).sum::<f64>();
    // Gaps of 1.0 and 0.8 to the grounded x faces.
    let lateral = |eh: f64, w: f64| EPS0_FF_PER_UM * w * eh * (1.0 + 1.0 / 0.8);
    vec![
        // L1a: interfaces 0.4 below and 0.3 above a horizontal plate.
        Case {
            name: "L1a",
            boxes: [0.0, 0.0, 1.0, 4.0, 4.0, 1.2],
            bounds: [0.0, 0.0, 0.0, 4.0, 4.0, 1.8],
            grounded: Axis::Z,
            substrate: Some(0.0),
            layers: vec![layer(0.4, 4.0), layer(1.5, 2.0)],
            eps_top: 7.0,
            exact: a * (series(&[(0.4, 4.0), (0.6, 2.0)]) + series(&[(0.3, 2.0), (0.3, 7.0)])),
        },
        // L1b: high contrast (κ = 0.1) below the plate.
        Case {
            name: "L1b",
            boxes: [0.0, 0.0, 1.0, 4.0, 4.0, 1.2],
            bounds: [0.0, 0.0, 0.0, 4.0, 4.0, 1.8],
            grounded: Axis::Z,
            substrate: Some(0.0),
            layers: vec![layer(0.5, 10.0)],
            eps_top: 1.0,
            exact: a * (series(&[(0.5, 10.0), (0.5, 1.0)]) + 1.0 / 0.6),
        },
        // L2: vertical plate across two interfaces.
        Case {
            name: "L2",
            boxes: [1.0, 0.0, 0.0, 1.2, 2.0, 2.0],
            bounds: [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            grounded: Axis::X,
            substrate: None,
            layers: vec![layer(0.7, 3.0), layer(1.2, 1.0)],
            eps_top: 6.0,
            exact: lateral(3.0 * 0.7 + 1.0 * 0.5 + 6.0 * 0.8, 2.0),
        },
        // L8: horizontal plate in a stack with two 50 nm layers.
        Case {
            name: "L8",
            boxes: [0.0, 0.0, 1.0, 4.0, 4.0, 1.2],
            bounds: [0.0, 0.0, 0.0, 4.0, 4.0, 2.0],
            grounded: Axis::Z,
            substrate: Some(0.0),
            layers: vec![
                layer(0.3, 3.9),
                layer(0.35, 7.3),
                layer(0.7, 4.05),
                layer(1.0, 4.5),
                layer(1.25, 4.2),
                layer(1.3, 7.5),
                layer(1.6, 4.1),
            ],
            eps_top: 3.0,
            exact: a
                * (series(&[(0.3, 3.9), (0.05, 7.3), (0.35, 4.05), (0.3, 4.5)])
                    + series(&[(0.05, 4.2), (0.05, 7.5), (0.3, 4.1), (0.4, 3.0)])),
        },
        // L9: vertical plate across two 50 nm high-k layers.
        Case {
            name: "L9",
            boxes: [1.0, 0.0, 0.0, 1.2, 2.0, 2.0],
            bounds: [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            grounded: Axis::X,
            substrate: None,
            layers: vec![
                layer(0.6, 3.0),
                layer(0.65, 7.5),
                layer(1.2, 1.0),
                layer(1.25, 7.3),
            ],
            eps_top: 6.0,
            exact: lateral(
                3.0 * 0.6 + 7.5 * 0.05 + 1.0 * 0.55 + 7.3 * 0.05 + 6.0 * 0.75,
                2.0,
            ),
        },
    ]
}

#[test]
#[ignore = "long: parameter study, minutes per variant"]
fn l4_parameter_study() {
    let walks = env_list("FARADICE_L4_WALKS", "20000000")[0];
    let seeds = env_list("FARADICE_L4_SEEDS", "1 2 3 4");
    for v in variants() {
        for c in cases() {
            let mut g = GeometryBuilder::new();
            let id = g.net("P").unwrap();
            g.add_box(id, Aabb::from_array(c.boxes).unwrap()).unwrap();
            let g = g.build().unwrap();
            let mut faces = [FaceKind::Reflect; 6];
            faces[Face::new(c.grounded, Side::Min).index()] = FaceKind::Ground;
            faces[Face::new(c.grounded, Side::Max).index()] = FaceKind::Ground;
            let bounds = Aabb::from_array(c.bounds).unwrap();
            for &seed in &seeds {
                let cfg = SolverConfig {
                    eps_r: c.eps_top,
                    layers: c.layers.clone(),
                    boundary: Boundary::Box { bounds, faces },
                    substrate_z: c.substrate,
                    accuracy: Accuracy {
                        total_rel_tol: 1e-9,
                        min_walks: walks,
                        max_walks: walks,
                        ..Accuracy::default()
                    },
                    walk: WalkOptions {
                        antithetic: v.antithetic,
                        stack: v.stack.clone(),
                        ..WalkOptions::default()
                    },
                    seed,
                    ground_nets: Vec::new(),
                    regions: Default::default(),
                };
                let t0 = Instant::now();
                let sol = solve(&g, &[NetId::new(0)], &cfg).unwrap();
                let secs = t0.elapsed().as_secs_f64();
                let r = &sol.nets[0];
                let est = r.total();
                #[allow(clippy::cast_precision_loss)]
                let steps = r.total_steps as f64 / r.walks as f64;
                println!(
                    "L4|{}|{}|{seed}|{:.9}|{:.9}|{:.9}|{}|{steps:.2}|{secs:.2}",
                    v.name, c.name, est.value, est.std_err, c.exact, r.walks
                );
            }
        }
    }
}
