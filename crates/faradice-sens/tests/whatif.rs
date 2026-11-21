//! Pillar C validation (design D3 §9: S5, S10).

#![allow(missing_docs)]

use faradice_core::{Axis, EPS0_FF_PER_UM, Face, NetId, Side};
use faradice_frw::cross::solve_with_cross_moments;
use faradice_frw::{Accuracy, Boundary, FaceKind, SolverConfig, WalkOptions};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};
use faradice_io::result::ResultFile;
use faradice_sens::whatif::{Edit, Value, WhatIf, apply};

fn geom(boxes: &[(&str, [f64; 6])]) -> Geometry {
    let mut g = GeometryBuilder::new();
    for (net, c) in boxes {
        let id = g.net(net).unwrap();
        g.add_box(id, Aabb::from_array(*c).unwrap()).unwrap();
    }
    g.build().unwrap()
}

fn cfg(
    boundary: Boundary,
    substrate_z: Option<f64>,
    eps_r: f64,
    walks: u64,
    seed: u64,
) -> SolverConfig {
    SolverConfig {
        eps_r,
        layers: Vec::new(),
        boundary,
        substrate_z,
        accuracy: Accuracy {
            min_walks: walks,
            max_walks: walks,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed,
        ground_nets: Vec::new(),
        regions: Default::default(),
    }
}

/// Solves `targets` with cross moments and returns the `result.json` contents.
fn result(g: &Geometry, targets: &[&str], c: &SolverConfig) -> ResultFile {
    let ids: Vec<NetId> = targets.iter().map(|t| g.net_id(t).unwrap()).collect();
    let (sol, cross) = solve_with_cross_moments(g, &ids, c, &()).unwrap();
    let mut r = ResultFile::from_solution(
        g,
        &sol,
        c.seed,
        Vec::new(),
        0.0,
        Vec::new(),
        "[output]\ncross_moments = true\n".into(),
    );
    r.set_cross(&cross);
    r
}

fn net<'a>(w: &'a WhatIf, name: &str) -> &'a faradice_sens::whatif::WhatIfNet {
    w.nets.iter().find(|n| n.net == name).unwrap()
}

fn coupling(w: &WhatIf, a: &str, b: &str) -> Value {
    net(w, a).couplings.iter().find(|c| c.net == b).unwrap().c
}

fn check(name: &str, v: Value, exact: f64, exact_sigma: f64, max_rel_sigma: f64) {
    let sigma = v.std_err.hypot(exact_sigma);
    eprintln!(
        "{name}: {:.6} ± {:.6} vs {exact:.6} ± {exact_sigma:.6} ({:+.2}σ)",
        v.value,
        v.std_err,
        (v.value - exact) / sigma
    );
    assert!((v.value - exact).abs() <= 4.0 * sigma, "{name}");
    assert!(sigma <= max_rel_sigma * exact.abs(), "{name}: σ too large");
}

/// S10: three plates between grounded top and bottom with reflecting sides
/// (exactly one-dimensional). Floating the middle plate puts its two gaps in
/// series.
#[test]
fn s10_floating_middle_plate() {
    let (l, t) = (4.0, 0.2);
    let z = [1.0, 1.8, 2.6]; // plate bottoms; gaps 1.0, 0.6, 0.6, 1.0
    let h = z[2] + t + 1.0;
    let g = geom(&[
        ("p1", [0.0, 0.0, z[0], l, l, z[0] + t]),
        ("m", [0.0, 0.0, z[1], l, l, z[1] + t]),
        ("p2", [0.0, 0.0, z[2], l, l, z[2] + t]),
    ]);
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, h]).unwrap();
    let c = cfg(Boundary::Box { bounds, faces }, None, 1.0, 400_000, 1);
    let r = result(&g, &["p1", "m", "p2"], &c);
    let w = apply(&r, &[Edit::Float { net: "m".into() }]).unwrap();
    assert!(w.warnings.is_empty(), "{:?}", w.warnings);
    let cap = |d: f64| EPS0_FF_PER_UM * l * l / d;
    let (c0, c1, c2, c3) = (
        cap(z[0]),
        cap(z[1] - z[0] - t),
        cap(z[2] - z[1] - t),
        cap(h - z[2] - t),
    );
    let series = c1 * c2 / (c1 + c2);
    check("S10 C'_p1p2", coupling(&w, "p1", "p2"), series, 0.0, 0.02);
    check("S10 C'_p1p1", net(&w, "p1").total, c0 + series, 0.0, 0.02);
    check("S10 C'_p2p2", net(&w, "p2").total, c3 + series, 0.0, 0.02);
    check("S10 ground p1", net(&w, "p1").ground, c0, 0.0, 0.05);
}

/// S5: grounding the crossing wire of the three-wire example is the same as
/// a re-solve with the crossing wire as a grounded, non-target conductor.
#[test]
fn s5_ground_matches_resolve() {
    let g = geom(&[
        ("left", [0.0, 0.00, 1.0, 10.0, 0.14, 1.36]),
        ("victim", [0.0, 0.28, 1.0, 10.0, 0.42, 1.36]),
        ("right", [0.0, 0.56, 1.0, 10.0, 0.70, 1.36]),
        ("cross", [4.93, -2.0, 1.76, 5.07, 2.7, 2.12]),
    ]);
    let three = ["left", "victim", "right"];
    let all = ["left", "victim", "right", "cross"];
    let ra = result(&g, &all, &cfg(Boundary::Open, Some(0.0), 3.9, 300_000, 1));
    let rb = result(&g, &three, &cfg(Boundary::Open, Some(0.0), 3.9, 300_000, 2));
    let w = apply(
        &ra,
        &[Edit::Ground {
            net: "cross".into(),
        }],
    )
    .unwrap();
    let re = apply(&rb, &[]).unwrap();
    assert_eq!(w.nets.len(), 3);
    for n in three {
        let (a, b) = (net(&w, n), net(&re, n));
        check(
            &format!("S5 total {n}"),
            a.total,
            b.total.value,
            b.total.std_err,
            0.01,
        );
        check(
            &format!("S5 ground {n}"),
            a.ground,
            b.ground.value,
            b.ground.std_err,
            0.05,
        );
    }
    for (a, b) in [("left", "victim"), ("victim", "right"), ("left", "right")] {
        let r = coupling(&re, a, b);
        check(
            &format!("S5 C_{a},{b}"),
            coupling(&w, a, b),
            r.value,
            r.std_err,
            0.1,
        );
    }
}

/// S13 for pillar C: the delta-method σ is calibrated. Over 50 independent
/// seed pairs of S5 (grounded crossing wire vs re-solve), and of floating the
/// crossing wire (vs the mean over all pairs), z-scores of every reported
/// quantity must fall within 1.645σ about 90 % of the time.
#[test]
#[ignore = "statistical coverage, ~2 CPU-minutes; run with --ignored"]
fn s13_whatif_sigma_coverage() {
    let g = geom(&[
        ("left", [0.0, 0.00, 1.0, 10.0, 0.14, 1.36]),
        ("victim", [0.0, 0.28, 1.0, 10.0, 0.42, 1.36]),
        ("right", [0.0, 0.56, 1.0, 10.0, 0.70, 1.36]),
        ("cross", [4.93, -2.0, 1.76, 5.07, 2.7, 2.12]),
    ]);
    let three = ["left", "victim", "right"];
    let all = ["left", "victim", "right", "cross"];
    let walks = 40_000;
    let flat = |w: &WhatIf| -> Vec<Value> {
        let mut v = Vec::new();
        for n in three {
            v.push(net(w, n).total);
            v.push(net(w, n).ground);
        }
        for (a, b) in [("left", "victim"), ("victim", "right"), ("left", "right")] {
            v.push(coupling(w, a, b));
        }
        v
    };
    let (mut inside, mut count) = (0, 0);
    let mut floats: Vec<Vec<Value>> = Vec::new();
    for k in 0..50u64 {
        let ra = result(
            &g,
            &all,
            &cfg(Boundary::Open, Some(0.0), 3.9, walks, 1000 + 2 * k),
        );
        let rb = result(
            &g,
            &three,
            &cfg(Boundary::Open, Some(0.0), 3.9, walks, 1001 + 2 * k),
        );
        let a = flat(
            &apply(
                &ra,
                &[Edit::Ground {
                    net: "cross".into(),
                }],
            )
            .unwrap(),
        );
        let b = flat(&apply(&rb, &[]).unwrap());
        for (x, y) in a.iter().zip(&b) {
            let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
            inside += usize::from(z.abs() <= 1.645);
            count += 1;
        }
        floats.push(flat(
            &apply(
                &ra,
                &[Edit::Float {
                    net: "cross".into(),
                }],
            )
            .unwrap(),
        ));
    }
    let ground_cov =
        f64::from(u32::try_from(inside).unwrap()) / f64::from(u32::try_from(count).unwrap());
    // Float: compare each run with the mean of the others (σ of that mean
    // from the spread is small, but include it).
    let (mut fin, mut fcount) = (0, 0);
    for q in 0..floats[0].len() {
        let vals: Vec<f64> = floats.iter().map(|f| f[q].value).collect();
        for (i, f) in floats.iter().enumerate() {
            let others: Vec<f64> = vals
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .collect();
            #[allow(clippy::cast_precision_loss)]
            let m = others.iter().sum::<f64>() / others.len() as f64;
            #[allow(clippy::cast_precision_loss)]
            let se = f[q].std_err / (others.len() as f64).sqrt();
            let z = (f[q].value - m) / f[q].std_err.hypot(se);
            fin += usize::from(z.abs() <= 1.645);
            fcount += 1;
        }
    }
    let float_cov =
        f64::from(u32::try_from(fin).unwrap()) / f64::from(u32::try_from(fcount).unwrap());
    eprintln!(
        "coverage at 1.645σ: ground {ground_cov:.3} ({count}), float {float_cov:.3} ({fcount})"
    );
    for c in [ground_cov, float_cov] {
        assert!((0.85..=0.95).contains(&c), "coverage {c}");
    }
}
