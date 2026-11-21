//! Sensitivity validation (design D3 §9: S1–S3, nominal identity).

#![cfg(feature = "sensitivity")]
#![allow(missing_docs)]

use faradice_core::{Axis, BoxId, EPS0_FF_PER_UM, Face, NetId, Side, Sink};
use faradice_frw::sens::{FaceStep, Param, SensOptions, solve_sensitivity};
use faradice_frw::{Accuracy, Boundary, FaceKind, SolverConfig, WalkOptions, solve};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};

fn geom(boxes: &[(&str, [f64; 6])]) -> Geometry {
    let mut g = GeometryBuilder::new();
    for (net, c) in boxes {
        let id = g.net(net).unwrap();
        g.add_box(id, Aabb::from_array(*c).unwrap()).unwrap();
    }
    g.build().unwrap()
}

fn cfg(boundary: Boundary, substrate_z: Option<f64>, walks: u64) -> SolverConfig {
    SolverConfig {
        eps_r: 1.0,
        boundary,
        substrate_z,
        accuracy: Accuracy {
            min_walks: walks,
            max_walks: walks,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed: 7,
        ground_nets: Vec::new(),
        layers: Vec::new(),
        regions: Default::default(),
    }
}

fn assert_close(name: &str, est: faradice_core::Estimate, exact: f64, bias_rel: f64) {
    let err = (est.value - exact).abs();
    let allowed = 4.0 * est.std_err + bias_rel * exact.abs();
    eprintln!(
        "{name}: {:.6} ± {:.6} (exact {exact:.6}, err {:.2}%, σ {:.2}%)",
        est.value,
        est.std_err,
        100.0 * err / exact.abs(),
        100.0 * est.std_err / exact.abs()
    );
    assert!(err <= allowed, "{name}: {est:?} vs exact {exact}");
}

/// Like [`assert_close`], and the estimate must be sharp enough to mean
/// something: σ below `max_rel_sigma` of the exact value.
fn assert_sharp(
    name: &str,
    est: faradice_core::Estimate,
    exact: f64,
    bias: f64,
    max_rel_sigma: f64,
) {
    assert_close(name, est, exact, bias);
    assert!(
        est.std_err <= max_rel_sigma * exact.abs(),
        "{name}: σ {} too large for a meaningful check",
        est.std_err
    );
}

const fn face(axis: Axis, side: Side) -> Face {
    Face::new(axis, side)
}

/// The V1 plate: substrate at z = 0, plate [d1, d1 + t], grounded top at
/// d1 + t + d2, reflecting sides, so C = εA(1/d1 + 1/d2) exactly.
fn plate() -> (Geometry, SolverConfig, f64, f64, f64) {
    let (l, d1, t, d2) = (4.0, 1.0, 0.2, 0.6);
    let g = geom(&[("P", [0.0, 0.0, d1, l, l, d1 + t])]);
    let mut faces = [FaceKind::Reflect; 6];
    faces[face(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[face(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, d1 + t + d2]).unwrap();
    (
        g,
        cfg(Boundary::Box { bounds, faces }, Some(0.0), 1_600_000),
        l * l,
        d1,
        d2,
    )
}

/// S1 + S2: moving the bottom (top) face of the plate changes d1 (d2).
#[test]
fn s1_s2_parallel_plate_gaps() {
    let (g, c, a, d1, d2) = plate();
    let eps = EPS0_FF_PER_UM;
    // 20 % of the plate thickness, ≤ 7 % of either gap: the plate's C(d) is
    // smooth, and Richardson removes the O(δ/d) bias of the step.
    let delta = 0.04;
    let steps = [
        FaceStep {
            box_id: BoxId::new(0),
            face: face(Axis::Z, Side::Min),
            delta,
            keep: None,
        },
        FaceStep {
            box_id: BoxId::new(0),
            face: face(Axis::Z, Side::Max),
            delta,
            keep: None,
        },
    ];
    let sol = solve_sensitivity(
        &g,
        &[NetId::new(0)],
        &c,
        &steps,
        &[],
        SensOptions::default(),
    )
    .unwrap();
    let r = &sol.nets[0];
    let n = r.nominal.walks;
    let p = Sink::Net(NetId::new(0));
    let top = Sink::Boundary(face(Axis::Z, Side::Max));
    let (bottom_face, top_face) = (&r.faces[0], &r.faces[1]);
    eprintln!(
        "continued: bottom {} top {} of {n}",
        bottom_face.continued_walks, top_face.continued_walks
    );
    // Outward displacement of the bottom face shrinks d1: dC/dn = εA/d1².
    assert_sharp(
        "S1 dC_PP",
        bottom_face.dc(p, n),
        eps * a / (d1 * d1),
        0.01,
        0.05,
    );
    assert!(bottom_face.converged() && top_face.converged());
    // One-sided and Richardson share their samples; they must agree.
    let (one, rich) = (bottom_face.dc_one_sided(p, n), bottom_face.dc(p, n));
    assert!(
        (one.value - rich.value).abs() <= 4.0 * rich.std_err,
        "{one:?} vs {rich:?}"
    );
    assert_close(
        "S1 dC_P,sub",
        bottom_face.dc(Sink::GroundPlane, n),
        -eps * a / (d1 * d1),
        0.01,
    );
    // The top gap is untouched by the bottom face.
    let e = bottom_face.dc(top, n);
    assert!(e.value.abs() <= 4.0 * e.std_err + 1e-12, "{e:?}");
    assert_sharp(
        "S2 dC_PP",
        top_face.dc(p, n),
        eps * a / (d2 * d2),
        0.01,
        0.05,
    );
    assert_close(
        "S2 dC_P,top",
        top_face.dc(top, n),
        -eps * a / (d2 * d2),
        0.01,
    );
}

/// S3: the free-space cube. Growing all six faces by dn grows the side by
/// 2·dn and C ∝ a, so Σ_faces dC/dn = 2C/a; by symmetry each face is C/(3a).
#[test]
fn s3_free_space_cube() {
    let g = geom(&[("C", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]);
    let c = cfg(Boundary::Open, None, 800_000);
    let steps: Vec<FaceStep> = Face::ALL
        .iter()
        .map(|&f| FaceStep {
            box_id: BoxId::new(0),
            face: f,
            delta: 0.1,
            keep: None,
        })
        .collect();
    // Uniform growth as one parameter: Σ_f dC/dn_f = 2C/a.
    let grow = Param {
        faces: (0..6).map(|i| (i, 1.0)).collect(),
    };
    let sol = solve_sensitivity(
        &g,
        &[NetId::new(0)],
        &c,
        &steps,
        &[grow],
        SensOptions::default(),
    )
    .unwrap();
    let r = &sol.nets[0];
    let n = r.nominal.walks;
    let cap = 0.660_678_15 * 4.0 * std::f64::consts::PI * EPS0_FF_PER_UM;
    let s = Sink::Net(NetId::new(0));
    for (i, f) in r.faces.iter().enumerate() {
        assert_sharp(&format!("S3 face {i}"), f.dc(s, n), cap / 3.0, 0.02, 0.1);
    }
    assert_sharp("S3 growth", r.params[0].dc(s, n), 2.0 * cap, 0.02, 0.05);
}

/// The nominal statistics are exactly those of a plain solve.
#[test]
fn nominal_is_bit_identical_to_solve() {
    let g = geom(&[
        ("A", [0.0, 0.0, 1.0, 5.0, 0.3, 1.3]),
        ("B", [0.0, 0.6, 1.0, 5.0, 0.9, 1.3]),
    ]);
    let bounds = Aabb::from_array([-3.0, -3.0, 0.0, 8.0, 4.0, 4.0]).unwrap();
    let mut c = cfg(Boundary::grounded_box(bounds), Some(0.0), 0);
    c.accuracy = Accuracy {
        min_walks: 20_000,
        ..Accuracy::default()
    };
    let targets = [NetId::new(0), NetId::new(1)];
    let plain = solve(&g, &targets, &c).unwrap();
    let steps = [FaceStep {
        box_id: BoxId::new(1),
        face: face(Axis::Y, Side::Min),
        delta: 0.006,
        keep: None,
    }];
    let sens = solve_sensitivity(&g, &targets, &c, &steps, &[], SensOptions::default()).unwrap();
    for (a, b) in plain.nets.iter().zip(&sens.nets) {
        let b = &b.nominal;
        assert_eq!(a.walks, b.walks);
        assert_eq!(a.total_steps, b.total_steps);
        assert_eq!(a.sinks.len(), b.sinks.len());
        for ((sa, ma), (sb, mb)) in a.sinks.iter().zip(&b.sinks) {
            assert_eq!(sa, sb);
            assert_eq!(ma.sum().to_bits(), mb.sum().to_bits());
            assert_eq!(ma.sum_sq().to_bits(), mb.sum_sq().to_bits());
        }
    }
    // Moving B's lower face outward narrows the gap and strengthens the
    // coupling, so the (negative) Maxwell entry C_AB decreases.
    let r = &sens.nets[0];
    let d = r.faces[0].dc(Sink::Net(NetId::new(1)), r.nominal.walks);
    eprintln!("dC_AB/dn = {d:?}");
    assert!(d.value < -3.0 * d.std_err, "{d:?}");
}

/// Face results do not depend on the thread count, on the order of the
/// requested faces or on which other faces are requested.
#[test]
fn deterministic_across_threads_and_face_order() {
    let g = geom(&[
        ("A", [0.0, 0.0, 1.0, 5.0, 0.3, 1.3]),
        ("B", [0.0, 0.6, 1.0, 5.0, 0.9, 1.3]),
    ]);
    // Open boundary with a substrate: exercises the odd-image signs.
    let c = cfg(Boundary::Open, Some(0.0), 30_000);
    let f0 = FaceStep {
        box_id: BoxId::new(1),
        face: face(Axis::Y, Side::Min),
        delta: 0.006,
        keep: None,
    };
    let f1 = FaceStep {
        box_id: BoxId::new(0),
        face: face(Axis::Z, Side::Min),
        delta: 0.006,
        keep: None,
    };
    let run = |threads: usize, steps: &[FaceStep]| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                solve_sensitivity(&g, &[NetId::new(0)], &c, steps, &[], SensOptions::default())
                    .unwrap()
            })
    };
    let a = run(1, &[f0, f1]);
    let b = run(4, &[f1, f0]);
    let only = run(3, &[f0]);
    let key = |r: &faradice_frw::sens::FaceResult| {
        r.sinks
            .iter()
            .map(|(s, m)| (*s, m.sum().to_bits(), m.sum_sq().to_bits()))
            .collect::<Vec<_>>()
    };
    let (a, b, only) = (&a.nets[0], &b.nets[0], &only.nets[0]);
    assert_eq!(key(&a.faces[0]), key(&b.faces[1]));
    assert_eq!(key(&a.faces[1]), key(&b.faces[0]));
    assert_eq!(key(&a.faces[0]), key(&only.faces[0]));
    assert!(!a.faces[0].sinks.is_empty());
}

/// Steps below 20 ε_stop and faces on reflecting walls are rejected.
#[test]
fn rejects_tiny_steps_and_wall_faces() {
    let (g, mut c, ..) = plate();
    c.accuracy.min_walks = 4096;
    c.accuracy.max_walks = 4096;
    let bad = |f: Face, delta: f64| {
        solve_sensitivity(
            &g,
            &[NetId::new(0)],
            &c,
            &[FaceStep {
                box_id: BoxId::new(0),
                face: f,
                delta,
                keep: None,
            }],
            &[],
            SensOptions::default(),
        )
        .is_err()
    };
    assert!(bad(face(Axis::Z, Side::Min), 1e-6));
    assert!(bad(face(Axis::X, Side::Max), 0.04));
    assert!(!bad(face(Axis::Z, Side::Min), 0.04));
}

/// The L-shaped net of S12: a horizontal arm A with a vertical arm B standing
/// on its left end, and a neighbour N above the exposed part of A's top face.
/// The y faces lie on reflecting walls, so the case is two-dimensional.
fn l_shape(arm_top: f64) -> Vec<(&'static str, [f64; 6])> {
    vec![
        ("L", [0.0, 0.0, 1.0, 2.0, 4.0, arm_top]),
        ("L", [0.0, 0.0, arm_top, 0.3, 4.0, 2.0]),
        ("N", [0.6, 0.0, 1.6, 2.0, 4.0, 1.9]),
    ]
}

fn l_cfg(walks: u64, seed: u64) -> SolverConfig {
    let bounds = Aabb::from_array([-1.0, 0.0, 0.0, 3.0, 4.0, 3.0]).unwrap();
    let mut faces = [FaceKind::Ground; 6];
    faces[face(Axis::Y, Side::Min).index()] = FaceKind::Reflect;
    faces[face(Axis::Y, Side::Max).index()] = FaceKind::Reflect;
    let mut c = cfg(Boundary::Box { bounds, faces }, None, walks);
    // Pinned: the re-solves must not change ε_stop with the geometry.
    c.walk.stop_dist = Some(5e-4);
    c.seed = seed;
    c
}

/// S12: only the exposed part of a partially covered face moves (the strip
/// under the other arm is filled back in). The one-sided A0 difference at a
/// large step must match independent re-solves of the explicitly shrunk
/// geometry at the same step.
#[test]
fn s12_partially_covered_face_matches_resolves() {
    let (top, delta, walks) = (1.3, 0.04, 4_000_000);
    let g = geom(&l_shape(top));
    let zmax = face(Axis::Z, Side::Max);
    let (l, n) = (Sink::Net(NetId::new(0)), Sink::Net(NetId::new(1)));
    let su = faradice_frw::sens::sens_setup(&g, &[NetId::new(0)], &l_cfg(walks, 1)).unwrap();
    let exposed = faradice_frw::sens::exposed_area(&g, &su.domain, BoxId::new(0), zmax).unwrap();
    assert!((exposed - 1.7 * 4.0).abs() < 1e-9, "{exposed}");
    let sol = solve_sensitivity(
        &g,
        &[NetId::new(0)],
        &l_cfg(walks, 1),
        &[FaceStep {
            box_id: BoxId::new(0),
            face: zmax,
            delta,
            keep: None,
        }],
        &[],
        SensOptions {
            richardson: false,
            ..SensOptions::default()
        },
    )
    .unwrap();
    let r = &sol.nets[0];
    let w = r.nominal.walks;
    // The shrunk geometry: A's top lowered by δ, B extended down to it.
    let shrunk = geom(&l_shape(top - delta));
    let nom = solve(&g, &[NetId::new(0)], &l_cfg(walks, 2)).unwrap();
    let pert = solve(&shrunk, &[NetId::new(0)], &l_cfg(walks, 3)).unwrap();
    for (name, s) in [("C_LL", l), ("C_LN", n)] {
        let a0 = r.faces[0].dc_one_sided(s, w);
        let (c0, c1) = (nom.nets[0].c(s), pert.nets[0].c(s));
        let fd = (c0.value - c1.value) / delta;
        let fd_sigma = c0.std_err.hypot(c1.std_err) / delta;
        let sigma = a0.std_err.hypot(fd_sigma);
        eprintln!(
            "S12 d{name}/dn: A0 {:.5} ± {:.5}, re-solve {fd:.5} ± {fd_sigma:.5}",
            a0.value, a0.std_err
        );
        assert!((a0.value - fd).abs() <= 4.0 * sigma, "{name}");
        assert!(sigma <= 0.05 * fd.abs(), "{name}: σ {sigma} too large");
    }
}

/// A face fully covered by another box of its net has no exposed area, and a
/// face flush with an overlapping box of its net is ambiguous: both are
/// rejected.
#[test]
fn rejects_covered_and_flush_faces() {
    let c = l_cfg(4096, 1);
    let try_face = |g: &Geometry, b: u32, f: Face| {
        solve_sensitivity(
            g,
            &[NetId::new(0)],
            &c,
            &[FaceStep {
                box_id: BoxId::new(b),
                face: f,
                delta: 0.02,
                keep: None,
            }],
            &[],
            SensOptions::default(),
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    let g = geom(&l_shape(1.3));
    // B's bottom face stands on A.
    let e = try_face(&g, 1, face(Axis::Z, Side::Min)).unwrap_err();
    assert!(e.contains("no exposed area"), "{e}");
    assert!(try_face(&g, 0, face(Axis::Z, Side::Max)).is_ok());
    // A box inside A that ends in A's top plane.
    let mut boxes = l_shape(1.3);
    boxes.push(("L", [1.0, 0.0, 1.1, 1.2, 4.0, 1.3]));
    let e = try_face(&geom(&boxes), 0, face(Axis::Z, Side::Max)).unwrap_err();
    assert!(e.contains("flush"), "{e}");
}

/// S9: open boundary with the substrate as an odd image (walks reflected
/// across the substrate flip sign, including inside continuations). Two
/// wires above the substrate: the one-sided A0 difference at a large step
/// must match independent re-solves of the shrunk geometry at the same step,
/// for the face toward the neighbour and the face toward the substrate.
#[test]
fn s9_open_substrate_matches_resolves() {
    let wires = |dy: f64, dz: f64| {
        geom(&[
            ("A", [0.0, 0.0, 0.5 + dz, 4.0, 0.3 - dy, 0.8]),
            ("B", [0.0, 0.6, 0.5, 4.0, 0.9, 0.8]),
        ])
    };
    let mk = |seed| {
        let mut c = cfg(Boundary::Open, Some(0.0), 4_000_000);
        c.walk.stop_dist = Some(3e-4);
        c.seed = seed;
        c
    };
    let g = wires(0.0, 0.0);
    // 15 % of the 0.3 µm gap to B, 9 % of the 0.5 µm gap to the substrate.
    let delta = 0.045;
    let faces = [face(Axis::Y, Side::Max), face(Axis::Z, Side::Min)];
    let steps: Vec<FaceStep> = faces
        .iter()
        .map(|&f| FaceStep {
            box_id: BoxId::new(0),
            face: f,
            delta,
            keep: None,
        })
        .collect();
    let sol = solve_sensitivity(
        &g,
        &[NetId::new(0)],
        &mk(1),
        &steps,
        &[],
        SensOptions {
            richardson: false,
            ..SensOptions::default()
        },
    )
    .unwrap();
    let r = &sol.nets[0];
    let w = r.nominal.walks;
    let nom = solve(&g, &[NetId::new(0)], &mk(2)).unwrap();
    let (a, b) = (Sink::Net(NetId::new(0)), Sink::Net(NetId::new(1)));
    for (fi, shrunk) in [wires(delta, 0.0), wires(0.0, delta)].iter().enumerate() {
        let pert = solve(shrunk, &[NetId::new(0)], &mk(3 + fi as u64)).unwrap();
        for (name, s) in [("C_AA", a), ("C_AB", b)] {
            let a0 = r.faces[fi].dc_one_sided(s, w);
            let (c0, c1) = (nom.nets[0].c(s), pert.nets[0].c(s));
            let fd = (c0.value - c1.value) / delta;
            let fd_sigma = c0.std_err.hypot(c1.std_err) / delta;
            let sigma = a0.std_err.hypot(fd_sigma);
            eprintln!(
                "S9 face {} d{name}/dn: A0 {:.5} ± {:.5}, re-solve {fd:.5} ± {fd_sigma:.5}",
                faces[fi].as_str(),
                a0.value,
                a0.std_err
            );
            assert!((a0.value - fd).abs() <= 4.0 * sigma, "{name}");
            assert!(sigma <= 0.1 * fd.abs(), "{name}: σ {sigma} too large");
        }
    }
}

/// The exposure rule does not depend on where the geometry sits: the L of
/// S12 translated far along the face normal has the same exposed area.
#[test]
fn exposure_is_translation_invariant() {
    let zmax = face(Axis::Z, Side::Max);
    let shift = |x0: f64| {
        let boxes: Vec<(&str, [f64; 6])> = l_shape(1.3)
            .into_iter()
            .map(|(n, mut c)| {
                c[2] += x0;
                c[5] += x0;
                (n, c)
            })
            .collect();
        let g = geom(&boxes);
        let bounds = Aabb::from_array([-1.0, 0.0, x0, 3.0, 4.0, x0 + 3.0]).unwrap();
        let mut faces = [FaceKind::Ground; 6];
        faces[face(Axis::Y, Side::Min).index()] = FaceKind::Reflect;
        faces[face(Axis::Y, Side::Max).index()] = FaceKind::Reflect;
        let c = cfg(Boundary::Box { bounds, faces }, None, 1000);
        let su = faradice_frw::sens::sens_setup(&g, &[NetId::new(0)], &c).unwrap();
        faradice_frw::sens::exposed_area(&g, &su.domain, BoxId::new(0), zmax).unwrap()
    };
    let a = shift(0.0);
    for x0 in [1e6, 1e9] {
        assert!((shift(x0) - a).abs() < 1e-6 * a, "x0 = {x0}");
    }
}

/// S6: deleting the crossing wire of the three-wire example. The edited
/// capacitances must match a re-solve without the wire, the paired
/// differences must be far sharper than the difference of two solves, and
/// the nominal statistics stay bit-identical to `solve`.
#[test]
fn s6_delete_crossing_wire() {
    use faradice_frw::sens::{Deletion, SensRequest, solve_sensitivity_with};
    let wires = [
        ("left", [0.0, 0.00, 1.0, 10.0, 0.14, 1.36]),
        ("victim", [0.0, 0.28, 1.0, 10.0, 0.42, 1.36]),
        ("right", [0.0, 0.56, 1.0, 10.0, 0.70, 1.36]),
    ];
    let mut all = wires.to_vec();
    all.push(("cross", [4.93, -2.0, 1.76, 5.07, 2.7, 2.12]));
    let (g, g3) = (geom(&all), geom(&wires));
    let mk = |seed| {
        let mut c = cfg(Boundary::Open, Some(0.0), 600_000);
        c.eps_r = 3.9;
        c.walk.stop_dist = Some(1.4e-4);
        c.seed = seed;
        c
    };
    let targets = [NetId::new(0), NetId::new(1), NetId::new(2)];
    let cross = NetId::new(3);
    let sol = solve_sensitivity_with(
        &g,
        &targets,
        &mk(1),
        &SensRequest {
            deletions: &[Deletion { nets: vec![cross] }],
            ..SensRequest::default()
        },
    )
    .unwrap();
    let plain = solve(&g, &targets, &mk(1)).unwrap();
    let re = solve(&g3, &targets, &mk(2)).unwrap();
    for (i, t) in targets.iter().enumerate() {
        let r = &sol.nets[i];
        let n = r.nominal.walks;
        assert_eq!(n, plain.nets[i].walks);
        assert_eq!(
            r.nominal.total().value.to_bits(),
            plain.nets[i].total().value.to_bits()
        );
        let d = &r.deletions[0];
        assert!(d.converged());
        assert_eq!(d.c(Sink::Net(cross), n).value, 0.0);
        for (name, s) in [
            ("total", Sink::Net(*t)),
            ("victim", Sink::Net(NetId::new(1))),
        ] {
            if s == Sink::Net(*t) && name != "total" {
                continue;
            }
            let (a, b) = (d.c(s, n), re.nets[i].c(s));
            let sigma = a.std_err.hypot(b.std_err);
            eprintln!(
                "S6 net {i} {name}: edited {:.5} ± {:.5}, re-solve {:.5} ± {:.5} ({:+.2}σ)",
                a.value,
                a.std_err,
                b.value,
                b.std_err,
                (a.value - b.value) / sigma
            );
            assert!((a.value - b.value).abs() <= 4.0 * sigma, "S6 {i} {name}");
        }
        // Paired: σ(ΔC_tt) well below the σ of a difference of two solves.
        let dt = d.delta_c(Sink::Net(*t), n);
        let unpaired = r.nominal.total().std_err.hypot(re.nets[i].total().std_err);
        eprintln!(
            "S6 net {i} ΔC_tt = {:.5} ± {:.5} (unpaired σ {unpaired:.5}), continued {}",
            dt.value, dt.std_err, d.continued_walks
        );
        assert!(dt.std_err < 0.5 * unpaired);
    }
}

/// Weights `w_i` over the steps `2^i δ` (i < `exps.len() + 1`) with
/// `Σ w_i = 1` and `Σ w_i (2^i)^p = 0` for every `p` in `exps`: extrapolation
/// to δ → 0 that cancels bias terms `δ^p`.
fn extrapolation_weights(exps: &[f64]) -> Vec<f64> {
    let m = exps.len() + 1;
    // Rows: [1, 1, …] = 1, then [(2^i)^p] = 0. Gaussian elimination.
    let mut a: Vec<Vec<f64>> = Vec::new();
    a.push((0..m).map(|_| 1.0).chain([1.0]).collect());
    for &p in exps {
        a.push(
            (0..m)
                .map(|i| 2f64.powi(i as i32).powf(p))
                .chain([0.0])
                .collect(),
        );
    }
    for c in 0..m {
        let piv = (c..m)
            .max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))
            .unwrap();
        a.swap(c, piv);
        for r in 0..m {
            if r != c {
                let f = a[r][c] / a[c][c];
                for k in c..=m {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    (0..m).map(|i| a[i][m] / a[i][i]).collect()
}

/// S7: bias and variance of A0 against the step. The victim's face toward
/// the right wire (gap g = 0.14 µm) in the three-wire example, on a ladder
/// δ/g = 0.25 % … 16 % (factor 2), one-sided, plus per-sample extrapolations
/// that cancel bias terms δ^{1/3} (convex 90° edges) and δ^{1/3}, δ^{2/3}.
/// Prints tables for `docs/validation/D3.md`.
#[test]
#[ignore = "δ sweep; S7_WALKS sets the walks (default 16M); run with --ignored --nocapture"]
fn s7_delta_sweep() {
    let walks: u64 = std::env::var("S7_WALKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16_000_000);
    let g = geom(&[
        ("left", [0.0, 0.00, 1.0, 10.0, 0.14, 1.36]),
        ("victim", [0.0, 0.28, 1.0, 10.0, 0.42, 1.36]),
        ("right", [0.0, 0.56, 1.0, 10.0, 0.70, 1.36]),
    ]);
    let mut c = cfg(Boundary::Open, Some(0.0), walks);
    c.eps_r = 3.9;
    c.walk.stop_dist = Some(5e-6);
    let gap = 0.14;
    let rels: Vec<f64> = (0..7).map(|i| 0.0025 * 2f64.powi(i)).collect();
    let steps: Vec<FaceStep> = rels
        .iter()
        .map(|r| FaceStep {
            box_id: BoxId::new(1),
            face: face(Axis::Y, Side::Max),
            delta: r * gap,
            keep: None,
        })
        .collect();
    let w1 = extrapolation_weights(&[1.0 / 3.0]);
    let w2 = extrapolation_weights(&[1.0 / 3.0, 2.0 / 3.0]);
    let wl = extrapolation_weights(&[1.0]);
    let mut params = Vec::new();
    for w in [&wl, &w1, &w2] {
        for i in 0..rels.len() + 1 - w.len() {
            params.push(Param {
                faces: w.iter().enumerate().map(|(k, &x)| (i + k, x)).collect(),
            });
        }
    }
    let sol = solve_sensitivity(
        &g,
        &[NetId::new(1)],
        &c,
        &steps,
        &params,
        SensOptions {
            richardson: false,
            ..SensOptions::default()
        },
    )
    .unwrap();
    let r = &sol.nets[0];
    let n = r.nominal.walks;
    eprintln!(
        "walks {n}, eps_stop {}, weights 1/3: {w1:?}, 1/3+2/3: {w2:?}",
        sol.eps_stop
    );
    for (name, s) in [
        ("C_vv", Sink::Net(NetId::new(1))),
        ("C_vr", Sink::Net(NetId::new(2))),
    ] {
        eprintln!(
            "\n{name}\n| δ/g | one-sided | linear Richardson | R(1/3) | R(1/3, 2/3) |\n|---|---|---|---|---|"
        );
        let mut pi = 0;
        let mut cols: Vec<Vec<String>> = vec![Vec::new(); 3];
        for (col, w) in [&wl, &w1, &w2].iter().enumerate() {
            for _ in 0..rels.len() + 1 - w.len() {
                let e = r.params[pi].dc(s, n);
                cols[col].push(format!("{:.3} ± {:.3}", e.value, e.std_err));
                pi += 1;
            }
        }
        for (i, rel) in rels.iter().enumerate() {
            let o = r.faces[i].dc(s, n);
            let get = |c: usize| cols[c].get(i).cloned().unwrap_or_default();
            eprintln!(
                "| {:.2} % | {:.3} ± {:.3} | {} | {} | {} |",
                100.0 * rel,
                o.value,
                o.std_err,
                get(0),
                get(1),
                get(2)
            );
        }
    }
    let s = Sink::Net(NetId::new(1));
    let ratio = r.faces[0].dc(s, n).std_err / r.faces[6].dc(s, n).std_err;
    eprintln!("σ ratio 0.25 %/16 %: {ratio:.2} (√64 = 8)");
    assert!((4.0..16.0).contains(&ratio));
}

/// S13 for A0: the reported σ is calibrated. 50 seeds of S1 (plate bottom
/// face, Richardson, δ = 0.04 so the remaining O(δ²) bias is ~0.2 % ≪ σ):
/// z-scores against the closed form fall within 1.645σ 90 % ± 13 % of the
/// time (three binomial standard deviations at n = 50).
#[test]
fn s13_a0_sigma_coverage() {
    let (g, mut c, a, d1, d2) = plate();
    c.accuracy.min_walks = 60_000;
    c.accuracy.max_walks = 60_000;
    let eps = EPS0_FF_PER_UM;
    let steps = [FaceStep {
        box_id: BoxId::new(0),
        face: face(Axis::Z, Side::Min),
        delta: 0.04,
        keep: None,
    }];
    // Only the plate entry: the substrate entry is its per-sample negative
    // here (the bottom face only trades the plate against the substrate).
    let p = Sink::Net(NetId::new(0));
    let exact = eps * a / (d1 * d1);
    let _ = d2;
    let (mut inside, mut n) = (0, 0);
    let mut zs = Vec::new();
    for seed in 0..50 {
        c.seed = 100 + seed;
        let sol = solve_sensitivity(
            &g,
            &[NetId::new(0)],
            &c,
            &steps,
            &[],
            SensOptions::default(),
        )
        .unwrap();
        let r = &sol.nets[0];
        let e = r.faces[0].dc(p, r.nominal.walks);
        let z = (e.value - exact) / e.std_err;
        zs.push(z);
        inside += usize::from(z.abs() <= 1.645);
        n += 1;
    }
    let cov = f64::from(u32::try_from(inside).unwrap()) / f64::from(u32::try_from(n).unwrap());
    let mean = zs.iter().sum::<f64>() / zs.len() as f64;
    eprintln!("S13 A0 coverage at 1.645σ: {cov:.3} over {n}, mean z {mean:+.3}");
    assert!((0.77..=1.0).contains(&cov), "coverage {cov}");
    // Mean of 50 unit normals: σ = 0.14.
    assert!(mean.abs() < 0.45, "mean z {mean}");
}

/// A0 and deletion in a stratified dielectric: the V1 plate inside a stack
/// (ε 2 below z = 0.5, ε 4 up to 1.6, ε 1 above). Each gap is a series of
/// layers, C = ε0 A / Σ t_k/ε_k, and growing a face by dn thins the layer
/// it touches: dC/dn = ε0 A / (ε_adj (Σ t_k/ε_k)²). Nominal statistics stay
/// bit-identical to a layered `solve`.
#[test]
fn layered_plate_faces() {
    use faradice_frw::DielectricLayer;
    let (g, mut c, a, _, _) = plate();
    c.eps_r = 1.0;
    c.layers = vec![
        DielectricLayer {
            z_top: 0.5,
            eps_r: 2.0,
        },
        DielectricLayer {
            z_top: 1.6,
            eps_r: 4.0,
        },
    ];
    c.accuracy.min_walks = 16_000_000;
    c.accuracy.max_walks = 16_000_000;
    let steps = [
        FaceStep {
            box_id: BoxId::new(0),
            face: face(Axis::Z, Side::Min),
            delta: 0.04,
            keep: None,
        },
        FaceStep {
            box_id: BoxId::new(0),
            face: face(Axis::Z, Side::Max),
            delta: 0.04,
            keep: None,
        },
    ];
    let sol = solve_sensitivity(
        &g,
        &[NetId::new(0)],
        &c,
        &steps,
        &[],
        SensOptions::default(),
    )
    .unwrap();
    let plain = solve(&g, &[NetId::new(0)], &c).unwrap();
    let r = &sol.nets[0];
    assert_eq!(
        r.nominal.total().value.to_bits(),
        plain.nets[0].total().value.to_bits()
    );
    let n = r.nominal.walks;
    let e0 = EPS0_FF_PER_UM;
    let s_bot = 0.5 / 2.0 + 0.5 / 4.0;
    let s_top = 0.4 / 4.0 + 0.2 / 1.0;
    let p = Sink::Net(NetId::new(0));
    assert_close(
        "layered C_PP",
        r.nominal.total(),
        e0 * a * (1.0 / s_bot + 1.0 / s_top),
        0.005,
    );
    assert_sharp(
        "layered bottom dC_PP/dn",
        r.faces[0].dc(p, n),
        e0 * a / (4.0 * s_bot * s_bot),
        0.01,
        0.05,
    );
    assert_sharp(
        "layered top dC_PP/dn",
        r.faces[1].dc(p, n),
        e0 * a / (4.0 * s_top * s_top),
        0.01,
        0.05,
    );
}

/// Deletion in a stratified dielectric: a small cube in the top gap of the
/// layered plate, deleted, against a layered re-solve without it.
#[test]
fn layered_deletion_matches_resolve() {
    use faradice_frw::DielectricLayer;
    use faradice_frw::sens::{Deletion, SensRequest, solve_sensitivity_with};
    let (l, d1, t, d2) = (4.0, 1.0, 0.2, 0.6);
    let plate_box = ("P", [0.0, 0.0, d1, l, l, d1 + t]);
    let cube = ("q", [1.8, 1.8, 1.4, 2.2, 2.2, 1.6]);
    let (g, g0) = (geom(&[plate_box, cube]), geom(&[plate_box]));
    let mut faces = [FaceKind::Reflect; 6];
    faces[face(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[face(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, d1 + t + d2]).unwrap();
    let mk = |seed| {
        let mut c = cfg(Boundary::Box { bounds, faces }, None, 2_000_000);
        c.layers = vec![
            DielectricLayer {
                z_top: 0.5,
                eps_r: 2.0,
            },
            DielectricLayer {
                z_top: 1.5,
                eps_r: 4.0,
            },
        ];
        c.walk.stop_dist = Some(2e-4);
        c.seed = seed;
        c
    };
    let p = NetId::new(0);
    let sol = solve_sensitivity_with(
        &g,
        &[p],
        &mk(1),
        &SensRequest {
            deletions: &[Deletion {
                nets: vec![NetId::new(1)],
            }],
            ..SensRequest::default()
        },
    )
    .unwrap();
    let re = solve(&g0, &[p], &mk(2)).unwrap();
    let r = &sol.nets[0];
    let (a, b) = (
        r.deletions[0].c(Sink::Net(p), r.nominal.walks),
        re.nets[0].total(),
    );
    let d = r.deletions[0].delta_c(Sink::Net(p), r.nominal.walks);
    eprintln!(
        "layered deletion: edited {:.5} ± {:.5}, re-solve {:.5} ± {:.5}, ΔC {:.5} ± {:.5}",
        a.value, a.std_err, b.value, b.std_err, d.value, d.std_err
    );
    assert!((a.value - b.value).abs() <= 4.0 * a.std_err.hypot(b.std_err));
    assert!(
        d.value < -3.0 * d.std_err,
        "removing a grounded cube lowers C_PP"
    );
}

/// Deleting a net that is also a target: its row is empty, and the other
/// rows are bit-identical to a run where it is not a target.
#[test]
fn deleting_a_target_drops_its_row() {
    use faradice_frw::sens::{Deletion, SensRequest, solve_sensitivity_with};
    let g = geom(&[
        ("a", [0.0, 0.0, 1.0, 4.0, 0.3, 1.3]),
        ("b", [0.0, 0.6, 1.0, 4.0, 0.9, 1.3]),
    ]);
    let c = cfg(Boundary::Open, Some(0.0), 50_000);
    let del = [Deletion {
        nets: vec![NetId::new(1)],
    }];
    let req = SensRequest {
        deletions: &del,
        ..SensRequest::default()
    };
    let both = solve_sensitivity_with(&g, &[NetId::new(0), NetId::new(1)], &c, &req).unwrap();
    let one = solve_sensitivity_with(&g, &[NetId::new(0)], &c, &req).unwrap();
    assert!(both.nets[1].deletions[0].edited.is_empty());
    let key = |r: &faradice_frw::sens::DeletionResult| -> Vec<(Sink, u64, u64)> {
        r.edited
            .iter()
            .map(|(s, m)| (*s, m.sum().to_bits(), m.sum_sq().to_bits()))
            .collect()
    };
    assert_eq!(
        key(&both.nets[0].deletions[0]),
        key(&one.nets[0].deletions[0])
    );
    assert!(!one.nets[0].deletions[0].edited.is_empty());
}

/// Deleting every conductor (here the only one, a target) gives an empty
/// row instead of failing on an empty geometry.
#[test]
fn deleting_every_conductor() {
    use faradice_frw::sens::{Deletion, SensRequest, solve_sensitivity_with};
    let g = geom(&[("a", [0.0, 0.0, 1.0, 4.0, 0.3, 1.3])]);
    let c = cfg(Boundary::Open, Some(0.0), 10_000);
    let del = [Deletion {
        nets: vec![NetId::new(0)],
    }];
    let r = solve_sensitivity_with(
        &g,
        &[NetId::new(0)],
        &c,
        &SensRequest {
            deletions: &del,
            ..SensRequest::default()
        },
    )
    .unwrap();
    assert!(r.nets[0].deletions[0].edited.is_empty());
    assert!(r.nets[0].nominal.total().value > 0.0);
}
