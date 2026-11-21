//! Pillar A (design D3 §6): shape derivatives on edge-free faces (S1, S2).

#![cfg(feature = "sensitivity")]
#![allow(missing_docs)]

use faradice_core::{Axis, BoxId, EPS0_FF_PER_UM, Face, NetId, Side};
use faradice_frw::shape::{ShapeFace, ShapeOptions, solve_shape_derivative};
use faradice_frw::{Accuracy, Boundary, FaceKind, SolverConfig, WalkOptions};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};

fn geom(boxes: &[(&str, [f64; 6])]) -> Geometry {
    let mut g = GeometryBuilder::new();
    for (net, c) in boxes {
        let id = g.net(net).unwrap();
        g.add_box(id, Aabb::from_array(*c).unwrap()).unwrap();
    }
    g.build().unwrap()
}

fn slab_cfg(l: f64, h: f64) -> SolverConfig {
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    SolverConfig {
        eps_r: 2.0,
        layers: Vec::new(),
        boundary: Boundary::Box {
            bounds: Aabb::from_array([0.0, 0.0, 0.0, l, l, h]).unwrap(),
            faces,
        },
        substrate_z: None,
        accuracy: Accuracy::default(),
        walk: WalkOptions::default(),
        seed: 9,
        ground_nets: Vec::new(),
        regions: Default::default(),
    }
}

fn check(name: &str, e: faradice_core::Estimate, exact: f64, max_rel_sigma: f64) {
    eprintln!(
        "{name}: {:.5} ± {:.5} (exact {exact:.5}, {:+.2}σ, σ {:.2} %)",
        e.value,
        e.std_err,
        (e.value - exact) / e.std_err,
        100.0 * e.std_err / exact.abs()
    );
    assert!((e.value - exact).abs() <= 4.0 * e.std_err, "{name}");
    assert!(
        e.std_err <= max_rel_sigma * exact.abs(),
        "{name}: σ too large"
    );
}

/// S1 + S2 with pillar A: the V1 plate (reflecting sides, so its faces have
/// no geometric edges). Growing the bottom face by dn shrinks d1:
/// dC/dn = εA/d1²; the top face likewise with d2.
#[test]
fn s1_s2_plate_faces() {
    let (l, d1, t, d2) = (4.0, 1.0, 0.2, 0.6);
    let g = geom(&[("P", [0.0, 0.0, d1, l, l, d1 + t])]);
    let c = slab_cfg(l, d1 + t + d2);
    let p = NetId::new(0);
    let faces = [
        ShapeFace {
            box_id: BoxId::new(0),
            face: Face::new(Axis::Z, Side::Min),
        },
        ShapeFace {
            box_id: BoxId::new(0),
            face: Face::new(Axis::Z, Side::Max),
        },
    ];
    let r = solve_shape_derivative(
        &g,
        &[p],
        &c,
        &faces,
        ShapeOptions {
            samples: 200_000,
            edge_margin: None,
        },
    )
    .unwrap();
    let eps = 2.0 * EPS0_FF_PER_UM;
    let a = l * l;
    eprintln!(
        "min half-sides: {} {}",
        r[0].min_half_side, r[1].min_half_side
    );
    check("A S1 dC_PP", r[0].dc(p, p), eps * a / (d1 * d1), 0.02);
    check("A S1 dGround", r[0].d_ground(p), eps * a / (d1 * d1), 0.02);
    check("A S2 dC_PP", r[1].dc(p, p), eps * a / (d2 * d2), 0.02);
    assert_eq!(r[0].aborted_walks + r[1].aborted_walks, 0);
}

/// Two plates: growing P1's face toward P2 by dn shrinks their gap d:
/// dC_11 = dC_22 = εA/d², dC_12 = −εA/d² (the coupling grows), and P1's
/// ground capacitance does not change on that face.
#[test]
fn two_plates_mutual_term() {
    let (l, d) = (3.0, 0.8);
    let g = geom(&[
        ("p1", [0.0, 0.0, 1.0, l, l, 1.2]),
        ("p2", [0.0, 0.0, 1.2 + d, l, l, 1.4 + d]),
    ]);
    let c = slab_cfg(l, 2.4 + d);
    let (p1, p2) = (NetId::new(0), NetId::new(1));
    let r = solve_shape_derivative(
        &g,
        &[p1, p2],
        &c,
        &[ShapeFace {
            box_id: BoxId::new(0),
            face: Face::new(Axis::Z, Side::Max),
        }],
        ShapeOptions {
            samples: 200_000,
            edge_margin: None,
        },
    )
    .unwrap();
    let k = 2.0 * EPS0_FF_PER_UM * l * l / (d * d);
    check("A dC_11", r[0].dc(p1, p1), k, 0.02);
    check("A dC_22", r[0].dc(p2, p2), k, 0.02);
    check("A dC_12", r[0].dc(p1, p2), -k, 0.02);
    let dg = r[0].d_ground(p1);
    eprintln!("A dGround_1: {:.5} ± {:.5} (exact 0)", dg.value, dg.std_err);
    assert!(dg.value.abs() <= 4.0 * dg.std_err + 1e-9);
}

/// With P2 declared a ground net, its column belongs to P1's ground: on the
/// face toward P2, dGround_1 = dC_11 = εA/d² (not 0).
#[test]
fn ground_nets_count_as_ground() {
    let (l, d) = (3.0, 0.8);
    let g = geom(&[
        ("p1", [0.0, 0.0, 1.0, l, l, 1.2]),
        ("p2", [0.0, 0.0, 1.2 + d, l, l, 1.4 + d]),
    ]);
    let mut c = slab_cfg(l, 2.4 + d);
    c.ground_nets = vec![NetId::new(1)];
    let r = solve_shape_derivative(
        &g,
        &[NetId::new(0)],
        &c,
        &[ShapeFace {
            box_id: BoxId::new(0),
            face: Face::new(Axis::Z, Side::Max),
        }],
        ShapeOptions {
            samples: 200_000,
            edge_margin: None,
        },
    )
    .unwrap();
    let k = 2.0 * EPS0_FF_PER_UM * l * l / (d * d);
    check(
        "A dGround_1 with P2 grounded",
        r[0].d_ground(NetId::new(0)),
        k,
        0.02,
    );
}

/// Faces with geometric edges need an edge margin; with one, only the
/// interior is integrated.
#[test]
fn edges_need_a_margin() {
    let g = geom(&[("c", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]);
    let c = SolverConfig {
        boundary: Boundary::Open,
        ..slab_cfg(1.0, 1.0)
    };
    let f = [ShapeFace {
        box_id: BoxId::new(0),
        face: Face::new(Axis::Z, Side::Max),
    }];
    let e =
        solve_shape_derivative(&g, &[NetId::new(0)], &c, &f, ShapeOptions::default()).unwrap_err();
    assert!(e.to_string().contains("edges"), "{e}");
    let r = solve_shape_derivative(
        &g,
        &[NetId::new(0)],
        &c,
        &f,
        ShapeOptions {
            samples: 2000,
            edge_margin: Some(0.2),
        },
    )
    .unwrap();
    assert!((r[0].area - 0.36).abs() < 1e-12);
    assert!(r[0].min_half_side >= 0.2 - 1e-12);
}

/// S8 (remedy 1 of §6.4): the derivative of a face with edges as pillar A
/// over the interior (edge margin r) plus A0 over the edge strip only (the
/// interior is kept in place). Checked against the exact cube face C/(3a)
/// and against whole-face A0 on a side face of the three-wire victim. The
/// strip step must satisfy δ ≪ r: moving only the strip leaves a step of
/// height δ at the strip's inner boundary, and at δ/r = 0.35 the strip
/// estimate is biased by −4 % (docs/validation/D3.md).
#[test]
#[ignore = "S8 hybrid; ~1 CPU-minute; run with --ignored --nocapture"]
fn s8_hybrid_edge_strip() {
    use faradice_core::Sink;
    use faradice_frw::sens::{FaceStep, SensOptions, solve_sensitivity};
    use std::time::Instant;
    let hybrid = |g: &Geometry,
                  c: &SolverConfig,
                  net: NetId,
                  b: u32,
                  f: Face,
                  r: f64,
                  delta: f64,
                  a_samples: u64,
                  a0_walks: u64| {
        let t0 = Instant::now();
        let a = solve_shape_derivative(
            g,
            &[net],
            c,
            &[ShapeFace {
                box_id: BoxId::new(b),
                face: f,
            }],
            ShapeOptions {
                samples: a_samples,
                edge_margin: Some(r),
            },
        )
        .unwrap();
        let ta = t0.elapsed().as_secs_f64();
        let bx = *g.box_(BoxId::new(b));
        let ax = f.axis.index();
        let t: Vec<usize> = (0..3).filter(|&i| i != ax).collect();
        let keep = (
            [bx.min()[t[0]] + r, bx.min()[t[1]] + r],
            [bx.max()[t[0]] - r, bx.max()[t[1]] - r],
        );
        let mut c0 = c.clone();
        c0.accuracy.min_walks = a0_walks;
        c0.accuracy.max_walks = a0_walks;
        let t1 = Instant::now();
        let strip = solve_sensitivity(
            g,
            &[net],
            &c0,
            &[
                FaceStep {
                    box_id: BoxId::new(b),
                    face: f,
                    delta,
                    keep: Some(keep),
                },
                FaceStep {
                    box_id: BoxId::new(b),
                    face: f,
                    delta,
                    keep: None,
                },
            ],
            &[],
            SensOptions::default(),
        )
        .unwrap();
        let ts = t1.elapsed().as_secs_f64();
        let n = strip.nets[0].nominal.walks;
        let ai = a[0].dc(net, net);
        let si = strip.nets[0].faces[0].dc(Sink::Net(net), n);
        let whole = strip.nets[0].faces[1].dc(Sink::Net(net), n);
        let sum = ai.value + si.value;
        let sig = ai.std_err.hypot(si.std_err);
        eprintln!(
            "  interior A {:.5} ± {:.5} ({ta:.2} s, min h {:.4}), strip A0 {:.5} ± {:.5}, hybrid {sum:.5} ± {sig:.5}; whole-face A0 {:.5} ± {:.5} ({ts:.2} s for both A0 faces)",
            ai.value,
            ai.std_err,
            a[0].min_half_side,
            si.value,
            si.std_err,
            whole.value,
            whole.std_err
        );
        (sum, sig, whole)
    };
    // Cube in free space: each face C/(3a).
    let g = geom(&[("c", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]);
    let mut c = SolverConfig {
        eps_r: 1.0,
        boundary: Boundary::Open,
        ..slab_cfg(1.0, 1.0)
    };
    c.walk.stop_dist = Some(1e-4);
    let exact = 0.660_678_15 * 4.0 * std::f64::consts::PI * EPS0_FF_PER_UM / 3.0;
    for r in [0.2, 0.1] {
        eprintln!("cube face, r = {r}, δ = 0.01 (exact {exact:.5}):");
        let (sum, sig, _) = hybrid(
            &g,
            &c,
            NetId::new(0),
            0,
            Face::new(Axis::Z, Side::Max),
            r,
            0.01,
            400_000,
            4_000_000,
        );
        assert!(
            (sum - exact).abs() <= 4.0 * sig + 0.01 * exact,
            "cube r = {r}"
        );
    }
    // Three-wire victim, face toward the right wire.
    let g = geom(&[
        ("left", [0.0, 0.00, 1.0, 10.0, 0.14, 1.36]),
        ("victim", [0.0, 0.28, 1.0, 10.0, 0.42, 1.36]),
        ("right", [0.0, 0.56, 1.0, 10.0, 0.70, 1.36]),
    ]);
    let mut c = SolverConfig {
        eps_r: 3.9,
        boundary: Boundary::Open,
        substrate_z: Some(0.0),
        ..slab_cfg(1.0, 1.0)
    };
    c.walk.stop_dist = Some(5e-6);
    {
        let r = 0.04;
        eprintln!("victim y+, r = {r}, δ = 0.0035:");
        let (sum, sig, whole) = hybrid(
            &g,
            &c,
            NetId::new(1),
            1,
            Face::new(Axis::Y, Side::Max),
            r,
            0.0035,
            2_000_000,
            16_000_000,
        );
        let s = sig.hypot(whole.std_err);
        assert!((sum - whole.value).abs() <= 4.0 * s, "victim r = {r}");
    }
}
