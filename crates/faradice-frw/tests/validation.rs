//! Analytical validation cases (design §9: V1, V2, V6).

#![allow(missing_docs)]

use faradice_core::{Axis, EPS0_FF_PER_UM, Face, NetId, Side, Sink};
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

fn cfg(boundary: Boundary, substrate_z: Option<f64>, tol: f64) -> SolverConfig {
    SolverConfig {
        eps_r: 1.0,
        layers: Vec::new(),
        boundary,
        substrate_z,
        accuracy: Accuracy {
            total_rel_tol: tol,
            min_walks: 20_000,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed: 1,
        ground_nets: Vec::new(),
        regions: Default::default(),
    }
}

fn assert_close(name: &str, est: faradice_core::Estimate, exact: f64, bias_rel: f64) {
    let err = (est.value - exact).abs();
    let allowed = 4.0 * est.std_err + bias_rel * exact.abs();
    eprintln!(
        "{name}: {:.6} ± {:.6} (exact {exact:.6}, err {:.3}%)",
        est.value,
        est.std_err,
        100.0 * err / exact
    );
    assert!(
        err <= allowed,
        "{name}: {} ± {} vs exact {exact}",
        est.value,
        est.std_err
    );
}

/// V1: an infinite parallel plate (reflecting sides) between the substrate
/// and a grounded top face has C = εA(1/d₁ + 1/d₂) exactly.
#[test]
fn v1_parallel_plate_is_exact() {
    let (l, d1, t, d2) = (4.0, 1.0, 0.2, 0.6);
    let g = geom(&[("P", [0.0, 0.0, d1, l, l, d1 + t])]);
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, d1 + t + d2]).unwrap();
    let c = cfg(Boundary::Box { bounds, faces }, Some(0.0), 0.003);
    let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
    let r = &sol.nets[0];
    assert!(r.converged);
    let eps = EPS0_FF_PER_UM;
    let a = l * l;
    assert_close("total", r.total(), eps * a * (1.0 / d1 + 1.0 / d2), 0.003);
    assert_close(
        "substrate",
        r.coupling(Sink::GroundPlane),
        eps * a / d1,
        0.005,
    );
    assert_close(
        "top",
        r.coupling(Sink::Boundary(Face::new(Axis::Z, Side::Max))),
        eps * a / d2,
        0.005,
    );
}

/// V2: unit cube in free space, C = 0.66067815 · 4πε·a.
#[test]
fn v2_free_space_cube() {
    let g = geom(&[("C", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]);
    let c = cfg(Boundary::Open, None, 0.003);
    let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
    let r = &sol.nets[0];
    eprintln!(
        "walks {} aborted {} steps/walk {:.1}",
        r.walks,
        r.aborted_walks,
        r.total_steps as f64 / r.walks as f64
    );
    assert_eq!(r.aborted_walks, 0);
    let exact = 0.660_678_15 * 4.0 * std::f64::consts::PI * EPS0_FF_PER_UM;
    assert_close("cube", r.total(), exact, 0.003);
    assert_close("infinity", r.coupling(Sink::Infinity), exact, 0.003);
}

/// V6: reciprocity C_12 = C_21 and thread-count independence.
#[test]
fn v6_reciprocity_and_determinism() {
    let g = geom(&[
        ("A", [0.0, 0.0, 1.0, 5.0, 0.3, 1.3]),
        ("B", [0.0, 0.6, 1.0, 5.0, 0.9, 1.3]),
    ]);
    let bounds = Aabb::from_array([-3.0, -3.0, 0.0, 8.0, 4.0, 4.0]).unwrap();
    let mut c = cfg(Boundary::grounded_box(bounds), Some(0.0), 0.01);
    c.accuracy.coupling_rel_tol = 0.02;
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| solve(&g, &[NetId::new(0), NetId::new(1)], &c).unwrap())
    };
    let s1 = run(1);
    let s4 = run(4);
    for (a, b) in s1.nets.iter().zip(&s4.nets) {
        assert_eq!(a.walks, b.walks);
        assert_eq!(a.total().value.to_bits(), b.total().value.to_bits());
    }
    let c12 = s1.nets[0].coupling(Sink::Net(NetId::new(1)));
    let c21 = s1.nets[1].coupling(Sink::Net(NetId::new(0)));
    let diff = (c12.value - c21.value).abs();
    eprintln!("C12 {c12:?} C21 {c21:?}");
    assert!(diff <= 4.0 * c12.std_err.hypot(c21.std_err) + 0.005 * c12.value);
}

/// Open boundary with substrate (odd images): a cube above the plane must equal
/// the free-space pair (cube, mirrored cube at −1 V): C = C_AA − C_AB.
#[test]
fn open_with_substrate_matches_image_pair() {
    let above = [0.0, 0.0, 0.5, 1.0, 1.0, 1.5];
    let below = [0.0, 0.0, -1.5, 1.0, 1.0, -0.5];
    let single = geom(&[("A", above)]);
    let pair = geom(&[("A", above), ("B", below)]);
    let c = cfg(Boundary::Open, Some(0.0), 0.003);
    let mut free = cfg(Boundary::Open, None, 0.003);
    free.accuracy.coupling_rel_tol = 0.01;
    free.accuracy.track_rel = 0.01;
    let s = solve(&single, &[NetId::new(0)], &c).unwrap();
    let f = solve(&pair, &[NetId::new(0)], &free).unwrap();
    let caa = f.nets[0].total();
    let cab = f.nets[0].c(Sink::Net(NetId::new(1)));
    let expected = caa.value - cab.value;
    let sigma = caa.std_err.hypot(cab.std_err);
    let got = s.nets[0].total();
    eprintln!("image: {got:?} vs {expected} ± {sigma}");
    assert!((got.value - expected).abs() <= 4.0 * got.std_err.hypot(sigma) + 0.003 * expected);
    // A lone cube: all of its capacitance goes to ground (substrate + infinity),
    // and no separate infinity tally is produced under the image extension.
    let g = s.nets[0].ground();
    eprintln!("image ground: {g:?}");
    assert!((g.value - got.value).abs() <= 4.0 * g.std_err.hypot(got.std_err));
    assert!(s.nets[0].sinks.iter().all(|(k, _)| *k != Sink::Infinity));
}

/// Duplicated and abutting boxes of one net must not change the result.
#[test]
fn gaussian_surface_ownership() {
    let bounds = Aabb::from_array([-3.0, -3.0, 0.0, 5.0, 4.0, 4.0]).unwrap();
    let c = cfg(Boundary::grounded_box(bounds), Some(0.0), 0.004);
    let one = geom(&[("A", [0.0, 0.0, 1.0, 2.0, 1.0, 1.5])]);
    // Includes an exact duplicate whose faces lie on the outer bound of the net.
    let split = geom(&[
        ("A", [0.0, 0.0, 1.0, 2.0, 1.0, 1.5]),
        ("A", [0.0, 0.0, 1.0, 1.0, 1.0, 1.5]),
        ("A", [1.0, 0.0, 1.0, 2.0, 1.0, 1.5]),
        ("A", [0.0, 0.0, 1.0, 2.0, 1.0, 1.5]),
    ]);
    let a = solve(&one, &[NetId::new(0)], &c).unwrap().nets[0].total();
    let b = solve(&split, &[NetId::new(0)], &c).unwrap().nets[0].total();
    eprintln!("ownership: {a:?} vs {b:?}");
    assert!((a.value - b.value).abs() <= 4.0 * a.std_err.hypot(b.std_err));
}

/// Accuracy study (run with `--ignored --nocapture`): bias vs Green table
/// resolution and stopping distance on the two exact cases.
#[test]
#[ignore = "study; prints a table"]
fn study_resolution_and_stop() {
    let plate = geom(&[("P", [0.0, 0.0, 1.0, 4.0, 4.0, 1.2])]);
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let pb = Boundary::Box {
        bounds: Aabb::from_array([0.0, 0.0, 0.0, 4.0, 4.0, 1.8]).unwrap(),
        faces,
    };
    let plate_exact = EPS0_FF_PER_UM * 16.0 * (1.0 + 1.0 / 0.6);
    let cube = geom(&[("C", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]);
    let cube_exact = 0.660_678_15 * 4.0 * std::f64::consts::PI * EPS0_FF_PER_UM;
    println!("| case | M | stop_ratio | rel. error (%) | σ (%) | samples | steps/walk |");
    println!("|---|---|---|---|---|---|---|");
    for (name, g, b, sub, exact) in [
        ("plate", &plate, pb.clone(), Some(0.0), plate_exact),
        ("cube", &cube, Boundary::Open, None, cube_exact),
    ] {
        for (m, stop) in [
            (16, 1e-3),
            (32, 1e-3),
            (64, 1e-3),
            (128, 1e-3),
            (64, 1e-2),
            (64, 1e-4),
        ] {
            let mut c = cfg(b.clone(), sub, 0.001);
            c.walk.green_resolution = m;
            c.walk.stop_ratio = stop;
            let r = solve(g, &[NetId::new(0)], &c).unwrap().nets.remove(0);
            let t = r.total();
            println!(
                "| {name} | {m} | {stop:e} | {:+.3} | {:.3} | {} | {:.1} |",
                100.0 * (t.value - exact) / exact,
                100.0 * t.std_err / exact,
                r.walks,
                r.total_steps as f64 / r.walks as f64 / 2.0
            );
        }
    }
}

/// Thin gaps: the default stopping distance must adapt to the smallest gap.
#[test]
fn thin_gap_parallel_plate() {
    let (l, d1, t, d2) = (1.0, 1e-3, 1.0, 1.0);
    let g = geom(&[("P", [0.0, 0.0, d1, l, l, d1 + t])]);
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, d1 + t + d2]).unwrap();
    let c = cfg(Boundary::Box { bounds, faces }, Some(0.0), 0.003);
    let r = solve(&g, &[NetId::new(0)], &c).unwrap().nets.remove(0);
    assert_close(
        "thin gap",
        r.total(),
        EPS0_FF_PER_UM * l * l * (1.0 / d1 + 1.0 / d2),
        0.003,
    );
}

/// Invalid walk options are rejected instead of panicking.
#[test]
fn invalid_walk_options() {
    let g = geom(&[("A", [0.0, 0.0, 1.0, 1.0, 1.0, 2.0])]);
    let mut c = cfg(Boundary::Open, None, 0.01);
    c.walk.green_resolution = 63;
    assert!(solve(&g, &[NetId::new(0)], &c).is_err());
    c.walk.green_resolution = 64;
    c.walk.gauss_alpha = 1.5;
    assert!(solve(&g, &[NetId::new(0)], &c).is_err());
}

/// The engine path (launch on the CPU, transitions in a [`faradice_frw::WalkEngine`])
/// agrees with [`solve`] and does not depend on how walks are grouped.
#[test]
fn engine_path_agrees_and_is_grouping_independent() {
    use faradice_frw::{CpuEngine, Launch, Outcome, Scene, WalkEngine, solve_with};
    struct Grouped(usize);
    impl WalkEngine for Grouped {
        fn run(&mut self, s: &Scene<'_>, l: &[Launch]) -> Result<Vec<Outcome>, String> {
            let mut out = Vec::new();
            for c in l.chunks(997) {
                out.extend(CpuEngine.run(s, c)?);
            }
            Ok(out)
        }
        fn preferred_walks(&self) -> usize {
            self.0
        }
    }
    let g = geom(&[
        ("a", [0.0, 0.0, 1.0, 6.0, 0.2, 1.3]),
        ("b", [0.0, 0.5, 1.0, 6.0, 0.7, 1.3]),
    ]);
    let bounds = Aabb::from_array([-2.0, -2.0, 0.0, 8.0, 2.7, 4.0]).unwrap();
    let c = cfg(Boundary::grounded_box(bounds), Some(0.0), 0.004);
    let t = [NetId::new(0)];
    let cpu = solve(&g, &t, &c).unwrap();
    let e1 = solve_with(&g, &t, &c, |_| Ok(CpuEngine)).unwrap();
    let e2 = solve_with(&g, &t, &c, |_| Ok(Grouped(50_000))).unwrap();
    let (a, b) = (&e1.nets[0], &e2.nets[0]);
    // Different wave sizes, same batches merged in the same order.
    assert_eq!(a.walks, b.walks);
    assert_eq!(a.total().value.to_bits(), b.total().value.to_bits());
    let (x, y) = (a.total(), cpu.nets[0].total());
    let s = x.std_err.hypot(y.std_err);
    assert!((x.value - y.value).abs() < 4.5 * s, "{x:?} vs {y:?}");
    let k = Sink::Net(NetId::new(1));
    let (x, y) = (a.coupling(k), cpu.nets[0].coupling(k));
    assert!((x.value - y.value).abs() < 4.5 * x.std_err.hypot(y.std_err));
}
