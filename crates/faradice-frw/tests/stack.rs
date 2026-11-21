//! Dielectric stacks at the solver level (design M2a, input stage).

#![allow(missing_docs)]

use faradice_core::{Axis, EPS0_FF_PER_UM, Estimate, Face, NetId, Side, Sink};
use faradice_frw::{
    Accuracy, Boundary, DielectricLayer, FaceKind, SolveError, SolverConfig, WalkOptions, solve,
};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};

fn wire() -> Geometry {
    let mut g = GeometryBuilder::new();
    let id = g.net("a").unwrap();
    g.add_box(
        id,
        Aabb::from_array([0.0, 0.0, 1.0, 1.0, 1.0, 1.5]).unwrap(),
    )
    .unwrap();
    g.build().unwrap()
}

fn cfg(boundary: Boundary, eps_r: f64, layers: Vec<DielectricLayer>) -> SolverConfig {
    SolverConfig {
        eps_r,
        layers,
        boundary,
        substrate_z: Some(0.0),
        accuracy: Accuracy {
            total_rel_tol: 0.02,
            min_walks: 2_000,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed: 3,
        ground_nets: Vec::new(),
        regions: Default::default(),
    }
}

fn domain() -> Boundary {
    Boundary::grounded_box(Aabb::new([-3.0, -3.0, 0.0], [4.0, 4.0, 4.0]).unwrap())
}

fn layer(z_top: f64, eps_r: f64) -> DielectricLayer {
    DielectricLayer { z_top, eps_r }
}

#[test]
fn dielectric_regions_are_validated() {
    let g = wire();
    let t = [NetId::new(0)];
    let mut c = cfg(domain(), 4.5, Vec::new());
    c.regions.conformal.push(faradice_frw::ConformalRule {
        layer: "met1".into(),
        boxes: vec![faradice_core::BoxId::new(0)],
        side: 0.0,
        top: 0.0,
        eps_r: 3.5,
    });
    let err = solve(&g, &t, &c).unwrap_err();
    assert!(
        matches!(err, SolveError::Config(ref m) if m.contains("both")),
        "{err}"
    );
    c.regions.conformal[0].side = 0.03;
    // Resolve the shell's effect (about 1 %) statistically.
    c.accuracy.total_rel_tol = 0.002;
    let shelled = solve(&g, &t, &c).unwrap();
    // A rule that selects nothing adds nothing.
    c.regions.conformal[0].boxes.clear();
    let plain = solve(&g, &t, &c).unwrap();
    // A thin shell of lower permittivity lowers the capacitance a little.
    let (a, b) = (shelled.nets[0].total(), plain.nets[0].total());
    let z = (b.value - a.value) / a.std_err.hypot(b.std_err);
    eprintln!(
        "shelled {} ± {}, plain {} ± {} (z {z:.1})",
        a.value, a.std_err, b.value, b.std_err
    );
    assert!(z > 3.0, "{} vs {} (z {z})", a.value, b.value);
}

#[test]
fn invalid_stacks_are_rejected() {
    let g = wire();
    let t = [NetId::new(0)];
    let err = solve(&g, &t, &cfg(Boundary::Open, 1.0, vec![layer(2.0, 3.9)])).unwrap_err();
    assert!(
        matches!(err, SolveError::Config(ref m) if m.contains("box")),
        "{err}"
    );
    let bad = vec![layer(2.0, 3.9), layer(1.0, 4.0)];
    let err = solve(&g, &t, &cfg(domain(), 1.0, bad)).unwrap_err();
    assert!(
        matches!(err, SolveError::Config(ref m) if m.contains("z_top")),
        "{err}"
    );
}

fn check(name: &str, est: Estimate, exact: f64) {
    let err = est.value - exact;
    eprintln!(
        "{name}: {:.6} ± {:.6} (exact {exact:.6}, {:+.3}%, {:+.1}σ)",
        est.value,
        est.std_err,
        100.0 * err / exact,
        err / est.std_err
    );
    assert!(
        err.abs() <= 4.0 * est.std_err + 2e-4 * exact.abs(),
        "{name}"
    );
}

fn plates_cfg(
    boundary: Boundary,
    substrate: Option<f64>,
    layers: Vec<DielectricLayer>,
    eps_top: f64,
) -> SolverConfig {
    let mut c = cfg(boundary, eps_top, layers);
    c.substrate_z = substrate;
    c.accuracy.total_rel_tol = 0.003;
    c.accuracy.min_walks = 20_000;
    c
}

/// L1: plate between the substrate and a grounded top face (reflecting sides),
/// interfaces parallel to the plate: C/A = 1 / Σ dᵢ/εᵢ on each side.
#[test]
fn l1_parallel_interfaces() {
    let (l, z0, z1, top) = (4.0, 1.0, 1.2, 1.8);
    for (stack, eps_top, below, above) in [
        // Interfaces 0.4 below and 0.3 above the plate, κ = 0.5 and 3.5.
        (
            vec![layer(0.4, 4.0), layer(1.5, 2.0)],
            7.0,
            1.0 / (0.4 / 4.0 + 0.6 / 2.0),
            1.0 / (0.3 / 2.0 + 0.3 / 7.0),
        ),
        // High contrast κ = 0.1 below the plate.
        (
            vec![layer(0.5, 10.0)],
            1.0,
            1.0 / (0.5 / 10.0 + 0.5),
            1.0 / 0.6,
        ),
    ] {
        let mut g = GeometryBuilder::new();
        let id = g.net("P").unwrap();
        g.add_box(id, Aabb::from_array([0.0, 0.0, z0, l, l, z1]).unwrap())
            .unwrap();
        let g = g.build().unwrap();
        let mut faces = [FaceKind::Reflect; 6];
        faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
        faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
        let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, top]).unwrap();
        let c = plates_cfg(Boundary::Box { bounds, faces }, Some(0.0), stack, eps_top);
        let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
        let r = &sol.nets[0];
        let a = l * l * EPS0_FF_PER_UM;
        check("L1 substrate", r.coupling(Sink::GroundPlane), a * below);
        check(
            "L1 top",
            r.coupling(Sink::Boundary(Face::new(Axis::Z, Side::Max))),
            a * above,
        );
        check("L1 total", r.total(), a * (below + above));
    }
}

/// Seeds of the long bias checks: `FARADICE_BIAS_SEEDS="2 3 4"` (default 2, 3).
fn bias_seeds() -> Vec<u64> {
    std::env::var("FARADICE_BIAS_SEEDS").map_or_else(
        |_| vec![2, 3],
        |v| v.split_whitespace().map(|s| s.parse().unwrap()).collect(),
    )
}

/// L8 (M2b): a plate in a stack of thin layers, its bottom face on an
/// interface (as metal levels in a PDK stack): C/A = 1 / Σ dᵢ/εᵢ on each side.
fn l8_case(tol: f64, seed: u64) -> (f64, f64) {
    let (l, z0, z1, top) = (4.0, 1.0, 1.2, 2.0);
    let stack = vec![
        layer(0.3, 3.9),
        layer(0.35, 7.3), // 50 nm
        layer(0.7, 4.05),
        layer(1.0, 4.5), // plate bottom on this interface
        layer(1.25, 4.2),
        layer(1.3, 7.5), // 50 nm
        layer(1.6, 4.1),
    ];
    let below = 1.0 / (0.3 / 3.9 + 0.05 / 7.3 + 0.35 / 4.05 + 0.3 / 4.5);
    let above = 1.0 / (0.05 / 4.2 + 0.05 / 7.5 + 0.3 / 4.1 + 0.4 / 3.0);
    let mut g = GeometryBuilder::new();
    let id = g.net("P").unwrap();
    g.add_box(id, Aabb::from_array([0.0, 0.0, z0, l, l, z1]).unwrap())
        .unwrap();
    let g = g.build().unwrap();
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, top]).unwrap();
    let mut c = plates_cfg(Boundary::Box { bounds, faces }, Some(0.0), stack, 3.0);
    c.accuracy.total_rel_tol = tol;
    c.seed = seed;
    let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
    let r = &sol.nets[0];
    let a = l * l * EPS0_FF_PER_UM;
    check("L8 substrate", r.coupling(Sink::GroundPlane), a * below);
    check(
        "L8 top",
        r.coupling(Sink::Boundary(Face::new(Axis::Z, Side::Max))),
        a * above,
    );
    check("L8 total", r.total(), a * (below + above));
    #[allow(clippy::cast_precision_loss)]
    let steps = r.total_steps as f64 / r.walks as f64;
    eprintln!("L8 walks {} steps/walk {steps:.1}", r.walks);
    #[allow(clippy::cast_precision_loss)]
    (r.walks as f64, steps)
}

#[test]
fn l8_thin_layers() {
    let (_, steps) = l8_case(0.003, 3);
    // Interface-anchored cubes cross the thin layers in a few steps.
    assert!(steps < 25.0, "{steps}");
}

#[test]
#[ignore = "bias check"]
fn l8_bias_check() {
    for seed in bias_seeds() {
        l8_case(0.0005, seed);
    }
}

/// L2: a vertical plate (normal x) spanning three layers between grounded x
/// faces, reflecting y and z faces: C = ε₀ W Σ εᵢhᵢ (1/d_left + 1/d_right).
#[test]
fn l2_perpendicular_interfaces() {
    let (w, height, x0, x1, len) = (2.0, 2.0, 1.0, 1.2, 2.0);
    let mut g = GeometryBuilder::new();
    let id = g.net("P").unwrap();
    g.add_box(id, Aabb::from_array([x0, 0.0, 0.0, x1, w, height]).unwrap())
        .unwrap();
    let g = g.build().unwrap();
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::X, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, len, w, height]).unwrap();
    let stack = vec![layer(0.7, 3.0), layer(1.2, 1.0)];
    let c = plates_cfg(Boundary::Box { bounds, faces }, None, stack, 6.0);
    let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
    let r = &sol.nets[0];
    let eh = 3.0 * 0.7 + 1.0 * 0.5 + 6.0 * 0.8;
    let side = |d: f64| EPS0_FF_PER_UM * w * eh / d;
    check(
        "L2 left",
        r.coupling(Sink::Boundary(Face::new(Axis::X, Side::Min))),
        side(x0),
    );
    check(
        "L2 right",
        r.coupling(Sink::Boundary(Face::new(Axis::X, Side::Max))),
        side(len - x1),
    );
    check("L2 total", r.total(), side(x0) + side(len - x1));
}

/// L9 (M2b): L2 with two 50 nm high-k layers: the lateral Gaussian faces
/// cross thin layers, whose first hops use half-bands of anchored cubes.
fn l9_case(tol: f64, seed: u64) -> f64 {
    let (w, height, x0, x1, len) = (2.0, 2.0, 1.0, 1.2, 2.0);
    let mut g = GeometryBuilder::new();
    let id = g.net("P").unwrap();
    g.add_box(id, Aabb::from_array([x0, 0.0, 0.0, x1, w, height]).unwrap())
        .unwrap();
    let g = g.build().unwrap();
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::X, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, len, w, height]).unwrap();
    let stack = vec![
        layer(0.6, 3.0),
        layer(0.65, 7.5),
        layer(1.2, 1.0),
        layer(1.25, 7.3),
    ];
    let mut c = plates_cfg(Boundary::Box { bounds, faces }, None, stack, 6.0);
    c.accuracy.total_rel_tol = tol;
    c.seed = seed;
    let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
    let r = &sol.nets[0];
    let eh = 3.0 * 0.6 + 7.5 * 0.05 + 1.0 * 0.55 + 7.3 * 0.05 + 6.0 * 0.75;
    let side = |d: f64| EPS0_FF_PER_UM * w * eh / d;
    check(
        "L9 left",
        r.coupling(Sink::Boundary(Face::new(Axis::X, Side::Min))),
        side(x0),
    );
    check(
        "L9 right",
        r.coupling(Sink::Boundary(Face::new(Axis::X, Side::Max))),
        side(len - x1),
    );
    check("L9 total", r.total(), side(x0) + side(len - x1));
    #[allow(clippy::cast_precision_loss)]
    let walks = r.walks as f64;
    eprintln!("L9 walks {walks}");
    walks
}

#[test]
fn l9_lateral_faces_across_thin_layers() {
    l9_case(0.003, 3);
}

/// Long L9 runs for a bias estimate (`--ignored`).
#[test]
#[ignore = "long: 50 M walks per seed"]
fn l9_bias_check() {
    for seed in bias_seeds() {
        l9_case(0.0005, seed);
    }
}

/// L3: conductors symmetric about an interface in a box symmetric about it:
/// the capacitance matrix is (ε₁+ε₂)/2 times the homogeneous one (ε = 1).
#[test]
fn l3_symmetric_about_the_interface() {
    let mut g = GeometryBuilder::new();
    for (name, x) in [("A", -1.2), ("B", 0.4)] {
        let id = g.net(name).unwrap();
        g.add_box(
            id,
            Aabb::from_array([x, -0.4, -0.4, x + 0.8, 0.4, 0.4]).unwrap(),
        )
        .unwrap();
    }
    let g = g.build().unwrap();
    let bounds = Aabb::from_array([-3.0, -2.0, -2.0, 3.0, 2.0, 2.0]).unwrap();
    let t = [NetId::new(0), NetId::new(1)];
    let mut hom = plates_cfg(Boundary::grounded_box(bounds), None, Vec::new(), 1.0);
    hom.seed = 11;
    let lay = plates_cfg(
        Boundary::grounded_box(bounds),
        None,
        vec![layer(0.0, 1.0)],
        5.0,
    );
    let (h, s) = (solve(&g, &t, &hom).unwrap(), solve(&g, &t, &lay).unwrap());
    let scaled = |e: Estimate| Estimate {
        value: 3.0 * e.value,
        std_err: 3.0 * e.std_err,
    };
    for i in 0..2 {
        let (a, b) = (&s.nets[i], &h.nets[i]);
        let diff = |x: Estimate, y: Estimate, name: &str| {
            let y = scaled(y);
            let sigma = x.std_err.hypot(y.std_err);
            eprintln!(
                "{name}: {:.6} vs {:.6} ({:+.1}σ)",
                x.value,
                y.value,
                (x.value - y.value) / sigma
            );
            assert!(
                (x.value - y.value).abs() <= 4.0 * sigma + 2e-4 * y.value.abs(),
                "{name}"
            );
        };
        diff(a.total(), b.total(), "L3 total");
        let other = Sink::Net(NetId::new(if i == 0 { 1 } else { 0 }));
        diff(a.coupling(other), b.coupling(other), "L3 coupling");
    }
}

/// L7: results with a stack do not depend on the thread count.
#[test]
fn l7_stack_is_deterministic() {
    let g = wire();
    let t = [NetId::new(0)];
    let c = cfg(domain(), 1.0, vec![layer(0.8, 4.0), layer(1.3, 2.0)]);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| solve(&g, &t, &c).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.nets[0].walks, b.nets[0].walks);
    assert_eq!(
        a.nets[0].total().value.to_bits(),
        b.nets[0].total().value.to_bits()
    );
}

#[test]
fn a_stack_that_is_homogeneous_in_the_domain_uses_its_permittivity() {
    let g = wire();
    let t = [NetId::new(0)];
    let reference = solve(&g, &t, &cfg(domain(), 3.9, Vec::new())).unwrap();
    // One layer covering the whole domain (its top is above the domain).
    let covered = solve(&g, &t, &cfg(domain(), 1.0, vec![layer(10.0, 3.9)])).unwrap();
    // Equal permittivities merge into no interface.
    let merged = solve(&g, &t, &cfg(domain(), 3.9, vec![layer(2.0, 3.9)])).unwrap();
    let c = |s: &faradice_frw::Solution| s.nets[0].total().value;
    assert_eq!(c(&reference), c(&covered));
    assert_eq!(c(&reference), c(&merged));
}

#[test]
#[ignore = "bias check"]
fn l2_bias_check() {
    let (w, height, x0, x1, len) = (2.0, 2.0, 1.0, 1.2, 2.0);
    let mut g = GeometryBuilder::new();
    let id = g.net("P").unwrap();
    g.add_box(id, Aabb::from_array([x0, 0.0, 0.0, x1, w, height]).unwrap())
        .unwrap();
    let g = g.build().unwrap();
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::X, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, len, w, height]).unwrap();
    let eh = 3.0 * 0.7 + 1.0 * 0.5 + 6.0 * 0.8;
    let side = |d: f64| EPS0_FF_PER_UM * w * eh / d;
    for seed in bias_seeds() {
        let mut c = plates_cfg(
            Boundary::Box { bounds, faces },
            None,
            vec![layer(0.7, 3.0), layer(1.2, 1.0)],
            6.0,
        );
        c.accuracy.total_rel_tol = 0.0006;
        c.seed = seed;
        let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
        let r = &sol.nets[0];
        check(
            "L2 left",
            r.coupling(Sink::Boundary(Face::new(Axis::X, Side::Min))),
            side(x0),
        );
        check(
            "L2 right",
            r.coupling(Sink::Boundary(Face::new(Axis::X, Side::Max))),
            side(len - x1),
        );
        check("L2 total", r.total(), side(x0) + side(len - x1));
        eprintln!(
            "walks {} steps/walk {:.1}",
            r.walks,
            r.total_steps as f64 / r.walks as f64
        );
    }
}

/// A horizontal face on an interface at every coarse trial offset still gets
/// a positive separation (no panic, finite result).
#[test]
fn gaussian_offset_avoids_interfaces() {
    let g = wire(); // top at z = 1.5, clearance to the box top 2.5
    let mut layers = Vec::new();
    for k in 0..40 {
        let z = 1.5 + 0.02 * f64::from(k) + 0.01;
        layers.push(layer(z, if k % 2 == 0 { 2.0 } else { 3.0 }));
    }
    let mut c = cfg(domain(), 1.0, layers);
    c.accuracy.min_walks = 1_000;
    c.accuracy.max_walks = 20_000;
    let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
    assert!(sol.nets[0].total().value.is_finite());
}
