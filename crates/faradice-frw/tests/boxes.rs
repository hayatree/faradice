//! Dielectric boxes at the solver level (design M2c §9: C1–C3, C7).

#![allow(missing_docs)]

use faradice_core::{Axis, EPS0_FF_PER_UM, Face, NetId, Side};
use faradice_frw::{
    Accuracy, Boundary, DielectricBox, DielectricLayer, DielectricRegions, FaceKind, SolverConfig,
    WalkOptions, solve,
};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};

fn plate(c: [f64; 6]) -> Geometry {
    let mut g = GeometryBuilder::new();
    let id = g.net("P").unwrap();
    g.add_box(id, Aabb::from_array(c).unwrap()).unwrap();
    g.build().unwrap()
}

fn dbox(c: [f64; 6], eps_r: f64) -> DielectricBox {
    DielectricBox {
        region: Aabb::from_array(c).unwrap(),
        eps_r,
    }
}

/// A box domain grounded on the two faces normal to `grounded`, reflecting
/// elsewhere.
fn cfg(
    bounds: [f64; 6],
    grounded: Axis,
    eps_r: f64,
    layers: Vec<DielectricLayer>,
    boxes: Vec<DielectricBox>,
    tol: f64,
) -> SolverConfig {
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(grounded, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(grounded, Side::Max).index()] = FaceKind::Ground;
    SolverConfig {
        eps_r,
        layers,
        boundary: Boundary::Box {
            bounds: Aabb::from_array(bounds).unwrap(),
            faces,
        },
        substrate_z: None,
        accuracy: Accuracy {
            total_rel_tol: tol,
            min_walks: 20_000,
            max_walks: 20_000_000,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed: 11,
        ground_nets: Vec::new(),
        regions: DielectricRegions {
            boxes,
            conformal: Vec::new(),
        },
    }
}

fn check(name: &str, g: &Geometry, c: &SolverConfig, exact: f64) {
    let sol = solve(g, &[NetId::new(0)], c).unwrap();
    let est = sol.nets[0].total();
    let z = (est.value - exact) / est.std_err;
    eprintln!(
        "{name}: {:.6} ± {:.6} vs {exact:.6} (z = {z:+.2}, {} walks)",
        est.value, est.std_err, sol.nets[0].walks
    );
    assert!(z.abs() < 4.0, "{name}: z = {z}");
}

/// C1: a vertical plate between grounded x faces, with box slabs normal to x
/// in both gaps (κ = 0.1, 0.78 and 10 against the background ε = 10):
/// series capacitors, exact.
#[test]
fn c1_vertical_plates_with_box_slabs() {
    let g = plate([1.0, 0.0, 0.0, 1.2, 2.0, 2.0]);
    let boxes = vec![
        dbox([0.1, 0.0, 0.0, 0.4, 2.0, 2.0], 1.0),
        dbox([1.7, 0.0, 0.0, 1.85, 2.0, 2.0], 7.8),
        dbox([1.85, 0.0, 0.0, 2.0, 2.0, 2.0], 100.0),
    ];
    let c = cfg(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        Axis::X,
        10.0,
        Vec::new(),
        boxes,
        2e-3,
    );
    let left = 0.3 / 1.0 + 0.7 / 10.0;
    let right = 0.5 / 10.0 + 0.15 / 7.8 + 0.15 / 100.0;
    let exact = EPS0_FF_PER_UM * 4.0 * (1.0 / left + 1.0 / right);
    check("C1", &g, &c, exact);
}

/// C2: L1a (horizontal plate between interfaces) with the bottom layer given
/// as a box spanning the domain instead of a stack layer.
#[test]
fn c2_a_stack_layer_as_a_box() {
    let g = plate([0.0, 0.0, 1.0, 4.0, 4.0, 1.2]);
    let series = |d: &[(f64, f64)]| 1.0 / d.iter().map(|(t, e)| t / e).sum::<f64>();
    let exact = EPS0_FF_PER_UM
        * 16.0
        * (series(&[(0.4, 4.0), (0.6, 2.0)]) + series(&[(0.3, 2.0), (0.3, 7.0)]));
    let c = cfg(
        [0.0, 0.0, 0.0, 4.0, 4.0, 1.8],
        Axis::Z,
        7.0,
        vec![DielectricLayer {
            z_top: 1.5,
            eps_r: 2.0,
        }],
        vec![dbox([0.0, 0.0, 0.0, 4.0, 4.0, 0.4], 4.0)],
        2e-3,
    );
    check("C2", &g, &c, exact);
}

/// C3: boxes with the permittivity of their surroundings have no interfaces;
/// the answer is the stack's.
#[test]
fn c3_boxes_matching_the_medium() {
    let g = plate([1.0, 0.0, 0.0, 1.2, 2.0, 2.0]);
    let boxes = vec![
        dbox([0.2, 0.5, 0.5, 0.5, 1.5, 1.5], 3.0),
        dbox([1.5, 0.0, 0.0, 1.8, 2.0, 2.0], 3.0),
    ];
    let c = cfg(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        Axis::X,
        3.0,
        Vec::new(),
        boxes,
        2e-3,
    );
    let exact = EPS0_FF_PER_UM * 3.0 * 4.0 * (1.0 / 1.0 + 1.0 / 0.8);
    check("C3", &g, &c, exact);
}

/// C7: results with boxes do not depend on the thread count.
#[test]
fn c7_boxes_are_deterministic() {
    let g = plate([1.0, 0.0, 0.0, 1.2, 2.0, 2.0]);
    let boxes = vec![dbox([1.7, 0.0, 0.5, 1.85, 2.0, 1.5], 3.5)];
    let c = cfg(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        Axis::X,
        4.5,
        Vec::new(),
        boxes,
        2e-2,
    );
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| solve(&g, &[NetId::new(0)], &c).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.nets[0].walks, b.nets[0].walks);
    assert_eq!(
        a.nets[0].total().value.to_bits(),
        b.nets[0].total().value.to_bits()
    );
}

/// Bias check of C1 and C2 over several seeds (long; `--ignored`).
#[test]
#[ignore = "long: bias check over seeds"]
fn c_bias_check() {
    let seeds: Vec<u64> = std::env::var("FARADICE_BIAS_SEEDS")
        .unwrap_or_else(|_| "1 2 3 4 5 6 7 8".into())
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let series = |d: &[(f64, f64)]| 1.0 / d.iter().map(|(t, e)| t / e).sum::<f64>();
    let c1 = (
        plate([1.0, 0.0, 0.0, 1.2, 2.0, 2.0]),
        cfg(
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            Axis::X,
            10.0,
            Vec::new(),
            vec![
                dbox([0.1, 0.0, 0.0, 0.4, 2.0, 2.0], 1.0),
                dbox([1.7, 0.0, 0.0, 1.85, 2.0, 2.0], 7.8),
                dbox([1.85, 0.0, 0.0, 2.0, 2.0, 2.0], 100.0),
            ],
            5e-4,
        ),
        EPS0_FF_PER_UM
            * 4.0
            * (1.0 / (0.3 / 1.0 + 0.7 / 10.0) + 1.0 / (0.5 / 10.0 + 0.15 / 7.8 + 0.15 / 100.0)),
    );
    let c2 = (
        plate([0.0, 0.0, 1.0, 4.0, 4.0, 1.2]),
        cfg(
            [0.0, 0.0, 0.0, 4.0, 4.0, 1.8],
            Axis::Z,
            7.0,
            vec![DielectricLayer {
                z_top: 1.5,
                eps_r: 2.0,
            }],
            vec![dbox([0.0, 0.0, 0.0, 4.0, 4.0, 0.4], 4.0)],
            5e-4,
        ),
        EPS0_FF_PER_UM
            * 16.0
            * (series(&[(0.4, 4.0), (0.6, 2.0)]) + series(&[(0.3, 2.0), (0.3, 7.0)])),
    );
    // Axis-cube first hops (normal kernel) next to a box face.
    let c1b = (
        plate([1.0, 0.0, 0.0, 1.25, 2.0, 2.0]),
        cfg(
            [0.0, 0.0, 0.0, 2.25, 2.0, 2.0],
            Axis::X,
            1.0,
            Vec::new(),
            vec![dbox([0.5, 0.0, 0.0, 1.0, 2.0, 2.0], 4.0)],
            5e-4,
        ),
        EPS0_FF_PER_UM * 4.0 * (1.0 / (0.5 / 1.0 + 0.5 / 4.0) + 1.0 / 1.0),
    );
    for (name, (g, base, exact)) in [("C1", c1), ("C2", c2), ("C1b", c1b)] {
        let (mut sum, mut var) = (0.0, 0.0);
        for &seed in &seeds {
            let mut c = base.clone();
            c.seed = seed;
            c.accuracy.max_walks = 200_000_000;
            let est = solve(&g, &[NetId::new(0)], &c).unwrap().nets[0].total();
            let rel = est.value / exact - 1.0;
            let se = est.std_err / exact;
            eprintln!(
                "{name} seed {seed}: {:+.4} % ± {:.4} %",
                100.0 * rel,
                100.0 * se
            );
            sum += rel;
            var += se * se;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = seeds.len() as f64;
        let (mean, se) = (sum / n, var.sqrt() / n);
        eprintln!(
            "{name}: mean {:+.4} % ± {:.4} % ({:+.1}σ)",
            100.0 * mean,
            100.0 * se,
            mean / se
        );
    }
}

/// The default Gaussian offset would put the left Gaussian face on a box
/// face; another offset is chosen (design M2c §5) and the answer is exact.
#[test]
fn gaussian_offsets_avoid_interfaces() {
    let g = plate([1.0, 0.0, 0.0, 1.25, 2.0, 2.0]);
    // δ₀ = 0.5 × min(clearance 1.0, extent 2.0) = 0.5: x = 0.5 is the box face.
    let c = cfg(
        [0.0, 0.0, 0.0, 2.25, 2.0, 2.0],
        Axis::X,
        1.0,
        Vec::new(),
        vec![dbox([0.5, 0.0, 0.0, 1.0, 2.0, 2.0], 4.0)],
        2e-3,
    );
    let exact = EPS0_FF_PER_UM * 4.0 * (1.0 / (0.5 / 1.0 + 0.5 / 4.0) + 1.0 / 1.0);
    check("offset", &g, &c, exact);
}

/// A box strip crossing every candidate Gaussian face transversely leaves no
/// offset: a typed error.
#[test]
fn gaussian_surfaces_crossing_interfaces_are_refused() {
    let g = plate([1.0, 0.0, 0.0, 1.25, 2.0, 2.0]);
    let c = cfg(
        [0.0, 0.0, 0.0, 2.25, 2.0, 2.0],
        Axis::X,
        1.0,
        Vec::new(),
        vec![dbox([0.4, 0.9, 0.0, 0.9, 1.1, 2.0], 4.0)],
        2e-2,
    );
    let err = solve(&g, &[NetId::new(0)], &c).unwrap_err();
    assert!(err.to_string().contains("Gaussian offset"), "{err}");
}

/// C6: potentials at and near a four-media edge (design M2c §4.6). Below a
/// plate spanning the domain, ε = A(y)·B(z) with A = 1, 4 (interface
/// y = 0.4) and B = 1, 2 (interface z = 1): flux continuity holds for
/// φ = φ(y), the series potential, with an edge of four media at
/// (y, z) = (0.4, 1). Map walks start on the edge, a few ulps off it, on its
/// planes and away from it.
#[cfg(feature = "sensitivity")]
#[test]
fn c6_potentials_at_a_four_media_edge() {
    use faradice_frw::map::{MapOptions, potential_map};
    let g = plate([0.0, 1.0, 0.0, 2.0, 1.2, 2.0]);
    let c = cfg(
        [0.0, 0.0, 0.0, 2.0, 2.2, 2.0],
        Axis::Y,
        1.0,
        Vec::new(),
        vec![
            dbox([0.0, 0.0, 1.0, 2.0, 0.4, 2.0], 2.0),
            dbox([0.0, 0.4, 0.0, 2.0, 1.0, 1.0], 4.0),
            dbox([0.0, 0.4, 1.0, 2.0, 1.0, 2.0], 8.0),
        ],
        2e-2,
    );
    let exact = |y: f64| {
        let t = if y <= 0.4 { y } else { 0.4 + (y - 0.4) / 4.0 };
        t / 0.55
    };
    // On the edge, a few ulps off it, on its planes, near it (landing boxes
    // and edge cubes, design M2d) and away from it.
    let points = [
        [1.0, 0.4, 1.0],
        [1.0, 0.4 + 1e-10, 1.0 - 1e-10],
        [0.5, 0.2, 1.0],
        [1.0, 0.4, 0.3],
        [1.0, 0.41, 1.02],
        [0.7, 0.39, 0.995],
        [1.2, 0.4, 1.03],
        [1.3, 0.7, 0.4],
    ];
    let opts = MapOptions {
        samples: 40_000,
        gradient: false,
    };
    let map = potential_map(&g, &[NetId::new(0)], &c, &points, opts).unwrap();
    for m in &map {
        let phi = m.phi.iter().find(|(n, _)| *n == NetId::new(0)).unwrap().1;
        let e = exact(m.point[1]);
        let z = (phi.value - e) / phi.std_err;
        eprintln!(
            "{:?}: {:.4} ± {:.4} vs {e:.4} (z {z:+.2})",
            m.point, phi.value, phi.std_err
        );
        assert!(z.abs() < 4.0, "{:?}: z = {z}", m.point);
    }
}

/// Map points on a dielectric vertex (three interface planes) are refused
/// (design M2c §4.2); points on an edge are not.
#[cfg(feature = "sensitivity")]
#[test]
fn map_points_on_dielectric_vertices_are_refused() {
    use faradice_frw::map::{MapOptions, potential_map};
    let g = plate([2.0, -1.0, -1.0, 3.0, 1.0, 1.0]);
    let mut c = cfg(
        [-4.0, -4.0, -4.0, 4.0, 4.0, 4.0],
        Axis::X,
        1.0,
        Vec::new(),
        vec![dbox([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 4.0)],
        2e-2,
    );
    c.boundary =
        Boundary::grounded_box(Aabb::from_array([-4.0, -4.0, -4.0, 4.0, 4.0, 4.0]).unwrap());
    let opts = MapOptions {
        samples: 16,
        gradient: false,
    };
    let err = potential_map(&g, &[NetId::new(0)], &c, &[[0.0, 0.0, 0.0]], opts).unwrap_err();
    assert!(err.to_string().contains("dielectric"), "{err}");
    for p in [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]] {
        assert!(potential_map(&g, &[NetId::new(0)], &c, &[p], opts).is_ok());
    }
}

/// C6: a solve across many dielectric edges (a strip box through a stack
/// interface, a conformal shell) does not depend on the stopping distance
/// beyond its statistics, and no walk aborts.
/// The C6 geometry: two wires, a strip box through a stack interface and a
/// conformal shell (many dielectric edges).
fn c6_scene() -> (Geometry, SolverConfig, NetId) {
    use faradice_core::BoxId;
    use faradice_frw::ConformalRule;
    let mut gb = GeometryBuilder::new();
    let a = gb.net("a").unwrap();
    gb.add_box(a, Aabb::from_array([0.0, 0.0, 1.0, 6.0, 0.2, 1.3]).unwrap())
        .unwrap();
    let b = gb.net("b").unwrap();
    gb.add_box(b, Aabb::from_array([0.0, 1.2, 1.0, 6.0, 1.4, 1.3]).unwrap())
        .unwrap();
    let g = gb.build().unwrap();
    let base = {
        let mut c = cfg(
            [-1.0, -1.0, 0.0, 7.0, 2.6, 3.0],
            Axis::Z,
            3.9,
            vec![DielectricLayer {
                z_top: 1.15,
                eps_r: 7.0,
            }],
            vec![dbox([-1.0, 0.45, 0.8, 7.0, 0.95, 1.5], 5.0)],
            1e-3,
        );
        c.regions.conformal = vec![ConformalRule {
            layer: "m1".into(),
            boxes: vec![BoxId::new(1)],
            side: 0.05,
            top: 0.05,
            eps_r: 3.0,
        }];
        c.accuracy.min_walks = 300_000;
        c.accuracy.max_walks = 300_000;
        c
    };
    (g, base, b)
}

#[test]
fn c6_edges_and_the_stopping_distance() {
    let (g, base, b) = c6_scene();
    let run = |stop: f64| {
        let mut c = base.clone();
        c.walk.stop_dist = Some(stop);
        let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
        let n = &sol.nets[0];
        assert_eq!(n.aborted_walks, 0, "stop {stop}");
        (n.total(), n.c(faradice_core::Sink::Net(b)))
    };
    let reference = run(1e-4);
    for stop in [3e-4, 1e-4 / 3.0] {
        let other = run(stop);
        for (name, x, y) in [
            ("total", reference.0, other.0),
            ("coupling", reference.1, other.1),
        ] {
            let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
            eprintln!(
                "stop {stop:.1e} {name}: {:.5} vs {:.5} (z {z:+.2})",
                y.value, x.value
            );
            assert!(z.abs() < 4.0, "stop {stop} {name}: z = {z}");
        }
    }
}

/// A finite plate over a bottom layer, three ways: the layer in the stack
/// (M2b), as a box (z axis cubes), and the whole problem rotated so that the
/// layer is a box normal to x. The lateral Gaussian faces use the rotated
/// tangential kernels (both cyclic orientations in the rotated case), the
/// faces facing the layer the rotated normal kernel.
#[test]
fn box_layers_agree_with_the_stack_in_any_orientation() {
    let solve_one = |g: &Geometry, c: &SolverConfig| {
        let sol = solve(g, &[NetId::new(0)], c).unwrap();
        sol.nets[0].total()
    };
    let z_plate = plate([1.0, 1.0, 1.0, 3.0, 3.0, 1.2]);
    let stack = solve_one(
        &z_plate,
        &cfg(
            [0.0, 0.0, 0.0, 4.0, 4.0, 1.8],
            Axis::Z,
            2.0,
            vec![DielectricLayer {
                z_top: 0.4,
                eps_r: 4.0,
            }],
            Vec::new(),
            2e-3,
        ),
    );
    let z_box = solve_one(
        &z_plate,
        &cfg(
            [0.0, 0.0, 0.0, 4.0, 4.0, 1.8],
            Axis::Z,
            2.0,
            Vec::new(),
            vec![dbox([0.0, 0.0, 0.0, 4.0, 4.0, 0.4], 4.0)],
            2e-3,
        ),
    );
    // (x, y, z) → (z, x, y): the layer normal becomes x.
    let x_box = solve_one(
        &plate([1.0, 1.0, 1.0, 1.2, 3.0, 3.0]),
        &cfg(
            [0.0, 0.0, 0.0, 1.8, 4.0, 4.0],
            Axis::X,
            2.0,
            Vec::new(),
            vec![dbox([0.0, 0.0, 0.0, 0.4, 4.0, 4.0], 4.0)],
            2e-3,
        ),
    );
    for (name, e) in [("z box", z_box), ("x box", x_box)] {
        let z = (e.value - stack.value) / e.std_err.hypot(stack.std_err);
        eprintln!(
            "{name}: {:.6} ± {:.6} vs stack {:.6} ± {:.6} (z = {z:+.2})",
            e.value, e.std_err, stack.value, stack.std_err
        );
        assert!(z.abs() < 4.0, "{name}: z = {z}");
    }
}

/// Design M2d: landing boxes and edge cubes leave the results of the C6
/// scene unchanged (against the M2c steps alone) and take fewer steps.
#[test]
fn edge_cubes_agree_with_the_m2c_steps() {
    let (g, base, b) = c6_scene();
    let run = |edge_cubes: bool| {
        let mut c = base.clone();
        c.walk.stack.edge_cubes = edge_cubes;
        let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
        let n = &sol.nets[0];
        assert_eq!(n.aborted_walks, 0, "edge cubes {edge_cubes}");
        #[allow(clippy::cast_precision_loss)]
        let steps = n.total_steps as f64 / n.walks as f64;
        (n.total(), n.c(faradice_core::Sink::Net(b)), steps)
    };
    let on = run(true);
    let off = run(false);
    for (name, x, y) in [("total", on.0, off.0), ("coupling", on.1, off.1)] {
        let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
        eprintln!("{name}: {:.5} vs {:.5} (z {z:+.2})", x.value, y.value);
        assert!(z.abs() < 4.0, "{name}: z = {z}");
    }
    eprintln!("steps per walk: {:.1} vs {:.1}", on.2, off.2);
    assert!(on.2 < off.2, "{} vs {}", on.2, off.2);
}

/// C9 (design M2e): thin shells (0.03) on both sides of a plate and slabs
/// further out, so that the cubes of most Gaussian points span several
/// interfaces parallel to the plate (layered first hops). Series capacitors,
/// exact.
#[test]
fn c9_plate_between_thin_shells() {
    let g = plate([1.0, 0.0, 0.0, 1.2, 2.0, 2.0]);
    let layers_left = [
        (0.0, 0.7, 4.0),
        (0.7, 0.9, 2.0),
        (0.9, 0.97, 4.0),
        (0.97, 1.0, 3.5),
    ];
    let layers_right = [
        (1.2, 1.23, 3.5),
        (1.23, 1.3, 4.0),
        (1.3, 1.45, 7.0),
        (1.45, 2.0, 4.0),
    ];
    let boxes = layers_left
        .iter()
        .chain(&layers_right)
        .filter(|&&(_, _, e): &&(f64, f64, f64)| e.to_bits() != 4.0f64.to_bits())
        .map(|&(lo, hi, e)| dbox([lo, 0.0, 0.0, hi, 2.0, 2.0], e))
        .collect();
    let c = cfg(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        Axis::X,
        4.0,
        Vec::new(),
        boxes,
        1e-3,
    );
    let r = |ls: &[(f64, f64, f64)]| ls.iter().map(|(lo, hi, e)| (hi - lo) / e).sum::<f64>();
    let exact = EPS0_FF_PER_UM * 4.0 * (1.0 / r(&layers_left) + 1.0 / r(&layers_right));
    check("C9", &g, &c, exact);
}

/// C10 (design M2f): two wires with conformal sidewall shells whose bottom
/// lies on a stack plane, as the bus of C5: window first hops leave the
/// results unchanged (against the point first hops alone) and reduce the
/// variance.
#[test]
fn c10_window_first_hops_agree_with_the_point_hops() {
    use faradice_core::BoxId;
    use faradice_frw::ConformalRule;
    let mut gb = GeometryBuilder::new();
    let a = gb.net("a").unwrap();
    gb.add_box(
        a,
        Aabb::from_array([0.0, 0.0, 1.0, 4.0, 0.14, 1.36]).unwrap(),
    )
    .unwrap();
    let b = gb.net("b").unwrap();
    gb.add_box(
        b,
        Aabb::from_array([0.0, 0.28, 1.0, 4.0, 0.42, 1.36]).unwrap(),
    )
    .unwrap();
    let g = gb.build().unwrap();
    let mut base = cfg(
        [-1.0, -1.0, 0.0, 5.0, 1.42, 3.0],
        Axis::Z,
        4.2,
        vec![DielectricLayer {
            z_top: 1.0,
            eps_r: 4.5,
        }],
        Vec::new(),
        1e-3,
    );
    base.regions.conformal = vec![ConformalRule {
        layer: "m".into(),
        boxes: vec![BoxId::new(0), BoxId::new(1)],
        side: 0.03,
        top: 0.0,
        eps_r: 3.5,
    }];
    base.accuracy.min_walks = 200_000;
    base.accuracy.max_walks = 200_000;
    let run = |windows: bool| {
        let mut c = base.clone();
        c.walk.stack.window_hops = windows;
        let sol = solve(&g, &[NetId::new(0)], &c).unwrap();
        let n = &sol.nets[0];
        assert_eq!(n.aborted_walks, 0, "windows {windows}");
        (n.total(), n.c(faradice_core::Sink::Net(b)))
    };
    let on = run(true);
    let off = run(false);
    for (name, x, y) in [("total", on.0, off.0), ("coupling", on.1, off.1)] {
        let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
        eprintln!(
            "{name}: {:.5} ± {:.5} vs {:.5} ± {:.5} (z {z:+.2})",
            x.value, x.std_err, y.value, y.std_err
        );
        assert!(z.abs() < 4.0, "{name}: z = {z}");
    }
    assert!(
        on.0.std_err < 0.95 * off.0.std_err,
        "{} vs {}",
        on.0.std_err,
        off.0.std_err
    );
}
