//! GPU engine vs analytic results and the CPU solver (statistical agreement).
//!
//! Tests are skipped (with a message) when no hardware Vulkan adapter exists.

#![allow(missing_docs)]

use faradice_core::{Axis, EPS0_FF_PER_UM, Estimate, Face, NetId, Side, Sink};
use faradice_frw::{
    Accuracy, Boundary, FaceKind, Solution, SolveError, SolverConfig, WalkOptions, solve,
    solve_with,
};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};
use faradice_gpu::{GpuEngine, GpuError, GpuOptions};

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
        seed: 7,
        ground_nets: Vec::new(),
        regions: Default::default(),
    }
}

/// `None` if no GPU is available.
fn gpu_solve(g: &Geometry, targets: &[NetId], c: &SolverConfig) -> Option<Solution> {
    let mut skipped = false;
    let r = solve_with(g, targets, c, |scene| {
        GpuEngine::new(scene, GpuOptions::default()).map_err(|e| {
            if matches!(e, GpuError::NoAdapter(_)) {
                skipped = true;
            }
            e.to_string()
        })
    });
    match r {
        Ok(s) => Some(s),
        Err(SolveError::Engine(m)) if skipped => {
            eprintln!("skipping GPU test: {m}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

fn agree(name: &str, a: Estimate, b: Estimate) {
    let s = (a.std_err * a.std_err + b.std_err * b.std_err).sqrt();
    eprintln!(
        "{name}: gpu {:.6} ± {:.6}, cpu {:.6} ± {:.6}",
        a.value, a.std_err, b.value, b.std_err
    );
    assert!(
        (a.value - b.value).abs() <= 4.5 * s + 1e-12,
        "{name}: {} vs {} (σ {s})",
        a.value,
        b.value
    );
}

#[test]
fn parallel_plate_matches_exact() {
    let (l, d1, t, d2) = (4.0, 1.0, 0.2, 0.6);
    let g = geom(&[("P", [0.0, 0.0, d1, l, l, d1 + t])]);
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, d1 + t + d2]).unwrap();
    let c = cfg(Boundary::Box { bounds, faces }, Some(0.0), 0.003);
    let Some(sol) = gpu_solve(&g, &[NetId::new(0)], &c) else {
        return;
    };
    let r = &sol.nets[0];
    assert!(r.converged && r.aborted_walks == 0);
    let exact = EPS0_FF_PER_UM * l * l * (1.0 / d1 + 1.0 / d2);
    let e = r.total();
    eprintln!("plate: {} ± {} (exact {exact})", e.value, e.std_err);
    assert!((e.value - exact).abs() <= 4.0 * e.std_err + 0.003 * exact);
}

#[test]
fn three_wires_agree_with_cpu() {
    // Three parallel wires over a substrate in a grounded box.
    let g = geom(&[
        ("a", [0.0, 0.0, 1.0, 10.0, 0.2, 1.3]),
        ("b", [0.0, 0.5, 1.0, 10.0, 0.7, 1.3]),
        ("c", [0.0, 1.0, 1.0, 10.0, 1.2, 1.3]),
    ]);
    let bounds = Aabb::from_array([-2.0, -2.0, 0.0, 12.0, 3.2, 4.0]).unwrap();
    let mut c = cfg(Boundary::grounded_box(bounds), Some(0.0), 0.004);
    c.accuracy.coupling_rel_tol = 0.02;
    let targets = [NetId::new(1)];
    let Some(gpu) = gpu_solve(&g, &targets, &c) else {
        return;
    };
    let cpu = solve(&g, &targets, &c).unwrap();
    let (rg, rc) = (&gpu.nets[0], &cpu.nets[0]);
    assert!(rg.converged);
    agree("total", rg.total(), rc.total());
    agree("ground", rg.ground(), rc.ground());
    for n in [0, 2] {
        let s = Sink::Net(NetId::new(n));
        agree(&format!("coupling {n}"), rg.coupling(s), rc.coupling(s));
    }
    // Same work per walk on average.
    #[allow(clippy::cast_precision_loss)]
    let steps = |r: &faradice_frw::NetResult| r.total_steps as f64 / r.walks as f64;
    let (sg, sc) = (steps(rg), steps(rc));
    assert!((sg / sc - 1.0).abs() < 0.05, "steps/walk gpu {sg} cpu {sc}");
}

#[test]
fn open_boundary_with_substrate_agrees_with_cpu() {
    // Odd image extension: walks change sign when reflected across the substrate.
    let g = geom(&[
        ("x", [0.0, 0.0, 0.5, 1.0, 1.0, 1.5]),
        ("y", [1.5, 0.0, 0.5, 2.5, 1.0, 1.5]),
    ]);
    let c = cfg(Boundary::Open, Some(0.0), 0.005);
    let targets = [NetId::new(0)];
    let Some(gpu) = gpu_solve(&g, &targets, &c) else {
        return;
    };
    let cpu = solve(&g, &targets, &c).unwrap();
    let (rg, rc) = (&gpu.nets[0], &cpu.nets[0]);
    agree("total", rg.total(), rc.total());
    agree("ground", rg.ground(), rc.ground());
    let y = Sink::Net(NetId::new(1));
    agree("coupling", rg.coupling(y), rc.coupling(y));
}

#[test]
fn deterministic_per_seed() {
    let g = geom(&[("p", [0.0, 0.0, 1.0, 2.0, 2.0, 1.5])]);
    let bounds = Aabb::from_array([-3.0, -3.0, 0.0, 5.0, 5.0, 4.0]).unwrap();
    let mut c = cfg(Boundary::grounded_box(bounds), Some(0.0), 0.02);
    c.accuracy.max_walks = 100_000;
    let Some(a) = gpu_solve(&g, &[NetId::new(0)], &c) else {
        return;
    };
    let b = gpu_solve(&g, &[NetId::new(0)], &c).unwrap();
    assert_eq!(a.nets[0].walks, b.nets[0].walks);
    assert_eq!(
        a.nets[0].total().value.to_bits(),
        b.nets[0].total().value.to_bits()
    );
}

#[test]
fn precision_guard_rejects_tiny_stop_distance() {
    let g = geom(&[("p", [0.0, 0.0, 1.0, 1000.0, 1.0, 1.5])]);
    let mut c = cfg(Boundary::Open, Some(0.0), 0.02);
    c.walk.stop_dist = Some(1e-6);
    let r = solve_with(&g, &[NetId::new(0)], &c, |scene| {
        GpuEngine::new(scene, GpuOptions::default()).map_err(|e| e.to_string())
    });
    match r {
        Err(SolveError::Engine(m)) => assert!(m.contains("precision"), "{m}"),
        other => panic!("expected a precision error, got {other:?}"),
    }
}
