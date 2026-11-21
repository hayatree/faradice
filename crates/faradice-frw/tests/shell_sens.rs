//! Sensitivities of conductors with conformal shells (design D3 §4.4c, S15).

#![cfg(feature = "sensitivity")]
#![allow(missing_docs)]

use faradice_core::{Axis, BoxId, EPS0_FF_PER_UM, Estimate, Face, NetId, Side, Sink};
use faradice_frw::sens::{FaceStep, SensOptions, solve_sensitivity};
use faradice_frw::{
    Accuracy, Boundary, ConformalRule, DielectricRegions, FaceKind, SolverConfig, WalkOptions,
};
use faradice_geom::{Aabb, GeometryBuilder};

/// Plate `x ∈ [X0, X1]` across the whole y/z extent, grounded x faces of the
/// domain `[0, L]³`, reflecting y/z faces.
const L: f64 = 2.0;
const X0: f64 = 0.9;
const X1: f64 = 1.1;
/// Shell thickness.
const T: f64 = 0.1;

fn cfg(eps_out: f64, eps_shell: f64, walks: u64) -> SolverConfig {
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::X, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Ground;
    SolverConfig {
        eps_r: eps_out,
        layers: Vec::new(),
        boundary: Boundary::Box {
            bounds: Aabb::from_array([0.0, 0.0, 0.0, L, L, L]).unwrap(),
            faces,
        },
        substrate_z: None,
        accuracy: Accuracy {
            min_walks: walks,
            max_walks: walks,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed: 15,
        ground_nets: Vec::new(),
        regions: DielectricRegions {
            boxes: Vec::new(),
            conformal: vec![ConformalRule {
                layer: "m".into(),
                boxes: vec![BoxId::new(0)],
                side: T,
                top: 0.0,
                eps_r: eps_shell,
            }],
        },
    }
}

/// Series resistance-like `t/ε_s + (g − t)/ε_o` of one gap.
fn gap(g: f64, eps_out: f64, eps_shell: f64) -> f64 {
    T / eps_shell + (g - T) / eps_out
}

fn report(name: &str, est: Estimate, exact: f64) -> f64 {
    let z = (est.value - exact) / est.std_err;
    eprintln!(
        "{name}: {:.6} ± {:.6} vs {exact:.6} (z = {z:+.2}, σ {:.2}%)",
        est.value,
        est.std_err,
        100.0 * est.std_err / exact.abs()
    );
    assert!(
        est.std_err <= 0.05 * exact.abs(),
        "{name}: σ too large to mean something"
    );
    z
}

/// S15: moving the plate's x+ face with its shell held in place thickens the
/// shell, so `T^cond = ε₀A / (ε_s R²)` with `R` the right gap's series sum.
/// The shell's outer face adds `T^Σ = ε₀A (1/ε_o − 1/ε_s) / R²`, and the sum
/// is `ε₀A / (ε_o R²)`. κ = 1 has no interface and no `T^Σ`.
#[test]
fn s15_a_shelled_plate() {
    let mut g = GeometryBuilder::new();
    let p = g.net("P").unwrap();
    g.add_box(p, Aabb::from_array([X0, 0.0, 0.0, X1, L, L]).unwrap())
        .unwrap();
    let g = g.build().unwrap();
    let area = L * L;
    let xp = Face::new(Axis::X, Side::Max);
    let step = FaceStep {
        box_id: BoxId::new(0),
        face: xp,
        delta: 0.02,
        keep: None,
    };
    for (eps_out, eps_shell) in [(4.5, 3.5), (2.0, 4.0), (4.0, 2.0), (3.0, 3.0)] {
        let c = cfg(eps_out, eps_shell, 2_000_000);
        let sol = solve_sensitivity(
            &g,
            &[NetId::new(0)],
            &c,
            &[step],
            &[],
            SensOptions {
                shell_conductor_term: true,
                ..SensOptions::default()
            },
        )
        .unwrap();
        let r = &sol.nets[0];
        let (left, right) = (gap(X0, eps_out, eps_shell), gap(L - X1, eps_out, eps_shell));
        let name = format!("S15 κ = {:.2}", eps_out / eps_shell);
        let c_exact = EPS0_FF_PER_UM * area * (1.0 / left + 1.0 / right);
        let zc = report(&format!("{name} C"), r.nominal.total(), c_exact);
        assert!(zc.abs() < 4.0, "{name}: nominal z = {zc}");
        let t_cond = EPS0_FF_PER_UM * area / (eps_shell * right * right);
        let est = r.faces[0].dc(Sink::Net(NetId::new(0)), r.nominal.walks);
        let z = report(&format!("{name} T^cond"), est, t_cond);
        assert!(z.abs() < 4.0, "{name}: T^cond z = {z}");
        assert!(r.faces[0].converged());
        assert!(r.faces[0].shell.is_none());
        // The full derivative: the shell's outer face moves too.
        let sol = solve_sensitivity(
            &g,
            &[NetId::new(0)],
            &c,
            &[step],
            &[],
            SensOptions {
                shell_points: 200_000,
                ..SensOptions::default()
            },
        )
        .unwrap();
        let f = &sol.nets[0].faces[0];
        let row = Sink::Net(NetId::new(0));
        let t_sigma = EPS0_FF_PER_UM * area * (1.0 / eps_out - 1.0 / eps_shell) / (right * right);
        let shell = f.dc_shell(row).unwrap();
        if (eps_out - eps_shell).abs() < 1e-12 {
            assert_eq!(shell.value, 0.0, "{name}: no interface, no T^Σ");
        } else {
            let z = report(&format!("{name} T^Σ"), shell, t_sigma);
            assert!(z.abs() < 4.0, "{name}: T^Σ z = {z}");
        }
        let total = EPS0_FF_PER_UM * area / (eps_out * right * right);
        let z = report(
            &format!("{name} dC/dn"),
            f.dc(row, sol.nets[0].nominal.walks),
            total,
        );
        assert!(z.abs() < 4.0, "{name}: dC/dn z = {z}");
        assert!(f.converged());
        let s = f.shell.as_ref().unwrap();
        eprintln!(
            "{name}: {} of {} shell points on the interface",
            s.on_interface, s.points
        );
    }
    // Richardson needs 2δ within the shell, so that both levels agree on the
    // material of the slab vacated at level 1.
    let mut c = cfg(4.5, 3.5, 20_000);
    c.regions.conformal[0].side = 0.04;
    let err = solve_sensitivity(
        &g,
        &[NetId::new(0)],
        &c,
        &[FaceStep {
            delta: 0.03,
            ..step
        }],
        &[],
        SensOptions {
            shell_conductor_term: true,
            ..SensOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("Richardson"), "{err}");
}

/// The C4 case (`bench/compare/cases/c4_met1_pair_conformal.faradice.toml`)
/// with wire a's x+ face (towards b) at `xa`, in the grounded box domain
/// that `faradice` builds for it (margin 4·z_max, substrate at z = 0).
fn c4(xa: f64, walks: u64, seed: u64, stop: f64) -> (faradice_geom::Geometry, SolverConfig) {
    use faradice_frw::DielectricLayer;
    let (z0, z1) = (1.3761, 1.7361);
    let mut g = GeometryBuilder::new();
    let a = g.net("a").unwrap();
    let b = g.net("b").unwrap();
    g.add_box(a, Aabb::from_array([0.0, 0.0, z0, xa, 6.0, z1]).unwrap())
        .unwrap();
    g.add_box(b, Aabb::from_array([0.28, 0.0, z0, 0.42, 6.0, z1]).unwrap())
        .unwrap();
    let g = g.build().unwrap();
    let m = 4.0 * z1;
    let layers = [
        (0.9361, 3.9),
        (1.0361, 7.3),
        (1.3761, 4.05),
        (2.0061, 4.5),
        (2.7861, 4.2),
        (4.0211, 4.1),
    ]
    .map(|(z_top, eps_r)| DielectricLayer { z_top, eps_r })
    .to_vec();
    let c = SolverConfig {
        eps_r: 4.0,
        layers,
        boundary: Boundary::Box {
            bounds: Aabb::from_array([-m, -m, 0.0, 0.42 + m, 6.0 + m, z1 + m]).unwrap(),
            faces: [FaceKind::Ground; 6],
        },
        substrate_z: Some(0.0),
        accuracy: Accuracy {
            min_walks: walks,
            max_walks: walks,
            ..Accuracy::default()
        },
        walk: WalkOptions {
            stop_dist: Some(stop),
            ..WalkOptions::default()
        },
        seed,
        ground_nets: Vec::new(),
        regions: DielectricRegions {
            boxes: Vec::new(),
            conformal: vec![ConformalRule {
                layer: "met1".into(),
                boxes: vec![BoxId::new(0), BoxId::new(1)],
                side: 0.030,
                top: 0.0,
                eps_r: 3.5,
            }],
        },
    };
    (g, c)
}

/// S16: SKY130 met1 pair with NILD3_C. dC/dn of wire a's face towards b
/// (A0 in M₁ with Richardson, plus `T^Σ`) against central differences of
/// re-solves at ±0.02 µm with the same stopping distance. `S16_WALKS` sets
/// the walks per solve.
#[test]
#[ignore = "long: several minutes"]
fn s16_sky130_pair_against_resolves() {
    let walks: u64 = std::env::var("S16_WALKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_000_000);
    let a = NetId::new(0);
    let (g, mut c) = c4(0.14, walks, 1, 0.0);
    c.walk.stop_dist = None;
    let su = faradice_frw::sens::sens_setup(&g, &[a], &c).unwrap();
    let stop = su.eps_stop;
    let (g, c) = c4(0.14, walks, 1, stop);
    let t0 = std::time::Instant::now();
    let sol = solve_sensitivity(
        &g,
        &[a],
        &c,
        &[FaceStep {
            box_id: BoxId::new(0),
            face: Face::new(Axis::X, Side::Max),
            delta: 0.012,
            keep: None,
        }],
        &[],
        SensOptions {
            shell_points: 400_000,
            ..SensOptions::default()
        },
    )
    .unwrap();
    eprintln!("S16 sensitivity run: {:.0} s", t0.elapsed().as_secs_f64());
    let r = &sol.nets[0];
    let f = &r.faces[0];
    assert!(f.converged());
    let d = 0.02;
    let plus = faradice_frw::solve(
        &g_of(0.14 + d, walks, stop),
        &[a],
        &c4(0.14 + d, walks, 2, stop).1,
    )
    .unwrap();
    let minus = faradice_frw::solve(
        &g_of(0.14 - d, walks, stop),
        &[a],
        &c4(0.14 - d, walks, 3, stop).1,
    )
    .unwrap();
    for (name, s) in [("C_aa", Sink::Net(a)), ("C_ab", Sink::Net(NetId::new(1)))] {
        let est = f.dc(s, r.nominal.walks);
        let shell = f.dc_shell(s).unwrap();
        let (cp, cm) = (plus.nets[0].c(s), minus.nets[0].c(s));
        let fd = (cp.value - cm.value) / (2.0 * d);
        let fd_sigma = cp.std_err.hypot(cm.std_err) / (2.0 * d);
        let sigma = est.std_err.hypot(fd_sigma);
        let z = (est.value - fd) / sigma;
        eprintln!(
            "S16 d{name}/dn: A0+T^Σ {:.4} ± {:.4} (T^Σ {:.4} ± {:.4}), re-solves {fd:.4} ± \
             {fd_sigma:.4} (z = {z:+.2})",
            est.value, est.std_err, shell.value, shell.std_err
        );
        assert!(z.abs() < 4.0, "{name}: z = {z}");
    }
}

fn g_of(xa: f64, walks: u64, stop: f64) -> faradice_geom::Geometry {
    c4(xa, walks, 0, stop).0
}

/// S16 edge margin: `T^Σ` of the C4 face for several edge margins (the
/// zeroth-order extrapolation over the edge strips has O(margin²) bias).
#[test]
#[ignore = "study"]
fn s16_edge_margin_study() {
    let a = NetId::new(0);
    let (g, mut c) = c4(0.14, 20_000, 1, 0.0);
    c.walk.stop_dist = None;
    for margin in [0.25, 0.5, 1.0, 2.0, 3.0] {
        let t0 = std::time::Instant::now();
        let sol = solve_sensitivity(
            &g,
            &[a],
            &c,
            &[FaceStep {
                box_id: BoxId::new(0),
                face: Face::new(Axis::X, Side::Max),
                delta: 0.012,
                keep: None,
            }],
            &[],
            SensOptions {
                shell_points: 1_000_000,
                shell_edge_margin: margin,
                ..SensOptions::default()
            },
        )
        .unwrap();
        let f = &sol.nets[0].faces[0];
        let (aa, ab) = (
            f.dc_shell(Sink::Net(a)).unwrap(),
            f.dc_shell(Sink::Net(NetId::new(1))).unwrap(),
        );
        eprintln!(
            "margin {margin}: T^Σ_aa {:.4} ± {:.4}, T^Σ_ab {:.4} ± {:.4} ({:.0} s)",
            aa.value,
            aa.std_err,
            ab.value,
            ab.std_err,
            t0.elapsed().as_secs_f64()
        );
    }
}
