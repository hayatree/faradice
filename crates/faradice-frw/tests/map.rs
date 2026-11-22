//! Potential maps (design D3 §7, pillar B): analytic potentials and S11.

#![cfg(feature = "sensitivity")]
#![allow(missing_docs)]

use faradice_core::{Axis, EPS0_FF_PER_UM, Face, NetId, Side, Sink};
use faradice_frw::map::{MapOptions, potential_map};
use faradice_frw::sens::{Deletion, SensRequest, solve_sensitivity_with};
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

/// Two full-width plates between grounded top and bottom faces, reflecting
/// sides: ground | 1.0 | P1 (0.2) | d | P2 (0.2) | 1.0 | ground.
fn plates(l: f64, d: f64, extra: &[(&str, [f64; 6])]) -> (Geometry, SolverConfig) {
    let mut boxes = vec![
        ("p1", [0.0, 0.0, 1.0, l, l, 1.2]),
        ("p2", [0.0, 0.0, 1.2 + d, l, l, 1.4 + d]),
    ];
    boxes.extend_from_slice(extra);
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([0.0, 0.0, 0.0, l, l, 2.4 + d]).unwrap();
    let cfg = SolverConfig {
        eps_r: 1.0,
        layers: Vec::new(),
        boundary: Boundary::Box { bounds, faces },
        substrate_z: None,
        accuracy: Accuracy {
            total_rel_tol: 0.005,
            ..Accuracy::default()
        },
        walk: WalkOptions {
            stop_dist: Some(1e-4),
            ..WalkOptions::default()
        },
        seed: 5,
        ground_nets: Vec::new(),
        regions: Default::default(),
    };
    (geom(&boxes), cfg)
}

fn check(name: &str, e: faradice_core::Estimate, exact: f64, abs_floor: f64) {
    eprintln!(
        "{name}: {:.5} ± {:.5} (exact {exact:.5})",
        e.value, e.std_err
    );
    assert!(
        (e.value - exact).abs() <= 4.0 * e.std_err + abs_floor,
        "{name}: {e:?} vs {exact}"
    );
}

/// Potentials between and below the plates are linear in z.
#[test]
fn linear_potentials_between_plates() {
    let (g, c) = plates(4.0, 1.0, &[]);
    let (p1, p2) = (NetId::new(0), NetId::new(1));
    let pts = [
        [2.0, 2.0, 1.7],
        [1.0, 3.0, 1.45],
        [2.0, 2.0, 0.5],
        [0.0, 0.0, 1.7],
    ];
    let m = potential_map(
        &g,
        &[p1, p2],
        &c,
        &pts,
        MapOptions {
            samples: 40_000,
            gradient: true,
        },
    )
    .unwrap();
    // φ and ∂φ/∂z: between the plates (gap 1), below P1 (gap 1 to ground).
    let exact = [
        (0.5, 0.5, -1.0, 1.0),
        (0.75, 0.25, -1.0, 1.0),
        (0.5, 0.0, 1.0, 0.0),
        (0.5, 0.5, -1.0, 1.0), // on the reflecting corner edge
    ];
    for (k, (p, e)) in m.iter().zip(exact).enumerate() {
        assert!(p.inside.is_none());
        check(&format!("pt{k} φ1"), p.phi(p1), e.0, 1e-3);
        check(&format!("pt{k} φ2"), p.phi(p2), e.1, 1e-3);
        check(&format!("pt{k} ∂zφ1"), p.grad(p1)[2], e.2, 0.01);
        check(&format!("pt{k} ∂zφ2"), p.grad(p2)[2], e.3, 0.01);
        for a in 0..2 {
            check(&format!("pt{k} ∂{a}φ1"), p.grad(p1)[a], 0.0, 0.01);
        }
    }
    // Inside a plate: φ = 1 exactly.
    let inside =
        potential_map(&g, &[p1, p2], &c, &[[2.0, 2.0, 1.1]], MapOptions::default()).unwrap();
    assert_eq!(inside[0].inside, Some(p1));
    assert_eq!(inside[0].phi(p1).value, 1.0);
}

/// S11 (grounded): a small grounded cube of edge a at the centre of the gap
/// changes C_12 by +c·φ1φ2 and C_11 by +c·φ1², c = 0.66068·4πεa. ΔC comes
/// from deleting the cube from an extraction with it present (paired).
#[test]
fn s11_grounded_cube_matches_topological_derivative() {
    let d = 2.0;
    let zc = 1.2 + 0.5 * d;
    let mut prefactors = Vec::new();
    for a in [0.4, 0.2, 0.1, 0.05] {
        let h = 0.5 * a;
        let cube = [2.0 - h, 2.0 - h, zc - h, 2.0 + h, 2.0 + h, zc + h];
        let (g, mut c) = plates(4.0, d, &[("q", cube)]);
        c.accuracy.min_walks = 1_000_000;
        c.accuracy.max_walks = 1_000_000;
        let (p1, p2) = (NetId::new(0), NetId::new(1));
        let sol = solve_sensitivity_with(
            &g,
            &[p1, p2],
            &c,
            &SensRequest {
                deletions: &[Deletion {
                    nets: vec![NetId::new(2)],
                }],
                ..SensRequest::default()
            },
        )
        .unwrap();
        let map = potential_map(
            &g,
            &[p1, p2],
            &c,
            &[[2.0, 2.0, zc]],
            MapOptions {
                samples: 20_000,
                gradient: false,
            },
        )
        .unwrap();
        // Map without the cube: the cube's own geometry must not be present.
        let (g0, c0) = plates(4.0, d, &[]);
        let m0 = potential_map(
            &g0,
            &[p1, p2],
            &c0,
            &[[2.0, 2.0, zc]],
            MapOptions {
                samples: 20_000,
                gradient: false,
            },
        )
        .unwrap();
        assert_eq!(map[0].inside, Some(NetId::new(2)));
        let (f1, f2) = (m0[0].phi(p1).value, m0[0].phi(p2).value);
        let r = &sol.nets[0];
        let n = r.nominal.walks;
        // ΔC(insert) = −ΔC(delete).
        let d12 = r.deletions[0].delta_c(Sink::Net(p2), n).scaled(-1.0);
        let d11 = r.deletions[0].delta_c(Sink::Net(p1), n).scaled(-1.0);
        let cap = 0.660_678_15 * 4.0 * std::f64::consts::PI * EPS0_FF_PER_UM * a;
        let pred12 = cap * f1 * f2;
        let pred11 = cap * f1 * f1;
        eprintln!(
            "a = {a}: ΔC12 {:.6} ± {:.6} (pred {pred12:.6}, ratio {:.3}), ΔC11 {:.6} ± {:.6} (pred {pred11:.6}, ratio {:.3})",
            d12.value,
            d12.std_err,
            d12.value / pred12,
            d11.value,
            d11.std_err,
            d11.value / pred11
        );
        prefactors.push((
            a,
            d12.value,
            d12.value / pred12,
            d12.std_err / d12.value.abs(),
        ));
    }
    // Log–log slope between the two smallest sizes ≈ 1, and the prefactor at
    // the smallest size within 20 % (design §9 S11).
    let (a1, v1, _, _) = prefactors[2];
    let (a2, v2, ratio, rel) = prefactors[3];
    let slope = (v1 / v2).ln() / (a1 / a2).ln();
    eprintln!("slope {slope:.3}, prefactor ratio {ratio:.3} ± {rel:.3}");
    assert!((slope - 1.0).abs() < 0.1, "slope {slope}");
    assert!((ratio - 1.0).abs() < 0.2, "ratio {ratio}");
}

/// Products of potentials are unbiased even with two samples per point
/// (the product of the two means would be off by a factor (1 − 1/N) = ½ for
/// φ1φ2 at the midpoint, where φ1 + φ2 = 1 per walk). Points outside the
/// domain are flagged, and one sample is rejected.
#[test]
fn unbiased_products_and_domain_checks() {
    let (g, c) = plates(4.0, 1.0, &[]);
    let (p1, p2) = (NetId::new(0), NetId::new(1));
    let pts = vec![[2.0, 2.0, 1.7]; 4000];
    let m = potential_map(
        &g,
        &[p1, p2],
        &c,
        &pts,
        MapOptions {
            samples: 2,
            gradient: true,
        },
    )
    .unwrap();
    #[allow(clippy::cast_precision_loss)]
    let n = m.len() as f64;
    let mean = |f: &dyn Fn(&faradice_frw::map::PointPotentials) -> f64| {
        let v: Vec<f64> = m.iter().map(f).collect();
        let mu = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
        (mu, (var / n).sqrt())
    };
    let (pp, s) = mean(&|p| p.phi_product(p1, p2));
    eprintln!("φ1φ2: {pp:.4} ± {s:.4} (exact 0.25)");
    assert!((pp - 0.25).abs() < 4.0 * s);
    let (gd, s) = mean(&|p| p.grad_dot(p1, p2));
    eprintln!("∇φ1·∇φ2: {gd:.4} ± {s:.4} (exact −1)");
    assert!((gd + 1.0).abs() < 4.0 * s);
    let out = potential_map(
        &g,
        &[p1, p2],
        &c,
        &[[2.0, 2.0, -1.0], [9.0, 2.0, 1.7]],
        MapOptions::default(),
    )
    .unwrap();
    assert!(out.iter().all(|p| p.outside && p.phi.is_empty()));
    assert!(
        potential_map(
            &g,
            &[p1, p2],
            &c,
            &[[2.0, 2.0, 1.7]],
            MapOptions {
                samples: 1,
                gradient: false
            }
        )
        .is_err()
    );
}
