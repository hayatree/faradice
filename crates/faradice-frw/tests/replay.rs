//! D-add (design D3 §8): additions by replay.

#![cfg(feature = "sensitivity")]
#![allow(missing_docs)]

use faradice_core::{NetId, Sink};
use faradice_frw::replay::{AddedNet, Addition, replay_addition, solve_recorded};
use faradice_frw::sens::{Deletion, SensRequest, solve_sensitivity_with};
use faradice_frw::{Accuracy, Boundary, SolverConfig, WalkOptions, solve};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};

fn geom(boxes: &[(&str, [f64; 6])]) -> Geometry {
    let mut g = GeometryBuilder::new();
    for (net, c) in boxes {
        let id = g.net(net).unwrap();
        g.add_box(id, Aabb::from_array(*c).unwrap()).unwrap();
    }
    g.build().unwrap()
}

fn cfg(walks: u64, seed: u64) -> SolverConfig {
    SolverConfig {
        eps_r: 3.9,
        boundary: Boundary::Open,
        substrate_z: Some(0.0),
        accuracy: Accuracy {
            min_walks: walks,
            max_walks: walks,
            ..Accuracy::default()
        },
        walk: WalkOptions {
            // Pinned, so all runs share the stopping distance.
            stop_dist: Some(1e-4),
            // Gaussian surfaces close to the targets leave room for additions.
            gauss_alpha: 0.2,
            ..WalkOptions::default()
        },
        seed,
        ground_nets: Vec::new(),
        layers: Vec::new(),
        regions: Default::default(),
    }
}

const A: [f64; 6] = [0.0, 0.0, 1.0, 6.0, 0.2, 1.3];
const B: [f64; 6] = [0.0, 1.2, 1.0, 6.0, 1.4, 1.3];
/// A grounded shield half-way between A and B (and a bit shorter).
const SHIELD: [f64; 6] = [0.5, 0.65, 1.0, 5.5, 0.75, 1.3];

/// Adding a shield between two wires: the replayed edited matrix matches a
/// re-solve with the shield, and the paired ΔC matches minus the paired
/// deletion ΔC of the superset run (two exact, independent estimators).
#[test]
fn shield_addition_matches_resolve_and_deletion() {
    let (a, b) = (NetId::new(0), NetId::new(1));
    let g = geom(&[("a", A), ("b", B)]);
    let c = cfg(400_000, 11);
    let (sol, rec) = solve_recorded(&g, &[a, b], &c).unwrap();
    assert_eq!(
        sol.nets[0].total().value.to_bits(),
        solve(&g, &[a, b], &c).unwrap().nets[0]
            .total()
            .value
            .to_bits()
    );
    let add = Addition {
        boxes: vec![Aabb::from_array(SHIELD).unwrap()],
        net: AddedNet::New,
        ground: false,
    };
    let r = replay_addition(&g, &c, &sol, &rec, &add).unwrap();
    assert_eq!(r.added_net, NetId::new(2));
    // Superset: the shield present, then deleted (paired).
    let gs = geom(&[("a", A), ("b", B), ("s", SHIELD)]);
    let cs = cfg(400_000, 12);
    let del = solve_sensitivity_with(
        &gs,
        &[a, b],
        &cs,
        &SensRequest {
            deletions: &[Deletion {
                nets: vec![NetId::new(2)],
            }],
            ..SensRequest::default()
        },
    )
    .unwrap();
    let re = solve(&gs, &[a, b], &cfg(400_000, 13)).unwrap();
    for (t, net) in [(0, a), (1, b)] {
        let x = &r.nets[t];
        let d = &del.nets[t];
        let n = d.nominal.walks;
        eprintln!(
            "target {t}: replayed {} of {} samples",
            x.replayed,
            rec.samples()
        );
        for k in [Sink::Net(a), Sink::Net(b), Sink::Net(NetId::new(2))] {
            let add_d = x.delta_c(k);
            let del_d = d.deletions[0].delta_c(k, n).scaled(-1.0);
            let ed = x.c(k);
            let rs = re.nets[t].c(k);
            let z1 = (add_d.value - del_d.value) / add_d.std_err.hypot(del_d.std_err);
            let z2 = (ed.value - rs.value) / ed.std_err.hypot(rs.std_err);
            eprintln!(
                "  {k:?}: ΔC add {:.5} ± {:.5}, −ΔC del {:.5} ± {:.5} (z {z1:+.2}); edited {:.5} ± {:.5}, re-solve {:.5} ± {:.5} (z {z2:+.2})",
                add_d.value,
                add_d.std_err,
                del_d.value,
                del_d.std_err,
                ed.value,
                ed.std_err,
                rs.value,
                rs.std_err
            );
            assert!(z1.abs() < 4.0 && z2.abs() < 4.0, "{t} {k:?}");
        }
        let cab = x.delta_c(Sink::Net(if net == a { b } else { a }));
        assert!(cab.value > 5.0 * cab.std_err, "the shield lowers |C_ab|");
    }
}

/// Boxes inside an existing conductor of the same net never change a walk
/// (they are never strictly nearer): ΔC is exactly zero although samples
/// are replayed. Invalid additions are refused.
#[test]
fn hidden_box_changes_nothing_and_bad_additions_fail() {
    let (a, b) = (NetId::new(0), NetId::new(1));
    let g = geom(&[("a", A), ("b", B)]);
    let c = cfg(20_000, 3);
    let (sol, rec) = solve_recorded(&g, &[a], &c).unwrap();
    let inner = Aabb::from_array([1.0, 1.25, 1.05, 5.0, 1.35, 1.25]).unwrap();
    let r = replay_addition(
        &g,
        &c,
        &sol,
        &rec,
        &Addition {
            boxes: vec![inner],
            net: AddedNet::Existing(b),
            ground: false,
        },
    )
    .unwrap();
    assert!(r.nets[0].replayed > 0);
    assert!(r.nets[0].delta.values().all(|m| m.sum() == 0.0));
    assert_eq!(r.nets[0].delta_ground.sum(), 0.0);
    let fail = |boxes: [f64; 6], net| {
        replay_addition(
            &g,
            &c,
            &sol,
            &rec,
            &Addition {
                boxes: vec![Aabb::from_array(boxes).unwrap()],
                net,
                ground: false,
            },
        )
        .unwrap_err()
        .to_string()
    };
    assert!(fail([1.0, 1.25, 1.05, 5.0, 1.35, 1.25], AddedNet::New).contains("overlaps"));
    assert!(fail([0.0, -1.0, -0.5, 1.0, -0.8, 0.5], AddedNet::New).contains("substrate"));
    assert!(fail([1.0, 0.21, 1.0, 2.0, 0.25, 1.3], AddedNet::New).contains("Gaussian"));
    // Touching another net is a short.
    assert!(fail([1.0, 1.4, 1.0, 2.0, 1.5, 1.3], AddedNet::New).contains("touches"));
    // Extending a target: its surface would not enclose the new box.
    assert!(fail([1.0, -3.0, 1.0, 2.0, -2.8, 1.3], AddedNet::Existing(a)).contains("target"));
    // A solution of another run with the same configuration shape.
    let (other_sol, _) = solve_recorded(&g, &[a], &cfg(20_000, 3)).unwrap();
    let mut wrong = other_sol.clone();
    wrong.nets[0].walks += 1;
    let e = replay_addition(
        &g,
        &c,
        &wrong,
        &rec,
        &Addition {
            boxes: vec![Aabb::from_array(SHIELD).unwrap()],
            net: AddedNet::New,
            ground: false,
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("solution"), "{e}");
    // A tampered ground baseline is caught too.
    let mut wrong = sol.clone();
    wrong.nets[0].ground_moments = faradice_core::Moments::new();
    let add = Addition {
        boxes: vec![Aabb::from_array(SHIELD).unwrap()],
        net: AddedNet::New,
        ground: false,
    };
    assert!(replay_addition(&g, &c, &wrong, &rec, &add).is_err());
    // Another configuration.
    let other = cfg(20_000, 4);
    let e = replay_addition(
        &g,
        &other,
        &sol,
        &rec,
        &Addition {
            boxes: vec![Aabb::from_array(SHIELD).unwrap()],
            net: AddedNet::New,
            ground: false,
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("recording"));
}

#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn recording_overhead() {
    use std::time::Instant;
    let g = geom(&[("a", A), ("b", B), ("s", SHIELD)]);
    let c = cfg(4_000_000, 1);
    let t = [NetId::new(0)];
    let _ = solve(&g, &t, &c).unwrap();
    let t0 = Instant::now();
    let _ = solve(&g, &t, &c).unwrap();
    let plain = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let (_, rec) = solve_recorded(&g, &t, &c).unwrap();
    let recorded = t1.elapsed().as_secs_f64();
    eprintln!(
        "solve {plain:.3} s, recorded {recorded:.3} s ({:.0} % overhead), {} samples, ~{} MB",
        100.0 * (recorded / plain - 1.0),
        rec.samples(),
        rec.samples() * 56 / 1_000_000
    );
}

/// The ground flag: a new net counted as ground matches a re-solve with it
/// in `ground_nets`.
#[test]
fn new_net_as_ground() {
    let (a, b) = (NetId::new(0), NetId::new(1));
    let g = geom(&[("a", A), ("b", B)]);
    let c = cfg(200_000, 21);
    let (sol, rec) = solve_recorded(&g, &[a], &c).unwrap();
    let r = replay_addition(
        &g,
        &c,
        &sol,
        &rec,
        &Addition {
            boxes: vec![Aabb::from_array(SHIELD).unwrap()],
            net: AddedNet::New,
            ground: true,
        },
    )
    .unwrap();
    let gs = geom(&[("a", A), ("b", B), ("s", SHIELD)]);
    let mut cs = cfg(400_000, 22);
    cs.ground_nets = vec![NetId::new(2)];
    let re = solve(&gs, &[a], &cs).unwrap();
    let (x, y) = (r.nets[0].ground(), re.nets[0].ground());
    let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
    eprintln!(
        "ground: replay {:.5} ± {:.5}, re-solve {:.5} ± {:.5} (z {z:+.2})",
        x.value, x.std_err, y.value, y.std_err
    );
    assert!(z.abs() < 4.0);
    let _ = b;
}

/// Layered dielectric (box domain): lateral Gaussian points near an
/// interface query a band cube at the interface; an addition there must be
/// replayed. ΔC by replay matches minus the paired deletion ΔC.
#[test]
fn layered_addition_matches_deletion() {
    use faradice_core::{Axis, Face, Side};
    use faradice_frw::{DielectricLayer, FaceKind};
    let (a, b) = (NetId::new(0), NetId::new(1));
    // Interface at z = 1.15, through the wires.
    let shield = [0.5, 0.65, 0.9, 5.5, 0.75, 1.4];
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([-1.0, -1.0, 0.0, 7.0, 2.6, 3.0]).unwrap();
    let mk = |walks, seed| {
        let mut c = cfg(walks, seed);
        c.boundary = Boundary::Box { bounds, faces };
        c.substrate_z = None;
        c.layers = vec![DielectricLayer {
            z_top: 1.15,
            eps_r: 7.0,
        }];
        c
    };
    let g = geom(&[("a", A), ("b", B)]);
    let c = mk(300_000, 31);
    let (sol, rec) = solve_recorded(&g, &[a], &c).unwrap();
    let r = replay_addition(
        &g,
        &c,
        &sol,
        &rec,
        &Addition {
            boxes: vec![Aabb::from_array(shield).unwrap()],
            net: AddedNet::New,
            ground: false,
        },
    )
    .unwrap();
    let gs = geom(&[("a", A), ("b", B), ("s", shield)]);
    let del = solve_sensitivity_with(
        &gs,
        &[a],
        &mk(300_000, 32),
        &SensRequest {
            deletions: &[Deletion {
                nets: vec![NetId::new(2)],
            }],
            ..SensRequest::default()
        },
    )
    .unwrap();
    let n = del.nets[0].nominal.walks;
    for k in [Sink::Net(a), Sink::Net(b), Sink::Net(NetId::new(2))] {
        let x = r.nets[0].delta_c(k);
        let y = del.nets[0].deletions[0].delta_c(k, n).scaled(-1.0);
        let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
        eprintln!(
            "layered {k:?}: add {:.5} ± {:.5}, −del {:.5} ± {:.5} (z {z:+.2})",
            x.value, x.std_err, y.value, y.std_err
        );
        assert!(z.abs() < 4.0, "{k:?}");
    }
}

/// As [`layered_addition_matches_deletion`], with dielectric boxes (design
/// M2c §7): an explicit strip between the wires that the shield crosses, and
/// a conformal shell on wire `b`. Neither changes with the shield, so the
/// replay and the paired deletion stay valid and agree. Moving a face of the
/// shelled wire, or deleting it, is refused.
#[test]
fn additions_and_deletions_with_dielectric_boxes() {
    use faradice_core::{Axis, BoxId, Face, Side};
    use faradice_frw::sens::FaceStep;
    use faradice_frw::{
        ConformalRule, DielectricBox, DielectricLayer, DielectricRegions, FaceKind,
    };
    let (a, b) = (NetId::new(0), NetId::new(1));
    let shield = [0.5, 0.65, 0.9, 5.5, 0.75, 1.4];
    let mut faces = [FaceKind::Reflect; 6];
    faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Ground;
    faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Ground;
    let bounds = Aabb::from_array([-1.0, -1.0, 0.0, 7.0, 2.6, 3.0]).unwrap();
    let mk = |walks, seed| {
        let mut c = cfg(walks, seed);
        c.boundary = Boundary::Box { bounds, faces };
        c.substrate_z = None;
        c.layers = vec![DielectricLayer {
            z_top: 1.15,
            eps_r: 7.0,
        }];
        c.regions = DielectricRegions {
            boxes: vec![DielectricBox {
                region: Aabb::from_array([-1.0, 0.45, 0.8, 7.0, 0.95, 1.5]).unwrap(),
                eps_r: 5.0,
            }],
            conformal: vec![ConformalRule {
                layer: "m1".into(),
                boxes: vec![BoxId::new(1)],
                side: 0.05,
                top: 0.05,
                eps_r: 3.0,
            }],
        };
        c
    };
    let g = geom(&[("a", A), ("b", B)]);
    let c = mk(300_000, 41);
    let (sol, rec) = solve_recorded(&g, &[a], &c).unwrap();
    let r = replay_addition(
        &g,
        &c,
        &sol,
        &rec,
        &Addition {
            boxes: vec![Aabb::from_array(shield).unwrap()],
            net: AddedNet::New,
            ground: false,
        },
    )
    .unwrap();
    let gs = geom(&[("a", A), ("b", B), ("s", shield)]);
    let cs = mk(300_000, 42);
    let del = solve_sensitivity_with(
        &gs,
        &[a],
        &cs,
        &SensRequest {
            deletions: &[Deletion {
                nets: vec![NetId::new(2)],
            }],
            ..SensRequest::default()
        },
    )
    .unwrap();
    let n = del.nets[0].nominal.walks;
    // Walks cross the strip's and the shell's edges (design M2c §4.6): none
    // aborts, in the solves or the continuations.
    let dd = &del.nets[0].deletions[0];
    assert_eq!(dd.aborted_continuations, 0);
    assert_eq!(del.nets[0].nominal.aborted_walks, 0);
    let re = solve(&gs, &[a], &mk(300_000, 43)).unwrap();
    assert_eq!(re.nets[0].aborted_walks, 0);
    for k in [Sink::Net(a), Sink::Net(b), Sink::Net(NetId::new(2))] {
        let x = r.nets[0].delta_c(k);
        let y = dd.delta_c(k, n).scaled(-1.0);
        let z = (x.value - y.value) / x.std_err.hypot(y.std_err);
        let (ed, rs) = (r.nets[0].c(k), re.nets[0].c(k));
        let z2 = (ed.value - rs.value) / ed.std_err.hypot(rs.std_err);
        eprintln!(
            "boxes {k:?}: add {:.5} ± {:.5}, −del {:.5} ± {:.5} (z {z:+.2}); edited {:.5}, re-solve {:.5} (z {z2:+.2})",
            x.value, x.std_err, y.value, y.std_err, ed.value, rs.value
        );
        assert!(z.abs() < 4.0 && z2.abs() < 4.0, "{k:?}");
    }
    // The shelled wire: its deletion changes the medium, and its bottom face
    // moves a shell ring at the conductor edge (design D3 §4.4c stage 3).
    let face = FaceStep {
        box_id: BoxId::new(1),
        face: Face::new(Axis::Z, Side::Min),
        delta: 0.01,
        keep: None,
    };
    for req in [
        SensRequest {
            faces: &[face],
            ..SensRequest::default()
        },
        SensRequest {
            deletions: &[Deletion { nets: vec![b] }],
            ..SensRequest::default()
        },
    ] {
        let err = solve_sensitivity_with(&gs, &[a], &cs, &req).unwrap_err();
        assert!(err.to_string().contains("conformal"), "{err}");
    }
}
