//! End-to-end GDS → trace tests on generated layouts.

#![allow(missing_docs)]

use faradice_gds::flatten::{FlattenOptions, flatten};
use faradice_gds::stream::{Element, Library, Strans, Units, parse_library};
use faradice_gds::tech::Tech;
use faradice_gds::trace::{Diagnostic, TraceOptions, trace};
use faradice_gds::writer::write_library;

fn tech() -> Tech {
    Tech::parse(include_str!("../../../tech/sky130_uniform.tech.toml")).unwrap()
}

fn rect(layer: i16, dt: i16, x0: i32, y0: i32, x1: i32, y1: i32) -> Element {
    Element::Boundary {
        layer,
        datatype: dt,
        xy: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
    }
}

fn text(layer: i16, tt: i16, x: i32, y: i32, s: &str) -> Element {
    Element::Text {
        layer,
        texttype: tt,
        xy: [x, y],
        string: s.into(),
    }
}

fn lib(structs: Vec<(&str, Vec<Element>)>) -> Library {
    let mut l = Library {
        name: "test".into(),
        units: Some(Units {
            db_in_user: 1e-3,
            db_in_meters: 1e-9,
        }),
        ..Library::default()
    };
    for (n, els) in structs {
        l.index.insert(n.into(), l.structs.len());
        l.structs.push((n.into(), els));
    }
    // Round-trip through the writer and the reader.
    parse_library(&write_library(&l)).unwrap()
}

#[test]
fn via_stack_connects_and_labels_name_nets() {
    // Net A: met1 bar + via + met2 bar (label on met2). Net B: separate met1 bar (label on met1).
    let top = vec![
        rect(68, 20, 0, 0, 10_000, 140),
        rect(68, 44, 9_000, 0, 9_150, 140),
        rect(69, 20, 8_900, -2_000, 9_200, 2_000),
        text(69, 5, 9_000, 1_500, "A"),
        rect(68, 20, 0, 420, 8_000, 560),
        text(68, 5, 100, 500, "B"),
    ];
    let l = lib(vec![("TOP", top)]);
    let flat = flatten(&l, None, FlattenOptions::default()).unwrap();
    let t = trace(&flat, l.units.unwrap(), &tech(), TraceOptions::default()).unwrap();
    assert!(t.diagnostics.is_empty(), "{:?}", t.diagnostics);
    let a = t.component("A").unwrap();
    assert_eq!(a.boxes.len(), 3);
    assert_eq!(t.component("B").unwrap().boxes.len(), 1);
    let g = t.to_geometry(|_| true).unwrap();
    assert_eq!(g.num_nets(), 2);
}

#[test]
fn corner_touch_does_not_connect_and_edge_touch_does() {
    let top = vec![
        rect(68, 20, 0, 0, 100, 100),
        rect(68, 20, 100, 100, 200, 200), // corner touch only
        rect(68, 20, 100, 0, 150, 50),    // edge touch with the first
        text(68, 5, 50, 50, "X"),
        text(68, 5, 150, 150, "Y"),
    ];
    let l = lib(vec![("TOP", top)]);
    let flat = flatten(&l, None, FlattenOptions::default()).unwrap();
    let t = trace(&flat, l.units.unwrap(), &tech(), TraceOptions::default()).unwrap();
    assert_eq!(t.component("X").unwrap().boxes.len(), 2);
    assert_eq!(t.component("Y").unwrap().boxes.len(), 1);
}

#[test]
fn hierarchy_rotation_and_arrays() {
    // Cell: horizontal met1 bar 1000 x 100 nm.
    let cell = vec![rect(68, 20, 0, 0, 1000, 100)];
    let strans90 = Strans {
        mag: 1.0,
        angle: 90.0,
        ..Strans::default()
    };
    let top = vec![
        Element::Sref {
            sname: "BAR".into(),
            strans: strans90,
            xy: [5000, 0],
        },
        Element::Aref {
            sname: "BAR".into(),
            strans: Strans {
                mag: 1.0,
                ..Strans::default()
            },
            colrow: [1, 3],
            xy: [[0, 2000], [1000, 2000], [0, 2600]],
        },
        text(68, 5, 4950, 500, "ROT"),
        text(68, 5, 500, 2250, "ROW1"),
    ];
    let l = lib(vec![("BAR", cell), ("TOP", top)]);
    let flat = flatten(&l, Some("TOP"), FlattenOptions::default()).unwrap();
    assert_eq!(flat.shapes.len(), 4);
    let t = trace(&flat, l.units.unwrap(), &tech(), TraceOptions::default()).unwrap();
    // Rotated bar spans x in [4900, 5000] nm, y in [0, 1000] nm.
    let rot = t.component("ROT").unwrap();
    let b = t.boxes[rot.boxes[0]];
    assert!((b.min()[0] - 4.9).abs() < 1e-12 && (b.max()[1] - 1.0).abs() < 1e-12);
    // Array rows at y = 2000, 2200, 2400 nm; ROW1 labels the second row.
    let row1 = t.component("ROW1").unwrap();
    assert!((t.boxes[row1.boxes[0]].min()[1] - 2.2).abs() < 1e-12);
    assert_eq!(t.components.len(), 4);
}

#[test]
fn label_diagnostics() {
    let top = vec![
        rect(68, 20, 0, 0, 100, 100),
        text(68, 5, 10, 10, "P"),
        text(68, 5, 20, 20, "Q"), // conflicting label on the same component
        rect(68, 20, 500, 0, 600, 100),
        text(68, 5, 550, 50, "P"), // same name on a disconnected component
        text(68, 5, 5000, 5000, "Z"), // dangling
    ];
    let l = lib(vec![("TOP", top)]);
    let flat = flatten(&l, None, FlattenOptions::default()).unwrap();
    let t = trace(&flat, l.units.unwrap(), &tech(), TraceOptions::default()).unwrap();
    assert!(
        t.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ConflictingLabels { .. }))
    );
    assert!(
        t.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DisconnectedNet { parts: 2, .. }))
    );
    assert!(
        t.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DanglingLabel { .. }))
    );
    assert!(t.component("P").is_some() && t.component("P:2").is_some());
    let joined = trace(
        &flat,
        l.units.unwrap(),
        &tech(),
        TraceOptions {
            join_same_name: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(joined.component("P").unwrap().boxes.len(), 2);
}

#[test]
fn paths_and_cycles() {
    let top = vec![
        Element::Path {
            layer: 68,
            datatype: 20,
            pathtype: 0,
            width: 140,
            bgn_ext: 0,
            end_ext: 0,
            xy: vec![[0, 0], [1000, 0], [1000, 1000]],
        },
        text(68, 5, 1000, 900, "L"),
    ];
    let l = lib(vec![("TOP", top)]);
    let flat = flatten(&l, None, FlattenOptions::default()).unwrap();
    let t = trace(&flat, l.units.unwrap(), &tech(), TraceOptions::default()).unwrap();
    assert_eq!(t.component("L").unwrap().boxes.len(), 2);

    let a = vec![Element::Sref {
        sname: "B".into(),
        strans: Strans {
            mag: 1.0,
            ..Strans::default()
        },
        xy: [0, 0],
    }];
    let b = vec![Element::Sref {
        sname: "A".into(),
        strans: Strans {
            mag: 1.0,
            ..Strans::default()
        },
        xy: [0, 0],
    }];
    let cyc = lib(vec![("A", a), ("B", b)]);
    assert!(flatten(&cyc, Some("A"), FlattenOptions::default()).is_err());
}

#[test]
fn stratified_sky130_tech_parses() {
    let t = Tech::parse(include_str!("../../../tech/sky130_stack.tech.toml")).unwrap();
    let u = tech();
    assert_eq!(t.dielectric.len(), 6);
    assert_eq!(t.layers.len(), u.layers.len());
    for (a, b) in t.layers.iter().zip(&u.layers) {
        assert_eq!(a.name, b.name);
        assert!((a.zmin - b.zmin).abs() < 1e-12 && (a.thickness - b.thickness).abs() < 1e-12);
    }
    // NILD3_C and NILD4_C on the met1 and met2 sidewalls only.
    for l in &t.layers {
        let c = l.conformal;
        if l.name == "met1" || l.name == "met2" {
            let c = c.unwrap();
            assert!((c.side - 0.03).abs() < 1e-12 && c.top == 0.0 && (c.eps_r - 3.5).abs() < 1e-12);
        } else {
            assert!(c.is_none(), "{}", l.name);
        }
    }
}
