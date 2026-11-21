//! Writes `examples/bus.gds`: a 4-bit met1 bus with a met2 crossing strap that
//! connects to bit 1 through a via (labelled `bus[1]` on met2 as well), plus a
//! VSS shield wire, using the sky130 label layers.
//!
//! Run: `cargo run -p faradice-gds --example make_bus -- examples/bus.gds`

#![allow(missing_docs)]

use faradice_gds::stream::{Element, Library, Strans, Units};
use faradice_gds::writer::write_library;

fn rect(layer: i16, dt: i16, x0: i32, y0: i32, x1: i32, y1: i32) -> Element {
    Element::Boundary {
        layer,
        datatype: dt,
        xy: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/bus.gds".into());
    // One bus bit: a 20 µm met1 wire, 0.14 µm wide.
    let bit = vec![rect(68, 20, 0, 0, 20_000, 140)];
    let mut top = vec![Element::Aref {
        sname: "BIT".into(),
        strans: Strans {
            mag: 1.0,
            ..Strans::default()
        },
        colrow: [1, 4],
        xy: [[0, 0], [20_000, 0], [0, 1_120]],
    }];
    for (i, y) in [70, 350, 630, 910].iter().enumerate() {
        top.push(Element::Text {
            layer: 68,
            texttype: 5,
            xy: [500, *y],
            string: format!("bus[{i}]"),
        });
    }
    // met2 strap crossing the bus at x = 10 µm, connected to bus[1] by a via.
    top.push(rect(69, 20, 9_850, -1_000, 10_150, 2_000));
    top.push(rect(68, 44, 9_925, 280, 10_075, 420));
    top.push(Element::Text {
        layer: 69,
        texttype: 5,
        xy: [10_000, 1_800],
        string: "bus[1]".into(),
    });
    // A grounded shield wire below the bus on met1 (labelled VSS).
    top.push(rect(68, 20, 0, -700, 20_000, -560));
    top.push(Element::Text {
        layer: 68,
        texttype: 5,
        xy: [500, -630],
        string: "VSS".into(),
    });
    let mut lib = Library {
        name: "bus".into(),
        units: Some(Units {
            db_in_user: 1e-3,
            db_in_meters: 1e-9,
        }),
        ..Library::default()
    };
    for (n, els) in [("BIT", bit), ("TOP", top)] {
        lib.index.insert(n.into(), lib.structs.len());
        lib.structs.push((n.into(), els));
    }
    std::fs::write(&out, write_library(&lib)).expect("write gds");
    println!("wrote {out}");
}
