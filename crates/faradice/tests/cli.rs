//! Command-line behaviour of the `faradice` binary.

#![allow(missing_docs)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh scratch directory for `test`.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn faradice(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_faradice"))
        .args(args)
        .output()
        .unwrap()
}

fn stdout(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

/// `check` applies the structure's `substrate_z`, as `solve` does (#9).
#[test]
fn check_uses_the_structure_substrate_height() {
    let dir = scratch("check_substrate_z");
    let case =
        std::fs::read_to_string(root().join("bench/compare/cases/l5_wires_in_stack.faradice.toml"))
            .unwrap()
            .replacen("units = \"um\"\n", "units = \"um\"\nsubstrate_z = 0.3\n", 1);
    let structure = dir.join("s.faradice.toml");
    std::fs::write(&structure, case).unwrap();
    let config = dir.join("c.toml");
    std::fs::write(
        &config,
        "schema_version = 1\n[ground]\nsubstrate = \"plane\"\n",
    )
    .unwrap();
    let out = stdout(&faradice(&[
        Path::new("check"),
        &structure,
        Path::new("-c"),
        &config,
    ]));
    let boundary = out.lines().find(|l| l.starts_with("boundary:")).unwrap();
    assert!(boundary.contains(", 0.3], max:"), "{boundary}");
}

/// `map` refuses dielectric boxes up front, with its own message (#18).
#[test]
fn map_refuses_dielectric_boxes_up_front() {
    let dir = scratch("map_dielectric_boxes");
    let case = std::fs::read_to_string(root().join("examples/three_wires.faradice.toml"))
        .unwrap()
        .replacen(
            "[[conductor]]",
            "[[dielectric.box]]\nmin = [-1.0, -1.0, 0.0]\nmax = [11.0, 2.0, 0.9]\neps_r = 7.0\n\n\
             [[conductor]]",
            1,
        );
    let structure = dir.join("s.faradice.toml");
    std::fs::write(&structure, case).unwrap();
    let out = faradice(&[
        Path::new("map"),
        &structure,
        Path::new("--nets"),
        Path::new("left,right"),
        Path::new("--grid"),
        Path::new("0:10:3,0.35,1.5"),
        Path::new("-o"),
        &dir.join("map.csv"),
    ]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("insertion maps need a uniform dielectric"),
        "{err}"
    );
}
