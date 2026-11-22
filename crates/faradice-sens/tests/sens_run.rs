//! `run_sensitivity` end to end: shape-derivative faces (`method`) and
//! additions (`[[add]]`), from structure and configuration files.

#![allow(missing_docs)]

use std::path::{Path, PathBuf};

use faradice_core::EPS0_FF_PER_UM;
use faradice_engine::{Project, RunRequest, WalkEngineKind};
use faradice_io::config::{Config, read_config};
use faradice_sens::{Entry, Method, ParamFile, SensOutput, SensRequest, run_sensitivity};

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("faradice-sens-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn project(d: &Path, structure: &str) -> Project {
    let p = d.join("s.faradice.toml");
    std::fs::write(&p, structure).unwrap();
    Project::load(&p, None, None).unwrap()
}

fn config(d: &Path, text: &str) -> Config {
    let p = d.join("faradice.toml");
    std::fs::write(&p, text).unwrap();
    read_config(&p).unwrap()
}

fn run(project: &Project, cfg: &Config, params: &str, method: Option<Method>) -> SensOutput {
    run_sensitivity(
        project,
        &SensRequest {
            run: RunRequest {
                config: cfg.clone(),
                engine: WalkEngineKind::Cpu,
            },
            params: ParamFile::parse(params, "test").unwrap(),
            method,
            shape_samples: None,
        },
    )
    .unwrap()
}

fn targets(o: &SensOutput) -> String {
    serde_json::to_string(&o.run.result.targets).unwrap()
}

fn entry<'a>(v: &'a [Entry], i: &str, j: &str) -> &'a Entry {
    v.iter()
        .find(|e| e.i == i && e.j == j)
        .unwrap_or_else(|| panic!("no entry ({i}, {j})"))
}

fn z(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0) / a.1.hypot(b.1)
}

/// Plate P between the substrate (gap 1) and a plate T (gap 0.6), both
/// spanning the domain with reflecting sides (margin 0), so P's faces have
/// no geometric edges. Growing P's top face shrinks the P–T gap:
/// dC_PP/dn = εA/d², dC_PT/dn = −εA/d², d(ground of P)/dn = 0.
const PLATES: &str = r#"
schema_version = 1
units = "um"

[dielectric]
eps_r = 2.0

[[conductor]]
net = "P"
boxes = [[0.0, 0.0, 1.0, 3.0, 3.0, 1.2]]

[[conductor]]
net = "T"
boxes = [[0.0, 0.0, 1.8, 3.0, 3.0, 2.0]]
"#;

const PLATES_CFG: &str = r#"
schema_version = 1

[extract]
nets = ["P"]

[domain]
margin = 0.0

[ground]
substrate = "plane"
substrate_z = 0.0
boundary = { xmin = "reflect", xmax = "reflect", ymin = "reflect", ymax = "reflect", zmax = "ground" }

[accuracy]
min_walks = 200000
max_walks = 200000

[run]
seed = 5
"#;

const PLATE_TOP: &str = r#"
schema_version = 1

[[param]]
name = "top"
faces = [{ net = "P", box = 0, face = "z+" }]
"#;

#[test]
fn shape_faces_match_the_closed_form_and_finite_differences() {
    let d = dir("plates");
    let pr = project(&d, PLATES);
    let cfg = config(&d, PLATES_CFG);
    let exact = 2.0 * EPS0_FF_PER_UM * 9.0 / (0.6 * 0.6);

    let shape = run(&pr, &cfg, PLATE_TOP, Some(Method::Auto));
    assert_eq!(shape.sens.method, "shape");
    assert_eq!(shape.sens.faces[0].method, "shape");
    assert_eq!(shape.sens.faces[0].samples, Some(200_000));
    let fd = run(&pr, &cfg, PLATE_TOP, None);
    assert_eq!(fd.sens.faces[0].method, "fd");
    // The nominal extraction does not depend on the method.
    assert_eq!(targets(&shape), targets(&fd));

    let p = |o: &SensOutput, j: &str| {
        let e = entry(&o.sens.params[0].dc, "P", j);
        (e.value, e.std_err)
    };
    for (j, want) in [("P", exact), ("T", -exact)] {
        let (s, f) = (p(&shape, j), p(&fd, j));
        eprintln!("dC_P{j}: shape {s:?}, fd {f:?}, exact {want:.5}");
        assert!((s.0 - want).abs() <= 4.0 * s.1, "shape dC_P{j}");
        assert!(z(s, f).abs() <= 4.0, "shape vs fd dC_P{j}");
        assert!(s.1 < f.1, "the shape derivative should be the sharper one");
    }
    let g = p(&shape, "<GND>");
    assert!(g.0.abs() <= 4.0 * g.1.max(1e-12), "ground of P: {g:?}");

    // A finite-difference step limit does not apply to shape faces.
    let tiny = PLATE_TOP.replace("face = \"z+\" }", "face = \"z+\", delta = 1e-12 }");
    let t = run(&pr, &cfg, &tiny, Some(Method::Auto));
    assert_eq!(t.sens.faces[0].method, "shape");
    assert!(
        run_sensitivity(
            &pr,
            &SensRequest {
                run: RunRequest {
                    config: cfg.clone(),
                    engine: WalkEngineKind::Cpu,
                },
                params: ParamFile::parse(&tiny, "test").unwrap(),
                method: Some(Method::Fd),
                shape_samples: None,
            },
        )
        .is_err()
    );

    // A conductor named `GND` (a ground net by default) keeps its own column
    // next to the ground sum `<GND>`. P's ground capacitance is its row sum
    // over non-ground conductors, here C_PP alone.
    let d3 = dir("plates-gnd-names");
    let pr3 = project(&d3, &PLATES.replace("net = \"T\"", "net = \"GND\""));
    let g3 = run(&pr3, &cfg, PLATE_TOP, Some(Method::Auto));
    let col = entry(&g3.sens.params[0].dc, "P", "GND");
    let sum = entry(&g3.sens.params[0].dc, "P", "<GND>");
    assert!(col.value < 0.0, "{col:?}");
    let pp = entry(&g3.sens.params[0].dc, "P", "P");
    assert!(
        (sum.value - pp.value).abs() < 1e-9 * pp.value.abs(),
        "{sum:?} {pp:?}"
    );

    // A parameter that moves no face reports an explicit zero.
    let still =
        "schema_version = 1\n[[param]]\nname = \"x\"\ntranslate = { net = \"P\", axis = \"x\" }\n";
    let z0 = run(&pr, &cfg, still, Some(Method::Auto));
    let e = entry(&z0.sens.params[0].dc, "P", "<GND>");
    assert!(e.value == 0.0 && e.std_err == 0.0);

    // Every face must be supported under `shape`; a face with edges is not.
    let mixed = r#"
schema_version = 1
[[param]]
name = "top"
faces = [{ net = "P", box = 0, face = "z+" }]
"#;
    let pr2 = project(
        &d,
        &PLATES.replace(
            "[0.0, 0.0, 1.8, 3.0, 3.0, 2.0]",
            "[0.5, 0.5, 1.8, 2.5, 2.5, 2.0]",
        ),
    );
    // P still spans the domain, but T no longer covers it: P's top face keeps
    // no edges; T's faces have edges.
    let e = run_sensitivity(
        &pr2,
        &SensRequest {
            run: RunRequest {
                config: cfg.clone(),
                engine: WalkEngineKind::Cpu,
            },
            params: ParamFile::parse(
                &format!(
                    "{mixed}\n[[param]]\nname = \"t\"\nfaces = [{{ net = \"T\", box = 0, face = \
                     \"z-\" }}]\n"
                ),
                "test",
            )
            .unwrap(),
            method: Some(Method::Shape),
            shape_samples: Some(1000),
        },
    )
    .err()
    .unwrap()
    .to_string();
    assert!(
        e.contains("net `T` box 0 face z-") && e.contains("geometric edges"),
        "{e}"
    );
    assert!(!e.contains("net `P`"), "{e}");
}

/// Three wires and a crossing wire; a grounded shield is added over the
/// victim.
const WIRES: &str = r#"
schema_version = 1
units = "um"

[dielectric]
eps_r = 3.9

[[conductor]]
net = "left"
boxes = [[0.0, 0.00, 1.0, 10.0, 0.14, 1.36]]

[[conductor]]
net = "victim"
boxes = [[0.0, 0.28, 1.0, 10.0, 0.42, 1.36]]

[[conductor]]
net = "right"
boxes = [[0.0, 0.56, 1.0, 10.0, 0.70, 1.36]]

[[conductor]]
net = "cross"
boxes = [[4.93, -2.0, 1.76, 5.07, 2.7, 2.12]]
"#;

const SHIELD: &str = "[[conductor]]\nnet = \"shield\"\nboxes = [[3.0, 0.2, 1.5, 7.0, 0.5, 1.6]]\n";

fn wires_cfg(seed: u64) -> String {
    format!(
        r#"
schema_version = 1

[extract]
nets = ["victim"]

[accuracy]
min_walks = 300000
max_walks = 300000

[run]
seed = {seed}
"#
    )
}

#[test]
fn additions_match_a_resolve() {
    let d = dir("wires");
    let pr = project(&d, WIRES);
    let cfg = config(&d, &wires_cfg(3));
    let params = r#"
schema_version = 1

[[add]]
name = "shield"
net = "shield"
boxes = [[3.0, 0.2, 1.5, 7.0, 0.5, 1.6]]
"#;
    let out = run(&pr, &cfg, params, None);
    // Additions alone: the recorded run is the nominal one, as `solve`.
    let plain = faradice_engine::run(
        &pr,
        &RunRequest {
            config: cfg.clone(),
            engine: WalkEngineKind::Cpu,
        },
        &(),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_string(&out.run.result.targets).unwrap(),
        serde_json::to_string(&plain.result.targets).unwrap()
    );
    let a = &out.sens.additions[0];
    assert!(a.new_net && a.ground);
    assert!(a.replayed_samples > 0 && a.replayed_samples < a.recorded_samples);

    // Re-solve with the shield present (independent seed).
    let d2 = dir("wires-shield");
    let pr2 = project(&d2, &format!("{WIRES}\n{SHIELD}"));
    let cfg2 = config(&d2, &wires_cfg(11));
    let re = faradice_engine::run(
        &pr2,
        &RunRequest {
            config: cfg2,
            engine: WalkEngineKind::Cpu,
        },
        &(),
    )
    .unwrap();
    let r = &re.result.targets[0];
    #[allow(clippy::cast_precision_loss)]
    let n = r.walks as f64;
    let resolved = |j: &str| -> (f64, f64) {
        let s = r.sinks.iter().find(|s| s.sink == j).unwrap();
        let m = s.sum / n;
        (m, ((s.sum_sq / n - m * m) / (n - 1.0)).sqrt())
    };
    for j in ["victim", "left", "right", "cross", "shield"] {
        let e = entry(&a.c, "victim", j);
        let want = resolved(j);
        let zz = z((e.value, e.std_err), want);
        eprintln!(
            "C_victim,{j}: replay {:.5} ± {:.5}, re-solve {want:?}, z {zz:+.2}",
            e.value, e.std_err
        );
        assert!(zz.abs() <= 4.0, "C_victim,{j}");
    }
    // The shield is large against the victim's total change.
    let dv = entry(&a.dc, "victim", "victim");
    assert!(dv.value > 5.0 * dv.std_err, "{dv:?}");
}

#[test]
fn bad_additions_are_reported() {
    let d = dir("bad-adds");
    let pr = project(&d, WIRES);
    let cfg = config(&d, &wires_cfg(1).replace("300000", "2000"));
    let err = |params: &str| {
        run_sensitivity(
            &pr,
            &SensRequest {
                run: RunRequest {
                    config: cfg.clone(),
                    engine: WalkEngineKind::Cpu,
                },
                params: ParamFile::parse(params, "test").unwrap(),
                method: None,
                shape_samples: None,
            },
        )
        .err()
        .unwrap()
        .to_string()
    };
    let add = |net: &str, extra: &str, b: &str| {
        format!(
            "schema_version = 1\n[[add]]\nname = \"a\"\nnet = \"{net}\"\n{extra}boxes = [{b}]\n"
        )
    };
    let far = "[3.0, 0.2, 1.5, 7.0, 0.5, 1.6]";
    // With a deletion too, both passes run and their nominals agree.
    let both = run(
        &pr,
        &cfg,
        &format!(
            "{}\n[[delete]]\nname = \"d\"\nnets = [\"cross\"]\n",
            add("shield", "", far)
        ),
        None,
    );
    assert_eq!(both.sens.deletions.len(), 1);
    assert_eq!(both.sens.additions.len(), 1);
    assert!(err(&add("victim", "", far)).contains("is a target"));
    assert!(err(&add("<GND>", "", far)).contains("reserved"));
    assert!(err(&add("<substrate>", "", far)).contains("reserved"));
    assert!(err(&add("left", "ground = true\n", far)).contains("new nets only"));
    assert!(
        err(&add("shield", "", "[3.0, 0.3, 1.2, 7.0, 0.4, 1.6]")).contains("touches a conductor")
    );
}

/// `WIRES` in `nm` (coordinates × 1000).
fn wires_nm() -> String {
    let mut out = String::new();
    for line in WIRES.lines() {
        if let Some(list) = line.strip_prefix("boxes = [[") {
            let v: Vec<String> = list
                .trim_end_matches("]]")
                .split(',')
                .map(|t| format!("{:.1}", t.trim().parse::<f64>().unwrap() * 1000.0))
                .collect();
            out += &format!("boxes = [[{}]]\n", v.join(", "));
        } else {
            out += &line.replace("units = \"um\"", "units = \"nm\"");
            out.push('\n');
        }
    }
    out
}

/// Sensitivity files and map grids are in the input's unit: the same run in
/// nm and in µm agrees.
#[test]
fn sensitivity_and_map_lengths_follow_the_input_units() {
    let um = r#"
schema_version = 1

[[param]]
name = "w"
sigma = 0.01
faces = [{ net = "victim", at = [5.0, 0.42, 1.2], face = "y+", delta = 0.02 }]

[[add]]
name = "shield"
net = "shield"
boxes = [[3.0, 0.2, 1.5, 7.0, 0.5, 1.6]]
"#;
    let nm = r#"
schema_version = 1

[[param]]
name = "w"
sigma = 10.0
faces = [{ net = "victim", at = [5000.0, 420.0, 1200.0], face = "y+", delta = 20.0 }]

[[add]]
name = "shield"
net = "shield"
boxes = [[3000.0, 200.0, 1500.0, 7000.0, 500.0, 1600.0]]
"#;
    let cfg_text = wires_cfg(5).replace("300000", "40000");
    let d = dir("units-um");
    let (pu, cu) = (project(&d, WIRES), config(&d, &cfg_text));
    let d = dir("units-nm");
    let (pn, cn) = (project(&d, &wires_nm()), config(&d, &cfg_text));
    let a = run(&pu, &cu, um, None);
    let b = run(&pn, &cn, nm, None);
    let (pa, pb) = (&a.sens.params[0], &b.sens.params[0]);
    assert!((pb.sigma_um.unwrap() - 0.01).abs() < 1e-12);
    for (x, y) in [
        (
            entry(&pa.dc, "victim", "victim"),
            entry(&pb.dc, "victim", "victim"),
        ),
        (
            entry(&pa.sigma_c, "victim", "victim"),
            entry(&pb.sigma_c, "victim", "victim"),
        ),
        (
            entry(&a.sens.additions[0].dc, "victim", "victim"),
            entry(&b.sens.additions[0].dc, "victim", "victim"),
        ),
    ] {
        let zz = z((x.value, x.std_err), (y.value, y.std_err));
        assert!(zz.abs() <= 4.0, "µm {x:?}, nm {y:?}");
    }

    let map = |p: &Project, cfg: &Config, k: f64| {
        let req = faradice_sens::map::MapRequest {
            run: RunRequest {
                config: cfg.clone(),
                engine: WalkEngineKind::Cpu,
            },
            nets: ("victim".into(), "left".into()),
            grid: faradice_sens::map::Grid::parse(&format!("{},{},{}", 5.0 * k, 0.21 * k, 1.5 * k))
                .unwrap(),
            samples: 4000,
            radius: 0.05 * k,
        };
        faradice_sens::map::insertion_map(p, &req).unwrap().0
    };
    let (ra, rb) = (map(&pu, &cu, 1.0), map(&pn, &cn, 1000.0));
    // Rows are in µm whatever the input unit.
    for (x, y) in ra[0].p.point.iter().zip(rb[0].p.point) {
        assert!(
            (x - y).abs() < 1e-9,
            "{:?} vs {:?}",
            ra[0].p.point,
            rb[0].p.point
        );
    }
    let (ga, gb) = (ra[0].grounded, rb[0].grounded);
    assert!(ga.is_finite() && gb.is_finite());
    assert!((ga - gb).abs() <= 0.2 * ga.abs(), "µm {ga}, nm {gb}");
}
