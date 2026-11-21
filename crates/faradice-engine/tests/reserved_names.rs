//! Net names of the reserved form `<…>` are refused when a project loads,
//! before any configuration could remove the net.

#![allow(missing_docs)]

use faradice_engine::Project;

#[test]
fn reserved_net_names_are_refused_on_load() {
    let d = std::env::temp_dir().join(format!("faradice-engine-reserved-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join("s.faradice.toml");
    let s = |name: &str| {
        format!(
            "schema_version = 1\nunits = \"um\"\n[dielectric]\neps_r = 1.0\n[[conductor]]\nnet = \
             \"a\"\nboxes = [[0, 0, 1, 1, 1, 2]]\n[[conductor]]\nnet = \"{name}\"\nboxes = [[2, 0, \
             1, 3, 1, 2]]\n"
        )
    };
    std::fs::write(&p, s("<GND>")).unwrap();
    let e = Project::load(&p, None, None).err().unwrap().to_string();
    assert!(e.contains("reserved"), "{e}");
    for ok in ["GND", "GND:substrate", "bus<3>"] {
        std::fs::write(&p, s(ok)).unwrap();
        Project::load(&p, None, None).unwrap();
    }
}
