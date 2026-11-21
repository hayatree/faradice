//! The CLI command equivalent to a GUI run (design D1 §3: CLI parity).

use std::path::Path;

use faradice_engine::WalkEngineKind;
use faradice_io::config::Preset;

/// What the GUI passes to the engine, in CLI terms.
pub(crate) struct Equivalent<'a> {
    pub(crate) input: &'a Path,
    pub(crate) tech: Option<&'a Path>,
    pub(crate) config: Option<&'a Path>,
    /// Target globs (exact names escaped with `glob_escape`).
    pub(crate) nets: Option<&'a [String]>,
    pub(crate) preset: Preset,
    pub(crate) seed: u64,
    pub(crate) threads: usize,
    pub(crate) engine: WalkEngineKind,
}

/// Quotes `s` for a POSIX shell if needed.
fn sh(s: &str) -> String {
    let safe = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./,:=+@%".contains(c));
    if safe {
        s.to_owned()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

pub(crate) fn preset_name(p: Preset) -> &'static str {
    match p {
        Preset::Fast => "fast",
        Preset::Default => "default",
        Preset::Signoff => "signoff",
        Preset::Reference => "reference",
    }
}

impl Equivalent<'_> {
    /// Renders the `faradice solve` command line.
    pub(crate) fn command(&self) -> String {
        let mut v = vec![
            "faradice".to_owned(),
            "solve".to_owned(),
            sh(&self.input.to_string_lossy()),
        ];
        if let Some(t) = self.tech {
            v.push("--tech".into());
            v.push(sh(&t.to_string_lossy()));
        }
        if let Some(c) = self.config {
            v.push("-c".into());
            v.push(sh(&c.to_string_lossy()));
        }
        if let Some(n) = self.nets {
            v.push("--nets".into());
            v.push(sh(&n.join(",")));
        }
        v.push("--preset".into());
        v.push(preset_name(self.preset).into());
        v.push("--seed".into());
        v.push(self.seed.to_string());
        if self.threads != 0 {
            v.push("--threads".into());
            v.push(self.threads.to_string());
        }
        if self.engine == WalkEngineKind::Gpu {
            v.push("--engine".into());
            v.push("gpu".into());
        }
        v.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_only_when_needed() {
        let nets = vec![faradice_io::glob_escape("bus[0]"), "clk".to_owned()];
        let e = Equivalent {
            input: Path::new("my layout.gds"),
            tech: Some(Path::new("tech/sky130.tech.toml")),
            config: None,
            nets: Some(&nets),
            preset: Preset::Signoff,
            seed: 7,
            threads: 0,
            engine: WalkEngineKind::Cpu,
        };
        assert_eq!(
            e.command(),
            "faradice solve 'my layout.gds' --tech tech/sky130.tech.toml --nets 'bus[[]0],clk' --preset signoff --seed 7"
        );
        assert_eq!(sh("it's"), r"'it'\''s'");
    }
}
