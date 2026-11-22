//! Raw result files (`result.json`).
//!
//! The file stores the per-sink moments of every target net, so that reports
//! (thresholds, couple-to-ground, symmetrization) can be regenerated without
//! re-solving.

use std::path::Path;

use faradice_core::{Face, GROUND_SUM_LABEL, Moments, Sink};
use faradice_frw::Solution;
use faradice_geom::Geometry;
use serde::{Deserialize, Serialize};

use crate::{IoError, Result, read_to_string};

/// Supported result schema version.
///
/// Version 2 labels the ground sum `<GND>` and the fixed ground sinks
/// `<substrate>`, `<boundary:FACE>` and `<infinity>` (version 1: `GND`,
/// `GND:substrate`, …, which could collide with net names). Version 1 files
/// are read and relabelled.
pub const RESULT_SCHEMA_VERSION: u32 = 2;

/// Stored sums of one sink.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkSums {
    /// Sink label: a net name, `<substrate>`, `<boundary:FACE>` or `<infinity>` (or `<GND>`
    /// for the ground sum).
    pub sink: String,
    /// `Σw` (fF).
    pub sum: f64,
    /// `Σw²` (fF²).
    pub sum_sq: f64,
}

/// Stored statistics of one target net.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetRecord {
    /// Net name.
    pub net: String,
    /// Samples.
    pub walks: u64,
    /// Aborted walks.
    pub aborted_walks: u64,
    /// Total steps.
    pub total_steps: u64,
    /// Converged before `max_walks`.
    pub converged: bool,
    /// Gaussian offset (µm).
    pub gauss_offset: f64,
    /// Per-sink sums.
    pub sinks: Vec<SinkSums>,
    /// Aggregate ground sums.
    pub ground: SinkSums,
    /// Sink cross moments `Σ c_a c_b` (pairs that never co-occurred are
    /// zero and omitted). Only written with `output.cross_moments`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cross: Vec<CrossSum>,
}

/// One stored sink cross moment of a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossSum {
    /// First sink label.
    pub a: String,
    /// Second sink label.
    pub b: String,
    /// `Σ c_a c_b` over samples (fF²).
    pub sum: f64,
}

/// Whether a result covers the whole requested run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Completeness {
    /// The run ended normally (each target converged or used its budget).
    #[default]
    Complete,
    /// The run was cancelled; statistics are those merged until then.
    Cancelled,
    /// A snapshot of a run that is still in progress.
    Partial,
}

impl Completeness {
    fn is_complete(&self) -> bool {
        *self == Self::Complete
    }
}

/// The whole result file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFile {
    /// Schema version.
    pub schema_version: u32,
    /// Producer version.
    pub faradice_version: String,
    /// Seed.
    pub seed: u64,
    /// Stopping distance (µm). `NaN` (written as `null`) when the run was
    /// cancelled before the solver set it up.
    #[serde(deserialize_with = "nan_from_null")]
    pub eps_stop: f64,
    /// Wall-clock seconds.
    pub elapsed_s: f64,
    /// All net names of the geometry (index = net id).
    pub nets_all: Vec<String>,
    /// Names of nets treated as ground.
    pub ground_nets: Vec<String>,
    /// Per-target statistics.
    pub targets: Vec<NetRecord>,
    /// Warnings emitted while resolving and solving.
    pub warnings: Vec<String>,
    /// The configuration used (as TOML text).
    pub config_toml: String,
    /// Design name (the input's file stem), used to name the SPEF file that
    /// `faradice report` regenerates. Absent in files of older versions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub design: Option<String>,
    /// Completeness; omitted from the file when complete.
    #[serde(default, skip_serializing_if = "Completeness::is_complete")]
    pub completeness: Completeness,
}

/// Reads a number, mapping `null` (how JSON stores `NaN`) back to `NaN`.
fn nan_from_null<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// Label of a sink.
#[must_use]
pub fn sink_label(geom_names: &[String], sink: Sink) -> String {
    match sink {
        Sink::Net(n) => geom_names[n.index()].clone(),
        other => other.to_string(),
    }
}

/// Parses a sink label back (`None` for nets: the caller resolves the name).
#[must_use]
pub fn parse_fixed_sink(label: &str) -> Option<Sink> {
    match label {
        "<substrate>" => Some(Sink::GroundPlane),
        "<infinity>" => Some(Sink::Infinity),
        _ => label
            .strip_prefix("<boundary:")
            .and_then(|f| f.strip_suffix('>'))
            .and_then(|f| f.parse::<Face>().ok())
            .map(Sink::Boundary),
    }
}

impl ResultFile {
    /// Builds a result file from a solution.
    #[must_use]
    pub fn from_solution(
        geom: &Geometry,
        sol: &Solution,
        seed: u64,
        ground_nets: Vec<String>,
        elapsed_s: f64,
        warnings: Vec<String>,
        config_toml: String,
    ) -> Self {
        let names: Vec<String> = geom.nets().map(|n| geom.net_name(n).to_owned()).collect();
        let sums = |label: String, m: &Moments| SinkSums {
            sink: label,
            sum: m.sum(),
            sum_sq: m.sum_sq(),
        };
        let targets = sol
            .nets
            .iter()
            .map(|r| NetRecord {
                net: names[r.net.index()].clone(),
                walks: r.walks,
                aborted_walks: r.aborted_walks,
                total_steps: r.total_steps,
                converged: r.converged,
                gauss_offset: r.gauss_offset,
                sinks: r
                    .sinks
                    .iter()
                    .map(|(s, m)| sums(sink_label(&names, *s), m))
                    .collect(),
                ground: sums(GROUND_SUM_LABEL.into(), &r.ground_moments),
                cross: Vec::new(),
            })
            .collect();
        Self {
            schema_version: RESULT_SCHEMA_VERSION,
            faradice_version: env!("CARGO_PKG_VERSION").to_owned(),
            seed,
            eps_stop: sol.eps_stop,
            elapsed_s,
            nets_all: names,
            ground_nets,
            targets,
            warnings,
            config_toml,
            design: None,
            completeness: if sol.cancelled {
                Completeness::Cancelled
            } else {
                Completeness::Complete
            },
        }
    }

    /// Stores the cross moments of each target (aligned with
    /// [`ResultFile::targets`]).
    pub fn set_cross(&mut self, cross: &[faradice_frw::cross::CrossMoments]) {
        for (t, c) in self.targets.iter_mut().zip(cross) {
            t.cross = c
                .pairs
                .iter()
                .map(|&((a, b), sum)| CrossSum {
                    a: sink_label(&self.nets_all, a),
                    b: sink_label(&self.nets_all, b),
                    sum,
                })
                .collect();
        }
    }

    /// Writes pretty JSON.
    pub fn write(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).map_err(|source| IoError::Json {
            path: path.display().to_string(),
            source,
        })?;
        std::fs::write(path, text).map_err(|source| IoError::File {
            path: path.display().to_string(),
            source,
        })
    }

    /// Relabels a version-1 file: a sink `GND:…` that is not a net name
    /// becomes `<…>`, and the ground sum `<GND>`. A file is ambiguous, and
    /// refused, if a net name is a version-1 fixed-sink label or of the
    /// reserved form.
    fn upgrade_v1(&mut self) -> Result<()> {
        let old_fixed = |n: &str| {
            n == "GND:substrate"
                || n == "GND:infinity"
                || n.strip_prefix("GND:boundary:")
                    .is_some_and(|f| f.parse::<Face>().is_ok())
        };
        if let Some(n) = self
            .nets_all
            .iter()
            .find(|n| old_fixed(n) || faradice_core::is_reserved_label(n))
        {
            return Err(IoError::Invalid(format!(
                "version 1 result with a net named `{n}`: its sinks cannot be told apart \
                 from ground sinks; solve again"
            )));
        }
        let nets: std::collections::HashSet<String> = self.nets_all.iter().cloned().collect();
        let fix = |s: &mut String| {
            if !nets.contains(s.as_str()) {
                if let Some(rest) = s.strip_prefix("GND:") {
                    *s = format!("<{rest}>");
                }
            }
        };
        for t in &mut self.targets {
            for k in &mut t.sinks {
                fix(&mut k.sink);
            }
            for c in &mut t.cross {
                fix(&mut c.a);
                fix(&mut c.b);
            }
            t.ground.sink = GROUND_SUM_LABEL.into();
        }
        self.schema_version = 2;
        Ok(())
    }

    /// Reads a result file (version 1 files are relabelled).
    pub fn read(path: &Path) -> Result<Self> {
        let mut r: Self =
            serde_json::from_str(&read_to_string(path)?).map_err(|source| IoError::Json {
                path: path.display().to_string(),
                source,
            })?;
        if r.schema_version == 1 {
            r.upgrade_v1().map_err(|e| match e {
                IoError::Invalid(m) => IoError::Invalid(format!("{}: {m}", path.display())),
                other => other,
            })?;
        }
        if r.schema_version != RESULT_SCHEMA_VERSION {
            return Err(IoError::Invalid(format!(
                "unsupported result schema_version {}",
                r.schema_version
            )));
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_1_files_are_relabelled() {
        let text = r#"{ "schema_version": 1, "faradice_version": "t", "seed": 1, "eps_stop": 0.001,
            "elapsed_s": 0, "nets_all": ["a", "GND:x"], "ground_nets": [], "warnings": [],
            "config_toml": "", "targets": [
              { "net": "a", "walks": 1, "aborted_walks": 0, "total_steps": 0, "converged": true,
                "gauss_offset": 0.1, "ground": { "sink": "GND", "sum": 0, "sum_sq": 0 },
                "sinks": [ { "sink": "GND:substrate", "sum": -1.0, "sum_sq": 1.0 },
                           { "sink": "GND:boundary:xmin", "sum": -1.0, "sum_sq": 1.0 },
                           { "sink": "GND:x", "sum": -1.0, "sum_sq": 1.0 } ] } ] }"#;
        let dir = std::env::temp_dir().join(format!("faradice-io-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("result.json");
        std::fs::write(&p, text).unwrap();
        let r = ResultFile::read(&p).unwrap();
        assert_eq!(r.schema_version, RESULT_SCHEMA_VERSION);
        let t = &r.targets[0];
        let labels: Vec<&str> = t.sinks.iter().map(|s| s.sink.as_str()).collect();
        assert_eq!(labels, ["<substrate>", "<boundary:xmin>", "GND:x"]);
        assert_eq!(t.ground.sink, GROUND_SUM_LABEL);
        assert_eq!(
            parse_fixed_sink("<boundary:xmin>"),
            Some(Sink::Boundary("xmin".parse().unwrap()))
        );
        assert_eq!(parse_fixed_sink("<substrate>"), Some(Sink::GroundPlane));
        assert_eq!(parse_fixed_sink("GND:substrate"), None);
        assert_eq!(parse_fixed_sink("<boundary:shield"), None);
        // A net named like a version-1 fixed sink makes the file ambiguous.
        for bad in ["GND:boundary:xmax", "GND:substrate", "<substrate>"] {
            std::fs::write(&p, text.replace("\"GND:x\"]", &format!("\"{bad}\"]"))).unwrap();
            let e = ResultFile::read(&p).unwrap_err().to_string();
            assert!(e.contains("solve again"), "{bad}: {e}");
        }
    }
}
