//! Loaded inputs.

use std::path::{Path, PathBuf};

use faradice_gds::{flatten, stream, tech::Tech, trace};
use faradice_io::config::Config;
use faradice_io::structure::{Structure, StructureFile, read_structure};

use crate::{EngineError, Result};

/// What a project was loaded from.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Source {
    /// A box structure file (`*.faradice.toml`).
    Structure(StructureFile),
    /// A GDSII layout traced with a technology file.
    Gds {
        /// Traced conductors and nets.
        traced: trace::Traced,
        /// Technology used for tracing.
        tech: Tech,
    },
}

/// A loaded input, solvable many times with different configurations.
#[derive(Debug, Clone)]
pub struct Project {
    /// Input path.
    pub path: PathBuf,
    /// Design name (file stem), used for SPEF and output names.
    pub design: String,
    /// Loaded contents.
    pub source: Source,
    /// Non-fatal problems found while loading (e.g. GDS label conflicts).
    pub diagnostics: Vec<String>,
}

/// Whether `path` names a GDSII file (by extension).
#[must_use]
pub fn is_gds(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "gds" | "gds2" | "gdsii"))
}

impl Project {
    /// Loads a structure file, or a GDSII file when `tech` is given (GDSII
    /// input requires a technology file).
    ///
    /// # Errors
    ///
    /// Read, parse, flatten or trace errors.
    pub fn load(path: &Path, tech: Option<&Path>, top: Option<&str>) -> Result<Self> {
        Self::load_with(path, tech, top, trace::TraceOptions::default())
    }

    /// [`Project::load`] with explicit trace options for GDSII input
    /// (ignored for structure files).
    ///
    /// # Errors
    ///
    /// Read, parse, flatten or trace errors.
    pub fn load_with(
        path: &Path,
        tech: Option<&Path>,
        top: Option<&str>,
        opts: trace::TraceOptions,
    ) -> Result<Self> {
        let design = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("design")
            .trim_end_matches(".faradice")
            .to_owned();
        if is_gds(path) {
            let tech = tech.ok_or_else(|| {
                EngineError::Other("GDSII input requires a technology file".into())
            })?;
            Self::load_gds(path, tech, top, opts, design)
        } else {
            Self {
                path: path.to_owned(),
                design,
                source: Source::Structure(read_structure(path)?),
                diagnostics: Vec::new(),
            }
            .checked()
        }
    }

    /// Refuses net names of the reserved form `<…>` (sink labels), whether or
    /// not a configuration later removes the net.
    fn checked(self) -> Result<Self> {
        if let Some(n) = self
            .net_names()
            .into_iter()
            .find(|n| faradice_core::is_reserved_label(n))
        {
            return Err(EngineError::Other(format!(
                "net name `{n}` is reserved: names of the form <…> label ground sinks"
            )));
        }
        Ok(self)
    }

    /// Loads and traces a GDSII file with explicit trace options.
    ///
    /// # Errors
    ///
    /// Read, flatten or trace errors.
    pub fn load_gds(
        path: &Path,
        tech: &Path,
        top: Option<&str>,
        opts: trace::TraceOptions,
        design: String,
    ) -> Result<Self> {
        let tech = Tech::read(tech)?;
        let lib = stream::read_library(path)?;
        let units = lib
            .units
            .ok_or_else(|| EngineError::Other("GDS file has no UNITS record".into()))?;
        let flat = flatten::flatten(&lib, top, flatten::FlattenOptions::default())?;
        let traced = trace::trace(&flat, units, &tech, opts)?;
        let diagnostics = traced.diagnostics.iter().map(ToString::to_string).collect();
        Self {
            path: path.to_owned(),
            design,
            source: Source::Gds { traced, tech },
            diagnostics,
        }
        .checked()
    }

    /// All net names, in input order. Unlabelled GDS components (`$…`) are
    /// included; see [`Self::labelled_nets`].
    #[must_use]
    pub fn net_names(&self) -> Vec<String> {
        match &self.source {
            Source::Structure(f) => f.net_names(),
            Source::Gds { traced, .. } => {
                traced.components.iter().map(|c| c.name.clone()).collect()
            }
        }
    }

    /// Net names without unlabelled GDS components.
    #[must_use]
    pub fn labelled_nets(&self) -> Vec<String> {
        self.net_names()
            .into_iter()
            .filter(|n| !n.starts_with('$'))
            .collect()
    }

    /// Builds the solver geometry for `cfg` (removed nets dropped) and applies
    /// the input's substrate height to `cfg.ground.substrate_z`.
    ///
    /// # Errors
    ///
    /// Geometry errors (e.g. invalid boxes).
    pub fn build(&self, cfg: &mut Config) -> Result<Structure> {
        let st = match &self.source {
            Source::Structure(f) => f.build(|n| !cfg.is_removed(n))?,
            Source::Gds { traced, tech } => {
                let (geometry, spans) = traced.to_geometry_with_spans(|n| !cfg.is_removed(n))?;
                let regions = conformal_regions(tech, &spans);
                regions
                    .validate(&geometry)
                    .map_err(faradice_io::IoError::Invalid)?;
                Structure {
                    geometry,
                    regions,
                    eps_r: tech.eps_r,
                    layers: tech
                        .dielectric
                        .iter()
                        .map(|d| faradice_frw::DielectricLayer {
                            z_top: d.z_top,
                            eps_r: d.eps_r,
                        })
                        .collect(),
                    substrate_z: Some(tech.substrate_z),
                }
            }
        };
        if let Some(z) = st.substrate_z {
            cfg.ground.substrate_z = z;
        }
        Ok(st)
    }
}

/// Conformal rules of the technology's layers (design M2c §6.3), selecting the
/// geometry boxes drawn on each layer (`spans` holds the span index of every
/// box, indexed by box id).
#[must_use]
pub(crate) fn conformal_regions(
    tech: &faradice_gds::tech::Tech,
    spans: &[usize],
) -> faradice_frw::DielectricRegions {
    let conformal = tech
        .layers
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let c = l.conformal?;
            let boxes = spans
                .iter()
                .enumerate()
                .filter(|&(_, &s)| s == i)
                .map(|(b, _)| {
                    faradice_core::BoxId::new(u32::try_from(b).expect("box ids fit in u32"))
                })
                .collect();
            Some(faradice_frw::ConformalRule {
                layer: l.name.clone(),
                boxes,
                side: c.side,
                top: c.top,
                eps_r: c.eps_r,
            })
        })
        .collect();
    faradice_frw::DielectricRegions {
        boxes: Vec::new(),
        conformal,
    }
}

#[cfg(test)]
mod conformal_tests {
    use super::*;

    #[test]
    fn rules_select_the_boxes_of_their_layer() {
        let tech = faradice_gds::tech::Tech::parse(
            r#"
schema_version = 1
eps_r = 4.0

[[layer]]
name = "met1"
gds = [68, 20]
zmin = 1.0
thickness = 0.36
conformal = { side = 0.03, eps_r = 3.5 }

[[layer]]
name = "met2"
gds = [69, 20]
zmin = 2.0
thickness = 0.36
"#,
        )
        .unwrap();
        let r = conformal_regions(&tech, &[0, 1, 0, 1]);
        assert!(r.boxes.is_empty());
        assert_eq!(r.conformal.len(), 1);
        let rule = &r.conformal[0];
        assert_eq!(rule.layer, "met1");
        assert_eq!(
            rule.boxes,
            vec![faradice_core::BoxId::new(0), faradice_core::BoxId::new(2)]
        );
        assert_eq!((rule.side, rule.top, rule.eps_r), (0.03, 0.0, 3.5));
    }
}
