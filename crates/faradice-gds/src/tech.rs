//! Technology description: GDS layers → conductor layers, vias and labels.
//!
//! ```toml
//! schema_version = 1
//! name = "example"
//! eps_r = 3.9            # homogeneous dielectric, or the medium above the stack
//! substrate_z = 0.0
//!
//! [[dielectric]]          # optional stratified stack, bottom-up (µm)
//! z_top = 1.3761
//! eps_r = 4.5
//!
//! [[layer]]
//! name = "met1"
//! gds = [68, 20]          # [layer, datatype]; more pairs via `also = [[68, 16]]`
//! zmin = 1.3761
//! thickness = 0.36
//! labels = [[68, 5]]      # text [layer, texttype] pairs that name nets on this layer
//!
//! [[via]]
//! name = "via"
//! gds = [68, 44]
//! bottom = "li1"
//! top = "met1"
//! ```

use std::path::Path;

use serde::Deserialize;

use crate::{GdsError, Result};

/// A conductor layer.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Name.
    pub name: String,
    /// Primary GDS `[layer, datatype]`.
    pub gds: [i16; 2],
    /// Additional drawing `[layer, datatype]` pairs (e.g. pin shapes).
    #[serde(default)]
    pub also: Vec<[i16; 2]>,
    /// Bottom z (µm).
    pub zmin: f64,
    /// Thickness (µm).
    pub thickness: f64,
    /// Text `[layer, texttype]` pairs whose labels attach to this layer.
    #[serde(default)]
    pub labels: Vec<[i16; 2]>,
    /// Conformal dielectric shell around this layer's shapes (design M2c).
    #[serde(default)]
    pub conformal: Option<Conformal>,
}

/// A conformal dielectric shell: grown laterally by `side` and upwards by
/// `top` around a layer's shapes, minus the shapes.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conformal {
    /// Lateral thickness (µm).
    pub side: f64,
    /// Thickness above the shapes' tops (µm).
    #[serde(default)]
    pub top: f64,
    /// Relative permittivity.
    pub eps_r: f64,
}

/// One stratified dielectric layer: the medium below `z_top`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DielectricLayer {
    /// Top interface height (µm).
    pub z_top: f64,
    /// Relative permittivity.
    pub eps_r: f64,
}

/// A via (cut) layer connecting two conductor layers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Via {
    /// Name.
    pub name: String,
    /// GDS `[layer, datatype]`.
    pub gds: [i16; 2],
    /// Lower conductor layer.
    pub bottom: String,
    /// Upper conductor layer.
    pub top: String,
}

/// Technology file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tech {
    /// Schema version (1).
    pub schema_version: u32,
    /// Display name.
    #[serde(default)]
    pub name: String,
    /// Homogeneous dielectric constant, or that of the medium above the last
    /// `[[dielectric]]` layer.
    pub eps_r: f64,
    /// Stratified dielectric layers (µm), bottom-up (design M2a).
    #[serde(default)]
    pub dielectric: Vec<DielectricLayer>,
    /// Substrate height (µm).
    #[serde(default)]
    pub substrate_z: f64,
    /// Conductor layers.
    #[serde(rename = "layer")]
    pub layers: Vec<Layer>,
    /// Vias.
    #[serde(default, rename = "via")]
    pub vias: Vec<Via>,
}

/// Z range and role of a GDS layer/datatype after resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZSpan {
    /// Index into [`Tech::layers`] (conductors) or `layers.len() + via index` (vias).
    pub index: usize,
    /// Bottom z.
    pub z0: f64,
    /// Top z.
    pub z1: f64,
}

impl Tech {
    /// Parses and validates TOML text.
    pub fn parse(text: &str) -> Result<Self> {
        let t: Self = toml::from_str(text).map_err(|e| GdsError::Tech(e.to_string()))?;
        if t.schema_version != 1 {
            return Err(GdsError::Tech(format!(
                "unsupported schema_version {}",
                t.schema_version
            )));
        }
        // As in structure files.
        if !(t.eps_r.is_finite() && t.eps_r >= 1.0) {
            return Err(GdsError::Tech(format!(
                "eps_r must be finite and >= 1 (got {})",
                t.eps_r
            )));
        }
        for l in &t.layers {
            if !(l.thickness > 0.0 && l.zmin.is_finite()) {
                return Err(GdsError::Tech(format!(
                    "layer `{}` needs finite zmin and positive thickness",
                    l.name
                )));
            }
            if let Some(c) = &l.conformal {
                let ok_t = |v: f64| v.is_finite() && v >= 0.0;
                if !(ok_t(c.side)
                    && ok_t(c.top)
                    && (c.side > 0.0 || c.top > 0.0)
                    && c.eps_r.is_finite()
                    && c.eps_r >= 1.0)
                {
                    return Err(GdsError::Tech(format!(
                        "layer `{}`: conformal side and top must be finite, >= 0 and not both 0, eps_r finite and >= 1",
                        l.name
                    )));
                }
            }
        }
        for (i, d) in t.dielectric.iter().enumerate() {
            let above = i == 0 || d.z_top > t.dielectric[i - 1].z_top;
            if !(d.z_top.is_finite() && above && d.eps_r.is_finite() && d.eps_r >= 1.0) {
                return Err(GdsError::Tech(format!(
                    "dielectric {i}: z_top must be finite and increasing, eps_r finite and >= 1"
                )));
            }
        }
        for v in &t.vias {
            let (b, tp) = (t.layer(&v.bottom), t.layer(&v.top));
            match (b, tp) {
                (Some(b), Some(tp)) if b.zmin + b.thickness < tp.zmin => {}
                (Some(_), Some(_)) => {
                    return Err(GdsError::Tech(format!(
                        "via `{}`: `{}` must lie below `{}`",
                        v.name, v.bottom, v.top
                    )));
                }
                _ => {
                    return Err(GdsError::Tech(format!(
                        "via `{}` references an unknown layer",
                        v.name
                    )));
                }
            }
        }
        Ok(t)
    }

    /// Reads a technology file.
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| GdsError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Looks up a conductor layer by name.
    #[must_use]
    pub fn layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Resolves a GDS `(layer, datatype)` to its z span, if mapped.
    #[must_use]
    pub fn span(&self, layer: i16, datatype: i16) -> Option<ZSpan> {
        let key = [layer, datatype];
        if let Some((i, l)) = self
            .layers
            .iter()
            .enumerate()
            .find(|(_, l)| l.gds == key || l.also.contains(&key))
        {
            return Some(ZSpan {
                index: i,
                z0: l.zmin,
                z1: l.zmin + l.thickness,
            });
        }
        self.vias
            .iter()
            .enumerate()
            .find(|(_, v)| v.gds == key)
            .and_then(|(i, v)| {
                let (b, t) = (self.layer(&v.bottom)?, self.layer(&v.top)?);
                Some(ZSpan {
                    index: self.layers.len() + i,
                    z0: b.zmin + b.thickness,
                    z1: t.zmin,
                })
            })
    }

    /// Conductor layer index that a text `(layer, texttype)` labels, if any.
    #[must_use]
    pub fn label_layer(&self, layer: i16, texttype: i16) -> Option<usize> {
        self.layers
            .iter()
            .position(|l| l.labels.contains(&[layer, texttype]))
    }

    /// Name of a span index (conductor or via).
    #[must_use]
    pub fn span_name(&self, index: usize) -> &str {
        if index < self.layers.len() {
            &self.layers[index].name
        } else {
            &self.vias[index - self.layers.len()].name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
schema_version = 1
eps_r = 1.0

[[layer]]
name = "met1"
gds = [68, 20]
zmin = 1.0
thickness = 0.5
"#;

    #[test]
    fn rejects_a_background_eps_r_below_one() {
        for bad in ["0.0", "-3.9", "0.5", "nan", "inf"] {
            let text = BASE.replace("eps_r = 1.0", &format!("eps_r = {bad}"));
            assert!(Tech::parse(&text).is_err(), "{bad}");
        }
    }

    #[test]
    fn reads_and_validates_a_dielectric_stack() {
        let ok = format!(
            "{BASE}\n[[dielectric]]\nz_top = 0.5\neps_r = 3.9\n\n[[dielectric]]\nz_top = 2.0\neps_r = 4.5\n"
        );
        let t = Tech::parse(&ok).unwrap();
        assert_eq!(t.dielectric.len(), 2);
        assert_eq!((t.dielectric[1].z_top, t.dielectric[1].eps_r), (2.0, 4.5));
        assert!(Tech::parse(BASE).unwrap().dielectric.is_empty());
        for bad in [
            "z_top = 0.5\neps_r = 3.9\n\n[[dielectric]]\nz_top = 0.5\neps_r = 4.5",
            "z_top = nan\neps_r = 3.9",
            "z_top = 0.5\neps_r = 0.5",
        ] {
            let text = format!("{BASE}\n[[dielectric]]\n{bad}\n");
            assert!(Tech::parse(&text).is_err(), "{bad}");
        }
    }

    #[test]
    fn reads_and_validates_conformal_layers() {
        let ok = format!("{BASE}conformal = {{ side = 0.03, eps_r = 3.5 }}\n");
        let c = Tech::parse(&ok).unwrap().layers[0].conformal.unwrap();
        assert_eq!((c.side, c.top, c.eps_r), (0.03, 0.0, 3.5));
        assert!(Tech::parse(BASE).unwrap().layers[0].conformal.is_none());
        for bad in [
            "{ side = 0.0, eps_r = 3.5 }",
            "{ side = -0.1, eps_r = 3.5 }",
            "{ side = 0.03, eps_r = 0.5 }",
            "{ side = 0.03, top = nan, eps_r = 3.5 }",
        ] {
            let text = format!("{BASE}conformal = {bad}\n");
            assert!(Tech::parse(&text).is_err(), "{bad}");
        }
    }
}
