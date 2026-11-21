//! Box-based structure files.
//!
//! ```toml
//! schema_version = 1
//! units = "um"            # nm | um | mm | m
//!
//! [dielectric]
//! eps_r = 3.9             # the whole domain, or the medium above the layers
//!
//! [[dielectric.layer]]    # optional stratified stack, bottom-up
//! z_top = 0.5             # top of this layer (same unit as the boxes)
//! eps_r = 4.5
//!
//! [[dielectric.box]]      # optional boxes that override the stack (design M2c)
//! min = [0.0, 0.0, 1.0]
//! max = [10.0, 0.5, 1.36]
//! eps_r = 3.5
//!
//! [[dielectric.conformal]] # optional shells around the conductors of a layer
//! layer = "met1"
//! side = 0.03             # lateral thickness
//! top = 0.0               # thickness above the conductor tops (default 0)
//! eps_r = 3.5
//!
//! [[conductor]]
//! net = "A"
//! layer = "met1"          # optional tag, selects the boxes for conformal rules
//! boxes = [[0.0, 0.0, 1.0, 10.0, 0.14, 1.36]]   # [xmin, ymin, zmin, xmax, ymax, zmax]
//! ```

use std::path::Path;

use faradice_core::{BoxId, LengthUnit};
use faradice_frw::{
    ConformalRule, DielectricBox, DielectricLayer, DielectricRegions, validate_layers,
};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};
use serde::{Deserialize, Serialize};

use crate::{IoError, Result, read_to_string};

/// The supported structure schema version.
pub const STRUCTURE_SCHEMA_VERSION: u32 = 1;

/// Raw structure file contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructureFile {
    /// Schema version (must be 1).
    pub schema_version: u32,
    /// Length unit of all coordinates.
    #[serde(default = "default_units")]
    pub units: String,
    /// Dielectric (homogeneous or a stratified stack).
    pub dielectric: Dielectric,
    /// Substrate plane height (µm) implied by the source (e.g. a GDS technology
    /// file). Overrides `ground.substrate_z` of the run configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub substrate_z: Option<f64>,
    /// Conductors.
    #[serde(default, rename = "conductor")]
    pub conductors: Vec<Conductor>,
}

fn default_units() -> String {
    "um".to_owned()
}

/// Dielectric: a homogeneous medium, or a stratified stack (design M2a).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dielectric {
    /// Relative permittivity of the whole domain, or of the medium above the
    /// last layer.
    pub eps_r: f64,
    /// Stratified layers, bottom-up; the first extends downwards to the domain
    /// bottom.
    #[serde(default, rename = "layer", skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<Layer>,
    /// Boxes of dielectric that override the stack (design M2c).
    #[serde(default, rename = "box", skip_serializing_if = "Vec::is_empty")]
    pub boxes: Vec<BoxSpec>,
    /// Conformal shells around the conductors of a layer (design M2c).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conformal: Vec<Conformal>,
}

/// A box of dielectric (coordinates in the file's unit).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxSpec {
    /// Lower corner.
    pub min: [f64; 3],
    /// Upper corner.
    pub max: [f64; 3],
    /// Relative permittivity.
    pub eps_r: f64,
}

/// Conformal shells around the conductor boxes tagged with `layer`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conformal {
    /// Layer tag of the conductor boxes (see [`Conductor::layer`]). A tag that
    /// no conductor uses selects nothing.
    pub layer: String,
    /// Lateral thickness (in the file's unit).
    pub side: f64,
    /// Thickness above the conductor tops (in the file's unit).
    #[serde(default)]
    pub top: f64,
    /// Relative permittivity of the shell.
    pub eps_r: f64,
}

/// One dielectric layer of a stratified stack.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Height of the layer's top interface (in the file's unit).
    pub z_top: f64,
    /// Relative permittivity.
    pub eps_r: f64,
}

/// One net's boxes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conductor {
    /// Net name.
    pub net: String,
    /// Layer tag of these boxes, used by conformal rules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<String>,
    /// Boxes `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub boxes: Vec<[f64; 6]>,
}

/// A parsed structure: geometry in µm plus the dielectric.
#[derive(Debug, Clone)]
pub struct Structure {
    /// Conductor geometry (µm).
    pub geometry: Geometry,
    /// Relative permittivity (homogeneous, or above the last layer).
    pub eps_r: f64,
    /// Stratified dielectric layers (µm), bottom-up; empty if homogeneous.
    pub layers: Vec<DielectricLayer>,
    /// Substrate height from the structure, if specified.
    pub substrate_z: Option<f64>,
    /// Dielectric boxes and conformal rules (µm).
    pub regions: DielectricRegions,
}

impl StructureFile {
    /// Parses TOML text.
    pub fn parse(text: &str, path: &str) -> Result<Self> {
        toml::from_str(text).map_err(|source| IoError::Toml {
            path: path.to_owned(),
            source,
        })
    }

    /// Converts to a validated [`Structure`], keeping only nets accepted by `keep`.
    pub fn build(&self, keep: impl Fn(&str) -> bool) -> Result<Structure> {
        if self.schema_version != STRUCTURE_SCHEMA_VERSION {
            return Err(IoError::Invalid(format!(
                "unsupported structure schema_version {} (expected {STRUCTURE_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        let unit: LengthUnit = self
            .units
            .parse()
            .map_err(|e: faradice_core::Error| IoError::Invalid(e.to_string()))?;
        let k = unit.to_um_factor();
        let mut b = GeometryBuilder::new();
        let mut tagged: Vec<(&str, BoxId)> = Vec::new();
        for c in self.conductors.iter().filter(|c| keep(&c.net)) {
            let net = b.net(&c.net)?;
            for bx in &c.boxes {
                let id = b.add_box(net, Aabb::from_array(bx.map(|v| v * k))?)?;
                if let Some(layer) = &c.layer {
                    tagged.push((layer, id));
                }
            }
        }
        if !(self.dielectric.eps_r.is_finite() && self.dielectric.eps_r >= 1.0) {
            return Err(IoError::Invalid(
                "dielectric.eps_r must be finite and >= 1".into(),
            ));
        }
        let layers: Vec<DielectricLayer> = self
            .dielectric
            .layers
            .iter()
            .map(|l| DielectricLayer {
                z_top: l.z_top * k,
                eps_r: l.eps_r,
            })
            .collect();
        validate_layers(&layers).map_err(IoError::Invalid)?;
        let d = &self.dielectric;
        let boxes = d
            .boxes
            .iter()
            .map(|bx| {
                Ok(DielectricBox {
                    region: Aabb::new(bx.min.map(|v| v * k), bx.max.map(|v| v * k))?,
                    eps_r: bx.eps_r,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let conformal = d
            .conformal
            .iter()
            .map(|r| ConformalRule {
                layer: r.layer.clone(),
                boxes: tagged
                    .iter()
                    .filter(|(l, _)| *l == r.layer)
                    .map(|&(_, id)| id)
                    .collect(),
                side: r.side * k,
                top: r.top * k,
                eps_r: r.eps_r,
            })
            .collect();
        let regions = DielectricRegions { boxes, conformal };
        let geometry = b.build()?;
        regions.validate(&geometry).map_err(IoError::Invalid)?;
        Ok(Structure {
            geometry,
            eps_r: self.dielectric.eps_r,
            layers,
            substrate_z: self.substrate_z,
            regions,
        })
    }

    /// All net names in file order (deduplicated).
    #[must_use]
    pub fn net_names(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for c in &self.conductors {
            if !v.contains(&c.net) {
                v.push(c.net.clone());
            }
        }
        v
    }
}

/// Reads a structure file.
pub fn read_structure(path: &Path) -> Result<StructureFile> {
    StructureFile::parse(&read_to_string(path)?, &path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const STACK: &str = r#"
schema_version = 1
units = "nm"

[dielectric]
eps_r = 1.0

[[dielectric.layer]]
z_top = 500.0
eps_r = 3.9

[[dielectric.layer]]
z_top = 1500.0
eps_r = 4.5

[[conductor]]
net = "a"
boxes = [[0.0, 0.0, 1000.0, 100.0, 100.0, 1200.0]]
"#;

    #[test]
    fn reads_a_stack_in_file_units() {
        let st = StructureFile::parse(STACK, "t")
            .unwrap()
            .build(|_| true)
            .unwrap();
        assert_eq!(st.eps_r, 1.0);
        assert_eq!(
            st.layers,
            vec![
                DielectricLayer {
                    z_top: 0.5,
                    eps_r: 3.9
                },
                DielectricLayer {
                    z_top: 1.5,
                    eps_r: 4.5
                }
            ]
        );
    }

    #[test]
    fn homogeneous_files_are_unchanged() {
        let text = STACK
            .split("[[dielectric.layer]]")
            .next()
            .unwrap()
            .to_owned()
            + "[[conductor]]\nnet = \"a\"\nboxes = [[0, 0, 1, 1, 1, 2]]\n";
        let f = StructureFile::parse(&text, "t").unwrap();
        assert!(f.build(|_| true).unwrap().layers.is_empty());
        // Serialisation omits the empty layer list.
        assert!(!toml::to_string(&f).unwrap().contains("layer"));
    }

    #[test]
    fn rejects_unordered_layers() {
        let text = STACK.replace("z_top = 1500.0", "z_top = 400.0");
        let err = StructureFile::parse(&text, "t")
            .unwrap()
            .build(|_| true)
            .unwrap_err();
        assert!(err.to_string().contains("z_top"), "{err}");
    }

    const REGIONS: &str = r#"
schema_version = 1
units = "nm"

[dielectric]
eps_r = 4.5

[[dielectric.box]]
min = [0.0, 0.0, 0.0]
max = [1000.0, 1000.0, 100.0]
eps_r = 7.0

[[dielectric.conformal]]
layer = "met1"
side = 30.0
eps_r = 3.5

[[dielectric.conformal]]
layer = "met9"
side = 30.0
top = 10.0
eps_r = 3.0

[[conductor]]
net = "a"
layer = "met1"
boxes = [[0.0, 0.0, 1000.0, 100.0, 100.0, 1200.0]]

[[conductor]]
net = "a"
boxes = [[0.0, 0.0, 1200.0, 100.0, 100.0, 1400.0]]

[[conductor]]
net = "b"
layer = "met1"
boxes = [[300.0, 0.0, 1000.0, 400.0, 100.0, 1200.0]]
"#;

    #[test]
    fn reads_boxes_and_conformal_rules() {
        let f = StructureFile::parse(REGIONS, "t").unwrap();
        let st = f.build(|_| true).unwrap();
        let r = &st.regions;
        assert_eq!(r.boxes.len(), 1);
        assert_eq!(r.boxes[0].region.max(), [1.0, 1.0, 0.1]);
        assert_eq!(r.boxes[0].eps_r, 7.0);
        assert_eq!(r.conformal.len(), 2);
        let m1 = &r.conformal[0];
        assert_eq!(
            (m1.layer.as_str(), m1.side, m1.top, m1.eps_r),
            ("met1", 0.03, 0.0, 3.5)
        );
        // Box ids 0 (net a, met1) and 2 (net b, met1); box 1 is untagged.
        assert_eq!(m1.boxes, vec![BoxId::new(0), BoxId::new(2)]);
        // A tag that no conductor uses selects nothing.
        assert!(r.conformal[1].boxes.is_empty());
        assert!((r.conformal[1].top - 0.01).abs() < 1e-15);
        // Dropping net `a` drops its boxes from the rule.
        let st = f.build(|n| n == "b").unwrap();
        assert_eq!(st.regions.conformal[0].boxes, vec![BoxId::new(0)]);
        // Round trip.
        let again = StructureFile::parse(&toml::to_string(&f).unwrap(), "t").unwrap();
        assert_eq!(again.build(|_| true).unwrap().regions, r.clone());
    }

    #[test]
    fn rejects_invalid_regions() {
        for (from, to) in [
            ("eps_r = 7.0", "eps_r = 0.5"),
            ("side = 30.0\neps_r = 3.5", "side = 0.0\neps_r = 3.5"),
            (
                "max = [1000.0, 1000.0, 100.0]",
                "max = [1000.0, 1000.0, 0.0]",
            ),
        ] {
            let text = REGIONS.replacen(from, to, 1);
            assert!(
                StructureFile::parse(&text, "t")
                    .unwrap()
                    .build(|_| true)
                    .is_err(),
                "{to}"
            );
        }
        let overlapping = REGIONS.replacen(
            "[[dielectric.conformal]]",
            "[[dielectric.box]]\nmin = [500.0, 500.0, 50.0]\nmax = [2000.0, 2000.0, 80.0]\neps_r = 2.0\n\n[[dielectric.conformal]]",
            1,
        );
        let err = StructureFile::parse(&overlapping, "t")
            .unwrap()
            .build(|_| true)
            .unwrap_err();
        assert!(err.to_string().contains("overlap"), "{err}");
        assert!(
            StructureFile::parse(&REGIONS.replacen("side = 30.0", "sides = 30.0", 1), "t").is_err()
        );
    }
}
