//! Run configuration (`faradice.toml`), see design §4.
//!
//! Every section and key is optional; defaults follow the design document.
//! Unknown keys are rejected so that typos are reported instead of ignored.

use std::path::Path;

use faradice_core::{Axis, Face, NetId, Side};
use faradice_frw::{
    Accuracy, Boundary, DielectricLayer, DielectricRegions, FaceKind, SolverConfig, StackOptions,
    WalkOptions,
};
use faradice_geom::{Aabb, Geometry};
use serde::{Deserialize, Serialize};

use crate::glob::glob_match;
use crate::{IoError, Result, read_to_string};

/// Supported config schema version.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

/// Top-level configuration file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// Schema version (1).
    pub schema_version: Option<u32>,
    /// Ground model (§4.1).
    pub ground: GroundConfig,
    /// Power nets.
    pub power: PowerConfig,
    /// Simulation domain.
    pub domain: DomainConfig,
    /// Environment scope (§4.2).
    pub scope: ScopeConfig,
    /// Target nets (§4.3).
    pub extract: ExtractConfig,
    /// Accuracy and stopping (§4.4).
    pub accuracy: AccuracyConfig,
    /// Coupling post-processing (§4.5).
    pub coupling: CouplingConfig,
    /// Walk tuning.
    pub walk: WalkConfig,
    /// Run options.
    pub run: RunConfig,
    /// Outputs.
    pub output: OutputConfig,
}

/// `ground.ignore` (§4.1.3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundIgnore {
    /// Model ground as configured.
    #[default]
    None,
    /// Solve with ground, drop ground capacitance from reports.
    Report,
    /// Remove ground from the physics.
    Physics,
}

/// Substrate model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubstrateKind {
    /// Infinite ground plane at `substrate_z`.
    #[default]
    Plane,
    /// No substrate.
    None,
}

/// What to do with ground nets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetTreatment {
    /// Conductors whose capacitance is reported as ground.
    #[default]
    Ground,
    /// Ordinary nets.
    Net,
    /// Removed from the geometry.
    Remove,
}

/// How ground capacitance is reported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundReport {
    /// One merged `C_g`.
    #[default]
    Merged,
    /// Merged plus per-source breakdown.
    BySource,
    /// Not reported.
    Drop,
}

/// Boundary condition of a domain face.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaceBc {
    /// Dirichlet 0 V.
    #[default]
    Ground,
    /// Neumann (mirror).
    Reflect,
}

/// `[ground]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GroundConfig {
    /// Shorthand for ignoring ground.
    pub ignore: GroundIgnore,
    /// Substrate model.
    pub substrate: SubstrateKind,
    /// Substrate plane height (µm).
    pub substrate_z: f64,
    /// Ground net globs.
    pub nets: Vec<String>,
    /// Treatment of ground nets.
    pub nets_treatment: NetTreatment,
    /// Ground reporting.
    pub report: GroundReport,
    /// Per-face boundary conditions of a box domain.
    pub boundary: FaceBcs,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            ignore: GroundIgnore::None,
            substrate: SubstrateKind::Plane,
            substrate_z: 0.0,
            nets: vec!["VSS".into(), "GND".into(), "vss*".into(), "VGND".into()],
            nets_treatment: NetTreatment::Ground,
            report: GroundReport::Merged,
            boundary: FaceBcs::default(),
        }
    }
}

/// `[ground.boundary]`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[allow(missing_docs)]
pub struct FaceBcs {
    pub xmin: FaceBc,
    pub xmax: FaceBc,
    pub ymin: FaceBc,
    pub ymax: FaceBc,
    pub zmax: FaceBc,
}

/// `[power]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PowerConfig {
    /// Power net globs.
    pub nets: Vec<String>,
    /// `ground` (default) or `net`.
    pub treat_as: PowerTreatment,
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            nets: vec!["VDD".into(), "VCC".into(), "vdd*".into(), "VPWR".into()],
            treat_as: PowerTreatment::Ground,
        }
    }
}

/// Power net treatment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerTreatment {
    /// Treated like ground nets.
    #[default]
    Ground,
    /// Ordinary nets.
    Net,
}

/// Domain kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainKind {
    /// Finite box around the conductors.
    #[default]
    Box,
    /// Unbounded space (infinity at 0 V).
    Open,
}

/// `[domain]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DomainConfig {
    /// Box or open.
    pub kind: DomainKind,
    /// Margin around the conductors (µm); `None` = automatic.
    pub margin: Option<f64>,
}

/// `[scope]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScopeConfig {
    /// Nets removed from the geometry (changes the physics; warned).
    pub ignore_nets: Vec<String>,
}

/// `[extract]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ExtractConfig {
    /// Target net globs.
    pub nets: Vec<String>,
    /// Excluded globs.
    pub exclude_nets: Vec<String>,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            nets: vec!["*".into()],
            exclude_nets: Vec::new(),
        }
    }
}

/// Accuracy presets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    /// 2 % total, 10 % couplings.
    Fast,
    /// 1 % total, 5 % couplings.
    #[default]
    Default,
    /// 0.5 % total, 3 % couplings.
    Signoff,
    /// 0.1 % total, 1 % couplings.
    Reference,
}

/// Coupling tracking rule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TrackConfig {
    /// Relative threshold (fraction of total); 0 disables.
    pub rel: f64,
    /// Absolute threshold (fF); 0 disables.
    pub abs_ff: f64,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            rel: 0.05,
            abs_ff: 0.0,
        }
    }
}

/// `[accuracy]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AccuracyConfig {
    /// Preset; explicit keys override it.
    pub preset: Preset,
    /// Total capacitance relative tolerance.
    pub total_rel_tol: Option<f64>,
    /// Tracked coupling relative tolerance.
    pub coupling_rel_tol: Option<f64>,
    /// Which couplings are tracked.
    pub coupling_track: TrackConfig,
    /// Confidence in σ.
    pub confidence_sigma: Option<f64>,
    /// Minimum samples per net.
    pub min_walks: Option<u64>,
    /// Maximum samples per net.
    pub max_walks: Option<u64>,
    /// Samples per batch.
    pub batch_walks: Option<u32>,
}

/// Coupling mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouplingMode {
    /// Keep couplings.
    #[default]
    Keep,
    /// Move all couplings to ground (couple-to-ground).
    Ground,
}

/// Which totals a relative threshold compares against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelRef {
    /// Below the threshold for either net.
    #[default]
    Either,
    /// Below the threshold for both nets.
    Both,
    /// Compared with the smaller total.
    Min,
    /// Compared with the larger total.
    Max,
}

/// Combination of enabled thresholds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdOp {
    /// Moved if all enabled thresholds say so.
    And,
    /// Moved if any enabled threshold says so.
    #[default]
    Or,
}

/// Coupling handling for non-target nets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonTarget {
    /// Keep as coupling.
    #[default]
    Keep,
    /// Move to ground.
    Ground,
}

/// Symmetrization of C_ij / C_ji when both nets are targets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Symmetrize {
    /// Inverse-variance weighted mean.
    #[default]
    InverseVariance,
    /// Plain mean.
    Average,
    /// The estimate with the smaller standard error.
    MaxSide,
    /// Keep both directions as computed.
    None,
}

/// `[coupling]` (§4.5). All thresholds equal to 0 are disabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CouplingConfig {
    /// Keep or couple-to-ground.
    pub mode: CouplingMode,
    /// Multiplier applied to couplings moved to ground.
    pub ground_multiplier: f64,
    /// Absolute threshold (fF).
    pub abs_threshold_ff: f64,
    /// Relative threshold (fraction of total).
    pub rel_threshold: f64,
    /// Reference for the relative threshold.
    pub rel_ref: RelRef,
    /// How abs and rel combine.
    pub threshold_op: ThresholdOp,
    /// Couplings below `k·σ` are moved to ground.
    pub significance_sigma: f64,
    /// Couplings to non-target nets.
    pub to_non_target: NonTarget,
    /// Symmetrization.
    pub symmetrize: Symmetrize,
}

impl Default for CouplingConfig {
    fn default() -> Self {
        Self {
            mode: CouplingMode::Keep,
            ground_multiplier: 1.0,
            abs_threshold_ff: 0.0,
            rel_threshold: 0.0,
            rel_ref: RelRef::Either,
            threshold_op: ThresholdOp::Or,
            significance_sigma: 0.0,
            to_non_target: NonTarget::Keep,
            symmetrize: Symmetrize::InverseVariance,
        }
    }
}

/// `[walk]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WalkConfig {
    /// Absolute stop distance (µm).
    pub stop_dist: Option<f64>,
    /// Relative stop distance.
    pub stop_ratio: Option<f64>,
    /// Gaussian offset fraction.
    pub gauss_alpha: Option<f64>,
    /// Step limit.
    pub max_steps: Option<u32>,
    /// Green table resolution.
    pub green_resolution: Option<usize>,
    /// Antithetic first hop.
    pub antithetic: Option<bool>,
    /// Stratified dielectrics: tabulated interface offsets `K`.
    pub interface_offsets: Option<usize>,
    /// Stratified dielectrics: cells per face edge of the layered tables.
    pub layered_resolution: Option<usize>,
    /// Stratified dielectrics: largest half-band fraction `η_max`.
    pub band_fraction: Option<f64>,
    /// Stratified dielectrics: landing threshold `α`.
    pub landing_alpha: Option<f64>,
    /// Dielectric boxes: edge cubes and landing on edge planes (design M2d).
    pub edge_cubes: Option<bool>,
    /// Dielectric boxes: window first hops for 2D cross-sections (design M2f).
    pub window_hops: Option<bool>,
}

/// `[run]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RunConfig {
    /// Master seed.
    pub seed: u64,
    /// Worker threads (0 = all cores).
    pub threads: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            threads: 0,
        }
    }
}

/// `[output]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OutputConfig {
    /// Report formats: `json`, `csv`, `spef`.
    pub formats: Vec<String>,
    /// Store per-target sink cross moments in `result.json` (needed for the
    /// σ of `faradice whatif`; CPU engine only).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cross_moments: bool,
}

/// The report formats `output.formats` accepts.
pub const OUTPUT_FORMATS: [&str; 3] = ["json", "csv", "spef"];

impl OutputConfig {
    /// Checks that every entry of `formats` is known, so a typo fails before
    /// the solve rather than when the outputs are written.
    ///
    /// # Errors
    ///
    /// [`IoError::Invalid`] naming the first unknown format.
    pub fn validate(&self) -> Result<()> {
        match self
            .formats
            .iter()
            .find(|f| !OUTPUT_FORMATS.contains(&f.as_str()))
        {
            Some(f) => Err(IoError::Invalid(format!(
                "unknown output format `{f}` in output.formats (expected one of: {})",
                OUTPUT_FORMATS.join(", ")
            ))),
            None => Ok(()),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            formats: vec!["json".into(), "csv".into(), "spef".into()],
            cross_moments: false,
        }
    }
}

/// Reads a config file.
pub fn read_config(path: &Path) -> Result<Config> {
    Config::parse(&read_to_string(path)?, &path.display().to_string())
}

fn any_match(globs: &[String], name: &str) -> bool {
    globs.iter().any(|g| glob_match(g, name))
}

/// A configuration resolved against a concrete geometry.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// Solver configuration.
    pub solver: SolverConfig,
    /// Target nets.
    pub targets: Vec<NetId>,
    /// Effective ground report mode (after `ground.ignore`).
    pub ground_report: GroundReport,
    /// Human-readable warnings.
    pub warnings: Vec<String>,
}

impl Config {
    /// Parses TOML text.
    pub fn parse(text: &str, path: &str) -> Result<Self> {
        let c: Self = toml::from_str(text).map_err(|source| IoError::Toml {
            path: path.to_owned(),
            source,
        })?;
        if let Some(v) = c.schema_version {
            if v != CONFIG_SCHEMA_VERSION {
                return Err(IoError::Invalid(format!(
                    "unsupported config schema_version {v}"
                )));
            }
        }
        c.output.validate()?;
        Ok(c)
    }

    /// Whether `name` is reported as ground: ground nets with
    /// `nets_treatment = "ground"`, and power nets with `treat_as = "ground"`.
    #[must_use]
    pub fn is_ground_net(&self, name: &str) -> bool {
        (self.ground.nets_treatment == NetTreatment::Ground && any_match(&self.ground.nets, name))
            || self.is_power_ground(name)
    }

    fn is_power_ground(&self, name: &str) -> bool {
        self.power.treat_as == PowerTreatment::Ground && any_match(&self.power.nets, name)
    }

    /// Whether a net is removed from the geometry before solving.
    #[must_use]
    pub fn is_removed(&self, name: &str) -> bool {
        let ground_glob = any_match(&self.ground.nets, name);
        let physics = self.ground.ignore == GroundIgnore::Physics
            && (ground_glob || self.is_power_ground(name));
        any_match(&self.scope.ignore_nets, name)
            || physics
            || (self.ground.nets_treatment == NetTreatment::Remove && ground_glob)
    }

    /// The solver accuracy settings: the preset's tolerances, overridden by
    /// explicit `[accuracy]` keys.
    #[must_use]
    pub fn accuracy(&self) -> Accuracy {
        let (tot, cpl) = match self.accuracy.preset {
            Preset::Fast => (0.02, 0.10),
            Preset::Default => (0.01, 0.05),
            Preset::Signoff => (0.005, 0.03),
            Preset::Reference => (0.001, 0.01),
        };
        let a = &self.accuracy;
        let disabled = |v: f64| if v > 0.0 { v } else { f64::INFINITY };
        let defaults = Accuracy::default();
        Accuracy {
            total_rel_tol: a.total_rel_tol.unwrap_or(tot),
            coupling_rel_tol: a.coupling_rel_tol.unwrap_or(cpl),
            track_rel: disabled(a.coupling_track.rel),
            track_abs_ff: disabled(a.coupling_track.abs_ff),
            confidence_sigma: a.confidence_sigma.unwrap_or(defaults.confidence_sigma),
            min_walks: a.min_walks.unwrap_or(defaults.min_walks),
            max_walks: a.max_walks.unwrap_or(defaults.max_walks),
            batch_walks: a.batch_walks.unwrap_or(defaults.batch_walks),
        }
    }

    /// Resolves against `geom` (which must already exclude removed nets) and
    /// the dielectric (`eps_r` above the stratified `layers`, bottom-up, plus
    /// boxes and conformal rules in `regions`).
    pub fn resolve(
        &self,
        geom: &Geometry,
        eps_r: f64,
        layers: &[DielectricLayer],
        regions: &DielectricRegions,
    ) -> Result<Resolved> {
        // Also here for configurations built in code rather than parsed.
        self.output.validate()?;
        let mut warnings = Vec::new();
        for r in regions.conformal.iter().filter(|r| r.boxes.is_empty()) {
            warnings.push(format!(
                "conformal rule `{}` selects no conductor box and has no effect",
                r.layer
            ));
        }
        if !self.scope.ignore_nets.is_empty() {
            warnings
                .push("scope.ignore_nets removes conductors and changes the physics".to_owned());
        }
        let physics = self.ground.ignore == GroundIgnore::Physics;
        let substrate_z = (self.ground.substrate == SubstrateKind::Plane && !physics)
            .then_some(self.ground.substrate_z);

        let ground_nets: Vec<NetId> = geom
            .nets()
            .filter(|&n| self.is_ground_net(geom.net_name(n)))
            .collect();
        let targets: Vec<NetId> = geom
            .nets()
            .filter(|&n| {
                let name = geom.net_name(n);
                any_match(&self.extract.nets, name)
                    && !any_match(&self.extract.exclude_nets, name)
                    && !ground_nets.contains(&n)
            })
            .collect();
        if targets.is_empty() {
            return Err(IoError::Invalid("no target nets match extract.nets".into()));
        }

        let b = geom.bounds();
        let boundary = match self.domain.kind {
            DomainKind::Open => {
                if physics {
                    warnings.push("open boundary: infinity remains at 0 V even with ground.ignore = \"physics\"".into());
                }
                Boundary::Open
            }
            DomainKind::Box => {
                let e = b.extent();
                let margin = self
                    .domain
                    .margin
                    .unwrap_or_else(|| (0.5 * e[0].max(e[1])).max(4.0 * b.max()[2]).max(1.0));
                let zmin = substrate_z.unwrap_or(b.min()[2] - margin);
                let bounds = Aabb::new(
                    [b.min()[0] - margin, b.min()[1] - margin, zmin],
                    [
                        b.max()[0] + margin,
                        b.max()[1] + margin,
                        b.max()[2] + margin,
                    ],
                )?;
                let bc = |f: FaceBc| match (f, physics) {
                    (_, true) | (FaceBc::Reflect, _) => FaceKind::Reflect,
                    (FaceBc::Ground, false) => FaceKind::Ground,
                };
                let fb = &self.ground.boundary;
                let mut faces = [FaceKind::Ground; 6];
                faces[Face::new(Axis::X, Side::Min).index()] = bc(fb.xmin);
                faces[Face::new(Axis::X, Side::Max).index()] = bc(fb.xmax);
                faces[Face::new(Axis::Y, Side::Min).index()] = bc(fb.ymin);
                faces[Face::new(Axis::Y, Side::Max).index()] = bc(fb.ymax);
                faces[Face::new(Axis::Z, Side::Max).index()] = bc(fb.zmax);
                // The bottom face coincides with the substrate if present; otherwise it follows physics.
                faces[Face::new(Axis::Z, Side::Min).index()] = if physics {
                    FaceKind::Reflect
                } else {
                    FaceKind::Ground
                };
                if physics && faces.iter().all(|f| *f == FaceKind::Reflect) {
                    warnings.push(
                        "ground.ignore = \"physics\": no ground reference; results depend on the domain size".into(),
                    );
                }
                Boundary::Box { bounds, faces }
            }
        };

        let accuracy = self.accuracy();
        let w = &self.walk;
        let wd = WalkOptions::default();
        let walk = WalkOptions {
            stop_dist: w.stop_dist,
            stop_ratio: w.stop_ratio.unwrap_or(wd.stop_ratio),
            gauss_alpha: w.gauss_alpha.unwrap_or(wd.gauss_alpha),
            max_steps: w.max_steps.unwrap_or(wd.max_steps),
            green_resolution: w.green_resolution.unwrap_or(wd.green_resolution),
            antithetic: w.antithetic.unwrap_or(wd.antithetic),
            stack: StackOptions {
                interface_offsets: w.interface_offsets.unwrap_or(wd.stack.interface_offsets),
                layered_resolution: w.layered_resolution.unwrap_or(wd.stack.layered_resolution),
                band_fraction: w.band_fraction.unwrap_or(wd.stack.band_fraction),
                landing_alpha: w.landing_alpha.unwrap_or(wd.stack.landing_alpha),
                edge_cubes: w.edge_cubes.unwrap_or(wd.stack.edge_cubes),
                window_hops: w.window_hops.unwrap_or(wd.stack.window_hops),
            },
        };
        let ground_report = if self.ground.ignore == GroundIgnore::Report {
            GroundReport::Drop
        } else {
            self.ground.report
        };
        Ok(Resolved {
            solver: SolverConfig {
                eps_r,
                layers: layers.to_vec(),
                boundary,
                substrate_z,
                accuracy,
                walk,
                seed: self.run.seed,
                ground_nets,
                regions: regions.clone(),
            },
            targets,
            ground_report,
            warnings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(Config::parse("[ground]\nsubstrat = \"plane\"\n", "t").is_err());
        assert!(Config::parse("[ground]\nignore = \"physics\"\n", "t").is_ok());
    }

    #[test]
    fn unknown_output_formats_are_rejected_when_parsing() {
        let e = Config::parse("[output]\nformats = [\"json\", \"xml\"]\n", "t").unwrap_err();
        assert!(e.to_string().contains("`xml`"), "{e}");
        assert!(Config::parse("[output]\nformats = [\"spef\", \"csv\"]\n", "t").is_ok());
        let mut c = Config::default();
        c.output.formats.push("xml".into());
        let mut b = faradice_geom::GeometryBuilder::new();
        let n = b.net("a").unwrap();
        b.add_box(n, Aabb::from_array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0]).unwrap())
            .unwrap();
        let g = b.build().unwrap();
        assert!(
            c.resolve(&g, 1.0, &[], &DielectricRegions::default())
                .is_err()
        );
    }

    #[test]
    fn ground_and_power_treatments_are_independent() {
        let c = Config::parse("[ground]\nnets_treatment = \"net\"\n", "t").unwrap();
        assert!(!c.is_ground_net("VSS"));
        assert!(c.is_ground_net("VDD")); // power.treat_as = ground still applies
        let p = Config::parse("[ground]\nignore = \"physics\"\n", "t").unwrap();
        assert!(p.is_removed("VSS") && p.is_removed("VDD") && !p.is_removed("clk"));
    }

    #[test]
    fn empty_config_is_default() {
        let c = Config::parse("", "t").unwrap();
        assert_eq!(c.ground.ignore, GroundIgnore::None);
        assert!(c.is_ground_net("VSS"));
        assert!(c.is_ground_net("VDD"));
        assert!(!c.is_ground_net("clk"));
    }

    #[test]
    fn empty_conformal_rules_are_reported() {
        let mut b = faradice_geom::GeometryBuilder::new();
        let n = b.net("a").unwrap();
        b.add_box(n, Aabb::from_array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0]).unwrap())
            .unwrap();
        let g = b.build().unwrap();
        let regions = DielectricRegions {
            boxes: Vec::new(),
            conformal: vec![faradice_frw::ConformalRule {
                layer: "metl".into(),
                boxes: Vec::new(),
                side: 0.03,
                top: 0.0,
                eps_r: 3.5,
            }],
        };
        let r = Config::default().resolve(&g, 4.0, &[], &regions).unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("`metl` selects no conductor box")),
            "{:?}",
            r.warnings
        );
    }
}
