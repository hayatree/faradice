//! Parameter files (`*.sens.toml`, design D3 §3) and their resolution into
//! face steps of the solver.

use std::collections::BTreeMap;
use std::path::Path;

use faradice_core::{Axis, BoxId, Face, NetId, Side};
use faradice_frw::BoundaryModel;
use faradice_frw::replay::{AddedNet, Addition};
use faradice_frw::sens::{Deletion, FaceStep, MIN_STEP_OVER_EPS_STOP, Param, exposed_area};
use faradice_geom::{Aabb, Geometry};
use serde::{Deserialize, Serialize};

use crate::{Result, SensError};

/// Default step as a fraction of the box extent and of the gap in front of
/// the face (D3 §4.3).
pub const DEFAULT_DELTA_REL: f64 = 0.05;

/// A parameter file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamFile {
    /// Format version (1).
    pub schema_version: u32,
    /// Default step relative to the box extent and the gap in front of each
    /// face (default [`DEFAULT_DELTA_REL`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_rel: Option<f64>,
    /// Richardson extrapolation over `δ` and `2δ` (default `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub richardson: Option<bool>,
    /// The parameters.
    #[serde(rename = "param", default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ParamSpec>,
    /// Conductor deletions (exact, from the same walks).
    #[serde(rename = "delete", default, skip_serializing_if = "Vec::is_empty")]
    pub deletes: Vec<DeleteSpec>,
    /// Conductor additions (exact, by replaying a recorded extraction).
    #[serde(rename = "add", default, skip_serializing_if = "Vec::is_empty")]
    pub adds: Vec<AddSpec>,
    /// Face derivative method: `fd` (default), `shape` or `auto` (see
    /// [`Method`]). The command line can override it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Face samples per face of the shape derivative (default
    /// [`DEFAULT_SHAPE_SAMPLES`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_samples: Option<u64>,
}

/// Default face samples per face of the shape derivative.
pub const DEFAULT_SHAPE_SAMPLES: u64 = 200_000;

/// How face derivatives are estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// Finite differences with path continuation (pillar A0): every face.
    #[default]
    Fd,
    /// Shape derivative (pillar A): every face must be supported (no
    /// geometric edge, uncovered, homogeneous dielectric), else an error.
    Shape,
    /// Shape derivative where supported, finite differences elsewhere.
    Auto,
}

impl std::str::FromStr for Method {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fd" => Ok(Self::Fd),
            "shape" => Ok(Self::Shape),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("unknown method `{s}` (fd, shape, auto)")),
        }
    }
}

/// Conductors added together (D3 §8, D-add): `C` of the geometry with them,
/// and the exact paired `ΔC`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddSpec {
    /// Unique name.
    pub name: String,
    /// Net of the boxes: an existing non-target net (extended), or a new name
    /// (a new conductor at 0 V, e.g. a shield).
    pub net: String,
    /// For a new net: count it as ground in the `<GND>` sums (default `true`).
    /// Not allowed for an existing net (its `ground.nets` membership applies).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground: Option<bool>,
    /// Boxes `[x0, y0, z0, x1, y1, z1]` (µm), like the structure file.
    pub boxes: Vec<[f64; 6]>,
}

/// Nets deleted together: `C` of the geometry without them, and `ΔC`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeleteSpec {
    /// Unique name.
    pub name: String,
    /// Nets to delete (not targets).
    pub nets: Vec<String>,
}

/// One parameter: explicit faces, a translation of a whole net, or both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSpec {
    /// Unique name.
    pub name: String,
    /// Weighted faces: `dp` moves each face by `weight · dp` along its
    /// outward normal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces: Vec<FaceSpec>,
    /// Shorthand for a rigid translation of a net along an axis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translate: Option<Translate>,
    /// 1σ process variation of the parameter (µm), for `σ_C`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
}

/// A face handle with a weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaceSpec {
    /// Net name.
    pub net: String,
    /// Index of the box within the net, in structure-file order. Give this
    /// or `at`.
    #[serde(rename = "box", default, skip_serializing_if = "Option::is_none")]
    pub box_index: Option<usize>,
    /// A point on the face (µm): selects the net's box whose `face` contains
    /// it (for GDS inputs, where box indices are opaque). Give this or `box`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<[f64; 3]>,
    /// `x-`, `x+`, …, `z+` (or `xmin`, …, `zmax`).
    pub face: String,
    /// Weight (default 1).
    #[serde(default = "one")]
    pub weight: f64,
    /// Explicit step `δ` in µm (default: from `delta_rel`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,
}

const fn one() -> f64 {
    1.0
}

/// A rigid translation of every exposed face of a net perpendicular to `axis`:
/// the `-` faces with weight −1 and the `+` faces with weight +1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Translate {
    /// Net name.
    pub net: String,
    /// `x`, `y` or `z`.
    pub axis: String,
}

impl ParamFile {
    /// Parses a parameter file; `origin` names it in errors.
    ///
    /// # Errors
    ///
    /// TOML syntax, unknown keys, an unsupported schema version, or invalid
    /// values (duplicate names, empty parameters, non-finite numbers).
    pub fn parse(text: &str, origin: &str) -> Result<Self> {
        let f: Self =
            toml::from_str(text).map_err(|e| SensError::Param(format!("{origin}: {e}")))?;
        f.validate()
            .map_err(|e| SensError::Param(format!("{origin}: {e}")))?;
        Ok(f)
    }

    /// Reads a parameter file.
    ///
    /// # Errors
    ///
    /// As [`ParamFile::parse`], and file system errors.
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SensError::Param(format!("reading {}: {e}", path.display())))?;
        Self::parse(&text, &path.display().to_string())
    }

    fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported schema_version {} (expected 1)",
                self.schema_version
            ));
        }
        if let Some(r) = self
            .delta_rel
            .filter(|r| !(r.is_finite() && *r > 0.0 && *r < 0.25))
        {
            return Err(format!("delta_rel {r} must be in (0, 0.25)"));
        }
        if self.params.is_empty() && self.deletes.is_empty() && self.adds.is_empty() {
            return Err("no [[param]], [[delete]] or [[add]] given".into());
        }
        if let Some(m) = &self.method {
            m.parse::<Method>()?;
        }
        if self.shape_samples.is_some_and(|n| n < 2) {
            return Err("shape_samples must be at least 2".into());
        }
        let mut seen_add = std::collections::BTreeSet::new();
        for a in &self.adds {
            if !seen_add.insert(a.name.as_str()) {
                return Err(format!("duplicate addition name `{}`", a.name));
            }
            if a.net.is_empty() {
                return Err(format!("addition `{}`: empty net name", a.name));
            }
            if a.boxes.is_empty() {
                return Err(format!("addition `{}` has no box", a.name));
            }
            for b in &a.boxes {
                if !b.iter().all(|v| v.is_finite()) || !(0..3).all(|i| b[i] < b[i + 3]) {
                    return Err(format!(
                        "addition `{}`: box {b:?} must be finite with x0 < x1, y0 < y1, z0 < z1",
                        a.name
                    ));
                }
            }
        }
        let mut seen_del = std::collections::BTreeSet::new();
        for d in &self.deletes {
            if !seen_del.insert(d.name.as_str()) {
                return Err(format!("duplicate deletion name `{}`", d.name));
            }
            if d.nets.is_empty() {
                return Err(format!("deletion `{}` names no net", d.name));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for p in &self.params {
            if !seen.insert(p.name.as_str()) {
                return Err(format!("duplicate parameter name `{}`", p.name));
            }
            if p.faces.is_empty() && p.translate.is_none() {
                return Err(format!(
                    "parameter `{}` has neither faces nor translate",
                    p.name
                ));
            }
            if p.sigma.is_some_and(|s| !(s.is_finite() && s >= 0.0)) {
                return Err(format!(
                    "parameter `{}`: sigma must be finite and >= 0",
                    p.name
                ));
            }
            for f in &p.faces {
                parse_face(&f.face).map_err(|e| format!("parameter `{}`: {e}", p.name))?;
                if f.box_index.is_some() == f.at.is_some() {
                    return Err(format!(
                        "parameter `{}`: give exactly one of `box` and `at` per face",
                        p.name
                    ));
                }
                if f.at.is_some_and(|a| !a.iter().all(|v| v.is_finite())) {
                    return Err(format!("parameter `{}`: non-finite `at`", p.name));
                }
                if !f.weight.is_finite() {
                    return Err(format!("parameter `{}`: non-finite weight", p.name));
                }
                if f.delta.is_some_and(|d| !(d.is_finite() && d > 0.0)) {
                    return Err(format!("parameter `{}`: delta must be positive", p.name));
                }
            }
            if let Some(t) = &p.translate {
                parse_axis(&t.axis).map_err(|e| format!("parameter `{}`: {e}", p.name))?;
            }
        }
        Ok(())
    }
}

/// Parses `x-`/`x+`/… or `xmin`/`xmax`/….
fn parse_face(s: &str) -> std::result::Result<Face, String> {
    let t = s.trim().to_ascii_lowercase();
    if let Some(a) = t.strip_suffix('-') {
        return Ok(Face::new(parse_axis(a)?, Side::Min));
    }
    if let Some(a) = t.strip_suffix('+') {
        return Ok(Face::new(parse_axis(a)?, Side::Max));
    }
    t.parse::<Face>()
        .map_err(|_| format!("unknown face `{s}` (x-, x+, y-, y+, z-, z+)"))
}

fn parse_axis(s: &str) -> std::result::Result<Axis, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "x" => Ok(Axis::X),
        "y" => Ok(Axis::Y),
        "z" => Ok(Axis::Z),
        _ => Err(format!("unknown axis `{s}` (x, y, z)")),
    }
}

/// Short face name (`x-`, …, `z+`).
#[must_use]
pub fn face_name(f: Face) -> &'static str {
    match (f.axis, f.side) {
        (Axis::X, Side::Min) => "x-",
        (Axis::X, Side::Max) => "x+",
        (Axis::Y, Side::Min) => "y-",
        (Axis::Y, Side::Max) => "y+",
        (Axis::Z, Side::Min) => "z-",
        (Axis::Z, Side::Max) => "z+",
    }
}

/// One face of the resolved request.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFace {
    /// Net name.
    pub net: String,
    /// Box index within the net.
    pub box_index: usize,
    /// The solver step.
    pub step: FaceStep,
    /// Exposed area of the face (µm²).
    pub exposed_area: f64,
    /// Gap in front of the face (µm), `None` if unbounded.
    pub gap: Option<f64>,
}

/// An addition resolved against a geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAdd {
    /// Name of the added net (existing or new).
    pub net: String,
    /// `true` for a new net.
    pub new_net: bool,
    /// The solver request.
    pub addition: Addition,
}

/// A parameter file resolved against a geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// Distinct faces, in order of first use.
    pub faces: Vec<ResolvedFace>,
    /// One per parameter of the file, in file order.
    pub params: Vec<Param>,
    /// One per deletion of the file, in file order.
    pub deletions: Vec<Deletion>,
    /// One per addition of the file, in file order.
    pub additions: Vec<ResolvedAdd>,
    /// Step relative to extent and gap used for defaults.
    pub delta_rel: f64,
    /// Richardson extrapolation.
    pub richardson: bool,
    /// Non-fatal notes (e.g. faces skipped by `translate`).
    pub warnings: Vec<String>,
}

/// Resolves `file` against `geom`: face handles to box ids, `translate` to
/// exposed faces, and default steps from the box extent and the gap in front
/// of each face.
///
/// # Errors
///
/// Unknown nets or boxes, faces without exposed area, steps below
/// [`MIN_STEP_OVER_EPS_STOP`] × `eps_stop`, or inconsistent explicit steps for
/// the same face.
pub fn resolve(
    file: &ParamFile,
    geom: &Geometry,
    domain: &BoundaryModel,
    eps_stop: f64,
) -> Result<Resolved> {
    let r = resolve_unchecked(file, geom, domain)?;
    check_steps(&r.faces, eps_stop)?;
    Ok(r)
}

/// [`resolve`] without the step check, for faces that may not take finite
/// differences (see [`check_steps`]).
///
/// # Errors
///
/// As [`resolve`], except for small steps.
pub fn resolve_unchecked(
    file: &ParamFile,
    geom: &Geometry,
    domain: &BoundaryModel,
) -> Result<Resolved> {
    let delta_rel = file.delta_rel.unwrap_or(DEFAULT_DELTA_REL);
    let richardson = file.richardson.unwrap_or(true);
    let mut faces: Vec<ResolvedFace> = Vec::new();
    let mut index: BTreeMap<(BoxId, usize), usize> = BTreeMap::new();
    let mut explicit: BTreeMap<usize, f64> = BTreeMap::new();
    let mut params = Vec::with_capacity(file.params.len());
    let mut warnings = Vec::new();
    let perr =
        |p: &ParamSpec, msg: String| SensError::Param(format!("parameter `{}`: {msg}", p.name));
    for p in &file.params {
        // (net, box index, face, weight, explicit δ)
        let mut handles: Vec<(NetId, usize, Face, f64, Option<f64>)> = Vec::new();
        for f in &p.faces {
            let net = net_id(geom, &f.net).map_err(|e| perr(p, e))?;
            let face = parse_face(&f.face).map_err(|e| perr(p, e))?;
            let bi = match (f.box_index, f.at) {
                (Some(bi), _) => bi,
                (None, Some(at)) => box_at(geom, net, face, at).map_err(|e| perr(p, e))?,
                (None, None) => unreachable!("validated"),
            };
            handles.push((net, bi, face, f.weight, f.delta));
        }
        if let Some(t) = &p.translate {
            let net = net_id(geom, &t.net).map_err(|e| perr(p, e))?;
            let axis = parse_axis(&t.axis).map_err(|e| perr(p, e))?;
            let mut n = 0;
            for (bi, &b) in geom.net_boxes(net).iter().enumerate() {
                for (side, w) in [(Side::Min, -1.0), (Side::Max, 1.0)] {
                    let face = Face::new(axis, side);
                    let area =
                        exposed_area(geom, domain, b, face).map_err(|e| perr(p, e.to_string()))?;
                    if area > 0.0 {
                        handles.push((net, bi, face, w, None));
                        n += 1;
                    }
                }
            }
            if n == 0 {
                warnings.push(format!(
                    "parameter `{}`: net `{}` has no exposed face across {}; its derivative is zero",
                    p.name, t.net, t.axis
                ));
            }
        }
        let mut param = Param { faces: Vec::new() };
        for (net, bi, face, w, delta) in handles {
            let boxes = geom.net_boxes(net);
            let &b = boxes.get(bi).ok_or_else(|| {
                perr(
                    p,
                    format!(
                        "net `{}` has {} boxes, box {bi} does not exist",
                        geom.net_name(net),
                        boxes.len()
                    ),
                )
            })?;
            let key = (b, face.index());
            let fi = if let Some(&i) = index.get(&key) {
                i
            } else {
                let area =
                    exposed_area(geom, domain, b, face).map_err(|e| perr(p, e.to_string()))?;
                if area <= 0.0 {
                    return Err(perr(
                        p,
                        format!(
                            "net `{}` box {bi} face {} has no exposed area (covered by its own \
                             net or on a reflecting wall)",
                            geom.net_name(net),
                            face_name(face)
                        ),
                    ));
                }
                let gap = front_gap(geom, domain, b, face);
                let extent = geom.box_(b).extent()[face.axis.index()];
                let delta = delta_rel * gap.map_or(extent, |g| g.min(extent));
                faces.push(ResolvedFace {
                    net: geom.net_name(net).to_owned(),
                    box_index: bi,
                    step: FaceStep {
                        box_id: b,
                        face,
                        delta,
                        keep: None,
                    },
                    exposed_area: area,
                    gap,
                });
                index.insert(key, faces.len() - 1);
                faces.len() - 1
            };
            if let Some(d) = delta {
                if explicit
                    .get(&fi)
                    .is_some_and(|&e| (e - d).abs() > 1e-12 * d)
                {
                    return Err(perr(
                        p,
                        format!(
                            "net `{}` box {bi} face {} is given two different steps",
                            geom.net_name(net),
                            face_name(face)
                        ),
                    ));
                }
                explicit.insert(fi, d);
                faces[fi].step.delta = d;
            }
            if w != 0.0 {
                match param.faces.iter_mut().find(|(i, _)| *i == fi) {
                    Some((_, acc)) => *acc += w,
                    None => param.faces.push((fi, w)),
                }
            }
        }
        params.push(param);
    }
    let deletions = file
        .deletes
        .iter()
        .map(|d| {
            d.nets
                .iter()
                .map(|n| net_id(geom, n))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map(|nets| Deletion { nets })
                .map_err(|e| SensError::Param(format!("deletion `{}`: {e}", d.name)))
        })
        .collect::<Result<Vec<_>>>()?;
    let additions = file
        .adds
        .iter()
        .map(|a| {
            let err = |m: String| SensError::Param(format!("addition `{}`: {m}", a.name));
            let boxes = a
                .boxes
                .iter()
                .map(|b| {
                    Aabb::new([b[0], b[1], b[2]], [b[3], b[4], b[5]])
                        .map_err(|e| err(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            let (net, ground) = match geom.net_id(&a.net) {
                Some(id) => {
                    if a.ground.is_some() {
                        return Err(err(format!(
                            "`ground` applies to new nets only; `{}` exists (use ground.nets)",
                            a.net
                        )));
                    }
                    (AddedNet::Existing(id), false)
                }
                None => {
                    if faradice_core::is_reserved_label(&a.net) {
                        return Err(err(format!(
                            "a new net cannot be named `{}`: names of the form <…> are \
                             reserved for ground sinks",
                            a.net
                        )));
                    }
                    (AddedNet::New, a.ground.unwrap_or(true))
                }
            };
            Ok(ResolvedAdd {
                net: a.net.clone(),
                new_net: matches!(net, AddedNet::New),
                addition: Addition { boxes, net, ground },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Resolved {
        faces,
        params,
        deletions,
        additions,
        delta_rel,
        richardson,
        warnings,
    })
}

/// Checks the finite-difference steps of `faces` against
/// [`MIN_STEP_OVER_EPS_STOP`] × `eps_stop`.
///
/// # Errors
///
/// The first face whose step is too small.
pub fn check_steps<'a>(
    faces: impl IntoIterator<Item = &'a ResolvedFace>,
    eps_stop: f64,
) -> Result<()> {
    let min = MIN_STEP_OVER_EPS_STOP * eps_stop;
    for f in faces {
        if f.step.delta < min {
            return Err(SensError::Param(format!(
                "net `{}` box {} face {}: step {:.3e} µm is below {MIN_STEP_OVER_EPS_STOP} × \
                 ε_stop = {min:.3e} µm. Give an explicit `delta`, raise `delta_rel`, or lower \
                 walk.stop_dist / walk.stop_ratio",
                f.net,
                f.box_index,
                face_name(f.step.face),
                f.step.delta
            )));
        }
    }
    Ok(())
}

/// The index (within `net`) of the box whose `face` contains `at`: on the
/// face plane (within 1e-9 of the box size) and inside the face rectangle.
/// A point on the shared edge of abutting boxes is ambiguous and rejected.
fn box_at(
    geom: &Geometry,
    net: NetId,
    face: Face,
    at: [f64; 3],
) -> std::result::Result<usize, String> {
    let a = face.axis.index();
    let hits: Vec<usize> = geom
        .net_boxes(net)
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            let bx = geom.box_(**b);
            let size = bx.extent().iter().copied().fold(0.0, f64::max);
            let tol = 1e-9 * size + 8.0 * f64::EPSILON * at[a].abs();
            (bx.face_coord(face) - at[a]).abs() <= tol
                && (0..3)
                    .filter(|&i| i != a)
                    .all(|i| bx.min()[i] - tol <= at[i] && at[i] <= bx.max()[i] + tol)
        })
        .map(|(i, _)| i)
        .collect();
    match hits.as_slice() {
        [] => Err(format!(
            "no box of net `{}` has a {} face through {at:?}",
            geom.net_name(net),
            face_name(face)
        )),
        [i] => Ok(*i),
        many => Err(format!(
            "{} boxes of net `{}` have a {} face through {at:?} (boxes {many:?}); pick a point \
             inside one face, or use `box`",
            many.len(),
            geom.net_name(net),
            face_name(face)
        )),
    }
}

fn net_id(geom: &Geometry, name: &str) -> std::result::Result<NetId, String> {
    geom.net_id(name)
        .ok_or_else(|| format!("unknown net `{name}` (not in the structure, or removed)"))
}

/// Distance from `face` of box `id` to the nearest thing in front of it: a
/// box (of any net, not touching the face) whose projection overlaps the face,
/// a grounded domain face, the substrate plane, or the mirror image of the
/// box in a reflecting domain face. `None` if nothing bounds it (open space).
#[must_use]
pub fn front_gap(geom: &Geometry, domain: &BoundaryModel, id: BoxId, face: Face) -> Option<f64> {
    let a = face.axis.index();
    let s = face.side.sign();
    let b = geom.box_(id);
    let coord = b.face_coord(face);
    let t: Vec<usize> = (0..3).filter(|&i| i != a).collect();
    let overlaps = |c: &Aabb| {
        t.iter()
            .all(|&i| c.min()[i].max(b.min()[i]) < c.max()[i].min(b.max()[i]))
    };
    let mut gap = f64::INFINITY;
    for (i, c) in geom.boxes().iter().enumerate() {
        if i == id.index() || !overlaps(c) {
            continue;
        }
        let d = if s > 0.0 {
            c.min()[a] - coord
        } else {
            coord - c.max()[a]
        };
        if d > 0.0 {
            gap = gap.min(d);
        }
    }
    for &(axis, g, _) in domain.ground_faces() {
        let d = s * (g - coord);
        if axis == a && d > 0.0 {
            gap = gap.min(d);
        }
    }
    if let Some(z) = domain
        .substrate_z()
        .filter(|&z| a == 2 && s < 0.0 && coord > z)
    {
        gap = gap.min(coord - z);
    }
    if let Some(bounds) = domain.bounds() {
        let (lo, hi) = domain.reflect()[a];
        let d = if s > 0.0 {
            hi.then(|| bounds.max()[a] - coord)
        } else {
            lo.then(|| coord - bounds.min()[a])
        };
        if let Some(d) = d.filter(|d| *d > 0.0) {
            gap = gap.min(2.0 * d);
        }
    }
    gap.is_finite().then_some(gap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faradice_frw::sens::sens_setup;
    use faradice_frw::{Accuracy, Boundary, SolverConfig, WalkOptions};
    use faradice_geom::GeometryBuilder;

    /// Three wires (as `examples/three_wires`) in open space above a substrate.
    fn wires() -> (Geometry, BoundaryModel, f64) {
        let mut g = GeometryBuilder::new();
        for (net, c) in [
            ("left", [0.0, 0.00, 1.0, 10.0, 0.14, 1.36]),
            ("victim", [0.0, 0.28, 1.0, 10.0, 0.42, 1.36]),
            ("right", [0.0, 0.56, 1.0, 10.0, 0.70, 1.36]),
        ] {
            let id = g.net(net).unwrap();
            g.add_box(id, Aabb::from_array(c).unwrap()).unwrap();
        }
        let g = g.build().unwrap();
        let cfg = SolverConfig {
            eps_r: 3.9,
            layers: Vec::new(),
            boundary: Boundary::Open,
            substrate_z: Some(0.0),
            accuracy: Accuracy::default(),
            walk: WalkOptions::default(),
            seed: 1,
            ground_nets: Vec::new(),
            regions: Default::default(),
        };
        let su = sens_setup(&g, &[g.net_id("victim").unwrap()], &cfg).unwrap();
        (g, su.domain, su.eps_stop)
    }

    const FILE: &str = r#"
schema_version = 1

[[param]]
name = "w_victim"
faces = [
  { net = "victim", box = 0, face = "y-", weight = 0.5 },
  { net = "victim", box = 0, face = "ymax", weight = 0.5 },
]
sigma = 0.005

[[param]]
name = "y_right"
translate = { net = "right", axis = "y" }
"#;

    #[test]
    fn parses_and_resolves() {
        let f = ParamFile::parse(FILE, "t").unwrap();
        let (g, domain, eps_stop) = wires();
        let r = resolve(&f, &g, &domain, eps_stop).unwrap();
        assert!(r.richardson);
        // victim y-, victim y+, right y-, right y+.
        assert_eq!(r.faces.len(), 4);
        assert_eq!(r.params[0].faces, vec![(0, 0.5), (1, 0.5)]);
        assert_eq!(r.params[1].faces, vec![(2, -1.0), (3, 1.0)]);
        // Side faces of the victim: 0.14 µm gap and 0.14 µm extent.
        let v = &r.faces[0];
        assert!((v.gap.unwrap() - 0.14).abs() < 1e-12);
        assert!((v.step.delta - 0.05 * 0.14).abs() < 1e-12);
        assert!((v.exposed_area - 10.0 * 0.36).abs() < 1e-12);
        // The right wire's outer face sees nothing in front of it.
        assert_eq!(r.faces[3].gap, None);
        assert!((r.faces[3].step.delta - 0.05 * 0.14).abs() < 1e-12);
    }

    #[test]
    fn faces_by_point() {
        let (g, domain, eps_stop) = wires();
        let text = "schema_version = 1\n[[param]]\nname = 'w'\n\
                    faces = [{ net = 'victim', at = [5.0, 0.42, 1.2], face = 'y+' }]\n";
        let r = resolve(&ParamFile::parse(text, "t").unwrap(), &g, &domain, eps_stop).unwrap();
        assert_eq!(r.faces[0].box_index, 0);
        assert_eq!(r.faces[0].step.face, Face::new(Axis::Y, Side::Max));
        let off = text.replace("0.42", "0.40");
        let e = resolve(&ParamFile::parse(&off, "t").unwrap(), &g, &domain, eps_stop).unwrap_err();
        assert!(e.to_string().contains("no box"), "{e}");
        let both = text.replace("at = ", "box = 0, at = ");
        assert!(
            ParamFile::parse(&both, "t")
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
    }

    #[test]
    fn front_gaps_of_boundaries() {
        let (g, domain, _) = wires();
        let v = g.net_boxes(g.net_id("victim").unwrap())[0];
        let bottom = front_gap(&g, &domain, v, Face::new(Axis::Z, Side::Min));
        assert!((bottom.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(
            front_gap(&g, &domain, v, Face::new(Axis::Z, Side::Max)),
            None
        );
    }

    #[test]
    fn rejects_bad_files() {
        let bad = |text: &str| ParamFile::parse(text, "t").unwrap_err().to_string();
        assert!(
            bad("schema_version = 2\n[[param]]\nname='a'\ntranslate={net='a',axis='x'}")
                .contains("schema_version")
        );
        assert!(bad("schema_version = 1").contains("no [[param]], [[delete]] or [[add]]"));
        let add = |extra: &str| format!("schema_version = 1\n{extra}");
        assert!(bad(&add("[[add]]\nname='s'\nnet='s'\nboxes=[]")).contains("has no box"));
        assert!(bad(&add("[[add]]\nname='s'\nnet='s'\nboxes=[[0,0,0,1,0,1]]")).contains("x0 < x1"));
        assert!(
            bad(&add("[[add]]\nname='s'\nnet='s'\nboxes=[[0,0,0,1,1,1]]\n\
                 [[add]]\nname='s'\nnet='t'\nboxes=[[0,0,0,1,1,1]]"))
            .contains("duplicate addition")
        );
        assert!(
            bad(&add("method='exact'\n[[delete]]\nname='d'\nnets=['a']"))
                .contains("unknown method")
        );
        assert!(
            bad(&add("shape_samples=1\n[[delete]]\nname='d'\nnets=['a']")).contains("at least 2")
        );
        let ok = ParamFile::parse(
            &add("method='Auto'\n[[add]]\nname='s'\nnet='s'\nboxes=[[0,0,0,1,1,1]]"),
            "t",
        )
        .unwrap();
        assert_eq!(
            ok.method.as_deref().map(str::parse::<Method>),
            Some(Ok(Method::Auto))
        );
        assert!(bad("schema_version = 1\n[[param]]\nname='a'").contains("neither"));
        assert!(
            bad("schema_version = 1\n[[param]]\nname='a'\nfaces=[{net='a',box=0,face='top'}]")
                .contains("unknown face")
        );
        assert!(
            bad("schema_version = 1\n[[param]]\nname='a'\ntranslate={net='a',axis='w'}")
                .contains("unknown axis")
        );
        assert!(bad("schema_version = 1\n[[param]]\nname='a'\ntranslate={net='a',axis='x'}\n[[param]]\nname='a'\ntranslate={net='a',axis='y'}").contains("duplicate"));
        assert!(
            bad("schema_version = 1\nfoo = 1\n[[param]]\nname='a'\ntranslate={net='a',axis='x'}")
                .contains("unknown field")
        );
    }

    #[test]
    fn rejects_unknown_handles_and_tiny_steps() {
        let (g, domain, eps_stop) = wires();
        let res = |text: &str| {
            resolve(&ParamFile::parse(text, "t").unwrap(), &g, &domain, eps_stop)
                .unwrap_err()
                .to_string()
        };
        let head = "schema_version = 1\n[[param]]\nname='a'\n";
        assert!(
            res(&format!("{head}faces=[{{net='nope',box=0,face='x+'}}]")).contains("unknown net")
        );
        assert!(
            res(&format!("{head}faces=[{{net='victim',box=1,face='x+'}}]"))
                .contains("does not exist")
        );
        assert!(
            res(&format!(
                "{head}faces=[{{net='victim',box=0,face='x+',delta=1e-9}}]"
            ))
            .contains("ε_stop")
        );
        let two = format!(
            "{head}faces=[{{net='victim',box=0,face='x+',delta=0.01}},{{net='victim',box=0,face='x+',delta=0.02}}]"
        );
        assert!(res(&two).contains("two different steps"));
    }
}
