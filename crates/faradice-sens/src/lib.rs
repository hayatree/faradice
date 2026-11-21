//! Capacitance sensitivities for faradice (design D3).
//!
//! - [`params`]: parameter files (`*.sens.toml`): face handles, weights,
//!   `translate` shorthand, process sigmas, and their resolution into solver
//!   face steps with default step sizes.
//! - [`run_sensitivity`]: one nominal extraction plus `dC_ij/dp` for every
//!   parameter by finite differences with path continuation (pillar A0) or
//!   the shape derivative (pillar A, see [`Method`]), deletions (D-del) and
//!   additions by replay (D-add), producing the usual [`RunOutput`] and a
//!   [`SensFile`] (`sens.json`).
//! - [`map`]: insertion maps of potentials and predicted ΔC (pillar B).
//! - [`whatif`]: exact swap/merge/ground/float edits of the Maxwell matrix of
//!   a `result.json` (pillar C).

pub mod map;
pub mod params;
pub mod whatif;

use std::path::Path;
use std::time::Instant;

use faradice_core::{Estimate, Sink};
use faradice_engine::{Project, RunOutput, RunRequest, WalkEngineKind};
use faradice_frw::Solution;
use faradice_frw::replay::{replay_addition, solve_recorded};
use faradice_frw::sens::{
    Param, SensOptions, SensRequest as FrwRequest, sens_setup, solve_sensitivity_with,
};
use faradice_frw::shape::{
    ShapeFace, ShapeFaceResult, ShapeOptions, shape_support, solve_shape_derivative,
};
use faradice_io::result::sink_label;
use serde::{Deserialize, Serialize};

pub use params::{Method, ParamFile, Resolved, resolve};

/// Sensitivity errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SensError {
    /// Invalid parameter file or parameter.
    #[error("{0}")]
    Param(String),
    /// Loading, resolving or solving failed.
    #[error(transparent)]
    Engine(#[from] faradice_engine::EngineError),
    /// The solver rejected the request.
    #[error(transparent)]
    Solve(#[from] faradice_frw::SolveError),
    /// Writing output failed.
    #[error("{0}")]
    Output(String),
}

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, SensError>;

/// Current `sens.json` schema version.
///
/// Version 2: the ground sum is `<GND>` and the fixed ground sinks are
/// `<substrate>`, `<boundary:FACE>` and `<infinity>` (version 1: `GND`,
/// `GND:…`).
pub const SCHEMA_VERSION: u32 = 2;

/// `sens.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensFile {
    /// Schema version ([`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Face estimator: `fd-continuation`, `shape` (every face) or `mixed`
    /// (see [`FaceOut::method`]).
    pub method: String,
    /// Richardson extrapolation over `δ` and `2δ`.
    pub richardson: bool,
    /// Default step relative to box extent and front gap.
    pub delta_rel: f64,
    /// Stopping distance of the walks (µm).
    pub eps_stop_um: f64,
    /// Master seed.
    pub seed: u64,
    /// Per-target walk statistics.
    pub nets: Vec<NetStats>,
    /// The differentiated faces, with their per-face derivatives.
    pub faces: Vec<FaceOut>,
    /// The parameters, in file order.
    pub params: Vec<ParamOut>,
    /// The deletions, in file order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deletions: Vec<DeletionOut>,
    /// The additions, in file order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additions: Vec<AdditionOut>,
    /// Warnings (non-converged continuations, skipped faces, …).
    pub warnings: Vec<String>,
}

/// Walk statistics of one target net.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetStats {
    /// Target net.
    pub net: String,
    /// Nominal walks.
    pub walks: u64,
}

/// One face and `dC_ij/dn` for an outward displacement of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceOut {
    /// Net of the box.
    pub net: String,
    /// Box index within the net (structure-file order).
    #[serde(rename = "box")]
    pub box_index: usize,
    /// Face (`x-`, …, `z+`).
    pub face: String,
    /// Estimator of this face: `fd` (finite differences with path
    /// continuation) or `shape` (shape derivative, no step).
    #[serde(default = "method_fd")]
    pub method: String,
    /// Step `δ` (µm); 0 for `shape`.
    pub delta_um: f64,
    /// Face samples of a `shape` face.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u64>,
    /// Exposed area (µm²).
    pub exposed_area_um2: f64,
    /// Gap in front of the face (µm), if bounded.
    pub gap_um: Option<f64>,
    /// Walks continued in the perturbed geometries, over all targets.
    pub continued_walks: u64,
    /// Continuations (or, for `shape`, kernel walks) that hit `max_steps`.
    pub aborted_continuations: u64,
    /// `dC_ij/dn` (fF/µm).
    #[serde(rename = "dC")]
    pub dc: Vec<Entry>,
}

/// One parameter: `dC_ij/dp` and, with a process sigma, `σ_C`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamOut {
    /// Parameter name.
    pub name: String,
    /// Unit of the derivatives.
    pub unit: String,
    /// Process sigma of the parameter (µm), if given.
    pub sigma_um: Option<f64>,
    /// `(face index, weight)` of the chain rule.
    pub faces: Vec<(usize, f64)>,
    /// `dC_ij/dp` (fF/µm).
    #[serde(rename = "dC")]
    pub dc: Vec<Entry>,
    /// `|dC_ij/dp| · σ_p` (fF), with a process sigma.
    #[serde(rename = "sigma_C", skip_serializing_if = "Vec::is_empty", default)]
    pub sigma_c: Vec<Entry>,
}

/// One deletion: capacitances of the geometry without some nets, and their
/// exact paired differences from the nominal ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeletionOut {
    /// Deletion name.
    pub name: String,
    /// Deleted nets.
    pub nets: Vec<String>,
    /// Walks continued in the edited geometry, over all targets.
    pub continued_walks: u64,
    /// Continuations that hit `max_steps` (the result is then biased).
    pub aborted_continuations: u64,
    /// `C_ij` of the edited geometry (fF), same conventions as [`Entry`].
    #[serde(rename = "C")]
    pub c: Vec<Entry>,
    /// `ΔC_ij = C_ij(edited) − C_ij(nominal)` (fF), paired.
    #[serde(rename = "dC")]
    pub dc: Vec<Entry>,
}

/// One addition: capacitances of the geometry with added conductors, and
/// their exact paired differences from the nominal ones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdditionOut {
    /// Addition name.
    pub name: String,
    /// Net of the added boxes.
    pub net: String,
    /// `true` if the net is new (a conductor at 0 V).
    pub new_net: bool,
    /// A new net counted in the `<GND>` sums.
    pub ground: bool,
    /// The added boxes (µm).
    pub boxes: Vec<[f64; 6]>,
    /// Samples replayed (twice each), over all targets.
    pub replayed_samples: u64,
    /// Samples recorded, over all targets.
    pub recorded_samples: u64,
    /// `C_ij` of the edited geometry (fF), same conventions as [`Entry`].
    #[serde(rename = "C")]
    pub c: Vec<Entry>,
    /// `ΔC_ij = C_ij(edited) − C_ij(nominal)` (fF), paired.
    #[serde(rename = "dC")]
    pub dc: Vec<Entry>,
}

/// [`FaceOut::method`] of finite differences.
pub const METHOD_FD: &str = "fd";
/// [`FaceOut::method`] of the shape derivative.
pub const METHOD_SHAPE: &str = "shape";

fn method_fd() -> String {
    METHOD_FD.into()
}

/// One derivative entry. `j == i` is the total capacitance `C_ii`,
/// `j == "<GND>"` the sum over ground-like sinks (positive convention, like the
/// report's ground capacitance), and other `j` the Maxwell entry `C_ij`
/// (negative of the coupling).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// Target net.
    pub i: String,
    /// Sink label.
    pub j: String,
    /// Value.
    pub value: f64,
    /// Standard error.
    pub std_err: f64,
}

/// Label of the ground sum in [`Entry::j`].
pub const GROUND_LABEL: &str = faradice_core::GROUND_SUM_LABEL;

/// A sensitivity request.
#[derive(Debug, Clone)]
pub struct SensRequest {
    /// The extraction (targets, accuracy, seed, …). Only the CPU engine is
    /// supported.
    pub run: RunRequest,
    /// The parameters.
    pub params: ParamFile,
    /// Face method, overriding the file's `method`.
    pub method: Option<Method>,
    /// Shape-derivative samples per face, overriding the file's
    /// `shape_samples`.
    pub shape_samples: Option<u64>,
}

/// Everything a sensitivity run produced.
#[derive(Debug, Clone)]
pub struct SensOutput {
    /// The nominal extraction, identical to `faradice solve` with the same
    /// configuration.
    pub run: RunOutput,
    /// The sensitivities.
    pub sens: SensFile,
}

/// Runs the nominal extraction of `req.run`, the derivatives of every
/// parameter, the deletions and the additions of `req.params`.
///
/// # Errors
///
/// Invalid input, parameters or configuration, the GPU engine, faces the
/// shape derivative does not support under [`Method::Shape`], or solver
/// errors.
#[allow(clippy::too_many_lines)]
pub fn run_sensitivity(project: &Project, req: &SensRequest) -> Result<SensOutput> {
    if req.run.engine != WalkEngineKind::Cpu {
        return Err(SensError::Param(
            "sensitivities run on the CPU engine only".into(),
        ));
    }
    let method = match (req.method, &req.params.method) {
        (Some(m), _) => m,
        (None, Some(m)) => m.parse().map_err(SensError::Param)?,
        (None, None) => Method::Fd,
    };
    let shape_samples = req
        .shape_samples
        .or(req.params.shape_samples)
        .unwrap_or(params::DEFAULT_SHAPE_SAMPLES);
    if shape_samples < 2 {
        return Err(SensError::Param("shape samples must be at least 2".into()));
    }
    let mut p = faradice_engine::prepare(project, &req.run)?;
    let mut warnings = Vec::new();
    if p.config.output.cross_moments {
        // Not collected on this path: the stored configuration must not claim
        // them, or `faradice whatif` would read missing pairs as zeros.
        p.config.output.cross_moments = false;
        let w = "output.cross_moments is ignored by `faradice sens`; use `faradice solve` for \
                 a result to edit with `faradice whatif`"
            .to_owned();
        p.warnings.push(w.clone());
        warnings.push(w);
    }
    let pool = faradice_engine::thread_pool(&p.config)?;
    let g = p.geometry.clone();
    let su = sens_setup(&g, &p.targets, &p.solver)?;
    let res = params::resolve_unchecked(&req.params, &g, &su.domain)?;
    warnings.extend(res.warnings.iter().cloned());

    // Which faces take the shape derivative.
    let use_shape: Vec<bool> = if method == Method::Fd || res.faces.is_empty() {
        vec![false; res.faces.len()]
    } else {
        let sf: Vec<ShapeFace> = res
            .faces
            .iter()
            .map(|f| ShapeFace {
                box_id: f.step.box_id,
                face: f.step.face,
            })
            .collect();
        let support = shape_support(&g, &p.targets, &p.solver, &sf)?;
        if method == Method::Shape {
            let bad: Vec<String> = res
                .faces
                .iter()
                .zip(&support)
                .filter_map(|(f, s)| {
                    s.as_ref().err().map(|e| {
                        format!(
                            "net `{}` box {} face {}: {e}",
                            f.net,
                            f.box_index,
                            params::face_name(f.step.face)
                        )
                    })
                })
                .collect();
            if !bad.is_empty() {
                return Err(SensError::Param(format!(
                    "method `shape` does not support these faces (use `auto` to take finite \
                     differences for them):\n  {}",
                    bad.join("\n  ")
                )));
            }
        }
        support.iter().map(std::result::Result::is_ok).collect()
    };
    params::check_steps(
        res.faces
            .iter()
            .zip(&use_shape)
            .filter(|(_, u)| !**u)
            .map(|(f, _)| f),
        su.eps_stop,
    )?;
    // The finite-difference request: the other faces, re-indexed.
    let mut fd_index = vec![usize::MAX; res.faces.len()];
    let mut fd_steps = Vec::new();
    for (fi, f) in res.faces.iter().enumerate() {
        if !use_shape[fi] {
            fd_index[fi] = fd_steps.len();
            fd_steps.push(f.step);
        }
    }
    let fd_params: Vec<Param> = res
        .params
        .iter()
        .map(|pr| Param {
            faces: pr
                .faces
                .iter()
                .filter(|(fi, _)| !use_shape[*fi])
                .map(|&(fi, w)| (fd_index[fi], w))
                .collect(),
        })
        .collect();
    let shape_sel: Vec<ShapeFace> = res
        .faces
        .iter()
        .zip(&use_shape)
        .filter(|(_, u)| **u)
        .map(|(f, _)| ShapeFace {
            box_id: f.step.box_id,
            face: f.step.face,
        })
        .collect();

    let t0 = Instant::now();
    // Additions replay a recorded extraction. When nothing else needs the
    // finite-difference pass, the recorded solution is the nominal one.
    let recorded = if res.additions.is_empty() {
        None
    } else {
        Some(pool.install(|| solve_recorded(&g, &p.targets, &p.solver))?)
    };
    let need_fd = recorded.is_none() || !fd_steps.is_empty() || !res.deletions.is_empty();
    let sol = if need_fd {
        Some(pool.install(|| {
            solve_sensitivity_with(
                &g,
                &p.targets,
                &p.solver,
                &FrwRequest {
                    faces: &fd_steps,
                    params: &fd_params,
                    deletions: &res.deletions,
                    opts: SensOptions {
                        richardson: res.richardson,
                        ..SensOptions::default()
                    },
                },
            )
        })?)
    } else {
        None
    };
    let shape = if shape_sel.is_empty() {
        Vec::new()
    } else {
        pool.install(|| {
            solve_shape_derivative(
                &g,
                &p.targets,
                &p.solver,
                &shape_sel,
                ShapeOptions {
                    samples: shape_samples,
                    edge_margin: None,
                },
            )
        })?
    };
    if let (Some(s), Some((r, _))) = (&sol, &recorded) {
        // Both passes reproduce `solve`; the additions build on the recorded
        // statistics, the written result on the finite-difference ones.
        let bits = |n: &faradice_frw::NetResult| {
            (
                n.walks,
                n.sinks
                    .iter()
                    .map(|(k, m)| (*k, m.sum().to_bits(), m.sum_sq().to_bits()))
                    .collect::<Vec<_>>(),
            )
        };
        if s.nets
            .iter()
            .zip(&r.nets)
            .any(|(a, b)| bits(&a.nominal) != bits(b))
        {
            return Err(SensError::Output(
                "internal: the recorded and finite-difference nominal runs differ".into(),
            ));
        }
    }
    let nominal = match (&sol, &recorded) {
        (Some(s), _) => Solution {
            nets: s.nets.iter().map(|n| n.nominal.clone()).collect(),
            eps_stop: s.eps_stop,
            cancelled: false,
        },
        (None, Some((s, _))) => s.clone(),
        (None, None) => unreachable!("one of the passes runs"),
    };
    let additions = match &recorded {
        Some((rs, rec)) => res
            .additions
            .iter()
            .zip(&req.params.adds)
            .map(|(a, spec)| {
                pool.install(|| replay_addition(&g, &p.solver, rs, rec, &a.addition))
                    .map_err(|e| match e {
                        faradice_frw::SolveError::Config(m) => SensError::Param(format!(
                            "addition `{}`: {}",
                            spec.name,
                            m.trim_start_matches("addition: ")
                                .trim_start_matches("addition ")
                        )),
                        other => other.into(),
                    })
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    let elapsed_s = t0.elapsed().as_secs_f64();

    let names: Vec<String> = g.nets().map(|n| g.net_name(n).to_owned()).collect();
    let target_names: Vec<String> = nominal
        .nets
        .iter()
        .map(|n| names[n.net.index()].clone())
        .collect();
    let label = |s: Sink| sink_label(&names, s);

    // Finite-difference entries of every target for one pick.
    let fd_entries = |pick: &dyn Fn(usize) -> (Vec<(Sink, Estimate)>, Estimate)| -> Vec<Entry> {
        let mut out = Vec::new();
        for (ti, i) in target_names.iter().enumerate() {
            let (sinks, ground) = pick(ti);
            for (s, e) in sinks {
                out.push(Entry {
                    i: i.clone(),
                    j: label(s),
                    value: e.value,
                    std_err: e.std_err,
                });
            }
            out.push(Entry {
                i: i.clone(),
                j: GROUND_LABEL.into(),
                value: ground.value,
                std_err: ground.std_err,
            });
        }
        out
    };
    // Shape-derivative entries of one face, scaled by `w`.
    let shape_entries = |r: &ShapeFaceResult, w: f64| -> Vec<Entry> {
        let mut out = Vec::new();
        for (n, i) in nominal.nets.iter().zip(&target_names) {
            for ((a, b), _) in &r.pairs {
                let other = if *a == n.net {
                    *b
                } else if *b == n.net {
                    *a
                } else {
                    continue;
                };
                let e = r.dc(n.net, other);
                out.push(Entry {
                    i: i.clone(),
                    j: names[other.index()].clone(),
                    value: w * e.value,
                    std_err: w.abs() * e.std_err,
                });
            }
            let e = r.d_ground(n.net);
            out.push(Entry {
                i: i.clone(),
                j: GROUND_LABEL.into(),
                value: w * e.value,
                std_err: w.abs() * e.std_err,
            });
        }
        out
    };
    let shape_pos = |fi: usize| use_shape[..fi].iter().filter(|u| **u).count();

    let faces = res
        .faces
        .iter()
        .enumerate()
        .map(|(fi, f)| {
            let base = FaceOut {
                net: f.net.clone(),
                box_index: f.box_index,
                face: params::face_name(f.step.face).into(),
                method: METHOD_FD.into(),
                delta_um: f.step.delta,
                samples: None,
                exposed_area_um2: f.exposed_area,
                gap_um: f.gap,
                continued_walks: 0,
                aborted_continuations: 0,
                dc: Vec::new(),
            };
            if use_shape[fi] {
                let r = &shape[shape_pos(fi)];
                if r.aborted_walks > 0 {
                    warnings.push(format!(
                        "net `{}` box {} face {}: {} shape-derivative walks exceeded max_steps \
                         (scored as reaching no net)",
                        f.net,
                        f.box_index,
                        params::face_name(f.step.face),
                        r.aborted_walks
                    ));
                }
                return FaceOut {
                    method: METHOD_SHAPE.into(),
                    delta_um: 0.0,
                    samples: Some(r.samples),
                    aborted_continuations: r.aborted_walks,
                    dc: shape_entries(r, 1.0),
                    ..base
                };
            }
            let sol = sol.as_ref().expect("finite differences ran");
            let k = fd_index[fi];
            let (continued, aborted) = sol.nets.iter().fold((0, 0), |(c, a), n| {
                (
                    c + n.faces[k].continued_walks,
                    a + n.faces[k].aborted_continuations,
                )
            });
            if aborted > 0 {
                warnings.push(format!(
                    "net `{}` box {} face {}: {aborted} continuations exceeded max_steps; the \
                     derivative is biased toward zero",
                    f.net,
                    f.box_index,
                    params::face_name(f.step.face)
                ));
            }
            let (skipped, shell_aborted) = sol
                .nets
                .iter()
                .filter_map(|n| n.faces[k].shell.as_ref())
                .fold((0, 0), |(s, a), r| (s + r.skipped, a + r.aborted));
            if skipped + shell_aborted > 0 {
                warnings.push(format!(
                    "net `{}` box {} face {}: the conformal-shell term lost {skipped} points \
                     without a field cube and {shell_aborted} walks over max_steps (counted as \
                     zero, biased)",
                    f.net,
                    f.box_index,
                    params::face_name(f.step.face)
                ));
            }
            FaceOut {
                continued_walks: continued,
                aborted_continuations: aborted,
                dc: fd_entries(&|ti| {
                    let n = &sol.nets[ti];
                    let r = &n.faces[k];
                    let w = n.nominal.walks;
                    (
                        r.sink_keys().into_iter().map(|s| (s, r.dc(s, w))).collect(),
                        r.d_ground(w),
                    )
                }),
                ..base
            }
        })
        .collect();

    let mut fixed_dropped = false;
    // Labels of the fixed ground sinks the finite differences reported.
    let fixed_labels: std::collections::BTreeSet<String> = sol
        .iter()
        .flat_map(|s| &s.nets)
        .flat_map(|n| {
            n.params
                .iter()
                .flat_map(|r| r.sink_keys())
                .chain(n.nominal.sinks.iter().map(|(k, _)| *k))
        })
        .filter(|k| k.is_fixed_ground())
        .map(label)
        .collect();
    let params = req
        .params
        .params
        .iter()
        .zip(&res.params)
        .enumerate()
        .map(|(pi, (spec, param))| {
            let mut parts: Vec<Vec<Entry>> = Vec::new();
            if let Some(sol) = &sol {
                if !fd_params[pi].faces.is_empty() {
                    parts.push(fd_entries(&|ti| {
                        let n = &sol.nets[ti];
                        let r = &n.params[pi];
                        let w = n.nominal.walks;
                        (
                            r.sink_keys().into_iter().map(|s| (s, r.dc(s, w))).collect(),
                            r.d_ground(w),
                        )
                    }));
                }
            }
            let mut has_shape = false;
            if parts.is_empty() && param.faces.is_empty() {
                // No face moves: an explicit zero per target.
                parts.push(
                    target_names
                        .iter()
                        .map(|i| Entry {
                            i: i.clone(),
                            j: GROUND_LABEL.into(),
                            value: 0.0,
                            std_err: 0.0,
                        })
                        .collect(),
                );
            }
            for &(fi, w) in &param.faces {
                if use_shape[fi] {
                    has_shape = true;
                    parts.push(shape_entries(&shape[shape_pos(fi)], w));
                }
            }
            let mut dc = sum_entries(&target_names, parts);
            if has_shape {
                // The shape derivative has no per-sink split of the fixed
                // ground sinks (substrate, walls, infinity): only `<GND>`.
                let before = dc.len();
                dc.retain(|e| !fixed_labels.contains(&e.j));
                fixed_dropped |= dc.len() != before;
            }
            let sigma_c = spec.sigma.map_or_else(Vec::new, |sp| {
                dc.iter()
                    .map(|e| Entry {
                        value: e.value.abs() * sp,
                        std_err: e.std_err * sp,
                        ..e.clone()
                    })
                    .collect()
            });
            ParamOut {
                name: spec.name.clone(),
                unit: "fF/um".into(),
                sigma_um: spec.sigma,
                faces: param.faces.clone(),
                dc,
                sigma_c,
            }
        })
        .collect();
    if fixed_dropped {
        warnings.push(
            "parameters with shape-derivative faces report the fixed ground sinks (<substrate>, \
             <boundary:…>, <infinity>) only in their <GND> sum"
                .into(),
        );
    }

    let deletions = req
        .params
        .deletes
        .iter()
        .enumerate()
        .map(|(di, spec)| {
            let sol = sol.as_ref().expect("finite differences ran");
            let (continued, aborted) = sol.nets.iter().fold((0, 0), |(c, a), n| {
                (
                    c + n.deletions[di].continued_walks,
                    a + n.deletions[di].aborted_continuations,
                )
            });
            if aborted > 0 {
                warnings.push(format!(
                    "deletion `{}`: {aborted} continuations exceeded max_steps; the result is biased",
                    spec.name
                ));
            }
            let pick = |edited: bool| {
                fd_entries(&|ti| {
                    let n = &sol.nets[ti];
                    let r = &n.deletions[di];
                    let w = n.nominal.walks;
                    let src = if edited { &r.edited } else { &r.delta };
                    (
                        src.iter()
                            .map(|(s, _)| (*s, if edited { r.c(*s, w) } else { r.delta_c(*s, w) }))
                            .collect(),
                        if edited { r.ground(w) } else { r.delta_ground(w) },
                    )
                })
            };
            // Deleted targets have no row in the edited geometry.
            let keep = |v: Vec<Entry>| -> Vec<Entry> {
                v.into_iter().filter(|e| !spec.nets.contains(&e.i)).collect()
            };
            DeletionOut {
                name: spec.name.clone(),
                nets: spec.nets.clone(),
                continued_walks: continued,
                aborted_continuations: aborted,
                c: keep(pick(true)),
                dc: keep(pick(false)),
            }
        })
        .collect();

    let recorded_samples = recorded.as_ref().map_or(0, |(_, r)| r.samples() as u64);
    let additions = req
        .params
        .adds
        .iter()
        .zip(&res.additions)
        .zip(&additions)
        .map(|((spec, ra), out)| {
            let mut all = names.clone();
            if ra.new_net {
                all.push(ra.net.clone());
            }
            let lab = |s: Sink| sink_label(&all, s);
            let replayed: u64 = out.nets.iter().map(|n| n.replayed).sum();
            if recorded_samples > 0 && 2 * replayed > recorded_samples {
                warnings.push(format!(
                    "addition `{}` replays {replayed} of {recorded_samples} samples twice; a \
                     re-solve would cost about as much",
                    spec.name
                ));
            }
            let (mut c, mut dc) = (Vec::new(), Vec::new());
            for (n, i) in out.nets.iter().zip(&target_names) {
                let e = |v: Estimate, j: String| Entry {
                    i: i.clone(),
                    j,
                    value: v.value,
                    std_err: v.std_err,
                };
                for s in n.edited.keys() {
                    c.push(e(n.c(*s), lab(*s)));
                }
                c.push(e(n.ground(), GROUND_LABEL.into()));
                for s in n.delta.keys() {
                    dc.push(e(n.delta_c(*s), lab(*s)));
                }
                dc.push(e(n.delta_ground(), GROUND_LABEL.into()));
            }
            AdditionOut {
                name: spec.name.clone(),
                net: ra.net.clone(),
                new_net: ra.new_net,
                ground: ra.addition.ground,
                boxes: spec.boxes.clone(),
                replayed_samples: replayed,
                recorded_samples,
                c,
                dc,
            }
        })
        .collect();

    let n_shape = use_shape.iter().filter(|u| **u).count();
    let sens = SensFile {
        schema_version: SCHEMA_VERSION,
        method: if n_shape == 0 {
            "fd-continuation"
        } else if n_shape == use_shape.len() {
            METHOD_SHAPE
        } else {
            "mixed"
        }
        .into(),
        richardson: res.richardson,
        delta_rel: res.delta_rel,
        eps_stop_um: nominal.eps_stop,
        seed: p.config.run.seed,
        nets: nominal
            .nets
            .iter()
            .map(|n| NetStats {
                net: names[n.net.index()].clone(),
                walks: n.walks,
            })
            .collect(),
        faces,
        params,
        deletions,
        additions,
        warnings,
    };
    let run = faradice_engine::finish(p, &nominal, elapsed_s, None)?;
    Ok(SensOutput { run, sens })
}

/// Sums entries with the same `(i, j)` over independent parts (standard
/// errors in quadrature), per target in `targets` order with the ground sum
/// last.
fn sum_entries(targets: &[String], parts: Vec<Vec<Entry>>) -> Vec<Entry> {
    let mut acc: Vec<(Entry, f64)> = Vec::new();
    for e in parts.into_iter().flatten() {
        let var = e.std_err * e.std_err;
        match acc.iter_mut().find(|(a, _)| a.i == e.i && a.j == e.j) {
            Some((a, v)) => {
                a.value += e.value;
                *v += var;
            }
            None => acc.push((e, var)),
        }
    }
    acc.sort_by_key(|(e, _)| {
        (
            targets.iter().position(|t| *t == e.i).unwrap_or(usize::MAX),
            e.j == GROUND_LABEL,
        )
    });
    acc.into_iter()
        .map(|(mut e, var)| {
            e.std_err = var.sqrt();
            e
        })
        .collect()
}

/// Writes `result.json`, the reports and `sens.json` into `dir`.
///
/// # Errors
///
/// File system errors.
pub fn write_outputs(out: &SensOutput, dir: &Path, design: &str) -> Result<()> {
    faradice_engine::write_outputs(&out.run, dir, design)?;
    let p = dir.join("sens.json");
    let text =
        serde_json::to_string_pretty(&out.sens).map_err(|e| SensError::Output(e.to_string()))?;
    std::fs::write(&p, text).map_err(|e| SensError::Output(format!("writing {}: {e}", p.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(i: &str, j: &str, value: f64, std_err: f64) -> Entry {
        Entry {
            i: i.into(),
            j: j.into(),
            value,
            std_err,
        }
    }

    #[test]
    fn a_conductor_named_gnd_stays_apart_from_the_ground_sum() {
        let t = vec!["a".to_owned()];
        let fd = vec![
            e("a", GROUND_LABEL, 1.0, 0.3),
            e("a", "a", 2.0, 0.1),
            e("a", "GND", 5.0, 0.4),
        ];
        let shape = vec![e("a", GROUND_LABEL, 10.0, 0.0), e("a", "GND", 20.0, 0.3)];
        let out = sum_entries(&t, vec![fd, shape]);
        assert_eq!(out.len(), 3);
        assert_eq!((out[0].j.as_str(), out[0].value), ("a", 2.0));
        assert_eq!((out[1].j.as_str(), out[1].value), ("GND", 25.0));
        assert!((out[1].std_err - 0.5).abs() < 1e-12);
        assert_eq!((out[2].j.as_str(), out[2].value), (GROUND_LABEL, 11.0));
        assert!((out[2].std_err - 0.3).abs() < 1e-12);
    }
}
