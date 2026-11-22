//! Blocking runs.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use faradice_core::NetId;
use faradice_frw::{Observer, Solution, SolverConfig};
use faradice_geom::Geometry;
use faradice_io::config::{Config, GroundIgnore, GroundReport};
use faradice_io::report::{Report, build_report};
use faradice_io::result::ResultFile;

use crate::{EngineError, Project, Result};

/// Which walk engine runs the transition phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum WalkEngineKind {
    /// Multi-threaded CPU solver (reference, bit-for-bit reproducible).
    #[default]
    Cpu,
    /// wgpu/Vulkan engine (needs the `gpu` feature).
    Gpu,
}

impl std::str::FromStr for WalkEngineKind {
    type Err = EngineError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(EngineError::Other(format!(
                "unknown engine `{other}` (expected `cpu` or `gpu`)"
            ))),
        }
    }
}

/// One extraction request.
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    /// Full run configuration (targets, accuracy, ground, reporting, …).
    pub config: Config,
    /// Walk engine.
    pub engine: WalkEngineKind,
}

/// Everything a run produced.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Raw statistics (`result.json`).
    pub result: ResultFile,
    /// Postprocessed report.
    pub report: Report,
    /// The configuration actually used (substrate height applied).
    pub config: Config,
    /// Solved geometry (for viewers).
    pub geometry: Arc<Geometry>,
    /// Target nets, in result order.
    pub targets: Vec<String>,
    /// Wall-clock solve time in seconds.
    pub elapsed_s: f64,
    /// Whether the run was cancelled; results are then partial.
    pub cancelled: bool,
    /// Engine description (e.g. the GPU adapter), if not the CPU.
    pub engine_info: Option<String>,
}

/// What [`prepare`] resolved before solving. Drivers that solve with their
/// own entry point (e.g. sensitivities) use it with [`finish`].
#[derive(Debug, Clone)]
pub struct Prepared {
    /// Design name of the project (stored in `result.json`).
    pub design: String,
    /// The configuration actually used (substrate height applied).
    pub config: Config,
    /// The geometry to solve.
    pub geometry: Arc<Geometry>,
    /// Target nets, in solve order.
    pub targets: Vec<NetId>,
    /// Solver configuration.
    pub solver: SolverConfig,
    /// Ground report mode.
    pub ground_report: GroundReport,
    /// Warnings of loading and resolving.
    pub warnings: Vec<String>,
}

/// Builds the geometry and resolves the configuration of `req`, without
/// solving.
///
/// # Errors
///
/// Invalid input or configuration.
pub fn prepare(project: &Project, req: &RunRequest) -> Result<Prepared> {
    let mut config = req.config.clone();
    let st = project.build(&mut config)?;
    let resolved = config.resolve(&st.geometry, st.eps_r, &st.layers, &st.regions)?;
    let mut warnings = project.diagnostics.clone();
    warnings.extend(resolved.warnings);
    Ok(Prepared {
        design: project.design.clone(),
        config,
        geometry: Arc::new(st.geometry),
        targets: resolved.targets,
        solver: resolved.solver,
        ground_report: resolved.ground_report,
        warnings,
    })
}

/// Runs one extraction, blocking the calling thread. Progress and
/// cancellation go through `observer` (use `&()` for none).
///
/// # Errors
///
/// Invalid input or configuration, solver or engine errors. Cancellation is
/// not an error: [`RunOutput::cancelled`] is set and results are partial.
pub fn run(project: &Project, req: &RunRequest, observer: &dyn Observer) -> Result<RunOutput> {
    let p = prepare(project, req)?;
    run_prepared(p, req.engine, observer)
}

pub(crate) fn run_prepared(
    p: Prepared,
    engine: WalkEngineKind,
    observer: &dyn Observer,
) -> Result<RunOutput> {
    let pool = thread_pool(&p.config)?;
    let t0 = Instant::now();
    if p.config.output.cross_moments {
        if engine != WalkEngineKind::Cpu {
            return Err(EngineError::Other(
                "output.cross_moments needs the CPU engine".into(),
            ));
        }
        let (sol, cross) = pool.install(|| {
            faradice_frw::cross::solve_with_cross_moments(
                &p.geometry,
                &p.targets,
                &p.solver,
                observer,
            )
        })?;
        let mut out = finish(p, &sol, t0.elapsed().as_secs_f64(), None)?;
        out.result.set_cross(&cross);
        return Ok(out);
    }
    let (sol, engine_info) =
        pool.install(|| solve(engine, &p.geometry, &p.targets, &p.solver, observer))?;
    finish(p, &sol, t0.elapsed().as_secs_f64(), engine_info)
}

/// The thread pool of `cfg.run.threads` (0 = all cores).
///
/// # Errors
///
/// The pool cannot be built.
pub fn thread_pool(cfg: &Config) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.run.threads)
        .build()
        .map_err(|e| EngineError::Other(format!("thread pool: {e}")))
}

/// Turns a solution of the prepared run into [`RunOutput`] (`result.json`
/// and the report).
///
/// # Errors
///
/// Serializing the configuration failed.
pub fn finish(
    p: Prepared,
    sol: &Solution,
    elapsed_s: f64,
    engine_info: Option<String>,
) -> Result<RunOutput> {
    let g = &p.geometry;
    let ground_names = p
        .solver
        .ground_nets
        .iter()
        .map(|&n| g.net_name(n).to_owned())
        .collect();
    let mut warnings = p.warnings;
    if sol.cancelled {
        warnings.push("run cancelled: results are partial and not converged".into());
    }
    let mut result = ResultFile::from_solution(
        g,
        sol,
        p.config.run.seed,
        ground_names,
        elapsed_s,
        warnings,
        config_toml(&p.config)?,
    );
    result.design = Some(p.design.clone());
    let report = build_report(&result, &p.config.coupling, p.ground_report);
    let targets = sol
        .nets
        .iter()
        .map(|n| g.net_name(n.net).to_owned())
        .collect();
    Ok(RunOutput {
        result,
        report,
        config: p.config,
        geometry: p.geometry,
        targets,
        elapsed_s,
        cancelled: sol.cancelled,
        engine_info,
    })
}

fn solve(
    engine: WalkEngineKind,
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    observer: &dyn Observer,
) -> Result<(Solution, Option<String>)> {
    match engine {
        WalkEngineKind::Cpu => Ok((
            faradice_frw::solve_observed(geom, targets, cfg, observer)?,
            None,
        )),
        WalkEngineKind::Gpu => solve_gpu(geom, targets, cfg, observer),
    }
}

#[cfg(feature = "gpu")]
fn solve_gpu(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    observer: &dyn Observer,
) -> Result<(Solution, Option<String>)> {
    let mut info = None;
    let sol = faradice_frw::solve_with_observed(
        geom,
        targets,
        cfg,
        |scene| {
            let e = faradice_gpu::GpuEngine::new(scene, faradice_gpu::GpuOptions::default())
                .map_err(|e| e.to_string())?;
            info = Some(format!("GPU, {}", e.adapter()));
            Ok(e)
        },
        observer,
    )?;
    Ok((sol, info))
}

#[cfg(not(feature = "gpu"))]
fn solve_gpu(
    _: &Geometry,
    _: &[NetId],
    _: &SolverConfig,
    _: &dyn Observer,
) -> Result<(Solution, Option<String>)> {
    Err(EngineError::Other(
        "this build has no GPU support; rebuild with the `gpu` feature".into(),
    ))
}

/// Writes `result.json` and the report formats of `out.config.output.formats`
/// into `dir`.
///
/// # Errors
///
/// File system errors or an unknown output format.
pub fn write_outputs(out: &RunOutput, dir: &Path, design: &str) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|e| EngineError::Other(format!("creating {}: {e}", dir.display())))?;
    out.result.write(&dir.join("result.json"))?;
    write_reports(&out.report, &out.config, dir, design)
}

/// Writes the report formats of `cfg.output.formats` into `dir`.
///
/// # Errors
///
/// File system errors or an unknown output format.
pub fn write_reports(report: &Report, cfg: &Config, dir: &Path, design: &str) -> Result<()> {
    let w = |name: String, text: String| {
        let p = dir.join(name);
        std::fs::write(&p, text)
            .map_err(|e| EngineError::Other(format!("writing {}: {e}", p.display())))
    };
    for f in &cfg.output.formats {
        match f.as_str() {
            "json" => w(
                "report.json".into(),
                serde_json::to_string_pretty(report)
                    .map_err(|e| EngineError::Other(e.to_string()))?,
            )?,
            "csv" => w("report.csv".into(), report.to_csv())?,
            "spef" => w(format!("{design}.spef"), report.to_spef(design))?,
            other => {
                return Err(EngineError::Other(format!(
                    "unknown output format `{other}` ({})",
                    faradice_io::config::OUTPUT_FORMATS.join(", ")
                )));
            }
        }
    }
    Ok(())
}

/// The effective ground report mode of a stored configuration (as used by
/// `faradice report`).
#[must_use]
pub fn ground_report_of(cfg: &Config) -> GroundReport {
    if cfg.ground.ignore == GroundIgnore::Report {
        GroundReport::Drop
    } else {
        cfg.ground.report
    }
}

/// Renders a configuration as TOML (unset options omitted).
///
/// # Errors
///
/// Serialization errors.
pub fn config_toml(cfg: &Config) -> Result<String> {
    let err = |e: &dyn std::fmt::Display| EngineError::Other(format!("config to TOML: {e}"));
    let v = serde_json::to_value(cfg).map_err(|e| err(&e))?;
    let t: toml::Value = serde_json::from_value(strip_nulls(v)).map_err(|e| err(&e))?;
    toml::to_string(&t).map_err(|e| err(&e))
}

fn strip_nulls(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => serde_json::Value::Object(
            m.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls(v)))
                .collect(),
        ),
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.into_iter().map(strip_nulls).collect())
        }
        other => other,
    }
}
