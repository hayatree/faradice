//! `faradice` command-line interface.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use faradice_engine::{Project, RunRequest, WalkEngineKind};
use faradice_io::config::{Config, Preset, read_config};
use faradice_io::report::{Report, build_report};
use faradice_io::result::ResultFile;

#[derive(Parser)]
#[command(
    name = "faradice",
    version,
    about = "Fast, accurate 3D capacitance extraction with floating random walks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solve a structure and write results and reports.
    Solve {
        /// Structure file (`*.faradice.toml`) or GDSII file (requires `--tech`).
        structure: PathBuf,
        /// Run configuration (`faradice.toml`).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Technology file, required for GDSII input.
        #[arg(long)]
        tech: Option<PathBuf>,
        /// Top cell for GDSII input.
        #[arg(long)]
        top: Option<String>,
        /// Output directory.
        #[arg(short, long, default_value = "faradice_out")]
        out: PathBuf,
        /// Target net globs (overrides `extract.nets`), comma separated.
        #[arg(long, value_delimiter = ',')]
        nets: Option<Vec<String>>,
        /// Accuracy preset (fast, default, signoff, reference).
        #[arg(long)]
        preset: Option<String>,
        /// Worker threads (0 = all cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Master seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Print the report as JSON on stdout.
        #[arg(long)]
        json: bool,
        /// Walk engine: `cpu` (default) or `gpu` (requires the `gpu` feature).
        #[arg(long, default_value = "cpu")]
        engine: String,
    },
    /// Capacitance sensitivities dC/dp, deletions and additions of a
    /// `*.sens.toml` file (design D3).
    #[cfg(feature = "sens")]
    Sens {
        /// Structure file (`*.faradice.toml`) or GDSII file (requires `--tech`).
        structure: PathBuf,
        /// Parameter file (`*.sens.toml`).
        #[arg(short, long)]
        params: PathBuf,
        /// Run configuration (`faradice.toml`).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Technology file, required for GDSII input.
        #[arg(long)]
        tech: Option<PathBuf>,
        /// Top cell for GDSII input.
        #[arg(long)]
        top: Option<String>,
        /// Output directory.
        #[arg(short, long, default_value = "faradice_out")]
        out: PathBuf,
        /// Target net globs (overrides `extract.nets`), comma separated.
        #[arg(long, value_delimiter = ',')]
        nets: Option<Vec<String>>,
        /// Accuracy preset (fast, default, signoff, reference).
        #[arg(long)]
        preset: Option<String>,
        /// Worker threads (0 = all cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Master seed.
        #[arg(long)]
        seed: Option<u64>,
        /// One-sided differences (no Richardson extrapolation).
        #[arg(long)]
        one_sided: bool,
        /// Face derivatives: `fd` (finite differences, default), `shape`
        /// (shape derivative; faces without geometric edges only) or `auto`
        /// (shape where supported). Overrides the file's `method`.
        #[arg(long)]
        method: Option<String>,
        /// Shape-derivative samples per face (overrides `shape_samples`).
        #[arg(long)]
        shape_samples: Option<u64>,
        /// Print `sens.json` on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Exact what-if edits of the extracted Maxwell matrix (ground, float,
    /// merge, swap conductors) without re-solving (design D3, pillar C).
    #[cfg(feature = "sens")]
    Whatif {
        /// Result file of a run where every edited conductor is a target
        /// (set `output.cross_moments = true` for exact σ).
        result: PathBuf,
        /// Ground these conductors.
        #[arg(long, value_delimiter = ',')]
        ground: Vec<String>,
        /// Leave these conductors floating (e.g. dummy fill).
        #[arg(long, value_delimiter = ',')]
        float: Vec<String>,
        /// Merge two conductors into one net (`a,b`); repeatable.
        #[arg(long, num_args = 1)]
        merge: Vec<String>,
        /// Swap the names of two conductors (`a,b`); repeatable.
        #[arg(long, num_args = 1)]
        swap: Vec<String>,
        /// Print the result as JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Insertion map: potentials of two nets on a grid and the predicted
    /// change of their coupling from a small grounded or floating object at
    /// each point (design D3, pillar B). Writes CSV.
    #[cfg(feature = "sens")]
    Map {
        /// Structure file (`*.faradice.toml`) or GDSII file (requires `--tech`).
        structure: PathBuf,
        /// The two nets `a,b` (equal for the total of one net).
        #[arg(long)]
        nets: String,
        /// Grid `x0:x1:nx,y0:y1:ny,z0:z1:nz` (µm); an axis may be one value.
        #[arg(long)]
        grid: String,
        /// Run configuration (`faradice.toml`).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Technology file, required for GDSII input.
        #[arg(long)]
        tech: Option<PathBuf>,
        /// Top cell for GDSII input.
        #[arg(long)]
        top: Option<String>,
        /// Samples per point.
        #[arg(long, default_value_t = 4000)]
        samples: u64,
        /// Radius (µm) of the sphere-equivalent object for the predictions.
        #[arg(long, default_value_t = 0.05)]
        radius: f64,
        /// Worker threads (0 = all cores).
        #[arg(long)]
        threads: Option<usize>,
        /// Master seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Output CSV file.
        #[arg(short, long, default_value = "map.csv")]
        out: PathBuf,
    },
    /// Regenerate reports from `result.json` without re-solving.
    Report {
        /// Result file.
        result: PathBuf,
        /// Configuration whose `[coupling]` and `[ground]` reporting options are applied
        /// (default: the configuration stored in the result).
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output directory (default: next to the result).
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Print the report as JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// GDSII tools: connectivity tracing and export.
    Gds {
        #[command(subcommand)]
        cmd: GdsCmd,
    },
    /// Validate a structure (and configuration) without solving.
    Check {
        /// Structure file.
        structure: PathBuf,
        /// Run configuration.
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}

#[derive(clap::Args, Clone)]
struct GdsInput {
    /// GDSII file.
    gds: PathBuf,
    /// Technology file (layer map, stack, vias, label layers).
    #[arg(long)]
    tech: PathBuf,
    /// Top cell (default: the unique top cell).
    #[arg(long)]
    top: Option<String>,
    /// Merge disconnected components that carry the same label.
    #[arg(long)]
    join_same_name: bool,
    /// Only use labels up to this hierarchy depth (0 = top cell only).
    #[arg(long)]
    label_depth: Option<u32>,
}

#[derive(Subcommand)]
enum GdsCmd {
    /// Trace connectivity and list nets (or show one net with --net).
    Trace {
        #[command(flatten)]
        input: GdsInput,
        /// Show details for this net.
        #[arg(long)]
        net: Option<String>,
        /// Print JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Export traced nets to a structure file for `faradice solve`.
    Export {
        #[command(flatten)]
        input: GdsInput,
        /// Output structure file.
        #[arg(short, long)]
        out: PathBuf,
        /// Net globs to export (default: all labelled nets).
        #[arg(long, value_delimiter = ',')]
        nets: Option<Vec<String>>,
    },
}

struct GdsLoaded {
    traced: faradice_gds::trace::Traced,
    tech: faradice_gds::tech::Tech,
}

fn load_gds(input: &GdsInput) -> Result<GdsLoaded> {
    use faradice_gds::{flatten, stream, tech, trace};
    let tech = tech::Tech::read(&input.tech)?;
    let lib = stream::read_library(&input.gds)?;
    let units = lib.units.context("GDS file has no UNITS record")?;
    let flat = flatten::flatten(
        &lib,
        input.top.as_deref(),
        flatten::FlattenOptions::default(),
    )?;
    let opts = trace::TraceOptions {
        max_label_depth: input.label_depth,
        join_same_name: input.join_same_name,
    };
    let traced = trace::trace(&flat, units, &tech, opts)?;
    for d in &traced.diagnostics {
        eprintln!("warning: {d}");
    }
    Ok(GdsLoaded { traced, tech })
}

fn component_json(g: &GdsLoaded, c: &faradice_gds::trace::Component) -> serde_json::Value {
    let mut per_layer: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut bb: Option<faradice_geom::Aabb> = None;
    for &b in &c.boxes {
        *per_layer
            .entry(g.tech.span_name(g.traced.box_span[b]).to_owned())
            .or_default() += 1;
        let x = g.traced.boxes[b];
        bb = Some(bb.map_or(x, |a| a.union(&x)));
    }
    let bb = bb.map(|b| serde_json::json!({ "min": b.min(), "max": b.max() }));
    serde_json::json!({ "net": c.name, "labels": c.labels, "boxes": c.boxes.len(), "boxes_per_layer": per_layer, "bbox_um": bb })
}

fn load_config(path: Option<&Path>) -> Result<Config> {
    Ok(match path {
        Some(p) => read_config(p)?,
        None => Config::default(),
    })
}

fn parse_preset(s: &str) -> Result<Preset> {
    Ok(match s {
        "fast" => Preset::Fast,
        "default" => Preset::Default,
        "signoff" => Preset::Signoff,
        "reference" => Preset::Reference,
        _ => bail!("unknown preset `{s}` (fast, default, signoff, reference)"),
    })
}

/// Prints the summary table and every warning (run and report).
fn print_summary(report: &Report) {
    eprintln!(
        "{:<24} {:>14} {:>9} {:>14} {:>8}  top coupling",
        "net", "total [fF]", "σ [%]", "ground [fF]", "walks"
    );
    for n in &report.nets {
        let rel = 100.0 * n.total.sigma / n.total.value.abs().max(f64::MIN_POSITIVE);
        let g = n
            .ground
            .map_or_else(|| "-".to_owned(), |g| format!("{:.6}", g.value));
        let top = n
            .couplings
            .first()
            .map_or_else(String::new, |c| format!("{} {:.6}", c.net, c.c.value));
        eprintln!(
            "{:<24} {:>14.6} {:>9.3} {:>14} {:>8}  {top}",
            n.net, n.total.value, rel, g, n.walks
        );
    }
    for w in &report.warnings {
        eprintln!("warning: {w}");
    }
}

/// Applies the command-line overrides shared by `solve` and `sens`.
fn run_config(
    config: Option<&Path>,
    nets: Option<Vec<String>>,
    preset: Option<&str>,
    threads: Option<usize>,
    seed: Option<u64>,
) -> Result<Config> {
    let mut cfg = load_config(config)?;
    if let Some(n) = nets {
        cfg.extract.nets = n;
    }
    if let Some(p) = preset {
        cfg.accuracy.preset = parse_preset(p)?;
    }
    if let Some(t) = threads {
        cfg.run.threads = t;
    }
    if let Some(s) = seed {
        cfg.run.seed = s;
    }
    Ok(cfg)
}

/// Prints the parameter derivatives of `sens.json`.
#[cfg(feature = "sens")]
fn print_sens(s: &faradice_sens::SensFile) {
    eprintln!(
        "{:<20} {:<16} {:<16} {:>14} {:>9}",
        "param", "i", "j", "dC/dp [fF/µm]", "σ [%]"
    );
    for p in &s.params {
        for e in &p.dc {
            let rel = 100.0 * e.std_err / e.value.abs().max(f64::MIN_POSITIVE);
            eprintln!(
                "{:<20} {:<16} {:<16} {:>14.6} {:>9.2}",
                p.name, e.i, e.j, e.value, rel
            );
        }
    }
    if !s.deletions.is_empty() {
        eprintln!(
            "{:<20} {:<16} {:<16} {:>14} {:>22}",
            "delete", "i", "j", "C [fF]", "ΔC [fF]"
        );
    }
    for d in &s.deletions {
        for c in &d.c {
            let dc = d.dc.iter().find(|e| e.i == c.i && e.j == c.j).map_or_else(
                || "0".to_owned(),
                |e| format!("{:.6} ± {:.6}", e.value, e.std_err),
            );
            eprintln!(
                "{:<20} {:<16} {:<16} {:>14.6} {:>22}",
                d.name, c.i, c.j, c.value, dc
            );
        }
    }
    if !s.additions.is_empty() {
        eprintln!(
            "{:<20} {:<16} {:<16} {:>14} {:>22}",
            "add", "i", "j", "C [fF]", "ΔC [fF]"
        );
    }
    for a in &s.additions {
        for c in &a.c {
            let dc = a.dc.iter().find(|e| e.i == c.i && e.j == c.j).map_or_else(
                || "0".to_owned(),
                |e| format!("{:.6} ± {:.6}", e.value, e.std_err),
            );
            eprintln!(
                "{:<20} {:<16} {:<16} {:>14.6} {:>22}",
                a.name, c.i, c.j, c.value, dc
            );
        }
    }
    for w in &s.warnings {
        eprintln!("warning: {w}");
    }
}

/// Parses `a,b`.
#[cfg(feature = "sens")]
fn pair(s: &str) -> Result<(String, String)> {
    match s.split_once(',') {
        Some((a, b)) if !a.is_empty() && !b.is_empty() && !b.contains(',') => {
            Ok((a.to_owned(), b.to_owned()))
        }
        _ => bail!("expected two conductors `a,b`, got `{s}`"),
    }
}

/// Prints an edited matrix.
#[cfg(feature = "sens")]
fn print_whatif(w: &faradice_sens::whatif::WhatIf) {
    eprintln!(
        "{:<24} {:>14} {:>9} {:>14} {:>9}  top coupling",
        "net", "total [fF]", "σ [%]", "ground [fF]", "σ [%]"
    );
    let rel =
        |v: &faradice_sens::whatif::Value| 100.0 * v.std_err / v.value.abs().max(f64::MIN_POSITIVE);
    for n in &w.nets {
        let top = n.couplings.first().map_or_else(String::new, |c| {
            format!("{} {:.6} ± {:.6}", c.net, c.c.value, c.c.std_err)
        });
        eprintln!(
            "{:<24} {:>14.6} {:>9.3} {:>14.6} {:>9.3}  {top}",
            n.net,
            n.total.value,
            rel(&n.total),
            n.ground.value,
            rel(&n.ground)
        );
    }
    for w in &w.warnings {
        eprintln!("warning: {w}");
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Solve {
            structure,
            config,
            tech,
            top,
            out,
            nets,
            preset,
            threads,
            seed,
            json,
            engine,
        } => {
            let cfg = run_config(config.as_deref(), nets, preset.as_deref(), threads, seed)?;
            let project = Project::load(&structure, tech.as_deref(), top.as_deref())?;
            let engine: WalkEngineKind = engine.parse()?;
            let out_run = faradice_engine::run(
                &project,
                &RunRequest {
                    config: cfg,
                    engine,
                },
                &(),
            )?;
            if let Some(info) = &out_run.engine_info {
                eprintln!("engine: {info}");
            }
            faradice_engine::write_outputs(&out_run, &out, &project.design)?;
            print_summary(&out_run.report);
            eprintln!(
                "solved {} nets in {:.2} s → {}",
                out_run.report.nets.len(),
                out_run.elapsed_s,
                out.display()
            );
            if json {
                println!("{}", serde_json::to_string_pretty(&out_run.report)?);
            }
        }
        #[cfg(feature = "sens")]
        Command::Sens {
            structure,
            params,
            config,
            tech,
            top,
            out,
            nets,
            preset,
            threads,
            seed,
            one_sided,
            method,
            shape_samples,
            json,
        } => {
            let cfg = run_config(config.as_deref(), nets, preset.as_deref(), threads, seed)?;
            let method = method
                .map(|m| m.parse::<faradice_sens::Method>())
                .transpose()
                .map_err(anyhow::Error::msg)?;
            let mut file = faradice_sens::ParamFile::read(&params)?;
            if one_sided {
                file.richardson = Some(false);
            }
            let project = Project::load(&structure, tech.as_deref(), top.as_deref())?;
            let res = faradice_sens::run_sensitivity(
                &project,
                &faradice_sens::SensRequest {
                    run: RunRequest {
                        config: cfg,
                        engine: WalkEngineKind::Cpu,
                    },
                    params: file,
                    method,
                    shape_samples,
                },
            )?;
            faradice_sens::write_outputs(&res, &out, &project.design)?;
            print_summary(&res.run.report);
            print_sens(&res.sens);
            eprintln!(
                "solved {} nets, {} parameters, {} deletions and {} additions in {:.2} s → {}",
                res.run.report.nets.len(),
                res.sens.params.len(),
                res.sens.deletions.len(),
                res.sens.additions.len(),
                res.run.elapsed_s,
                out.display()
            );
            if json {
                println!("{}", serde_json::to_string_pretty(&res.sens)?);
            }
        }
        #[cfg(feature = "sens")]
        Command::Whatif {
            result,
            ground,
            float,
            merge,
            swap,
            json,
        } => {
            use faradice_sens::whatif::{Edit, apply};
            let r = ResultFile::read(&result)?;
            let mut edits = Vec::new();
            for s in &swap {
                let (a, b) = pair(s)?;
                edits.push(Edit::Swap { a, b });
            }
            for s in &merge {
                let (a, b) = pair(s)?;
                edits.push(Edit::Merge { a, b });
            }
            edits.extend(ground.into_iter().map(|net| Edit::Ground { net }));
            edits.extend(float.into_iter().map(|net| Edit::Float { net }));
            let w = apply(&r, &edits)?;
            print_whatif(&w);
            if json {
                println!("{}", serde_json::to_string_pretty(&w)?);
            }
        }
        #[cfg(feature = "sens")]
        Command::Map {
            structure,
            nets,
            grid,
            config,
            tech,
            top,
            samples,
            radius,
            threads,
            seed,
            out,
        } => {
            let (a, b) = match nets.split_once(',') {
                Some((a, b)) => (a.to_owned(), b.to_owned()),
                None => (nets.clone(), nets.clone()),
            };
            if !(radius.is_finite() && radius > 0.0) {
                bail!("--radius must be positive");
            }
            let cfg = run_config(config.as_deref(), None, None, threads, seed)?;
            let project = Project::load(&structure, tech.as_deref(), top.as_deref())?;
            let req = faradice_sens::map::MapRequest {
                run: RunRequest {
                    config: cfg,
                    engine: WalkEngineKind::Cpu,
                },
                nets: (a.clone(), b.clone()),
                grid: faradice_sens::map::Grid::parse(&grid)?,
                samples,
                radius,
            };
            let t0 = std::time::Instant::now();
            let (rows, ids) = faradice_sens::map::insertion_map(&project, &req)?;
            std::fs::write(&out, faradice_sens::map::to_csv(&rows, ids, (&a, &b)))
                .with_context(|| format!("writing {}", out.display()))?;
            let best = |f: &dyn Fn(&faradice_sens::map::MapRow) -> f64| {
                rows.iter()
                    .filter(|r| f(r).is_finite())
                    .max_by(|x, y| f(x).abs().total_cmp(&f(y).abs()))
                    .map(|r| (r.p.point, f(r)))
            };
            if let Some((p, v)) = best(&|r| r.grounded) {
                eprintln!("largest |ΔC| grounded: {v:.6e} fF at {p:?}");
            }
            if let Some((p, v)) = best(&|r| r.floating) {
                eprintln!("largest |ΔC| floating: {v:.6e} fF at {p:?}");
            }
            eprintln!(
                "mapped {} points in {:.2} s → {}",
                rows.len(),
                t0.elapsed().as_secs_f64(),
                out.display()
            );
        }
        Command::Report {
            result,
            config,
            out,
            json,
        } => {
            let r = ResultFile::read(&result)?;
            let cfg = match config {
                Some(p) => read_config(&p)?,
                None => Config::parse(&r.config_toml, "result.json:config_toml")?,
            };
            let report = build_report(&r, &cfg.coupling, faradice_engine::ground_report_of(&cfg));
            let dir = out.unwrap_or_else(|| {
                result
                    .parent()
                    .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
            });
            std::fs::create_dir_all(&dir)?;
            faradice_engine::write_reports(&report, &cfg, &dir, "design")?;
            print_summary(&report);
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        }
        Command::Gds { cmd } => match cmd {
            GdsCmd::Trace { input, net, json } => {
                let g = load_gds(&input)?;
                let comps: Vec<&faradice_gds::trace::Component> = match &net {
                    Some(n) => vec![g.traced.component(n).with_context(|| {
                        let close: Vec<&str> = g
                            .traced
                            .components
                            .iter()
                            .map(|c| c.name.as_str())
                            .filter(|c| c.to_lowercase().contains(&n.to_lowercase()))
                            .take(5)
                            .collect();
                        format!("net `{n}` not found (similar: {close:?})")
                    })?],
                    None => g
                        .traced
                        .components
                        .iter()
                        .filter(|c| !c.name.starts_with('$'))
                        .collect(),
                };
                let items: Vec<serde_json::Value> =
                    comps.iter().map(|c| component_json(&g, c)).collect();
                if json {
                    println!("{}", serde_json::to_string_pretty(&items)?);
                } else {
                    let unlabeled = g
                        .traced
                        .components
                        .iter()
                        .filter(|c| c.name.starts_with('$'))
                        .count();
                    println!(
                        "{} components ({} labelled, {unlabeled} unlabelled), {} boxes",
                        g.traced.components.len(),
                        g.traced.components.len() - unlabeled,
                        g.traced.boxes.len()
                    );
                    for v in &items {
                        println!(
                            "{:<24} boxes {:>6}  layers {}  bbox {}",
                            v["net"].as_str().unwrap_or(""),
                            v["boxes"],
                            v["boxes_per_layer"],
                            v["bbox_um"]
                        );
                    }
                }
            }
            GdsCmd::Export { input, out, nets } => {
                let g = load_gds(&input)?;
                let keep = |name: &str| match &nets {
                    Some(globs) => globs.iter().any(|p| faradice_io::glob_match(p, name)),
                    None => !name.starts_with('$'),
                };
                // One entry per (net, layer or via), tagged with the layer's name
                // so that conformal rules can select its boxes.
                let mut conductors = Vec::new();
                for c in g.traced.components.iter().filter(|c| keep(&c.name)) {
                    let mut spans: Vec<usize> =
                        c.boxes.iter().map(|&b| g.traced.box_span[b]).collect();
                    spans.sort_unstable();
                    spans.dedup();
                    for span in spans {
                        let boxes = c
                            .boxes
                            .iter()
                            .filter(|&&b| g.traced.box_span[b] == span)
                            .map(|&b| {
                                let x = g.traced.boxes[b];
                                let (lo, hi) = (x.min(), x.max());
                                [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
                            })
                            .collect();
                        conductors.push(faradice_io::structure::Conductor {
                            net: c.name.clone(),
                            layer: Some(g.tech.span_name(span).to_owned()),
                            boxes,
                        });
                    }
                }
                let conformal = g
                    .tech
                    .layers
                    .iter()
                    .filter_map(|l| {
                        l.conformal.map(|c| faradice_io::structure::Conformal {
                            layer: l.name.clone(),
                            side: c.side,
                            top: c.top,
                            eps_r: c.eps_r,
                        })
                    })
                    .collect();
                let file = faradice_io::structure::StructureFile {
                    schema_version: 1,
                    units: "um".into(),
                    dielectric: faradice_io::structure::Dielectric {
                        eps_r: g.tech.eps_r,
                        layers: g
                            .tech
                            .dielectric
                            .iter()
                            .map(|d| faradice_io::structure::Layer {
                                z_top: d.z_top,
                                eps_r: d.eps_r,
                            })
                            .collect(),
                        boxes: Vec::new(),
                        conformal,
                    },
                    substrate_z: Some(g.tech.substrate_z),
                    conductors,
                };
                let text = format!(
                    "# Exported by faradice from {} (tech: {}; substrate_z = {} µm)\n{}",
                    input.gds.display(),
                    g.tech.name,
                    g.tech.substrate_z,
                    toml::to_string(&file)?
                );
                std::fs::write(&out, text)?;
                let nets = file
                    .conductors
                    .iter()
                    .map(|c| c.net.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                eprintln!("wrote {nets} nets to {}", out.display());
            }
        },
        Command::Check { structure, config } => {
            let mut cfg = load_config(config.as_deref())?;
            // The same loading as `solve`, so the structure's substrate height
            // applies to the configuration.
            let st = Project::load(&structure, None, None)?.build(&mut cfg)?;
            let resolved = cfg.resolve(&st.geometry, st.eps_r, &st.layers, &st.regions)?;
            let g = &st.geometry;
            let b = g.bounds();
            println!(
                "nets: {}  boxes: {}  eps_r: {}  dielectric layers: {}",
                g.num_nets(),
                g.boxes().len(),
                st.eps_r,
                st.layers.len()
            );
            if !st.regions.boxes.is_empty() || !st.regions.conformal.is_empty() {
                let rules: Vec<String> = st
                    .regions
                    .conformal
                    .iter()
                    .map(|r| format!("{} ({} boxes)", r.layer, r.boxes.len()))
                    .collect();
                println!(
                    "dielectric boxes: {}  conformal rules: {}",
                    st.regions.boxes.len(),
                    if rules.is_empty() {
                        "none".to_owned()
                    } else {
                        rules.join(", ")
                    }
                );
            }
            println!("bounds: {:?} .. {:?} µm", b.min(), b.max());
            let names = |v: &[faradice_core::NetId]| {
                v.iter()
                    .map(|&n| g.net_name(n))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            println!(
                "targets ({}): {}",
                resolved.targets.len(),
                names(&resolved.targets)
            );
            println!("ground nets: {}", names(&resolved.solver.ground_nets));
            println!("boundary: {:?}", resolved.solver.boundary);
            for w in &resolved.warnings {
                println!("warning: {w}");
            }
        }
    }
    Ok(())
}
