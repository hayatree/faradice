//! Application state, messages, update and view.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use faradice_engine::{Job, JobEvent, JobState, Project, RunOutput, RunRequest, WalkEngineKind};
use faradice_io::config::{Config, Preset, read_config};
use faradice_io::report::Report;
use iced::futures::channel::{mpsc, oneshot};
use iced::widget::{
    button, canvas, checkbox, column, container, pick_list, row, rule, scrollable, shader, space,
    text, text_input, tooltip,
};
use iced::{Element, Fill, Subscription, Task, Theme, keyboard, window};

use crate::cli::{Equivalent, preset_name};
use crate::matrix::{self, Columns};
use crate::plot::Convergence;
use crate::table::{self, Rows, Scroll, col};
use crate::viewport::Scene;

/// Runs `f` on its own thread and resolves with its result, so blocking work
/// (GDS tracing, file I/O) never runs on the UI or executor threads.
fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> impl Future<Output = T> {
    let (tx, rx) = oneshot::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    async move { rx.await.expect("worker thread ended without a result") }
}

const PRESETS: [PresetChoice; 4] = [
    PresetChoice(Preset::Fast),
    PresetChoice(Preset::Default),
    PresetChoice(Preset::Signoff),
    PresetChoice(Preset::Reference),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresetChoice(Preset);

impl std::fmt::Display for PresetChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            Preset::Fast => "fast — 2 % total, 10 % couplings",
            Preset::Default => "default — 1 % / 5 %",
            Preset::Signoff => "signoff — 0.5 % / 3 %",
            Preset::Reference => "reference — 0.1 % / 1 %",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineChoice(WalkEngineKind);

impl std::fmt::Display for EngineChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0 == WalkEngineKind::Gpu {
            "GPU (wgpu)"
        } else {
            "CPU"
        })
    }
}

fn engines() -> Vec<EngineChoice> {
    let mut v = vec![EngineChoice(WalkEngineKind::Cpu)];
    if cfg!(feature = "gpu") {
        v.push(EngineChoice(WalkEngineKind::Gpu));
    }
    v
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Progress,
    Results,
    Matrix,
    View3d,
    Messages,
}

/// One row of the net picker.
#[derive(Debug, Clone)]
struct NetRow {
    name: String,
    selected: bool,
}

/// Live state of one target during a run.
#[derive(Debug, Clone, Default)]
struct Live {
    walks: u64,
    value: f64,
    rel: f64,
    finished: bool,
    converged: bool,
    history: Vec<(u64, f64)>,
}

/// A run in progress or finished.
struct RunView {
    job: Option<Job>,
    started: Instant,
    targets: Vec<String>,
    live: Vec<Live>,
    warnings: Vec<String>,
    state: JobState,
    error: Option<String>,
    partial: Option<Report>,
    partial_version: u64,
    last_refresh: Option<Instant>,
    output: Option<Box<RunOutput>>,
    elapsed: f64,
    command: String,
    tolerance: f64,
    confidence: f64,
    /// Columns of the matrix view, rebuilt whenever the report changes.
    columns: Columns,
}

impl RunView {
    fn report(&self) -> Option<&Report> {
        self.output
            .as_ref()
            .map(|o| &o.report)
            .or(self.partial.as_ref())
    }

    fn running(&self) -> bool {
        !self.state.is_terminal()
    }

    fn report_changed(&mut self) {
        self.columns = self
            .report()
            .map(|r| Columns::new(r, matrix::MAX_COLUMNS))
            .unwrap_or_default();
    }
}

pub(crate) struct App {
    input: Option<PathBuf>,
    tech: Option<PathBuf>,
    config_path: Option<PathBuf>,
    base_config: Config,
    project: Option<Arc<Project>>,
    loading: bool,
    nets: Vec<NetRow>,
    filter: String,
    preset: PresetChoice,
    engine: EngineChoice,
    seed: String,
    threads: String,
    run: Option<RunView>,
    tab: Tab,
    /// Selected net (by name, so it maps into every table).
    selected: Option<String>,
    /// Input being loaded (committed to `input` only when loading succeeds).
    pending_input: Option<PathBuf>,
    /// Generation of the latest load; older `Loaded` messages are ignored.
    load_gen: u64,
    /// Id of the latest run; events of older jobs are ignored.
    run_id: u64,
    scroll_progress: Scroll,
    scroll_results: Scroll,
    scroll_couplings: Scroll,
    scroll_matrix: Scroll,
    scene: Scene,
    status: String,
    /// Self-test mode (`FARADICE_GUI_SHOTS=<dir>`): run, screenshot each
    /// view into `<dir>`, exit.
    shots: Option<PathBuf>,
    /// Frame benchmark (`FARADICE_GUI_BENCH=1`): spin the 3D view, report
    /// frame intervals, exit.
    bench: Option<Vec<Instant>>,
}

/// Id of the coupling table's scrollable (reset when the selection changes).
const COUPLINGS_ID: &str = "couplings";

/// Steps of the self-test after a run finished.
const SHOTS: [(&str, Tab, bool); 5] = [
    ("results", Tab::Results, false),
    ("results_selected", Tab::Results, true),
    ("matrix", Tab::Matrix, true),
    ("view3d", Tab::View3d, true),
    ("progress", Tab::Progress, true),
];

#[derive(Debug, Clone)]
pub(crate) enum Message {
    PickInput,
    PickTech,
    PickConfig,
    InputPicked(Option<PathBuf>),
    TechPicked(Option<PathBuf>),
    ConfigPicked(Option<PathBuf>),
    Loaded(u64, PathBuf, Result<(Arc<Project>, Scene), String>),
    Filter(String),
    ToggleNet(usize, bool),
    SelectVisible(bool),
    Preset(PresetChoice),
    Engine(EngineChoice),
    Seed(String),
    Threads(String),
    Run,
    Cancel,
    Job(u64, JobEvent),
    Tab(Tab),
    Select(String),
    ScrollProgress(scrollable::Viewport),
    ScrollResults(scrollable::Viewport),
    ScrollCouplings(scrollable::Viewport),
    ScrollMatrix(scrollable::Viewport),
    CopyCommand,
    Export,
    ExportTo(Option<PathBuf>, Box<RunOutput>, String),
    Exported(Result<PathBuf, String>),
    FileDropped(PathBuf),
    DimOthers(bool),
    Frame(Instant),
    Shot(usize),
    Shot2(usize, window::Screenshot),
}

impl App {
    pub(crate) fn boot(args: crate::Args) -> (Self, Task<Message>) {
        let mut app = Self {
            input: None,
            tech: args.tech,
            config_path: None,
            base_config: Config::default(),
            project: None,
            loading: false,
            nets: Vec::new(),
            filter: String::new(),
            preset: PresetChoice(Preset::Default),
            engine: EngineChoice(WalkEngineKind::Cpu),
            seed: "1".into(),
            threads: "0".into(),
            run: None,
            tab: Tab::Progress,
            selected: None,
            pending_input: None,
            load_gen: 0,
            run_id: 0,
            scroll_progress: Scroll::default(),
            scroll_results: Scroll::default(),
            scroll_couplings: Scroll::default(),
            scroll_matrix: Scroll::default(),
            scene: Scene::default(),
            shots: std::env::var_os("FARADICE_GUI_SHOTS").map(PathBuf::from),
            bench: std::env::var_os("FARADICE_GUI_BENCH").map(|_| Vec::new()),
            status: "Open a structure (.faradice.toml) or a GDSII layout to start. Files can also be dropped on the window.".into(),
        };
        if let Some(c) = args.config {
            app.set_config(c);
        }
        let task = match args.input {
            Some(p) => app.set_input(p),
            None => Task::none(),
        };
        (app, task)
    }

    pub(crate) fn title(&self) -> String {
        match &self.input {
            Some(p) => format!(
                "{} — faradice",
                p.file_name().map_or_else(
                    || p.display().to_string(),
                    |n| n.to_string_lossy().into_owned()
                )
            ),
            None => "faradice".into(),
        }
    }

    pub(crate) fn theme(&self) -> Option<Theme> {
        // `None`: follow the desktop's light/dark preference.
        None
    }

    pub(crate) fn subscription(&self) -> Subscription<Message> {
        let frames = if self.bench.is_some() && self.scene.len() > 0 {
            window::frames().map(Message::Frame)
        } else {
            Subscription::none()
        };
        let events = iced::event::listen_with(|event, status, _id| match event {
            iced::Event::Window(window::Event::FileDropped(p)) => Some(Message::FileDropped(p)),
            iced::Event::Keyboard(keyboard::Event::KeyPressed { key, modifiers, .. })
                if modifiers.command() && status == iced::event::Status::Ignored =>
            {
                match key.as_ref() {
                    keyboard::Key::Character("o") => Some(Message::PickInput),
                    keyboard::Key::Named(keyboard::key::Named::Enter) => Some(Message::Run),
                    keyboard::Key::Character("e") => Some(Message::Export),
                    _ => None,
                }
            }
            iced::Event::Keyboard(keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(keyboard::key::Named::Escape),
                ..
            }) => Some(Message::Cancel),
            _ => None,
        });
        Subscription::batch([events, frames])
    }

    fn set_config(&mut self, path: PathBuf) {
        match read_config(&path) {
            Ok(c) => {
                self.preset = PresetChoice(c.accuracy.preset);
                self.seed = c.run.seed.to_string();
                self.threads = c.run.threads.to_string();
                self.base_config = c;
                self.status = format!("Configuration {} loaded.", path.display());
                self.config_path = Some(path);
            }
            Err(e) => self.status = format!("Cannot read configuration: {e}"),
        }
    }

    fn set_input(&mut self, path: PathBuf) -> Task<Message> {
        let gds = faradice_engine::is_gds(&path);
        self.pending_input = Some(path);
        if gds && self.tech.is_none() {
            self.status = "GDSII input needs a technology file: choose one with “Tech…”.".into();
            return Task::none();
        }
        self.load()
    }

    /// The input a (re)load applies to: the pending one, else the current one.
    fn load_target(&self) -> Option<&Path> {
        self.pending_input.as_deref().or(self.input.as_deref())
    }

    fn load(&mut self) -> Task<Message> {
        let Some(input) = self.load_target().map(Path::to_path_buf) else {
            return Task::none();
        };
        self.load_gen += 1;
        let generation = self.load_gen;
        let path = input.clone();
        let tech = self
            .tech
            .clone()
            .filter(|_| faradice_engine::is_gds(&input));
        self.loading = true;
        self.status = format!("Loading {} …", input.display());
        let mut config = self.base_config.clone();
        Task::perform(
            blocking(move || {
                let p = Project::load(&input, tech.as_deref(), None).map_err(|e| e.to_string())?;
                // Show the geometry right away; targets are marked after a run.
                let scene = p
                    .build(&mut config)
                    .map(|st| Scene::new(&st.geometry, |_| true))
                    .unwrap_or_default();
                Ok((Arc::new(p), scene))
            }),
            move |r| Message::Loaded(generation, path.clone(), r),
        )
    }

    fn pick(
        title: &'static str,
        filters: &'static [(&'static str, &'static [&'static str])],
    ) -> Task<Option<PathBuf>> {
        Task::perform(
            async move {
                let mut d = rfd::AsyncFileDialog::new().set_title(title);
                for (name, ext) in filters {
                    d = d.add_filter(*name, ext);
                }
                d.pick_file().await.map(|f| f.path().to_owned())
            },
            |p| p,
        )
    }

    fn visible_nets(&self) -> impl Iterator<Item = usize> + '_ {
        let f = self.filter.to_lowercase();
        self.nets
            .iter()
            .enumerate()
            .filter(move |(_, n)| {
                f.is_empty()
                    || n.name.to_lowercase().contains(&f)
                    || faradice_io::glob_match(&self.filter, &n.name)
            })
            .map(|(i, _)| i)
    }

    fn selected_globs(&self) -> Option<Vec<String>> {
        let sel: Vec<&NetRow> = self.nets.iter().filter(|n| n.selected).collect();
        if sel.len() == self.nets.len() {
            None // all nets: keep the configuration's `extract.nets`
        } else {
            Some(
                sel.iter()
                    .map(|n| faradice_io::glob_escape(&n.name))
                    .collect(),
            )
        }
    }

    fn request(&self) -> Result<(RunRequest, String), String> {
        let mut config = self.base_config.clone();
        let globs = self.selected_globs();
        if let Some(g) = &globs {
            if g.is_empty() {
                return Err("Select at least one net.".into());
            }
            config.extract.nets.clone_from(g);
        }
        config.accuracy.preset = self.preset.0;
        config.run.seed = self
            .seed
            .trim()
            .parse()
            .map_err(|_| "Seed must be a non-negative integer.".to_owned())?;
        config.run.threads =
            self.threads.trim().parse().map_err(|_| {
                "Threads must be a non-negative integer (0 = all cores).".to_owned()
            })?;
        let input = self.input.as_deref().ok_or("No input loaded.")?;
        let command = Equivalent {
            input,
            tech: self
                .tech
                .as_deref()
                .filter(|_| faradice_engine::is_gds(input)),
            config: self.config_path.as_deref(),
            nets: globs.as_deref(),
            preset: config.accuracy.preset,
            seed: config.run.seed,
            threads: config.run.threads,
            engine: self.engine.0,
        }
        .command();
        Ok((
            RunRequest {
                config,
                engine: self.engine.0,
            },
            command,
        ))
    }

    fn start(&mut self) -> Task<Message> {
        if self.run.as_ref().is_some_and(RunView::running) || self.loading {
            return Task::none();
        }
        let Some(project) = self.project.clone() else {
            self.status = "Load an input first.".into();
            return Task::none();
        };
        let (req, command) = match self.request() {
            Ok(r) => r,
            Err(e) => {
                self.status = e;
                return Task::none();
            }
        };
        let acc = req.config.accuracy();
        let (tolerance, confidence) = (acc.total_rel_tol, acc.confidence_sigma);
        self.run_id += 1;
        let id = self.run_id;
        let (tx, rx) = mpsc::unbounded();
        let job = Job::start(project, req, move |e| {
            let _ = tx.unbounded_send(e);
        });
        self.run = Some(RunView {
            job: Some(job),
            started: Instant::now(),
            targets: Vec::new(),
            live: Vec::new(),
            warnings: Vec::new(),
            state: JobState::Queued,
            error: None,
            partial: None,
            partial_version: 0,
            last_refresh: None,
            output: None,
            elapsed: 0.0,
            columns: Columns::default(),
            command,
            tolerance,
            confidence,
        });
        self.selected = None;
        self.tab = Tab::Progress;
        self.status = "Running…".into();
        Task::run(rx, move |e| Message::Job(id, e))
    }

    /// Cancels the running job, if any (its remaining events are ignored).
    fn abandon_run(&mut self) {
        if let Some(job) = self.run.as_ref().and_then(|r| r.job.as_ref()) {
            job.cancel();
        }
        self.run_id += 1;
    }

    fn on_job(&mut self, e: JobEvent) {
        let Some(run) = self.run.as_mut() else { return };
        run.state = e.state();
        match e {
            JobEvent::Started { targets, warnings } => {
                run.live = vec![Live::default(); targets.len()];
                run.targets = targets;
                run.warnings = warnings;
            }
            JobEvent::Progress(p) => {
                if let Some(l) = run.live.get_mut(p.target) {
                    l.walks = p.walks;
                    l.value = p.total.value;
                    l.rel = run.confidence * p.total.rel_err();
                    l.finished = p.finished;
                    l.converged = p.converged;
                    if l.rel.is_finite() {
                        l.history.push((p.walks, l.rel));
                    }
                }
                // Refresh the partial report at most 4 times per second.
                let now = Instant::now();
                if run
                    .last_refresh
                    .is_none_or(|t| now.duration_since(t).as_millis() >= 250)
                {
                    run.last_refresh = Some(now);
                    if let Some(job) = &run.job {
                        let v = job.version();
                        if v != run.partial_version {
                            run.partial_version = v;
                            run.partial = job.partial().map(|p| p.report);
                            run.report_changed();
                        }
                    }
                }
                run.elapsed = run.started.elapsed().as_secs_f64();
            }
            JobEvent::Finished(out) => {
                run.elapsed = out.elapsed_s;
                run.warnings.clone_from(&out.report.warnings);
                let label = match run.state {
                    JobState::Succeeded => "converged",
                    JobState::NotConverged => "finished, some nets not converged",
                    JobState::Cancelled => "cancelled (partial results)",
                    _ => "finished",
                };
                let engine = out
                    .engine_info
                    .as_deref()
                    .map(|i| format!(" on {i}"))
                    .unwrap_or_default();
                self.status = format!(
                    "{} nets {label} in {:.2} s{engine}.",
                    out.report.nets.len(),
                    out.elapsed_s
                );
                let targets: std::collections::HashSet<&str> =
                    out.targets.iter().map(String::as_str).collect();
                let g = &out.geometry;
                self.scene = Scene::new(g, |i| {
                    #[allow(clippy::cast_possible_truncation)]
                    targets.contains(g.net_name(faradice_core::NetId::new(i as u32)))
                });
                self.scene.selected = None;
                run.output = Some(out);
                run.report_changed();
                run.job = None;
                self.tab = Tab::Results;
            }
            JobEvent::Failed(err) => {
                run.error = Some(err.to_string());
                run.job = None;
                self.status = format!("Run failed: {err}");
                self.tab = Tab::Messages;
            }
            _ => {}
        }
    }

    pub(crate) fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PickInput => {
                return Self::pick(
                    "Open structure or layout",
                    &[(
                        "faradice structure or GDSII",
                        &["toml", "gds", "gds2", "gdsii"],
                    )],
                )
                .map(Message::InputPicked);
            }
            Message::PickTech => {
                return Self::pick("Technology file", &[("faradice tech", &["toml"])])
                    .map(Message::TechPicked);
            }
            Message::PickConfig => {
                return Self::pick("Run configuration", &[("faradice config", &["toml"])])
                    .map(Message::ConfigPicked);
            }
            Message::InputPicked(Some(p)) | Message::FileDropped(p) => {
                return self.dropped_or_picked(p);
            }
            Message::TechPicked(Some(p)) => {
                self.tech = Some(p);
                if self.load_target().is_some_and(faradice_engine::is_gds) {
                    return self.load();
                }
            }
            Message::ConfigPicked(Some(p)) => self.set_config(p),
            Message::InputPicked(None)
            | Message::TechPicked(None)
            | Message::ConfigPicked(None)
            | Message::ExportTo(None, ..) => {}
            Message::Loaded(generation, _, _) if generation != self.load_gen => {}
            Message::Loaded(_, path, r) => {
                self.loading = false;
                match r {
                    Ok((p, scene)) => {
                        self.abandon_run();
                        self.input = Some(path);
                        self.pending_input = None;
                        self.selected = None;
                        self.nets = p
                            .labelled_nets()
                            .into_iter()
                            .map(|name| NetRow {
                                name,
                                selected: true,
                            })
                            .collect();
                        let diag = p.diagnostics.len();
                        self.status = format!(
                            "{}: {} nets{}.",
                            p.design,
                            self.nets.len(),
                            if diag > 0 {
                                format!(", {diag} load warnings (see Messages)")
                            } else {
                                String::new()
                            }
                        );
                        self.project = Some(p);
                        self.run = None;
                        self.scene = scene;
                        if self.shots.is_some() {
                            return self.start();
                        }
                    }
                    Err(e) => {
                        // Keep the previous project and input consistent.
                        self.pending_input = None;
                        self.status = format!("Cannot load {}: {e}", path.display());
                    }
                }
            }
            Message::Filter(f) => self.filter = f,
            Message::ToggleNet(i, on) => {
                if let Some(n) = self.nets.get_mut(i) {
                    n.selected = on;
                }
            }
            Message::SelectVisible(on) => {
                let vis: Vec<usize> = self.visible_nets().collect();
                for i in vis {
                    self.nets[i].selected = on;
                }
            }
            Message::Preset(p) => self.preset = p,
            Message::Engine(e) => self.engine = e,
            Message::Seed(s) => self.seed = s,
            Message::Threads(s) => self.threads = s,
            Message::Run => return self.start(),
            Message::Cancel => {
                if let Some(job) = self.run.as_ref().and_then(|r| r.job.as_ref()) {
                    job.cancel();
                    self.status = "Cancelling: finishing the current wave…".into();
                }
            }
            Message::Job(id, _) if id != self.run_id => {}
            Message::Job(_, e) => {
                let finished = matches!(e, JobEvent::Finished(_));
                self.on_job(e);
                if finished && self.shots.is_some() {
                    return Task::done(Message::Shot(0));
                }
            }
            Message::Tab(t) => self.tab = t,
            Message::Select(name) => {
                if let Some(out) = self.run.as_ref().and_then(|r| r.output.as_ref()) {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        self.scene.selected = out.geometry.net_id(&name).map(|n| n.index() as u32);
                    }
                }
                self.selected = Some(name);
                self.scroll_couplings = Scroll::default();
                return iced::widget::operation::snap_to(
                    COUPLINGS_ID,
                    iced::widget::operation::RelativeOffset::START,
                );
            }
            Message::ScrollProgress(v) => self.scroll_progress.set(&v),
            Message::ScrollResults(v) => self.scroll_results.set(&v),
            Message::ScrollCouplings(v) => self.scroll_couplings.set(&v),
            Message::ScrollMatrix(v) => self.scroll_matrix.set(&v),
            Message::CopyCommand => {
                if let Ok((_, cmd)) = self.request() {
                    self.status = "CLI command copied to the clipboard.".into();
                    return iced::clipboard::write(cmd);
                }
            }
            Message::Export => {
                // Bind the export to the results shown now, not to whatever
                // is current when the dialog closes.
                if let (Some(out), Some(p)) = (
                    self.run.as_ref().and_then(|r| r.output.clone()),
                    self.project.as_ref(),
                ) {
                    let design = p.design.clone();
                    return Task::perform(
                        async {
                            rfd::AsyncFileDialog::new()
                                .set_title("Export results to folder")
                                .pick_folder()
                                .await
                                .map(|f| f.path().to_owned())
                        },
                        move |dir| Message::ExportTo(dir, out.clone(), design.clone()),
                    );
                }
            }
            Message::ExportTo(Some(dir), mut out, design) => {
                // Always write every report format from the GUI.
                out.config.output.formats = vec!["json".into(), "csv".into(), "spef".into()];
                return Task::perform(
                    blocking(move || {
                        faradice_engine::write_outputs(&out, &dir, &design)
                            .map(|()| dir)
                            .map_err(|e| e.to_string())
                    }),
                    Message::Exported,
                );
            }
            Message::Exported(r) => {
                self.status = match r {
                    Ok(d) => format!(
                        "Exported result.json, report.json, report.csv and SPEF to {}.",
                        d.display()
                    ),
                    Err(e) => format!("Export failed: {e}"),
                }
            }
            Message::DimOthers(on) => self.scene.dim_non_targets = on,
            Message::Frame(at) => {
                self.tab = Tab::View3d;
                let Some(times) = self.bench.as_mut() else {
                    return Task::none();
                };
                times.push(at);
                self.scene.spin += 0.01;
                if times.len() > 300 {
                    // Skip warm-up frames (pipeline creation, upload).
                    let mut d: Vec<f64> = times[60..]
                        .windows(2)
                        .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1e3)
                        .collect();
                    d.sort_by(f64::total_cmp);
                    let q = |p: f64| d[((d.len() - 1) as f64 * p).round() as usize];
                    eprintln!(
                        "frame bench: {} boxes, {} frames: p50 {:.2} ms, p95 {:.2} ms, p99 {:.2} ms, max {:.2} ms",
                        self.scene.len(),
                        d.len(),
                        q(0.5),
                        q(0.95),
                        q(0.99),
                        d[d.len() - 1]
                    );
                    return iced::exit();
                }
            }
            Message::Shot(i) => {
                let Some((_, tab, select)) = SHOTS.get(i).copied() else {
                    return iced::exit();
                };
                self.tab = tab;
                if select && self.selected.is_none() {
                    let first = self
                        .run
                        .as_ref()
                        .and_then(RunView::report)
                        .and_then(|r| r.nets.first())
                        .map(|n| n.net.clone());
                    if let Some(name) = first {
                        return self
                            .update(Message::Select(name))
                            .chain(Task::done(Message::Shot(i)));
                    }
                }
                // Let the view render before capturing it.
                return Task::perform(
                    blocking(|| std::thread::sleep(std::time::Duration::from_millis(600))),
                    |()| (),
                )
                .then(|()| window::latest())
                .then(|id| id.map_or_else(Task::none, window::screenshot))
                .map(move |s| Message::Shot2(i, s));
            }
            Message::Shot2(i, s) => {
                if let Some(dir) = &self.shots {
                    let name = SHOTS[i].0;
                    let mut ppm =
                        format!("P6\n{} {}\n255\n", s.size.width, s.size.height).into_bytes();
                    ppm.extend(
                        s.rgba
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|p| [p[0], p[1], p[2]]),
                    );
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(dir.join(format!("{name}.ppm")), ppm);
                }
                return Task::done(Message::Shot(i + 1));
            }
        }
        Task::none()
    }

    fn dropped_or_picked(&mut self, p: PathBuf) -> Task<Message> {
        let name = p.to_string_lossy().to_lowercase();
        if name.ends_with(".tech.toml") {
            self.tech = Some(p);
            if self.load_target().is_some_and(faradice_engine::is_gds) {
                return self.load();
            }
            Task::none()
        } else if name.ends_with(".toml") && !name.ends_with(".faradice.toml") {
            self.set_config(p);
            Task::none()
        } else {
            self.set_input(p)
        }
    }

    // ----- view -----

    pub(crate) fn view(&self) -> Element<'_, Message> {
        let body = row![self.sidebar(), rule::vertical(1), self.main_area()].height(Fill);
        column![
            self.toolbar(),
            rule::horizontal(1),
            body,
            rule::horizontal(1),
            self.status_bar()
        ]
        .into()
    }

    fn toolbar(&self) -> Element<'_, Message> {
        let running = self.run.as_ref().is_some_and(RunView::running);
        let can_run = self.project.is_some() && !running && !self.loading;
        let file = |p: &Option<PathBuf>| {
            p.as_deref()
                .and_then(Path::file_name)
                .map_or_else(|| "—".to_owned(), |n| n.to_string_lossy().into_owned())
        };
        let tip = |w: iced::widget::Button<'static, Message>, t: &'static str| {
            tooltip(
                w,
                container(text(t).size(12))
                    .padding(6)
                    .style(container::rounded_box),
                tooltip::Position::Bottom,
            )
        };
        row![
            tip(
                button("Open…").on_press(Message::PickInput),
                "Open a structure or GDSII file (Ctrl+O)"
            ),
            text(file(&self.input)).size(13),
            tip(
                button("Tech…")
                    .style(button::secondary)
                    .on_press(Message::PickTech),
                "Technology file for GDSII input"
            ),
            text(file(&self.tech)).size(13),
            tip(
                button("Config…")
                    .style(button::secondary)
                    .on_press(Message::PickConfig),
                "Optional run configuration (faradice.toml)"
            ),
            text(file(&self.config_path)).size(13),
            space().width(Fill),
            if running {
                tip(
                    button("Cancel")
                        .style(button::danger)
                        .on_press(Message::Cancel),
                    "Stop at the next wave; keep partial results (Esc)",
                )
            } else {
                tip(
                    button("Run")
                        .style(button::success)
                        .on_press_maybe(can_run.then_some(Message::Run)),
                    "Extract the selected nets (Ctrl+Enter)",
                )
            },
            tip(
                button("Export…").style(button::secondary).on_press_maybe(
                    self.run
                        .as_ref()
                        .and_then(|r| r.output.as_ref())
                        .map(|_| Message::Export)
                ),
                "Write result.json, report.json, report.csv and SPEF (Ctrl+E)"
            ),
        ]
        .spacing(8)
        .padding(8)
        .align_y(iced::Center)
        .into()
    }

    fn sidebar(&self) -> Element<'_, Message> {
        let vis: Vec<usize> = self.visible_nets().collect();
        let n_sel = self.nets.iter().filter(|n| n.selected).count();
        let list = column(vis.iter().map(|&i| {
            let n = &self.nets[i];
            checkbox(n.selected)
                .label(n.name.as_str())
                .text_size(13)
                .on_toggle(move |on| Message::ToggleNet(i, on))
                .into()
        }))
        .spacing(4);
        let settings = column![
            text("Accuracy").size(13),
            pick_list(&PRESETS[..], Some(self.preset), Message::Preset)
                .text_size(13)
                .width(Fill),
            row![
                column![
                    text("Seed").size(13),
                    text_input("1", &self.seed).on_input(Message::Seed).size(13)
                ]
                .spacing(4),
                column![
                    text("Threads").size(13),
                    text_input("0 = all", &self.threads)
                        .on_input(Message::Threads)
                        .size(13)
                ]
                .spacing(4),
            ]
            .spacing(8),
            text("Walk engine").size(13),
            pick_list(engines(), Some(self.engine), Message::Engine)
                .text_size(13)
                .width(Fill),
            button(text("Copy CLI command").size(13))
                .style(button::secondary)
                .on_press_maybe(self.project.as_ref().map(|_| Message::CopyCommand)),
        ]
        .spacing(6);
        container(
            column![
                text(format!("Nets ({n_sel}/{} selected)", self.nets.len())).size(15),
                text_input("Filter (text or glob)…", &self.filter)
                    .on_input(Message::Filter)
                    .size(13),
                row![
                    button(text("All").size(12))
                        .style(button::secondary)
                        .on_press(Message::SelectVisible(true)),
                    button(text("None").size(12))
                        .style(button::secondary)
                        .on_press(Message::SelectVisible(false)),
                ]
                .spacing(6),
                scrollable(list).height(Fill).width(Fill),
                rule::horizontal(1),
                settings,
            ]
            .spacing(8),
        )
        .width(300)
        .padding(10)
        .into()
    }

    fn tabs(&self) -> Element<'_, Message> {
        let t = |label: &'static str, tab: Tab| {
            button(text(label).size(13))
                .style(if self.tab == tab {
                    button::primary
                } else {
                    button::text
                })
                .on_press(Message::Tab(tab))
        };
        let msgs = self
            .run
            .as_ref()
            .map_or(0, |r| r.warnings.len() + usize::from(r.error.is_some()))
            + self.project.as_ref().map_or(0, |p| p.diagnostics.len());
        row![
            t("Progress", Tab::Progress),
            t("Results", Tab::Results),
            t("Matrix", Tab::Matrix),
            t("3D view", Tab::View3d),
            button(
                text(if msgs > 0 {
                    format!("Messages ({msgs})")
                } else {
                    "Messages".into()
                })
                .size(13)
            )
            .style(if self.tab == Tab::Messages {
                button::primary
            } else {
                button::text
            })
            .on_press(Message::Tab(Tab::Messages)),
        ]
        .spacing(4)
        .padding([6, 8])
        .into()
    }

    fn main_area(&self) -> Element<'_, Message> {
        let content = match self.tab {
            Tab::Progress => self.progress_view(),
            Tab::Results => self.results_view(),
            Tab::Matrix => self.matrix_view(),
            Tab::View3d => self.view_3d(),
            Tab::Messages => self.messages_view(),
        };
        column![
            self.tabs(),
            rule::horizontal(1),
            container(content).padding(8).height(Fill)
        ]
        .width(Fill)
        .into()
    }

    fn progress_view(&self) -> Element<'_, Message> {
        let Some(run) = &self.run else {
            return table::empty(
                "No run yet",
                "Select nets on the left and press Run (Ctrl+Enter).",
            );
        };
        let cols = [
            col("net", 4, false),
            col("walks", 2, true),
            col("C total [fF]", 3, true),
            col("error", 2, true),
            col("target", 2, true),
            col("state", 3, false),
        ];
        let tol = run.tolerance;
        let sel = self
            .selected
            .as_ref()
            .and_then(|s| run.targets.iter().position(|t| t == s));
        let tbl = table::view(
            &cols,
            Rows {
                len: run.live.len(),
                scroll: self.scroll_progress,
                selected: sel,
                id: None,
            },
            |i| {
                let l = &run.live[i];
                let state = if l.finished {
                    if l.converged {
                        "converged"
                    } else {
                        "not converged"
                    }
                } else if l.walks > 0 {
                    "running"
                } else {
                    "queued"
                };
                vec![
                    run.targets[i].clone(),
                    l.walks.to_string(),
                    if l.walks > 0 {
                        format!("{:.5}", l.value)
                    } else {
                        "—".into()
                    },
                    if l.rel.is_finite() && l.walks > 0 {
                        format!("{:.3} %", 100.0 * l.rel)
                    } else {
                        "—".into()
                    },
                    format!("{:.2} %", 100.0 * tol),
                    state.into(),
                ]
            },
            |i| Message::Select(run.targets[i].clone()),
            Message::ScrollProgress,
        );
        let done = run.live.iter().filter(|l| l.finished).count();
        let header = text(format!(
            "{done}/{} nets finished · {:.1} s · error at {}σ · {}",
            run.live.len(),
            run.elapsed,
            run.confidence,
            run.command
        ))
        .size(12);
        let plot = sel
            .and_then(|i| run.live.get(i))
            .map(|l| Convergence {
                points: l.history.clone(),
                target: Some(tol),
            })
            .unwrap_or_default();
        column![
            header,
            tbl,
            container(canvas(plot).width(Fill).height(220)).style(container::bordered_box),
        ]
        .spacing(8)
        .into()
    }

    fn results_view(&self) -> Element<'_, Message> {
        let Some(report) = self.run.as_ref().and_then(RunView::report) else {
            return table::empty(
                "No results yet",
                "Results appear here while the run progresses.",
            );
        };
        let cols = [
            col("net", 4, false),
            col("C total [fF]", 3, true),
            col("σ [%]", 2, true),
            col("C ground [fF]", 3, true),
            col("walks", 3, true),
            col("ok", 1, false),
            col("top coupling", 5, false),
        ];
        let sel = self
            .selected
            .as_ref()
            .and_then(|s| report.nets.iter().position(|n| &n.net == s));
        let tbl = table::view(
            &cols,
            Rows {
                len: report.nets.len(),
                scroll: self.scroll_results,
                selected: sel,
                id: None,
            },
            |i| {
                let n = &report.nets[i];
                vec![
                    n.net.clone(),
                    format!("{:.5}", n.total.value),
                    format!(
                        "{:.3}",
                        100.0 * n.total.sigma / n.total.value.abs().max(f64::MIN_POSITIVE)
                    ),
                    n.ground
                        .map_or_else(|| "—".into(), |g| format!("{:.5}", g.value)),
                    n.walks.to_string(),
                    if n.converged {
                        "✓".into()
                    } else {
                        "·".into()
                    },
                    n.couplings
                        .first()
                        .map_or_else(String::new, |c| format!("{}  {:.5}", c.net, c.c.value)),
                ]
            },
            |i| Message::Select(report.nets[i].net.clone()),
            Message::ScrollResults,
        );
        let partial = self.run.as_ref().is_some_and(|r| r.output.is_none());
        let mut left = column![].spacing(6);
        if partial {
            left = left.push(text("Live partial results (not final)").size(12));
        }
        left = left.push(tbl);
        let detail: Element<'_, Message> = match sel.and_then(|i| report.nets.get(i)) {
            None => table::empty("Couplings", "Select a net to list its couplings."),
            Some(n) => {
                let ccols = [
                    col("to net", 4, false),
                    col("C [fF]", 3, true),
                    col("σ [%]", 2, true),
                    col("share", 2, true),
                ];
                let total = n.total.value.abs().max(f64::MIN_POSITIVE);
                column![
                    text(format!("{} — {} couplings", n.net, n.couplings.len())).size(14),
                    table::view(
                        &ccols,
                        Rows {
                            len: n.couplings.len(),
                            scroll: self.scroll_couplings,
                            selected: None,
                            id: Some(COUPLINGS_ID),
                        },
                        |j| {
                            let c = &n.couplings[j];
                            vec![
                                c.net.clone(),
                                format!("{:.5}", c.c.value),
                                format!(
                                    "{:.2}",
                                    100.0 * c.c.sigma / c.c.value.abs().max(f64::MIN_POSITIVE)
                                ),
                                format!("{:.1} %", 100.0 * c.c.value / total),
                            ]
                        },
                        |_| Message::Tab(Tab::Results),
                        Message::ScrollCouplings,
                    ),
                ]
                .spacing(6)
                .into()
            }
        };
        row![
            container(left).width(iced::FillPortion(3)),
            container(detail).width(iced::FillPortion(2))
        ]
        .spacing(10)
        .into()
    }

    fn matrix_view(&self) -> Element<'_, Message> {
        const LABEL_W: f32 = 160.0;
        const CELL_W: f32 = 92.0;
        let Some(run) = &self.run else {
            return table::empty(
                "No results yet",
                "The matrix appears here while the run progresses.",
            );
        };
        let Some(report) = run.report() else {
            return table::empty(
                "No results yet",
                "The matrix appears here while the run progresses.",
            );
        };
        let cols = &run.columns;
        let confidence = run.confidence;
        let fixed = |s: String, w: f32, right: bool| {
            let t = text(s)
                .size(12)
                .wrapping(iced::widget::text::Wrapping::None);
            let t = if right {
                t.align_x(iced::alignment::Horizontal::Right)
            } else {
                t
            };
            container(t.width(Fill))
                .width(w)
                .height(table::ROW_HEIGHT)
                .padding([0, 6])
                .align_y(iced::Center)
                .clip(true)
        };
        let header = iced::widget::Row::with_children(
            std::iter::once(fixed("target \\ net".into(), LABEL_W, false).into()).chain(
                cols.nets
                    .iter()
                    .map(|n| fixed(n.clone(), CELL_W, true).into())
                    .chain(std::iter::once(fixed("ground".into(), CELL_W, true).into())),
            ),
        );
        let sel = self
            .selected
            .as_ref()
            .and_then(|s| report.nets.iter().position(|n| &n.net == s));
        let len = report.nets.len();
        let range = self.scroll_matrix.visible(len);
        #[allow(clippy::cast_precision_loss)]
        let top = range.start as f32 * table::ROW_HEIGHT;
        #[allow(clippy::cast_precision_loss)]
        let bottom = (len - range.end) as f32 * table::ROW_HEIGHT;
        let mut body = column![space().height(top)];
        for i in range {
            let n = &report.nets[i];
            let cells = cols.cells(n, confidence).into_iter().map(|c| {
                let Some(c) = c else {
                    return fixed(String::new(), CELL_W, true).into();
                };
                fixed(significant_digits(c.value), CELL_W, true)
                    .style(move |theme: &Theme| matrix_cell_style(theme, c))
                    .into()
            });
            let r = iced::widget::Row::with_children(
                std::iter::once(fixed(n.net.clone(), LABEL_W, false).into()).chain(cells),
            );
            let is_sel = sel == Some(i);
            body = body.push(
                button(r)
                    .padding(0)
                    .on_press(Message::Select(n.net.clone()))
                    .style(move |theme: &Theme, status| {
                        let base = if is_sel {
                            button::primary(theme, status)
                        } else {
                            button::text(theme, status)
                        };
                        button::Style {
                            border: iced::Border::default(),
                            ..base
                        }
                    }),
            );
        }
        body = body.push(space().height(bottom));
        #[allow(clippy::cast_precision_loss)]
        let width = LABEL_W + CELL_W * cols.len() as f32;
        let grid = column![
            container(header).width(width).style(container::rounded_box),
            scrollable(body)
                .on_scroll(Message::ScrollMatrix)
                .width(width)
                .height(Fill),
        ];
        let mut caption = format!(
            "{len} target nets × {} nets + ground · diagonal: total C, other cells: coupling C \
             [fF] · shading: share of the row's total · faded: within {confidence}σ of zero",
            cols.nets.len()
        );
        if cols.hidden > 0 {
            caption += &format!(" · {} more nets not shown", cols.hidden);
        }
        let partial = run.output.is_none();
        let mut out = column![].spacing(6);
        if partial {
            out = out.push(text("Live partial results (not final)").size(12));
        }
        out.push(text(caption).size(12))
            .push(
                scrollable(grid)
                    .direction(scrollable::Direction::Horizontal(
                        scrollable::Scrollbar::default(),
                    ))
                    .width(Fill)
                    .height(Fill),
            )
            .into()
    }

    fn view_3d(&self) -> Element<'_, Message> {
        if self.scene.len() == 0 {
            return table::empty("3D view", "The solved geometry appears here after a run.");
        }
        column![
            row![
                text(format!("{} boxes · drag: orbit · right/middle drag: pan · wheel: zoom · select a net in Results to highlight it", self.scene.len())).size(12),
                space().width(Fill),
                checkbox(self.scene.dim_non_targets).label("Dim non-target nets").text_size(12).on_toggle(Message::DimOthers),
            ]
            .align_y(iced::Center),
            shader(&self.scene).width(Fill).height(Fill),
        ]
        .spacing(6)
        .into()
    }

    fn messages_view(&self) -> Element<'_, Message> {
        let mut items: Vec<(String, &str)> = Vec::new();
        if let Some(r) = &self.run {
            if let Some(e) = &r.error {
                items.push((e.clone(), "error"));
            }
            for w in &r.warnings {
                items.push((w.clone(), "warning"));
            }
        } else if let Some(p) = &self.project {
            for d in &p.diagnostics {
                items.push((d.clone(), "load"));
            }
        }
        if items.is_empty() {
            return table::empty(
                "No messages",
                "Warnings from loading, configuration and solving appear here.",
            );
        }
        scrollable(
            column(items.into_iter().map(|(m, kind)| {
                row![text(kind).size(12).width(70), text(m).size(13)]
                    .spacing(8)
                    .into()
            }))
            .spacing(6),
        )
        .height(Fill)
        .into()
    }

    fn status_bar(&self) -> Element<'_, Message> {
        let mut r = row![text(&self.status).size(12)]
            .spacing(12)
            .padding([4, 10]);
        if let Some(run) = &self.run {
            r = r.push(space().width(Fill));
            r = r.push(
                text(format!(
                    "seed {} · {}",
                    self.seed,
                    preset_name(self.preset.0)
                ))
                .size(12),
            );
            if run.running() {
                r = r.push(text("●  running").size(12));
            }
        }
        r.into()
    }
}

/// `v` with four significant digits (fixed point, as in the other tables).
fn significant_digits(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let decimals = (3 - v.abs().log10().floor() as i32).clamp(0, 6) as usize;
    format!("{v:.decimals$}")
}

/// Shading of a matrix cell: the diagonal in a neutral tone, couplings and
/// ground by their share of the row's total, values within the confidence
/// band faded.
fn matrix_cell_style(theme: &Theme, c: matrix::Cell) -> container::Style {
    let p = theme.extended_palette();
    let background = if c.diagonal {
        p.background.strong.color
    } else {
        #[allow(clippy::cast_possible_truncation)]
        let share = c.share.clamp(0.0, 1.0) as f32;
        p.primary.base.color.scale_alpha(0.06 + 0.7 * share.sqrt())
    };
    let text = theme.palette().text;
    container::Style {
        background: Some(background.into()),
        text_color: Some(if c.significant {
            text
        } else {
            text.scale_alpha(0.4)
        }),
        ..container::Style::default()
    }
}

#[cfg(test)]
mod tests {
    use super::significant_digits;

    #[test]
    fn four_significant_digits() {
        assert_eq!(significant_digits(12.3456), "12.35");
        assert_eq!(significant_digits(0.0012346), "0.001235");
        assert_eq!(significant_digits(-2678.18), "-2678");
        assert_eq!(significant_digits(123_456.0), "123456");
        assert_eq!(significant_digits(0.0), "0");
    }
}
