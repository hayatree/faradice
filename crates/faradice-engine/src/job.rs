//! Background jobs and their events.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use faradice_core::{Estimate, NetId};
use faradice_frw::{NetResult, Observer, Progress, Solution};
use faradice_geom::Geometry;
use faradice_io::config::{Config, GroundReport};
use faradice_io::report::{Report, build_report};
use faradice_io::result::{Completeness, ResultFile};

use crate::run::{config_toml, prepare, run_prepared};
use crate::{EngineError, Project, RunOutput, RunRequest};

/// Job lifecycle states (D2 §3; also the state set every transport maps onto).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JobState {
    /// Created, not yet solving.
    Queued,
    /// Solving.
    Running,
    /// Finished; every target converged.
    Succeeded,
    /// Finished within budget limits, but some target did not converge.
    NotConverged,
    /// Stopped by [`Job::cancel`]; partial results are available.
    Cancelled,
    /// Failed with an error.
    Failed,
}

impl JobState {
    /// Whether the job can no longer change state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Queued | Self::Running)
    }
}

/// Progress of one target net.
#[derive(Debug, Clone, PartialEq)]
pub struct NetProgress {
    /// Index into [`JobEvent::Started::targets`].
    pub target: usize,
    /// Net name.
    pub net: String,
    /// Walks merged so far.
    pub walks: u64,
    /// Current total-capacitance estimate (fF).
    pub total: Estimate,
    /// `true` for the last report of this net.
    pub finished: bool,
    /// Whether the net converged (only meaningful when `finished`).
    pub converged: bool,
}

/// Events of a job, delivered in order to the job's callback.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum JobEvent {
    /// Input resolved and solving started.
    Started {
        /// Target nets, in solve order.
        targets: Vec<String>,
        /// Configuration and input warnings.
        warnings: Vec<String>,
    },
    /// Progress of one net. Intermediate reports are throttled (at most one
    /// every 100 ms per net); the final report of each net is always sent.
    /// Statistics behind the latest report are available from
    /// [`Job::partial`].
    Progress(NetProgress),
    /// Terminal: finished, possibly cancelled or not converged (see
    /// [`RunOutput::cancelled`] and the per-net `converged` flags).
    Finished(Box<RunOutput>),
    /// Terminal: the run failed.
    Failed(Arc<EngineError>),
}

impl JobEvent {
    /// The state a job is in after this event.
    #[must_use]
    pub fn state(&self) -> JobState {
        match self {
            Self::Started { .. } | Self::Progress(_) => JobState::Running,
            Self::Finished(out) if out.cancelled => JobState::Cancelled,
            Self::Finished(out) if out.result.targets.iter().all(|t| t.converged) => {
                JobState::Succeeded
            }
            Self::Finished(_) => JobState::NotConverged,
            Self::Failed(_) => JobState::Failed,
        }
    }
}

/// A snapshot of a running job's statistics ([`Job::partial`]).
#[derive(Debug, Clone)]
pub struct Partial {
    /// Raw statistics so far ([`Completeness::Partial`]; nets without any
    /// merged walks are omitted).
    pub result: ResultFile,
    /// Report built from them.
    pub report: Report,
}

/// What a job knows once its input is prepared.
struct Live {
    geometry: Arc<Geometry>,
    config: Config,
    targets: Vec<NetId>,
    ground_nets: Vec<String>,
    ground_report: GroundReport,
    warnings: Vec<String>,
    eps_stop: f64,
    nets: Vec<Option<NetResult>>,
    last_sent: Vec<Option<Instant>>,
    version: u64,
}

type Shared = Arc<Mutex<Option<Live>>>;

fn lock(s: &Shared) -> std::sync::MutexGuard<'_, Option<Live>> {
    s.lock().unwrap_or_else(PoisonError::into_inner)
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

const THROTTLE: Duration = Duration::from_millis(100);

/// A running extraction. Dropping the handle does not stop the job; call
/// [`Job::cancel`] (and optionally [`Job::join`]).
#[derive(Debug)]
pub struct Job {
    id: u64,
    cancel: Arc<AtomicBool>,
    live: Shared,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Live {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Live")
            .field("targets", &self.targets.len())
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

struct Forwarder<F> {
    names: Vec<String>,
    sink: Arc<F>,
    cancel: Arc<AtomicBool>,
    live: Shared,
}

impl<F: Fn(JobEvent) + Send + Sync> Observer for Forwarder<F> {
    fn progress(&self, p: &Progress<'_>) {
        {
            let mut guard = lock(&self.live);
            let Some(live) = guard.as_mut() else { return };
            let now = Instant::now();
            let slot = &mut live.last_sent[p.target];
            if !p.finished && slot.is_some_and(|t| now.duration_since(t) < THROTTLE) {
                return;
            }
            *slot = Some(now);
            live.eps_stop = p.eps_stop;
            live.nets[p.target] = Some(p.snapshot());
            live.version += 1;
        }
        (self.sink)(JobEvent::Progress(NetProgress {
            target: p.target,
            net: self.names[p.target].clone(),
            walks: p.walks,
            total: p.total,
            finished: p.finished,
            converged: p.converged,
        }));
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

fn panic_message(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| e.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

impl Job {
    /// Starts `req` on `project` in a background thread.
    ///
    /// `sink` receives every event in order, from the job thread or solver
    /// worker threads: at most one [`JobEvent::Started`], then
    /// [`JobEvent::Progress`] reports, then exactly one terminal
    /// [`JobEvent::Finished`] or [`JobEvent::Failed`]. A panic in the solver
    /// or in `sink` itself is turned into `Failed`; if `sink` also panics on
    /// that, the job ends without a terminal event and [`Job::join`] returns
    /// the error.
    pub fn start<F>(project: Arc<Project>, req: RunRequest, sink: F) -> Self
    where
        F: Fn(JobEvent) + Send + Sync + 'static,
    {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        let live: Shared = Arc::new(Mutex::new(None));
        let sink = Arc::new(sink);
        let (flag, shared, on_spawn_error) =
            (Arc::clone(&cancel), Arc::clone(&live), Arc::clone(&sink));
        let thread = std::thread::Builder::new()
            .name(format!("faradice-job-{id}"))
            .spawn(move || {
                let body = AssertUnwindSafe(|| run_job(&project, &req, &sink, flag, shared));
                let event = match catch_unwind(body) {
                    Ok(Ok(out)) => JobEvent::Finished(Box::new(out)),
                    Ok(Err(e)) => JobEvent::Failed(Arc::new(e)),
                    Err(p) => JobEvent::Failed(Arc::new(EngineError::Panic(panic_message(&*p)))),
                };
                sink(event);
            });
        let thread = match thread {
            Ok(t) => Some(t),
            Err(e) => {
                on_spawn_error(JobEvent::Failed(Arc::new(EngineError::Other(format!(
                    "cannot start job thread: {e}"
                )))));
                None
            }
        };
        Self {
            id,
            cancel,
            live,
            thread,
        }
    }

    /// Process-unique job id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Requests cancellation; the solver stops at its next wave boundary and
    /// the job finishes with partial results.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation was requested.
    #[must_use]
    pub fn is_cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// A counter that increases whenever [`Job::partial`] would change
    /// (0 before solving starts). Lets clients skip rebuilding views.
    #[must_use]
    pub fn version(&self) -> u64 {
        lock(&self.live).as_ref().map_or(0, |l| l.version)
    }

    /// Statistics merged so far (as of the latest forwarded progress report),
    /// or `None` before solving starts. Builds a report, so call it at UI rates
    /// (a few Hz), not per event.
    #[must_use]
    pub fn partial(&self) -> Option<Partial> {
        let guard = lock(&self.live);
        let live = guard.as_ref()?;
        let sol = Solution {
            nets: live.nets.iter().flatten().cloned().collect(),
            eps_stop: live.eps_stop,
            cancelled: false,
        };
        let mut result = ResultFile::from_solution(
            &live.geometry,
            &sol,
            live.config.run.seed,
            live.ground_nets.clone(),
            0.0,
            live.warnings.clone(),
            config_toml(&live.config).unwrap_or_default(),
        );
        result.completeness = Completeness::Partial;
        let report = build_report(&result, &live.config.coupling, live.ground_report);
        Some(Partial { result, report })
    }

    /// Waits for the job thread to end (after its terminal event).
    ///
    /// # Errors
    ///
    /// The panic message if the job thread died without a terminal event.
    pub fn join(mut self) -> Result<(), String> {
        match self.thread.take() {
            Some(t) => t.join().map_err(|p| panic_message(&*p)),
            None => Ok(()),
        }
    }
}

fn run_job<F: Fn(JobEvent) + Send + Sync>(
    project: &Project,
    req: &RunRequest,
    sink: &Arc<F>,
    cancel: Arc<AtomicBool>,
    live: Shared,
) -> crate::Result<RunOutput> {
    let p = prepare(project, req)?;
    let g = &p.geometry;
    let names: Vec<String> = p
        .targets
        .iter()
        .map(|&n| g.net_name(n).to_owned())
        .collect();
    *lock(&live) = Some(Live {
        geometry: Arc::clone(&p.geometry),
        config: p.config.clone(),
        targets: p.targets.clone(),
        ground_nets: p
            .solver
            .ground_nets
            .iter()
            .map(|&n| g.net_name(n).to_owned())
            .collect(),
        ground_report: p.ground_report,
        warnings: p.warnings.clone(),
        eps_stop: f64::NAN,
        nets: vec![None; names.len()],
        last_sent: vec![None; names.len()],
        version: 0,
    });
    sink(JobEvent::Started {
        targets: names.clone(),
        warnings: p.warnings.clone(),
    });
    let fwd = Forwarder {
        names,
        sink: Arc::clone(sink),
        cancel,
        live,
    };
    run_prepared(p, req.engine, &fwd)
}
