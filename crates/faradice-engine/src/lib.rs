//! The faradice run engine, shared by the CLI, the GUI and (later) the
//! JSON-RPC/MCP servers (design D1 §5, D2 §2).
//!
//! - [`Project`]: a loaded input (structure file, or GDSII traced with a
//!   technology file) that can be solved many times with different configs.
//! - [`run`]: one blocking extraction with progress and cancellation through a
//!   [`faradice_frw::Observer`].
//! - [`Job`]: the same run on a background thread, reporting [`JobEvent`]s to a
//!   callback and cancellable at any time. Every transport (in-process GUI,
//!   JSON-RPC, MCP, A2A) maps onto this one job model.

mod job;
mod project;
mod run;

pub use job::{Job, JobEvent, JobState, NetProgress, Partial};
pub use project::is_gds;
pub use project::{Project, Source};
pub use run::{
    Prepared, RunOutput, RunRequest, WalkEngineKind, config_toml, finish, ground_report_of,
    prepare, run, thread_pool, write_outputs, write_reports,
};

/// Engine errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// Reading or validating input or configuration failed.
    #[error(transparent)]
    Io(#[from] faradice_io::IoError),
    /// GDSII reading, flattening or tracing failed.
    #[error(transparent)]
    Gds(#[from] faradice_gds::GdsError),
    /// The solver rejected the run.
    #[error(transparent)]
    Solve(#[from] faradice_frw::SolveError),
    /// A job thread panicked (a bug, or a panicking event callback).
    #[error("internal error: {0}")]
    Panic(String),
    /// Anything else (invalid request, thread pool, file system).
    #[error("{0}")]
    Other(String),
}

/// Engine result type.
pub type Result<T> = std::result::Result<T, EngineError>;
