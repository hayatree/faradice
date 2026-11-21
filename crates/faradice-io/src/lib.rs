//! Input/output formats, configuration and post-processing for faradice.
//!
//! - [`structure`]: the box-based structure file (`*.faradice.toml`).
//! - [`config`]: the run configuration (`faradice.toml`), resolved into a
//!   [`faradice_frw::SolverConfig`] plus reporting options.
//! - [`result`]: raw statistics (`result.json`), from which reports can be
//!   regenerated without re-solving.
//! - [`report`] — post-processing (thresholds, couple-to-ground, symmetrization)
//!   and report writers (JSON, CSV, SPEF).

#![forbid(unsafe_code)]

pub mod config;
mod glob;
pub mod report;
pub mod result;
pub mod structure;

pub use glob::{glob_escape, glob_match};

/// I/O errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IoError {
    /// File system error.
    #[error("{path}: {source}")]
    File {
        /// Path involved.
        path: String,
        /// Underlying error.
        source: std::io::Error,
    },
    /// TOML parse error.
    #[error("{path}: {source}")]
    Toml {
        /// Path involved.
        path: String,
        /// Underlying error.
        source: toml::de::Error,
    },
    /// JSON error.
    #[error("{path}: {source}")]
    Json {
        /// Path involved.
        path: String,
        /// Underlying error.
        source: serde_json::Error,
    },
    /// Geometry error.
    #[error(transparent)]
    Geom(#[from] faradice_geom::GeomError),
    /// Invalid content.
    #[error("{0}")]
    Invalid(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, IoError>;

pub(crate) fn read_to_string(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|source| IoError::File {
        path: path.display().to_string(),
        source,
    })
}
