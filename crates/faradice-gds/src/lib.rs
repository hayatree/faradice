//! GDSII input and connectivity tracing for faradice.
//!
//! Pipeline: [`stream::read_library`] → [`flatten::flatten`] (exact integer
//! coordinates in half database units) → [`manhattan`] decomposition into
//! rectangles → [`trace::trace`] with a [`tech::Tech`] description, which
//! returns physically connected components named by text labels.

#![forbid(unsafe_code)]

pub mod flatten;
pub mod manhattan;
pub mod stream;
pub mod tech;
pub mod trace;
pub mod writer;

/// GDS input errors.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GdsError {
    /// I/O error.
    #[error("{path}: {source}")]
    Io {
        /// Path.
        path: String,
        /// Source.
        source: std::io::Error,
    },
    /// Malformed stream.
    #[error("malformed GDSII at byte {offset}: {msg}")]
    Format {
        /// Byte offset.
        offset: usize,
        /// Message.
        msg: String,
    },
    /// Unsupported construct.
    #[error("unsupported GDSII construct: {0}")]
    Unsupported(String),
    /// Hierarchy problem.
    #[error("hierarchy error: {0}")]
    Hierarchy(String),
    /// Technology file problem.
    #[error("technology file: {0}")]
    Tech(String),
    /// Geometry error.
    #[error(transparent)]
    Geom(#[from] faradice_geom::GeomError),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, GdsError>;
