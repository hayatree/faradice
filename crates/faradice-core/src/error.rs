//! Shared error type.

/// Errors produced by faradice-core parsing and validation helpers.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A length unit string was not recognized.
    #[error("unknown length unit `{0}` (expected one of: nm, um, mm, m)")]
    UnknownLengthUnit(String),
    /// A face name was not recognized.
    #[error("unknown face `{0}` (expected one of: xmin, xmax, ymin, ymax, zmin, zmax)")]
    UnknownFace(String),
    /// An identifier index does not fit into the identifier type.
    #[error("identifier index {0} exceeds the supported range (u32)")]
    IdOverflow(usize),
}

/// Convenience alias for `Result<T, faradice_core::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
