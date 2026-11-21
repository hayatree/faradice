//! Geometry errors.

use faradice_core::BoxId;

/// Errors produced while building or validating geometry.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum GeomError {
    /// A box has a non-finite coordinate or a non-positive extent.
    #[error("invalid box {min:?}..{max:?}: coordinates must be finite and min < max on every axis")]
    InvalidBox {
        /// Lower corner.
        min: [f64; 3],
        /// Upper corner.
        max: [f64; 3],
    },
    /// A rectangle needs exactly one axis with zero extent and finite coordinates.
    #[error(
        "invalid rectangle {min:?}..{max:?}: coordinates must be finite, with min == max on exactly one axis and min < max on the others"
    )]
    InvalidRect {
        /// Lower corner.
        min: [f64; 3],
        /// Upper corner.
        max: [f64; 3],
    },
    /// Two boxes of different nets overlap or touch, which would short the nets.
    #[error("boxes {a} (net `{net_a}`) and {b} (net `{net_b}`) overlap or touch")]
    NetsShorted {
        /// First box.
        a: BoxId,
        /// Net of the first box.
        net_a: String,
        /// Second box.
        b: BoxId,
        /// Net of the second box.
        net_b: String,
    },
    /// The geometry contains no boxes.
    #[error("geometry contains no conductor boxes")]
    Empty,
    /// A net name of the reserved form `<…>` (sink labels).
    #[error("net name `{0}` is reserved: names of the form <…> label ground sinks")]
    ReservedNetName(String),
    /// A net has no boxes.
    #[error("net `{0}` has no boxes")]
    EmptyNet(String),
    /// An identifier overflowed.
    #[error(transparent)]
    Core(#[from] faradice_core::Error),
}

/// Convenience alias for `Result<T, GeomError>`.
pub type Result<T> = std::result::Result<T, GeomError>;
