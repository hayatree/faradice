//! Core types shared by all faradice crates.
//!
//! This crate is intentionally small and dependency-light. It defines the
//! vocabulary used across the solver, I/O and CLI layers:
//!
//! - [`units`]: length units and the vacuum permittivity in the internal unit system
//!   (lengths in µm, capacitances in fF).
//! - [`ids`]: strongly typed identifiers ([`NetId`], [`BoxId`]).
//! - [`geom`]: axis and face enums ([`Axis`], [`Side`], [`Face`]).
//! - [`sink`]: where a random walk terminates ([`Sink`]).
//! - [`stats`]: deterministic, mergeable accumulators for weighted Monte Carlo samples.
//! - [`error`]: the shared [`Error`] type.

#![forbid(unsafe_code)]

pub mod error;
pub mod geom;
pub mod ids;
pub mod sink;
pub mod stats;
pub mod units;

pub use error::{Error, Result};
pub use geom::{Axis, Face, Side};
pub use ids::{BoxId, NetId};
pub use sink::{GROUND_SUM_LABEL, Sink, is_reserved_label};
pub use stats::{Estimate, Moments, NeumaierSum};
pub use units::{EPS0_FF_PER_UM, LengthUnit};
