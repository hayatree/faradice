//! Axis-aligned conductor geometry and nearest-conductor queries.
//!
//! The floating random walk places, around each walk position, the largest cube
//! that does not intersect any conductor. For axis-aligned boxes the half-side
//! of that cube is exactly the L∞ (Chebyshev) distance from the point to the
//! nearest box, which is what [`SpatialIndex::nearest_linf`] computes.
//!
//! - [`Aabb`]: a validated axis-aligned box.
//! - [`Geometry`] / [`GeometryBuilder`]: boxes grouped into named nets.
//! - [`SpatialIndex`]: the query trait, with [`UniformGrid`] (fast) and
//!   [`BruteForce`] (reference) implementations.
//! - [`Rect`] / [`RectIndex`]: axis-aligned rectangles (dielectric interfaces)
//!   and exact L∞ nearest-rectangle queries.
//! - [`paint()`]: labelled Boolean operations on boxes (later items win),
//!   producing disjoint labelled boxes.

#![forbid(unsafe_code)]

mod aabb;
mod error;
mod geometry;
mod grid;
mod index;
mod paint;
mod rect;

pub use aabb::Aabb;
pub use error::{GeomError, Result};
pub use geometry::{Geometry, GeometryBuilder};
pub use grid::{GridData, GridOptions, LocalBoxes, UniformGrid};
pub use index::{BruteForce, Nearest, SpatialIndex};
pub use paint::{Paint, PaintItem, Painted, paint};
pub use rect::{RECT_CELL_FACTOR, Rect, RectIndex};
