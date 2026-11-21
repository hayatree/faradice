//! Floating random walk (FRW) capacitance solver core.
//!
//! - [`rng`]: small, fast, reproducible random number streams keyed by
//!   `(seed, net, batch)`.
//! - [`alias`]: Walker/Vose alias tables for O(1) discrete sampling.
//! - [`green`]: surface Green-function tables of the transition cube
//!   (transition probabilities and first-hop gradient kernels).
//! - [`layered`]: kernels of a cube crossing one dielectric interface.
//! - [`medium`]: cube selection and first hops in a stratified medium.
//! - [`multilayer`]: anchored multi-layer cubes and landing boxes (design M2b).
//! - [`boundary`]: the simulation domain (grounded/reflecting box or open space)
//!   and the optional substrate ground plane.
//! - [`gauss`]: Gaussian surfaces around target nets.
//! - [`stack`]: stratified dielectric stacks (effective ε(z) of the domain).
//! - [`regions`]: dielectric boxes and conformal rules (design M2c).
//! - [`dielectric`]: material boxes and their interface faces (design M2c).
//! - [`axis`]: walk steps with dielectric boxes (axis cubes, design M2c §4).
//! - [`section`]: window-flux kernels of boxes with a 2D cross-section, for
//!   the window first hops (design M2f).
//! - [`solver`]: the walk, the estimator and the stopping rules.
//! - `map` (feature `sensitivity`): potentials and their gradients at points,
//!   for insertion maps (design D3 §7).
//! - `sens` (feature `sensitivity`): capacitance sensitivities by finite
//!   differences with path continuation (design D3).
//! - [`cross`]: per-target sink cross moments (row covariances, design D3 §5.2).
//! - `shape` (feature `sensitivity`): shape derivatives with a face
//!   normal-derivative kernel (design D3 §6, interior of faces).
//! - [`engine`]: pluggable transition-walk engines (CPU reference; GPU in
//!   `faradice-gpu`).

#![forbid(unsafe_code)]

pub mod alias;
pub mod axis;
pub mod boundary;
pub mod cross;
mod dense;
pub mod dielectric;
pub mod edge;
pub mod engine;
pub mod gauss;
pub mod green;
pub mod layered;
#[cfg(feature = "sensitivity")]
pub mod map;
pub mod medium;
pub mod multilayer;
pub mod regions;
#[cfg(feature = "sensitivity")]
pub mod replay;
pub mod rng;
pub mod section;
#[cfg(feature = "sensitivity")]
pub mod sens;
#[cfg(feature = "sensitivity")]
pub mod shape;
#[cfg(feature = "sensitivity")]
mod shell;
pub mod solver;
pub mod stack;
mod walk;
mod windows;

pub use boundary::{Boundary, BoundaryModel, FaceKind};
pub use dielectric::BoxMedium;
pub use engine::{CpuEngine, Launch, Outcome, Scene, WalkEngine};
pub use medium::Medium;
pub use regions::{ConformalRule, DielectricBox, DielectricRegions};
pub use solver::{
    Accuracy, NetResult, Observer, Progress, Solution, SolveError, SolverConfig, StackOptions,
    WalkOptions, solve, solve_observed, solve_with, solve_with_observed,
};
pub use stack::{DielectricLayer, Stack, validate_layers};
