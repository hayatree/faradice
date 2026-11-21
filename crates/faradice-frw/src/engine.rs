//! Pluggable walk engines.
//!
//! The solver splits every sample into two phases:
//!
//! 1. **Launch** (always on the CPU, in `f64`): sample the Gaussian surface,
//!    take the first hop and compute the sample weight.
//! 2. **Transition**: walk from the launch point until a sink is reached.
//!
//! The transition phase is where almost all the time goes and is delegated to
//! a [`WalkEngine`]. [`CpuEngine`] runs it on the CPU with [`rayon`]; the
//! `faradice-gpu` crate provides a GPU engine. Each launch carries a 64-bit
//! key from which the engine derives an independent random stream, so the
//! outcome of a walk does not depend on how launches are grouped into
//! dispatches.

use faradice_core::Sink;
use faradice_geom::{Geometry, UniformGrid};
use rayon::prelude::*;

use crate::axis::BoxSteps;
use crate::boundary::BoundaryModel;
use crate::green::GreenTables;
use crate::medium::Medium;
use crate::rng::Rng;
use crate::walk::Walker;

/// Everything a walk engine needs to reproduce the CPU walk.
#[derive(Debug, Clone, Copy)]
pub struct Scene<'a> {
    /// Conductors.
    pub geometry: &'a Geometry,
    /// Spatial index over [`Geometry::boxes`].
    pub grid: &'a UniformGrid,
    /// Transition tables.
    pub green: &'a GreenTables,
    /// Domain boundary.
    pub boundary: &'a BoundaryModel,
    /// Stratified dielectric; `None` if homogeneous. Engines that do not
    /// implement layered cubes must refuse scenes that have one.
    pub medium: Option<&'a Medium>,
    /// Dielectric boxes (design M2c); `None` if there are none. Engines that do
    /// not implement them must refuse scenes that have them.
    pub boxes: Option<&'a BoxSteps>,
    /// Walks closer than this to a sink are absorbed (µm).
    pub eps_stop: f64,
    /// Lower clearance bounds at least this large are used directly as hop sizes.
    pub far_step: f64,
    /// Transition steps after which a walk is aborted.
    pub max_steps: u32,
}

/// One walk to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Launch {
    /// Start point (after the first hop; inside the domain).
    pub start: [f64; 3],
    /// Random stream key.
    pub key: u64,
}

/// How a walk ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// The sink, or `None` if the walk was aborted after `max_steps`.
    pub sink: Option<Sink>,
    /// `±1`: the sign flips when the walk is reflected across the substrate in
    /// the odd image extension (see [`BoundaryModel::is_odd_image`]).
    pub sign: f64,
    /// Transition steps taken.
    pub steps: u32,
}

/// Runs transition walks.
pub trait WalkEngine {
    /// Runs every launch and returns one outcome per launch, in order.
    ///
    /// # Errors
    ///
    /// Returns a message if the device fails; the solve is then aborted.
    fn run(&mut self, scene: &Scene<'_>, launches: &[Launch]) -> Result<Vec<Outcome>, String>;

    /// Preferred number of walks per call (the solver groups whole batches).
    fn preferred_walks(&self) -> usize;
}

/// The reference engine: the CPU walk, parallel over launches.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuEngine;

impl WalkEngine for CpuEngine {
    fn run(&mut self, s: &Scene<'_>, launches: &[Launch]) -> Result<Vec<Outcome>, String> {
        let walker = Walker {
            geom: s.geometry,
            index: s.grid,
            green: s.green,
            domain: s.boundary,
            medium: s.medium,
            boxes: s.boxes,
            eps_stop: s.eps_stop,
            far_step: s.far_step,
            max_steps: s.max_steps,
            overlay: None,
            added: None,
            windows: false,
        };
        Ok(launches
            .par_iter()
            .with_min_len(64)
            .map(|l| {
                let mut rng = Rng::from_seed(l.key);
                let end = walker.run(l.start, &mut rng);
                Outcome {
                    sink: end.sink,
                    sign: end.sign,
                    steps: end.steps,
                }
            })
            .collect())
    }

    fn preferred_walks(&self) -> usize {
        rayon::current_num_threads().max(1) * 16_384
    }
}
