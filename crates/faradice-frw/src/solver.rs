//! The FRW estimator, stopping rules and deterministic parallel driver.
//!
//! For target net `i`, `C_ij` is the charge on `i` when net `j` is at 1 V and
//! everything else at 0 V:
//!
//! ```text
//! Q_i = ∮_G ε (−∂φ/∂n_out) dS
//! ```
//!
//! A walk starts at a uniformly sampled Gaussian-surface point `r` (outward
//! normal `s·e_a`), takes a first hop in the cube of half-side `h` centred at
//! `r` sampled from `|∂P/∂x_a|`, and then walks until it reaches a sink `k`.
//! Its weight
//!
//! ```text
//! w = −A_cand · ε · s · sign(K) · ‖K‖₁ / (2h)
//! ```
//!
//! is added to sink `k`. The mean over all walks (rejected candidates count as
//! zero) estimates `C_ik`: positive for `k = i`, negative otherwise. Reported
//! couplings are `−C_ik`.

use std::collections::BTreeMap;

use faradice_core::{EPS0_FF_PER_UM, Estimate, Face, Moments, NetId, Side, Sink};
use faradice_geom::{Aabb, Geometry, GridOptions, Paint, PaintItem, UniformGrid, paint};
use rayon::prelude::*;

use crate::axis::BoxSteps;
use crate::boundary::{Boundary, BoundaryModel};
use crate::dielectric::BoxMedium;
use crate::engine::{Launch, Outcome, Scene, WalkEngine};
use crate::gauss::{GaussPoint, GaussSurface, net_clearance};
use crate::green::{DEFAULT_RESOLUTION, GreenTables};
use crate::medium::{Hop, Medium, homogeneous_hop};
use crate::regions::DielectricRegions;
use crate::rng::Rng;
use crate::stack::{DielectricLayer, Stack, validate_layers};
use crate::walk::{WalkEnd, Walker};

/// Accuracy targets and walk budgets.
#[derive(Debug, Clone, PartialEq)]
pub struct Accuracy {
    /// Target relative error of the total capacitance.
    pub total_rel_tol: f64,
    /// Target relative error of tracked couplings.
    pub coupling_rel_tol: f64,
    /// Track couplings with `|C_ij| ≥ track_rel · C_total` …
    pub track_rel: f64,
    /// … or `|C_ij| ≥ track_abs_ff`.
    pub track_abs_ff: f64,
    /// Errors are compared at `confidence_sigma` standard errors.
    pub confidence_sigma: f64,
    /// Minimum walks per net before stopping is considered.
    pub min_walks: u64,
    /// Maximum walks per net; reaching it marks the net as not converged.
    pub max_walks: u64,
    /// Walks per batch (the unit of statistics merging and stopping checks).
    pub batch_walks: u32,
}

/// The `default` accuracy preset of the run configuration: 1 % on totals and
/// 5 % on couplings, at 3σ.
impl Default for Accuracy {
    fn default() -> Self {
        Self {
            total_rel_tol: 0.01,
            coupling_rel_tol: 0.05,
            track_rel: 0.05,
            track_abs_ff: f64::INFINITY,
            confidence_sigma: 3.0,
            min_walks: 10_000,
            max_walks: 50_000_000,
            batch_walks: 4096,
        }
    }
}

/// Walk tuning parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct WalkOptions {
    /// Absolute stopping distance in µm; `None` = `stop_ratio × median smallest box extent`.
    pub stop_dist: Option<f64>,
    /// Relative stopping distance used when `stop_dist` is `None`.
    pub stop_ratio: f64,
    /// Gaussian surface offset as a fraction of the net's clearance.
    pub gauss_alpha: f64,
    /// Steps after which a walk is aborted (counted, never silently dropped).
    pub max_steps: u32,
    /// Green table resolution (cells per face edge, even).
    pub green_resolution: usize,
    /// Antithetic first hop: each Gaussian sample launches a pair of walks from
    /// `r ± h·u` with opposite kernel signs. The gradient kernel is odd under
    /// point reflection through the cube centre, so the pair cancels the
    /// locally constant part of the potential.
    pub antithetic: bool,
    /// Tables and step rules in stratified dielectrics (designs M2a, M2b).
    pub stack: StackOptions,
}

/// Parameters of the walk in a stratified dielectric (designs M2a §3, §4 and
/// M2b §3); the defaults come from the L4 study (`docs/validation/M2.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct StackOptions {
    /// Tabulated interface offsets `K` of one-interface cubes (M2a §3.1),
    /// at least 8.
    pub interface_offsets: usize,
    /// Cells per face edge `M` of the layered tables, a multiple of
    /// `2 · interface_offsets`.
    pub layered_resolution: usize,
    /// Largest half-band length `η_max` of lateral first hops, as a fraction
    /// of the cube half-side: a multiple of 1/32 in `(0, ½]` (M2a §4).
    pub band_fraction: f64,
    /// An off-interface cube smaller than this fraction of the clearance is
    /// replaced by a landing box (M2b §3.1), in `(0, 1]`. With dielectric
    /// boxes, also the threshold of edge cubes (design M2d §3.5).
    pub landing_alpha: f64,
    /// With dielectric boxes: land on dielectric edge planes and take edge
    /// cubes there (design M2d). Off: the M2c steps only.
    pub edge_cubes: bool,
    /// With dielectric boxes: window first hops where the box around a
    /// window of the Gaussian surface has a 2D cross-section (design M2f).
    /// Off: point first hops only.
    pub window_hops: bool,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            interface_offsets: 16,
            layered_resolution: 32,
            band_fraction: 0.5,
            landing_alpha: 0.7,
            edge_cubes: true,
            window_hops: true,
        }
    }
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            stop_dist: None,
            stop_ratio: 1e-3,
            gauss_alpha: 0.5,
            max_steps: 100_000,
            green_resolution: DEFAULT_RESOLUTION,
            antithetic: true,
            stack: StackOptions::default(),
        }
    }
}

/// Full solver configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverConfig {
    /// Relative permittivity of the homogeneous dielectric, or of the medium
    /// above the last entry of `layers`.
    pub eps_r: f64,
    /// Stratified dielectric layers, bottom-up (empty: homogeneous `eps_r`).
    /// A stack that is not homogeneous after clipping and merging (see
    /// [`Stack::effective`]) requires a box domain and the CPU engine
    /// (design M2a).
    pub layers: Vec<DielectricLayer>,
    /// Outer boundary.
    pub boundary: Boundary,
    /// Substrate ground plane height, if any.
    pub substrate_z: Option<f64>,
    /// Accuracy and budgets.
    pub accuracy: Accuracy,
    /// Walk parameters.
    pub walk: WalkOptions,
    /// Master seed.
    pub seed: u64,
    /// Nets treated as ground. Their sinks are still reported individually, and
    /// an aggregate ground estimate (with correct variance, accumulated per
    /// sample rather than summed afterwards) is reported in
    /// [`NetResult::ground`].
    pub ground_nets: Vec<NetId>,
    /// Dielectric boxes and conformal rules on top of the stack (design M2c).
    /// Not supported by the walk yet: a non-empty value is a configuration error.
    pub regions: DielectricRegions,
}

/// Solver errors.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SolveError {
    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// A target net touches another net or a grounded boundary.
    #[error("net `{0}` has zero clearance to another conductor or a grounded boundary")]
    NoClearance(String),
    /// The walk engine failed (see [`solve_with`]).
    #[error("walk engine: {0}")]
    Engine(String),
}

/// Raw statistics for one target net.
#[derive(Debug, Clone)]
pub struct NetResult {
    /// The target net.
    pub net: NetId,
    /// Walks launched (including rejected Gaussian candidates and aborted walks).
    pub walks: u64,
    /// Walks that exceeded `max_steps`.
    pub aborted_walks: u64,
    /// Total steps over all walks.
    pub total_steps: u64,
    /// Whether the accuracy targets were met before `max_walks` with no aborted walks.
    pub converged: bool,
    /// Gaussian surface offset used.
    pub gauss_offset: f64,
    /// Per-sink weight moments (weights in fF).
    pub sinks: Vec<(Sink, Moments)>,
    /// Moments of the per-sample sum over all ground-like sinks (substrate,
    /// grounded faces, infinity and `ground_nets`).
    pub ground_moments: Moments,
}

impl NetResult {
    /// Estimate of `C_ik` (fF) for sink `k` (zero if never hit).
    #[must_use]
    pub fn c(&self, sink: Sink) -> Estimate {
        self.sinks
            .iter()
            .find(|(s, _)| *s == sink)
            .map_or(Moments::new().estimate(self.walks), |(_, m)| {
                m.estimate(self.walks)
            })
    }

    /// Total (self) capacitance `C_ii` in fF.
    #[must_use]
    pub fn total(&self) -> Estimate {
        self.c(Sink::Net(self.net))
    }

    /// Total ground capacitance (fF, positive): `−Σ C_ik` over ground-like sinks,
    /// with the standard error of the sum (covariances included).
    #[must_use]
    pub fn ground(&self) -> Estimate {
        self.ground_moments.estimate(self.walks).scaled(-1.0)
    }

    /// Coupling `−C_ik` in fF.
    #[must_use]
    pub fn coupling(&self, sink: Sink) -> Estimate {
        self.c(sink).scaled(-1.0)
    }
}

/// Result of a solve.
#[derive(Debug, Clone)]
pub struct Solution {
    /// One entry per requested target, in request order. After a cancelled
    /// solve, targets that never started are omitted and the others hold the
    /// statistics merged so far (not converged).
    pub nets: Vec<NetResult>,
    /// Stopping distance used (µm).
    pub eps_stop: f64,
    /// Whether the solve stopped early because [`Observer::cancelled`]
    /// returned `true`.
    pub cancelled: bool,
}

/// Progress of one target net, reported after each merged wave of batches.
#[derive(Debug)]
#[non_exhaustive]
pub struct Progress<'a> {
    /// Index of the net in the requested targets.
    pub target: usize,
    /// The target net.
    pub net: NetId,
    /// Walks merged so far.
    pub walks: u64,
    /// Current estimate of the total capacitance `C_ii` (fF).
    pub total: Estimate,
    /// `true` for the last report of this net.
    pub finished: bool,
    /// Whether the accuracy targets are met (only meaningful when `finished`).
    pub converged: bool,
    /// Stopping distance of this solve (µm).
    pub eps_stop: f64,
    stats: &'a BatchStats,
    gauss_offset: f64,
}

impl Progress<'_> {
    /// The statistics merged so far, as a [`NetResult`] (allocates; call it
    /// only for reports you forward, e.g. throttled live views).
    #[must_use]
    pub fn snapshot(&self) -> NetResult {
        NetResult {
            net: self.net,
            walks: self.stats.walks,
            aborted_walks: self.stats.aborted,
            total_steps: self.stats.steps,
            converged: self.finished && self.converged,
            gauss_offset: self.gauss_offset,
            sinks: self.stats.sinks.iter().map(|(k, m)| (*k, *m)).collect(),
            ground_moments: self.stats.ground,
        }
    }
}

/// Observes and controls a running solve ([`solve_observed`],
/// [`solve_with_observed`]).
///
/// Methods are called from solver worker threads, possibly concurrently for
/// different nets, so implementations must be cheap and thread safe.
/// Observation never changes the results: a solve that is not cancelled is
/// identical to one without an observer.
pub trait Observer: Sync {
    /// Called after each merged wave of a net, and once more with
    /// [`Progress::finished`] set.
    fn progress(&self, _progress: &Progress<'_>) {}

    /// Polled between waves; returning `true` stops the solve at the next
    /// wave boundary, keeping every batch merged so far.
    fn cancelled(&self) -> bool {
        false
    }
}

/// The observer that does nothing ([`solve`], [`solve_with`]).
impl Observer for () {}

#[derive(Debug, Default)]
struct BatchStats {
    walks: u64,
    aborted: u64,
    steps: u64,
    sinks: BTreeMap<Sink, Moments>,
    ground: Moments,
}

impl BatchStats {
    fn merge(&mut self, o: &Self) {
        self.walks += o.walks;
        self.aborted += o.aborted;
        self.steps += o.steps;
        for (k, m) in &o.sinks {
            self.sinks.entry(*k).or_default().merge(m);
        }
        self.ground.merge(&o.ground);
    }
}

fn median_min_extent(geom: &Geometry) -> f64 {
    let mut v: Vec<f64> = geom
        .boxes()
        .iter()
        .map(faradice_geom::Aabb::min_extent)
        .collect();
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, f64::total_cmp).1
}

pub(crate) fn validate(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
) -> Result<(), SolveError> {
    let bad = |m: &str| Err(SolveError::Config(m.to_owned()));
    if !(cfg.eps_r.is_finite() && cfg.eps_r > 0.0) {
        return bad("eps_r must be positive");
    }
    if targets.is_empty() {
        return bad("no target nets");
    }
    let a = &cfg.accuracy;
    if a.batch_walks == 0
        || a.max_walks == 0
        || a.confidence_sigma.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        return bad("batch_walks, max_walks and confidence_sigma must be positive");
    }
    let w = &cfg.walk;
    if !(w.gauss_alpha > 0.0 && w.gauss_alpha < 1.0) {
        return bad("walk.gauss_alpha must be in (0, 1)");
    }
    if w.green_resolution < 2 || w.green_resolution % 2 != 0 {
        return bad("walk.green_resolution must be even and >= 2");
    }
    if w.stop_dist.is_some_and(|d| !(d.is_finite() && d > 0.0))
        || !(w.stop_ratio.is_finite() && w.stop_ratio > 0.0)
    {
        return bad("walk.stop_dist and walk.stop_ratio must be finite and positive");
    }
    if w.max_steps == 0 {
        return bad("walk.max_steps must be positive");
    }
    let st = &w.stack;
    // K ≥ 8: smaller K can make walks drift towards an interface (M2a §3.1).
    if st.interface_offsets < 8
        || st.layered_resolution == 0
        || st.layered_resolution % (2 * st.interface_offsets) != 0
    {
        return bad(
            "walk.interface_offsets must be >= 8 and walk.layered_resolution a positive multiple of twice it",
        );
    }
    let j = st.band_fraction * 32.0;
    #[allow(clippy::float_cmp)] // multiples of 1/32 are exact
    if !((1.0..=16.0).contains(&j) && j == j.round()) {
        return bad("walk.band_fraction must be a multiple of 1/32 in (0, 1/2]");
    }
    if !(st.landing_alpha > 0.0 && st.landing_alpha <= 1.0) {
        return bad("walk.landing_alpha must be in (0, 1]");
    }
    if targets.iter().any(|t| t.index() >= geom.num_nets()) {
        return bad("unknown target net id");
    }
    let b = geom.bounds();
    if let Some(z0) = cfg.substrate_z {
        if b.min()[2] <= z0 {
            return bad("all conductors must lie strictly above the substrate plane");
        }
    }
    if let Boundary::Box { bounds, .. } = &cfg.boundary {
        let ok = (0..3).all(|t| bounds.min()[t] <= b.min()[t] && b.max()[t] <= bounds.max()[t]);
        if !ok {
            return bad("the domain box must contain all conductors");
        }
        if let Some(z0) = cfg.substrate_z {
            if bounds.min()[2] < z0 {
                return bad("the domain box must not extend below the substrate plane");
            }
        }
    }
    if let Err(e) = validate_layers(&cfg.layers) {
        return bad(&e);
    }
    if let Err(e) = cfg.regions.validate(geom) {
        return bad(&e);
    }
    let domain = BoundaryModel::new(&cfg.boundary, cfg.substrate_z, b);
    if !Stack::effective(cfg.eps_r, &cfg.layers, &domain).is_homogeneous()
        && matches!(cfg.boundary, Boundary::Open)
    {
        return bad("a dielectric stack requires domain.kind = \"box\"");
    }
    Ok(())
}

pub(crate) struct Setup {
    pub(crate) domain: BoundaryModel,
    /// Effective dielectric stack.
    pub(crate) stack: Stack,
    /// Layered tables if the stack has interfaces.
    pub(crate) medium: Option<Medium>,
    /// Dielectric boxes (design M2c), if any lie in the domain.
    pub(crate) boxes: Option<BoxSteps>,
    pub(crate) green: GreenTables,
    pub(crate) index: UniformGrid,
    pub(crate) clearances: Vec<f64>,
    pub(crate) eps_stop: f64,
}

pub(crate) fn setup(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
) -> Result<Setup, SolveError> {
    validate(geom, targets, cfg)?;
    // The open boundary's escape sphere must enclose the dielectric boxes too.
    let domain = BoundaryModel::new(
        &cfg.boundary,
        cfg.substrate_z,
        cfg.regions
            .extent(geom)
            .map_or(geom.bounds(), |e| e.union(&geom.bounds())),
    );
    let stack = Stack::effective(cfg.eps_r, &cfg.layers, &domain);
    let medium = Medium::new(stack.clone(), &cfg.walk.stack).map_err(SolveError::Config)?;
    let boxes = if cfg.regions.is_empty() {
        None
    } else {
        BoxMedium::build(&cfg.regions, geom, &stack, &domain)
            .map(|m| {
                BoxSteps::new(
                    m,
                    &stack,
                    cfg.walk.stack.interface_offsets,
                    cfg.walk.stack.layered_resolution,
                    cfg.walk
                        .stack
                        .edge_cubes
                        .then_some(cfg.walk.stack.landing_alpha),
                )
            })
            .transpose()
            .map_err(SolveError::Config)?
    };
    let green = GreenTables::new(cfg.walk.green_resolution);
    let index = UniformGrid::new(geom, GridOptions::default());
    // Clearances of all targets (also needed for the Gaussian offsets). The default
    // stopping distance is relative to the smaller of the typical box size and the
    // smallest target gap, so thin gaps are never swallowed by the snap distance.
    let clearances: Vec<f64> = targets
        .iter()
        .map(|&n| net_clearance(geom, n, &domain))
        .collect();
    if let Some((i, _)) = clearances.iter().enumerate().find(|(_, c)| **c <= 0.0) {
        return Err(SolveError::NoClearance(
            geom.net_name(targets[i]).to_owned(),
        ));
    }
    let min_gap = clearances.iter().copied().fold(f64::INFINITY, f64::min);
    let eps_stop = cfg
        .walk
        .stop_dist
        .unwrap_or_else(|| cfg.walk.stop_ratio * median_min_extent(geom).min(min_gap));
    Ok(Setup {
        domain,
        stack,
        medium,
        boxes,
        green,
        index,
        clearances,
        eps_stop,
    })
}

impl Setup {
    pub(crate) fn walker<'a>(&'a self, geom: &'a Geometry, cfg: &SolverConfig) -> Walker<'a> {
        Walker {
            geom,
            index: &self.index,
            green: &self.green,
            domain: &self.domain,
            medium: self.medium.as_ref(),
            boxes: self.boxes.as_ref(),
            eps_stop: self.eps_stop,
            far_step: 2.0 * self.index.cell_size(),
            max_steps: cfg.walk.max_steps,
            overlay: None,
            added: None,
            // Window first hops are switched on by the solve paths that
            // neither record nor replay samples (design M2f §3.4).
            windows: false,
        }
    }

    fn scene<'a>(&'a self, geom: &'a Geometry, cfg: &SolverConfig) -> Scene<'a> {
        Scene {
            geometry: geom,
            grid: &self.index,
            green: &self.green,
            boundary: &self.domain,
            medium: self.medium.as_ref(),
            boxes: self.boxes.as_ref(),
            eps_stop: self.eps_stop,
            far_step: 2.0 * self.index.cell_size(),
            max_steps: cfg.walk.max_steps,
        }
    }

    /// The permittivity factor of the sample weight: `ε₀ ε_r` for a
    /// homogeneous medium; `ε₀` otherwise (the first hop carries `ε_r(r)`).
    pub(crate) fn eps_unit(&self) -> f64 {
        if self.medium.is_some() || self.boxes.is_some() {
            EPS0_FF_PER_UM
        } else {
            self.stack.eps()[0] * EPS0_FF_PER_UM
        }
    }
}

/// The Gaussian surface of `net`. With dielectric boxes, the offset is chosen
/// so that the surface touches no interface (design M2c §5); a typed error if
/// no offset in `[δ₀/4, δ₀]` does.
pub(crate) fn gauss_surface(
    geom: &Geometry,
    net: NetId,
    clearance: f64,
    cfg: &SolverConfig,
    medium: Option<&Medium>,
    boxes: Option<&BoxSteps>,
) -> Result<GaussSurface, SolveError> {
    let net_extent = geom
        .net_boxes(net)
        .iter()
        .map(|&b| geom.box_(b).extent().into_iter().fold(0.0, f64::max))
        .fold(0.0, f64::max);
    let offset = cfg.walk.gauss_alpha * clearance.min(net_extent);
    let ok = |d: f64| boxes.is_none_or(|b| !surface_touches_interfaces(geom, net, d, b));
    // With boxes, the first cube is also bounded by the distance from the
    // Gaussian faces to the box interfaces (sampled on a 3×3 grid per face).
    let box_size = |d: f64| {
        let Some(b) = boxes else { return d };
        let mut worst = d;
        for &id in geom.net_boxes(net) {
            let bx = geom.box_(id).dilated(d);
            for face in Face::ALL {
                let a = face.axis.index();
                let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                for i in 0..3 {
                    for j in 0..3 {
                        let t = |k: usize, s: i32| {
                            bx.min()[k] + 0.5 * f64::from(s) * (bx.max()[k] - bx.min()[k])
                        };
                        let mut p = [0.0; 3];
                        p[a] = bx.face_coord(face);
                        p[u] = t(u, i);
                        p[v] = t(v, j);
                        worst = worst.min(b.interface_distance(p, worst));
                    }
                }
            }
        }
        worst
    };
    interface_offset(geom, net, offset, medium, boxes.is_some(), &ok, &box_size)
        .map(|d| GaussSurface::new(geom, net, d))
        .ok_or_else(|| {
            SolveError::Config(format!(
                "no Gaussian offset keeps the surface of net `{}` off the dielectric box \
                 interfaces (design M2c §5)",
                geom.net_name(net)
            ))
        })
}

/// `true` if the Gaussian surface of `net` at offset `d` (the exposed faces
/// of its dilated boxes, closed) meets an interface face of `boxes`.
fn surface_touches_interfaces(geom: &Geometry, net: NetId, d: f64, boxes: &BoxSteps) -> bool {
    let dilated: Vec<Aabb> = geom
        .net_boxes(net)
        .iter()
        .map(|&b| geom.box_(b).dilated(d))
        .collect();
    let m = boxes.medium();
    for (i, bx) in dilated.iter().enumerate() {
        for face in Face::ALL {
            let a = face.axis.index();
            let c = bx.face_coord(face);
            let (mut lo, mut hi) = (bx.min(), bx.max());
            lo[a] = c;
            hi[a] = c;
            if !m.touches(lo, hi) {
                continue;
            }
            // Only the exposed part counts: remove the parts covered by the
            // other dilated boxes on the outer side of the face.
            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
            let flat =
                |l: [f64; 3], h: [f64; 3]| Aabb::new([l[u], l[v], 0.0], [h[u], h[v], 1.0]).ok();
            let Some(region) = flat(lo, hi) else { continue };
            let mut items = vec![PaintItem {
                region,
                paint: Paint::Set(0),
            }];
            let outer = |o: &Aabb| match face.side {
                Side::Max => o.min()[a] <= c && c < o.max()[a],
                Side::Min => o.min()[a] < c && c <= o.max()[a],
            };
            for (j, o) in dilated.iter().enumerate() {
                if j != i && outer(o) {
                    if let Some(region) = flat(o.min(), o.max()) {
                        items.push(PaintItem {
                            region,
                            paint: Paint::Clear,
                        });
                    }
                }
            }
            for (piece, _) in paint(&items).boxes {
                let (mut pl, mut ph) = ([c; 3], [c; 3]);
                pl[u] = piece.min()[0];
                ph[u] = piece.max()[0];
                pl[v] = piece.min()[1];
                ph[v] = piece.max()[1];
                if m.touches(pl, ph) {
                    return true;
                }
            }
        }
    }
    false
}

/// Design M2a §4: the Gaussian offset in `[δ₀/4, δ₀]` whose smallest first
/// cube (over the net's horizontal Gaussian faces, and `δ` itself for the
/// lateral faces) is largest; the first-hop weight is ∝ 1/(cube size). Ties
/// keep the larger offset. Candidates are a uniform grid plus the midpoints
/// between the offsets at which a face meets an interface, so a face never
/// lies on an interface. Only candidates accepted by `ok` are considered
/// (`None` if there is none); `search` (dielectric boxes) forces the search
/// without a stack and bounds the score by `extra`.
fn interface_offset(
    geom: &Geometry,
    net: NetId,
    delta0: f64,
    medium: Option<&Medium>,
    search: bool,
    ok: &dyn Fn(f64) -> bool,
    extra: &dyn Fn(f64) -> f64,
) -> Option<f64> {
    if medium.is_none() && !search {
        return Some(delta0);
    }
    let lo = 0.25 * delta0;
    let size = |d: f64| {
        let Some(medium) = medium else { return d };
        let mut worst = d;
        for &b in geom.net_boxes(net) {
            let bx = geom.box_(b);
            for z in [bx.min()[2] - d, bx.max()[2] + d] {
                worst = worst.min(medium.horizontal_hop_size(z, d));
            }
        }
        worst
    };
    let mut cand: Vec<f64> = (0..=96)
        .map(|k| delta0 * (1.0 - 0.75 * f64::from(k) / 96.0))
        .collect();
    let mut cuts = vec![lo, delta0];
    if let Some(medium) = medium {
        for &b in geom.net_boxes(net) {
            let bx = geom.box_(b);
            for &zi in medium.stack().interfaces() {
                for d in [zi - bx.max()[2], bx.min()[2] - zi] {
                    if lo < d && d < delta0 {
                        cuts.push(d);
                    }
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cand.extend(cuts.windows(2).map(|w| 0.5 * (w[0] + w[1])));
    cand.sort_by(|a, b| b.total_cmp(a));
    let mut best: Option<(f64, f64)> = None;
    for d in cand {
        let s = if search {
            size(d).min(extra(d))
        } else {
            size(d)
        };
        // With boxes, a zero-size first cube (a Gaussian face on a stack
        // plane) is no candidate.
        if best.is_none_or(|(bs, _)| s > bs) && (!search || s > 0.0) && ok(d) {
            best = Some((s, d));
        }
    }
    best.map(|(_, d)| d)
}

/// Runs the FRW solver for `targets`.
///
/// Results are bit-for-bit reproducible for a given `cfg.seed`, independent of
/// the number of threads.
pub fn solve(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
) -> Result<Solution, SolveError> {
    solve_observed(geom, targets, cfg, &())
}

/// [`solve`] with progress reports and cooperative cancellation.
///
/// # Errors
///
/// Like [`solve`]. Cancellation is not an error: the returned
/// [`Solution::cancelled`] is set instead.
pub fn solve_observed(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    observer: &dyn Observer,
) -> Result<Solution, SolveError> {
    solve_hooked(geom, targets, cfg, observer, &()).map(|(sol, _)| sol)
}

/// [`solve_observed`] with a per-sample hook; returns the hook statistics of
/// each target (default for targets that never started).
pub(crate) fn solve_hooked<H: SampleHook>(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    observer: &dyn Observer,
    hook: &H,
) -> Result<(Solution, Vec<H::Stats>), SolveError> {
    let none = || (0..targets.len()).map(|_| H::Stats::default()).collect();
    validate(geom, targets, cfg)?;
    if observer.cancelled() {
        return Ok((cancelled_before_start(targets), none()));
    }
    let su = setup(geom, targets, cfg)?;
    if observer.cancelled() {
        return Ok((cancelled_before_start(targets), none()));
    }
    let mut walker = su.walker(geom, cfg);
    // Recorded samples are replayed under edits with point first hops
    // (design M2f §3.4).
    walker.windows = cfg.walk.stack.window_hops && !H::RECORD;
    let eps = su.eps_unit();
    let surfaces = targets
        .iter()
        .zip(&su.clearances)
        .map(|(&net, &clearance)| {
            gauss_surface(
                geom,
                net,
                clearance,
                cfg,
                su.medium.as_ref(),
                su.boxes.as_ref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (nets, stats): (Vec<_>, Vec<_>) = targets
        .par_iter()
        .zip(su.clearances.par_iter())
        .enumerate()
        .map(|(i, (&net, _))| {
            if observer.cancelled() {
                return (None, H::Stats::default());
            }
            let surface = &surfaces[i];
            let ctx = NetCtx {
                target: i,
                net,
                eps,
                eps_stop: su.eps_stop,
            };
            let (r, stopped, hs) = run_net(&walker, surface, ctx, cfg, observer, hook);
            (Some((r, stopped)), hs)
        })
        .unzip();
    Ok((finish_solution(nets, su.eps_stop), stats))
}

/// The solution of a solve cancelled before any target started.
fn cancelled_before_start(targets: &[NetId]) -> Solution {
    finish_solution(vec![None; targets.len()], f64::NAN)
}

/// Assembles a [`Solution`] from per-target `(result, stopped early)` pairs,
/// dropping targets that never started (`None`).
pub(crate) fn finish_solution(nets: Vec<Option<(NetResult, bool)>>, eps_stop: f64) -> Solution {
    let mut cancelled = false;
    let mut out = Vec::with_capacity(nets.len());
    for n in nets {
        match n {
            Some((r, stopped)) => {
                cancelled |= stopped;
                out.push(r);
            }
            None => cancelled = true,
        }
    }
    Solution {
        nets: out,
        eps_stop,
        cancelled,
    }
}

/// Runs the FRW solver with the transition phase delegated to a [`WalkEngine`]
/// built by `make_engine` from the prepared [`Scene`].
///
/// The estimator, stopping rules and launch sampling are identical to
/// [`solve`], but each walk draws its random numbers from its own stream (keyed
/// per launch), so results agree with [`solve`] statistically, not bit for bit.
/// For a given seed and engine, results do not depend on how the engine groups
/// walks. Targets are processed one after another; each engine call carries
/// many batches of one net.
///
/// # Errors
///
/// Returns [`SolveError::Config`] / [`SolveError::NoClearance`] like [`solve`],
/// and [`SolveError::Engine`] if the engine cannot be created or fails.
pub fn solve_with<E, F>(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    make_engine: F,
) -> Result<Solution, SolveError>
where
    E: WalkEngine + Send,
    F: FnOnce(&Scene<'_>) -> Result<E, String>,
{
    solve_with_observed(geom, targets, cfg, make_engine, &())
}

/// [`solve_with`] with progress reports and cooperative cancellation (see
/// [`solve_observed`]).
///
/// # Errors
///
/// Like [`solve_with`].
pub fn solve_with_observed<E, F>(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    make_engine: F,
    observer: &dyn Observer,
) -> Result<Solution, SolveError>
where
    E: WalkEngine + Send,
    F: FnOnce(&Scene<'_>) -> Result<E, String>,
{
    validate(geom, targets, cfg)?;
    if observer.cancelled() {
        return Ok(cancelled_before_start(targets));
    }
    let su = setup(geom, targets, cfg)?;
    // Never create the engine (possibly a GPU device) for a cancelled solve.
    if observer.cancelled() {
        return Ok(cancelled_before_start(targets));
    }
    let mut walker = su.walker(geom, cfg);
    walker.windows = cfg.walk.stack.window_hops;
    let scene = su.scene(geom, cfg);
    let mut engine = make_engine(&scene).map_err(SolveError::Engine)?;
    let eps = su.eps_unit();
    let mut nets = Vec::with_capacity(targets.len());
    for (i, (&net, &clearance)) in targets.iter().zip(&su.clearances).enumerate() {
        if observer.cancelled() {
            nets.push(None);
            continue;
        }
        let surface = gauss_surface(
            geom,
            net,
            clearance,
            cfg,
            su.medium.as_ref(),
            su.boxes.as_ref(),
        )?;
        nets.push(Some(run_net_engine(
            &walker,
            &scene,
            &mut engine,
            &surface,
            NetCtx {
                target: i,
                net,
                eps,
                eps_stop: su.eps_stop,
            },
            cfg,
            observer,
        )?));
    }
    Ok(finish_solution(nets, su.eps_stop))
}

/// Identifies the net a loop works on.
#[derive(Clone, Copy)]
pub(crate) struct NetCtx {
    pub(crate) target: usize,
    pub(crate) net: NetId,
    pub(crate) eps: f64,
    pub(crate) eps_stop: f64,
}

fn report(
    observer: &dyn Observer,
    ctx: NetCtx,
    surface: &GaussSurface,
    total: &BatchStats,
    end: Option<bool>,
) {
    let est = total
        .sinks
        .get(&Sink::Net(ctx.net))
        .map_or(Moments::new().estimate(total.walks), |m| {
            m.estimate(total.walks)
        });
    observer.progress(&Progress {
        target: ctx.target,
        net: ctx.net,
        walks: total.walks,
        total: est,
        finished: end.is_some(),
        converged: end.unwrap_or(false),
        eps_stop: ctx.eps_stop,
        stats: total,
        gauss_offset: surface.offset(),
    });
}

/// Launches of one sample: weights of up to two walks (antithetic pair).
struct PlannedSample {
    /// Launch weights (`w_pair` in [`record`]).
    w_pair: [f64; 2],
    /// Number of launches (1 or 2).
    n: u8,
}

/// Launch phase of one batch.
struct PlannedBatch {
    walks: u64,
    samples: Vec<PlannedSample>,
    launches: Vec<Launch>,
}

fn plan_batch(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    net: NetId,
    eps: f64,
    cfg: &SolverConfig,
    batch: u64,
) -> PlannedBatch {
    let mut rng = Rng::for_stream(cfg.seed, net.raw(), batch);
    let mut out = PlannedBatch {
        walks: 0,
        samples: Vec::new(),
        launches: Vec::new(),
    };
    let a_cand = surface.candidate_area();
    for _ in 0..cfg.accuracy.batch_walks {
        out.walks += 1;
        let Some(g) = surface.sample(&mut rng, walker.domain) else {
            continue;
        };
        let (h, _) = walker.clearance(g.p);
        if h <= 0.0 {
            continue;
        }
        let hop = first_hop(
            walker,
            surface,
            &g,
            h,
            cfg.walk.antithetic,
            &mut rng,
            &mut |_, _| {},
        );
        let w0 = -a_cand * eps * g.sign * hop.factor;
        for &start in &hop.starts[..hop.n] {
            let mut start = start;
            walker.domain.fold(&mut start);
            out.launches.push(Launch {
                start,
                key: rng.next_u64(),
            });
        }
        let w_pair = if hop.n == 2 {
            [0.5 * w0, -0.5 * w0]
        } else {
            [w0, 0.0]
        };
        #[allow(clippy::cast_possible_truncation)]
        out.samples.push(PlannedSample {
            w_pair,
            n: hop.n as u8,
        });
    }
    out
}

/// The first hop from Gaussian point `g` with geometric clearance `h` (shared
/// by both solver paths). The sample weight is `−A·ε·s·factor`, with `ε` the
/// solver's `eps` (it includes `ε_r` only for a homogeneous medium; layered
/// hops carry `ε_r(r)` in their factor).
fn first_hop(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    g: &GaussPoint,
    h: f64,
    antithetic: bool,
    rng: &mut Rng,
    trace: &mut dyn FnMut([f64; 3], f64),
) -> Hop {
    let Some(b) = walker.boxes else {
        return match walker.medium {
            None => homogeneous_hop(walker.green, g.p, g.axis, h, 1.0, antithetic, rng),
            Some(m) => m.first_hop(walker, surface, g, h, antithetic, rng, trace),
        };
    };
    if walker.windows {
        if let Some(hop) = crate::windows::first_hop(walker, surface, g, b, antithetic, rng, trace)
        {
            return hop;
        }
    }
    // Dielectric boxes (design M2c §5): the weight carries ε_r at the Gaussian
    // point. Outside the boxes the stack hop applies within the box-free
    // cube; otherwise a homogeneous cube clear of every interface.
    let eps_r = b.eps_at(g.p);
    let v = b.box_distance(g.p, h);
    if let (Some(m), true) = (walker.medium, v > 0.0) {
        // Bands first (their points never lie in a box: the band is within
        // half the anchor's box distance), then the larger of the stack cube
        // and the best axis cube.
        let hv = h.min(v);
        // The half-band estimator needs every point of its band to fall
        // back alike after their windows (design M2f §3.1).
        let mut allow = |lo: f64, hi: f64| {
            !walker.windows || crate::windows::band_allowed(walker, surface, g, b, lo, hi)
        };
        if let Some(hop) =
            m.band_first_hop(walker, surface, g, hv, antithetic, rng, trace, &mut allow)
        {
            return hop;
        }
        let than = m.first_cube_size(g.p[2], hv);
        if let Some(hop) =
            b.axis_first_hop(walker.green, g.p, g.axis, h, eps_r, than, antithetic, rng)
        {
            return hop;
        }
        return m.cube_first_hop(walker.green, g, hv, antithetic, rng);
    }
    // Homogeneous: clear of interface faces and stack planes, or inside the
    // material box piece holding the point (stack planes there are overridden).
    let mut clear = b.interface_distance(g.p, h);
    if let Some(m) = walker.medium {
        clear = clear.min(m.stack().nearest_two(g.p[2]).0);
    }
    let s = h.min(clear.max(b.inner_clearance(g.p)));
    if let Some(hop) = b.axis_first_hop(walker.green, g.p, g.axis, h, eps_r, s, antithetic, rng) {
        return hop;
    }
    if s <= 0.0 {
        // A Gaussian point on an interface (measure zero unless a Gaussian face
        // lies in an interface plane; the offset choice avoids that): no
        // contribution.
        return Hop {
            starts: [g.p, g.p],
            n: 1,
            factor: 0.0,
        };
    }
    homogeneous_hop(walker.green, g.p, g.axis, s, eps_r, antithetic, rng)
}

fn finish_batch(
    plan: &PlannedBatch,
    outcomes: &[Outcome],
    ground_nets: &[NetId],
    odd_image: bool,
) -> BatchStats {
    let mut st = BatchStats {
        walks: plan.walks,
        ..BatchStats::default()
    };
    let mut it = outcomes.iter();
    for s in &plan.samples {
        let mut hits = [None, None];
        for (slot, hit) in hits.iter_mut().enumerate().take(usize::from(s.n)) {
            let o = it.next().expect("one outcome per launch");
            st.steps += u64::from(o.steps) + 1;
            match o.sink {
                Some(sink) => *hit = Some((sink, s.w_pair[slot] * o.sign)),
                None => st.aborted += 1,
            }
        }
        record(&mut st, hits, ground_nets, odd_image, s.w_pair);
    }
    st
}

#[allow(clippy::too_many_arguments)]
fn run_net_engine<E: WalkEngine + Send>(
    walker: &Walker<'_>,
    scene: &Scene<'_>,
    engine: &mut E,
    surface: &GaussSurface,
    ctx: NetCtx,
    cfg: &SolverConfig,
    observer: &dyn Observer,
) -> Result<(NetResult, bool), SolveError> {
    let NetCtx { net, eps, .. } = ctx;
    let acc = &cfg.accuracy;
    let per_batch = (u64::from(acc.batch_walks) * if cfg.walk.antithetic { 2 } else { 1 }).max(1);
    let max_wave = (engine.preferred_walks() as u64 / per_batch).max(1);
    let batch_walks = u64::from(acc.batch_walks);
    // Never plan past the walk budget.
    let max_batches = acc.max_walks.div_ceil(batch_walks).max(1);
    // Waves start just large enough for `min_walks` and double up to the
    // engine's preferred size, so small nets do not pay for a full wave.
    // Results do not depend on wave sizes.
    let mut wave = acc.min_walks.div_ceil(batch_walks).clamp(1, max_wave);
    let plan_wave = |first: u64, size: u64| -> Vec<PlannedBatch> {
        (first..(first + size).min(max_batches))
            .into_par_iter()
            .map(|b| plan_batch(walker, surface, net, eps, cfg, b))
            .collect()
    };
    let odd = walker.domain.odd_image();
    let finish_wave = |plans: &[PlannedBatch], outcomes: &[Outcome]| -> Vec<BatchStats> {
        let mut offsets = Vec::with_capacity(plans.len());
        let mut offset = 0;
        for p in plans {
            offsets.push(offset);
            offset += p.launches.len();
        }
        plans
            .par_iter()
            .zip(offsets.par_iter())
            .map(|(p, &o)| {
                finish_batch(p, &outcomes[o..o + p.launches.len()], &cfg.ground_nets, odd)
            })
            .collect()
    };
    let mut total = BatchStats::default();
    let mut is_converged = false;
    let mut plans = plan_wave(0, wave);
    let mut next_batch = wave;
    // The previous wave, whose statistics are computed while the engine runs
    // the current one. Batches are still merged strictly in order, so the
    // result does not depend on this overlap (a converged run just discards
    // the wave in flight).
    let mut prev: Option<(Vec<PlannedBatch>, Vec<Outcome>)> = None;
    let mut stopped = false;
    loop {
        let launches: Vec<Launch> = plans
            .iter()
            .flat_map(|p| p.launches.iter().copied())
            .collect();
        let (outcomes, next, stats) = std::thread::scope(|s| {
            let h = s.spawn(|| {
                if launches.is_empty() {
                    Ok(Vec::new())
                } else {
                    engine.run(scene, &launches)
                }
            });
            let next = plan_wave(next_batch, (2 * wave).min(max_wave));
            let stats = prev.as_ref().map(|(p, o)| finish_wave(p, o));
            (h.join(), next, stats)
        });
        let outcomes = outcomes
            .map_err(|_| SolveError::Engine("walk engine panicked".into()))?
            .map_err(SolveError::Engine)?;
        if outcomes.len() != launches.len() {
            return Err(SolveError::Engine(format!(
                "engine returned {} outcomes for {} launches",
                outcomes.len(),
                launches.len()
            )));
        }
        wave = (2 * wave).min(max_wave);
        next_batch += wave;
        let mut done = false;
        for b in stats.iter().flatten() {
            total.merge(b);
            if converged(&total, net, acc) {
                is_converged = true;
                done = true;
                break;
            }
            if total.walks >= acc.max_walks {
                done = true;
                break;
            }
        }
        // Stop when converged, or when the wave just run was empty (every
        // planned batch has been merged).
        if done || plans.is_empty() {
            break;
        }
        if stats.is_some() {
            report(observer, ctx, surface, &total, None);
        }
        if observer.cancelled() {
            stopped = true;
            break;
        }
        prev = Some((std::mem::replace(&mut plans, next), outcomes));
    }
    let converged = is_converged && total.aborted == 0;
    report(observer, ctx, surface, &total, Some(converged));
    Ok((
        NetResult {
            net,
            walks: total.walks,
            aborted_walks: total.aborted,
            total_steps: total.steps,
            converged,
            gauss_offset: surface.offset(),
            sinks: total.sinks.into_iter().collect(),
            ground_moments: total.ground,
        },
        stopped,
    ))
}

/// One finished sample, as seen by a [`SampleHook`].
#[cfg_attr(not(feature = "sensitivity"), allow(dead_code))]
pub(crate) struct HookSample {
    /// Batch index.
    pub(crate) batch: u64,
    /// Sample index within the batch (counts rejected candidates too).
    pub(crate) index: u32,
    /// Walk ends of the launches (antithetic pair; `[end, None]` otherwise).
    pub(crate) ends: [Option<WalkEnd>; 2],
    /// Launch weights (`w_pair` in [`record`]).
    pub(crate) w_pair: [f64; 2],
    /// With [`SampleHook::RECORD`]: the batch stream at the start of the
    /// sample, and the bounding box `[min, max]` of every clearance cube of
    /// the sample (first hop and all walks).
    pub(crate) record: Option<(Rng, [[f64; 3]; 2])>,
}

/// One Gaussian sample with its launches (the body of the batch loop).
pub(crate) struct OneSample {
    pub(crate) ends: [Option<WalkEnd>; 2],
    pub(crate) w_pair: [f64; 2],
}

/// Draws one Gaussian point from `rng` and runs its launches; `None` if the
/// candidate is rejected. `trace` sees every clearance cube. Shared by the
/// batch loop and by replays (which must reproduce it exactly).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_once<T: FnMut([f64; 3], f64)>(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    a_cand: f64,
    eps: f64,
    antithetic: bool,
    rng: &mut Rng,
    trace: &mut T,
) -> Option<OneSample> {
    let g = surface.sample(rng, walker.domain)?;
    let (h, _) = walker.clearance(g.p);
    if h <= 0.0 {
        return None;
    }
    trace(g.p, h);
    let hop = first_hop(walker, surface, &g, h, antithetic, rng, trace);
    let w0 = -a_cand * eps * g.sign * hop.factor;
    let mut launch = |k: usize, rng: &mut Rng| {
        let mut start = hop.starts[k];
        walker.domain.fold(&mut start);
        walker.run_traced(start, rng, trace)
    };
    Some(if hop.n == 2 {
        let e0 = launch(0, rng);
        let e1 = launch(1, rng);
        OneSample {
            ends: [Some(e0), Some(e1)],
            w_pair: [0.5 * w0, -0.5 * w0],
        }
    } else {
        OneSample {
            ends: [Some(launch(0, rng)), None],
            w_pair: [w0, 0.0],
        }
    })
}

/// Per-sample extension point of the batch loop (D3 sensitivities).
///
/// The hook only observes finished samples and must not touch the batch
/// stream, so the statistics of [`solve`] are unchanged by it. The `()` hook
/// does nothing and compiles away.
pub(crate) trait SampleHook: Sync {
    /// Per-batch statistics, merged in batch order like [`BatchStats`].
    type Stats: Default + Send;
    /// Fill [`HookSample::record`] (costs a stream copy and the cube
    /// bounds per sample).
    const RECORD: bool = false;
    /// Observes one sample that produced launches.
    fn sample(&self, stats: &mut Self::Stats, sample: &HookSample);
    /// Merges `from` into `into`.
    fn merge(into: &mut Self::Stats, from: &Self::Stats);
}

impl SampleHook for () {
    type Stats = ();
    #[inline]
    fn sample(&self, (): &mut (), _sample: &HookSample) {}
    #[inline]
    fn merge((): &mut (), (): &()) {}
}

#[allow(clippy::too_many_arguments)]
fn run_batch<H: SampleHook>(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    net: NetId,
    eps: f64,
    cfg: &SolverConfig,
    batch: u64,
    hook: &H,
) -> (BatchStats, H::Stats) {
    let mut rng = Rng::for_stream(cfg.seed, net.raw(), batch);
    let mut st = BatchStats::default();
    let mut hs = H::Stats::default();
    let a_cand = surface.candidate_area();
    for index in 0..cfg.accuracy.batch_walks {
        st.walks += 1;
        let (out, rec) = if H::RECORD {
            let start = rng.clone();
            let mut bb = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
            let out = sample_once(
                walker,
                surface,
                a_cand,
                eps,
                cfg.walk.antithetic,
                &mut rng,
                &mut |x, h| {
                    for a in 0..3 {
                        bb[0][a] = bb[0][a].min(x[a] - h);
                        bb[1][a] = bb[1][a].max(x[a] + h);
                    }
                },
            );
            (out, Some((start, bb)))
        } else {
            let out = sample_once(
                walker,
                surface,
                a_cand,
                eps,
                cfg.walk.antithetic,
                &mut rng,
                &mut |_, _| {},
            );
            (out, None)
        };
        let Some(OneSample { ends, w_pair }) = out else {
            continue;
        };
        for e in ends.iter().flatten() {
            st.steps += u64::from(e.steps) + 1;
            if e.sink.is_none() {
                st.aborted += 1;
            }
        }
        let hits = hits_of(&ends, w_pair);
        record(
            &mut st,
            hits,
            &cfg.ground_nets,
            walker.domain.odd_image(),
            w_pair,
        );
        hook.sample(
            &mut hs,
            &HookSample {
                batch,
                index,
                ends,
                w_pair,
                record: rec,
            },
        );
    }
    (st, hs)
}

/// Sink hits `(sink, weight × sign)` of a sample's walk ends (aborted walks: `None`).
#[inline]
pub(crate) fn hits_of(ends: &[Option<WalkEnd>; 2], w_pair: [f64; 2]) -> [Option<(Sink, f64)>; 2] {
    std::array::from_fn(|slot| {
        let e = ends[slot]?;
        e.sink.map(|sink| (sink, w_pair[slot] * e.sign))
    })
}

/// Adds one sample's contributions (up to two sinks) to the statistics.
///
/// With the odd image extension (open boundary + substrate), weights are only
/// meaningful for conductor sinks: the image flips the sign of walks that
/// cross the substrate, which is valid for conductor potentials but not for
/// the fixed-ground indicators (substrate, infinity). Their combined
/// contribution is recovered from the partition of unity `Σ_k C_ik = 0`: per
/// walk, the fixed-ground part is `w_launch − w_conductor`, recorded under
/// [`Sink::GroundPlane`].
fn record(
    st: &mut BatchStats,
    hits: [Option<(Sink, f64)>; 2],
    ground_nets: &[NetId],
    odd_image: bool,
    w_pair: [f64; 2],
) {
    let c = sample_contribs(hits, odd_image, w_pair);
    let mut ground = 0.0;
    let mut any_ground = false;
    for &(sink, w) in c.as_slice() {
        st.sinks.entry(sink).or_default().add(w);
        if is_ground_sink(sink, ground_nets) {
            ground += w;
            any_ground = true;
        }
    }
    if any_ground {
        st.ground.add(ground);
    }
}

/// `true` for sinks that count towards the aggregate ground estimate.
#[inline]
pub(crate) fn is_ground_sink(sink: Sink, ground_nets: &[NetId]) -> bool {
    sink.is_fixed_ground() || sink.net().is_some_and(|n| ground_nets.contains(&n))
}

/// Per-sink contributions of one sample (at most two hits plus the
/// reconstructed ground part per hit; see [`record`]).
#[derive(Clone, Copy)]
pub(crate) struct Contribs {
    buf: [(Sink, f64); 4],
    len: usize,
}

impl Contribs {
    pub(crate) fn as_slice(&self) -> &[(Sink, f64)] {
        &self.buf[..self.len]
    }
}

#[inline]
pub(crate) fn sample_contribs(
    hits: [Option<(Sink, f64)>; 2],
    odd_image: bool,
    w_pair: [f64; 2],
) -> Contribs {
    let mut c = Contribs {
        buf: [(Sink::Infinity, 0.0); 4],
        len: 0,
    };
    let mut add = |sink: Sink, w: f64| {
        if let Some(e) = c.buf[..c.len].iter_mut().find(|(s, _)| *s == sink) {
            e.1 += w;
        } else {
            c.buf[c.len] = (sink, w);
            c.len += 1;
        }
    };
    for (slot, hit) in hits.into_iter().enumerate() {
        let Some((sink, w)) = hit else { continue };
        if odd_image {
            if sink.is_fixed_ground() {
                add(Sink::GroundPlane, w_pair[slot]);
            } else {
                add(sink, w);
                add(Sink::GroundPlane, w_pair[slot] - w);
            }
        } else {
            add(sink, w);
        }
    }
    c
}

fn converged(st: &BatchStats, net: NetId, acc: &Accuracy) -> bool {
    if st.walks < acc.min_walks {
        return false;
    }
    let get = |s: Sink| {
        st.sinks
            .get(&s)
            .map_or(Moments::new().estimate(st.walks), |m| m.estimate(st.walks))
    };
    let total = get(Sink::Net(net));
    if total.value.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        || !total.meets(acc.total_rel_tol, acc.confidence_sigma)
    {
        return false;
    }
    st.sinks.iter().all(|(&sink, m)| {
        if sink == Sink::Net(net) || sink.net().is_none() {
            return true;
        }
        let e = m.estimate(st.walks);
        let tracked =
            e.value.abs() >= acc.track_rel * total.value || e.value.abs() >= acc.track_abs_ff;
        !tracked || e.meets(acc.coupling_rel_tol, acc.confidence_sigma)
    })
}

pub(crate) fn run_net<H: SampleHook>(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    ctx: NetCtx,
    cfg: &SolverConfig,
    observer: &dyn Observer,
    hook: &H,
) -> (NetResult, bool, H::Stats) {
    let NetCtx { net, eps, .. } = ctx;
    let acc = &cfg.accuracy;
    let wave = rayon::current_num_threads().max(1) as u64;
    let mut total = BatchStats::default();
    let mut total_hook = H::Stats::default();
    let mut next_batch = 0u64;
    let mut done = false;
    let mut is_converged = false;
    let mut stopped = false;
    while !done {
        // Compute a wave of batches in parallel, then merge strictly in batch order.
        let batches: Vec<(BatchStats, H::Stats)> = (next_batch..next_batch + wave)
            .into_par_iter()
            .map(|b| run_batch(walker, surface, net, eps, cfg, b, hook))
            .collect();
        next_batch += wave;
        for (b, hb) in &batches {
            total.merge(b);
            H::merge(&mut total_hook, hb);
            if converged(&total, net, acc) {
                is_converged = true;
                done = true;
                break;
            }
            if total.walks >= acc.max_walks {
                done = true;
                break;
            }
        }
        if !done {
            report(observer, ctx, surface, &total, None);
            stopped = observer.cancelled();
            done = stopped;
        }
    }
    // Aborted walks bias the estimate, so they never count as converged.
    let converged = is_converged && total.aborted == 0;
    report(observer, ctx, surface, &total, Some(converged));
    (
        NetResult {
            net,
            walks: total.walks,
            aborted_walks: total.aborted,
            total_steps: total.steps,
            converged,
            gauss_offset: surface.offset(),
            sinks: total.sinks.into_iter().collect(),
            ground_moments: total.ground,
        },
        stopped,
        total_hook,
    )
}
