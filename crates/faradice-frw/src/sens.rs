//! Capacitance sensitivities by finite differences with path continuation
//! (design D3, pillar A0).
//!
//! Each requested face of a box is moved **inward** by `δ` (and `2δ` for
//! Richardson extrapolation). Shrinking keeps every transition cube of every
//! nominal walk conductor-free, and keeps every Gaussian surface valid, so a
//! nominal path is also a valid path in the perturbed geometry. Only its
//! termination can change: a walk that stopped within `ε_stop` of the moved
//! box is re-examined in the perturbed geometry and, if it is no longer
//! absorbed there, *continued* until it is. Each sample therefore yields an
//! exact paired difference `Δw = w(perturbed) − w(nominal)`, and the
//! derivative with respect to an *outward* displacement of the face is
//!
//! ```text
//! dC_ik/dn ≈ −E[Δw_k(δ)]/δ                          (one-sided)
//! dC_ik/dn ≈ −E[2Δw_k(δ) − Δw_k(2δ)/2]/δ            (Richardson, O(δ²))
//! ```
//!
//! The nominal walks are exactly those of [`solve`](crate::solve):
//! continuations draw from their own streams, so the nominal results are bit
//! for bit identical to a plain solve with the same configuration.

use std::collections::BTreeMap;

use faradice_core::{Axis, BoxId, Estimate, Face, Moments, NetId, Side, Sink};
use faradice_geom::{Aabb, Geometry, GeometryBuilder, GridOptions, UniformGrid};
use rayon::prelude::*;

use crate::axis::BoxSteps;
use crate::boundary::BoundaryModel;
use crate::dielectric::BoxMedium;
use crate::regions::{ConformalRule, DielectricRegions};
use crate::rng::{Rng, splitmix64};
pub use crate::shell::ShellResult;
use crate::shell::{ShellCtx, ShellFace, shell_face, shell_term};
use crate::solver::{
    HookSample, NetCtx, NetResult, SampleHook, SolveError, SolverConfig, gauss_surface, hits_of,
    is_ground_sink, run_net, sample_contribs, setup,
};
use crate::stack::Stack;
use crate::walk::{Overlay, WalkEnd, Walker};

/// Domain-separation tag of the continuation streams.
const CONTINUATION_TAG: u64 = 0x5E45_C0A7_1A7E_D3A0;

/// Domain-separation tag of the deletion continuation streams.
const DELETION_TAG: u64 = 0xD3DE_1E7E_5E7D_E1E7;

/// Conductors deleted together (design D3 §8). Removing conductors is a
/// shrink to nothing: every nominal path stays valid, and only walks absorbed
/// by a deleted conductor are continued. The result is the exact paired
/// difference `ΔC_ik` and the capacitances of the edited geometry, with no
/// step and no bias. A deleted net that is also a target has no row in the
/// edited geometry: its [`DeletionResult`] stays empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deletion {
    /// The nets to delete (not targets).
    pub nets: Vec<NetId>,
}

/// Statistics of one target net for one [`Deletion`].
#[derive(Debug, Clone, Default)]
pub struct DeletionResult {
    /// Per-sink moments of the per-sample difference `w(edited) − w(nominal)` (fF).
    pub delta: Vec<(Sink, Moments)>,
    /// Moments of the per-sample difference of the ground sum.
    pub delta_ground: Moments,
    /// Per-sink moments in the edited geometry (fF).
    pub edited: Vec<(Sink, Moments)>,
    /// Moments of the per-sample ground sum in the edited geometry.
    pub edited_ground: Moments,
    /// Walks that were continued in the edited geometry.
    pub continued_walks: u64,
    /// Continuations that exceeded `max_steps` (their walk counts as unchanged).
    pub aborted_continuations: u64,
    /// Transition steps spent in continuations.
    pub continuation_steps: u64,
}

impl DeletionResult {
    /// `ΔC_ik` (fF) for sink `k`, over `walks` nominal samples.
    #[must_use]
    pub fn delta_c(&self, sink: Sink, walks: u64) -> Estimate {
        lookup(&self.delta, sink, walks)
    }

    /// Change of the total ground capacitance (positive convention).
    #[must_use]
    pub fn delta_ground(&self, walks: u64) -> Estimate {
        self.delta_ground.estimate(walks).scaled(-1.0)
    }

    /// `C_ik` (fF) in the edited geometry.
    #[must_use]
    pub fn c(&self, sink: Sink, walks: u64) -> Estimate {
        lookup(&self.edited, sink, walks)
    }

    /// Total ground capacitance in the edited geometry (positive convention).
    #[must_use]
    pub fn ground(&self, walks: u64) -> Estimate {
        self.edited_ground.estimate(walks).scaled(-1.0)
    }

    /// `false` if any continuation was aborted (the result is then biased).
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.aborted_continuations == 0
    }
}

/// Everything [`solve_sensitivity_with`] computes besides the nominal solve.
#[derive(Debug, Clone, Copy, Default)]
pub struct SensRequest<'a> {
    /// Faces to differentiate with respect to.
    pub faces: &'a [FaceStep],
    /// Weighted face combinations.
    pub params: &'a [Param],
    /// Conductor sets to delete.
    pub deletions: &'a [Deletion],
    /// Options of the face derivatives.
    pub opts: SensOptions,
}

/// One face to differentiate with respect to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceStep {
    /// The box.
    pub box_id: BoxId,
    /// The face of the box; the derivative is with respect to its outward
    /// displacement.
    pub face: Face,
    /// Inward finite-difference step `δ` in µm.
    pub delta: f64,
    /// A rectangle of the face (tangential coordinates, as in
    /// [`covered_parts`]) that does **not** move: only the rest of the face
    /// shrinks. Used to differentiate with respect to part of a face, e.g. an
    /// edge strip next to an interior handled by pillar A.
    pub keep: Option<FaceRect>,
}

/// Options of [`solve_sensitivity`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensOptions {
    /// Combine the steps `δ` and `2δ` to cancel the `O(δ)` bias.
    pub richardson: bool,
    /// For faces of boxes tagged by a conformal rule, return only the
    /// conductor term `T^cond` of design D3 §4.4c: the conductor face moves,
    /// its shell does not. This is **not** `dC/dn` (the shell interface term
    /// `T^Σ` is left out); it exists to validate the split.
    pub shell_conductor_term: bool,
    /// Points on the moving shell face per face and target for `T^Σ`
    /// (design D3 §4.4c). Each point costs 12 walks.
    pub shell_points: u64,
    /// Edge margin of `T^Σ` in units of the shell thickness: fields at
    /// points of the moving shell face closer than this to its edges are
    /// taken at the nearest point at that distance (design D3 §4.4c).
    pub shell_edge_margin: f64,
}

impl Default for SensOptions {
    fn default() -> Self {
        Self {
            richardson: true,
            shell_conductor_term: false,
            shell_points: 20_000,
            shell_edge_margin: 1.0,
        }
    }
}

/// A parameter: a linear combination of requested faces,
/// `dC/dp = Σ w_f · dC/dn_f`.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// `(index into the requested faces, weight)`.
    pub faces: Vec<(usize, f64)>,
}

/// Derivative statistics of one target net with respect to one [`Param`].
#[derive(Debug, Clone, Default)]
pub struct ParamResult {
    /// Per-sink moments of the per-sample combined estimate (fF/µm).
    pub sinks: Vec<(Sink, Moments)>,
    /// Moments of the per-sample sum over ground-like sinks.
    pub ground: Moments,
    /// `Σ_f w_f T_f^Σ` per sink for faces with conformal shells (independent
    /// of the walks above), and its ground sum (sign as in `ground`).
    pub shell: Vec<(Sink, Estimate)>,
    /// See `shell`.
    pub shell_ground: Option<Estimate>,
}

impl ParamResult {
    /// The sinks with a derivative (finite differences or shell term).
    #[must_use]
    pub fn sink_keys(&self) -> Vec<Sink> {
        union_keys(
            self.sinks.iter().map(|(k, _)| *k),
            self.shell.iter().map(|(k, _)| *k),
        )
    }

    /// `dC_ik/dp` for sink `k`, over `walks` nominal samples.
    #[must_use]
    pub fn dc(&self, sink: Sink, walks: u64) -> Estimate {
        let e = lookup(&self.sinks, sink, walks);
        match self.shell.iter().find(|(k, _)| *k == sink) {
            Some(&(_, s)) => sum(e, s),
            None => e,
        }
    }

    /// Derivative of the total ground capacitance (positive convention).
    #[must_use]
    pub fn d_ground(&self, walks: u64) -> Estimate {
        let e = self.ground.estimate(walks);
        self.shell_ground.map_or(e, |s| sum(e, s)).scaled(-1.0)
    }
}

fn union_keys(a: impl Iterator<Item = Sink>, b: impl Iterator<Item = Sink>) -> Vec<Sink> {
    let mut v: Vec<Sink> = a.chain(b).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Sum of independent estimates.
fn sum(a: Estimate, b: Estimate) -> Estimate {
    Estimate {
        value: a.value + b.value,
        std_err: a.std_err.hypot(b.std_err),
    }
}

/// Raw derivative statistics of one target net with respect to one face.
#[derive(Debug, Clone, Default)]
pub struct FaceResult {
    /// Per-sink moments of the per-sample derivative estimate (fF/µm), in the
    /// method selected by [`SensOptions::richardson`].
    pub sinks: Vec<(Sink, Moments)>,
    /// Moments of the per-sample sum over ground-like sinks.
    pub ground: Moments,
    /// One-sided estimate (step `δ` only), always available.
    pub one_sided: Vec<(Sink, Moments)>,
    /// Walks that were continued in a perturbed geometry.
    pub continued_walks: u64,
    /// Continuations that exceeded `max_steps` (their sample counts as unchanged).
    pub aborted_continuations: u64,
    /// Transition steps spent in continuations.
    pub continuation_steps: u64,
    /// The shell-interface term `T^Σ` (design D3 §4.4c) for a face of a box
    /// with a conformal shell; included in [`Self::dc`], [`Self::dc_one_sided`]
    /// and [`Self::d_ground`]. `None` otherwise (and with
    /// [`SensOptions::shell_conductor_term`]).
    pub shell: Option<ShellResult>,
}

impl FaceResult {
    /// `false` if any continuation or shell-term walk was aborted, or a shell
    /// point had no field cube (the estimate is then biased).
    #[must_use]
    pub fn converged(&self) -> bool {
        self.aborted_continuations == 0
            && self
                .shell
                .as_ref()
                .is_none_or(|s| s.aborted == 0 && s.skipped == 0)
    }

    fn with_shell(&self, e: Estimate, sink: Sink) -> Estimate {
        match &self.shell {
            Some(s) => sum(e, lookup(&s.sinks, sink, s.points)),
            None => e,
        }
    }

    /// The sinks with a derivative (finite differences or shell term).
    #[must_use]
    pub fn sink_keys(&self) -> Vec<Sink> {
        union_keys(
            self.sinks.iter().map(|(k, _)| *k),
            self.shell
                .iter()
                .flat_map(|s| s.sinks.iter().map(|(k, _)| *k)),
        )
    }

    /// `T^Σ` alone for sink `k` (`None` without a shell term).
    #[must_use]
    pub fn dc_shell(&self, sink: Sink) -> Option<Estimate> {
        self.shell
            .as_ref()
            .map(|s| lookup(&s.sinks, sink, s.points))
    }
}

impl FaceResult {
    /// `dC_ik/dn` (fF/µm) for sink `k`, over `walks` nominal samples.
    #[must_use]
    pub fn dc(&self, sink: Sink, walks: u64) -> Estimate {
        self.with_shell(lookup(&self.sinks, sink, walks), sink)
    }

    /// One-sided `dC_ik/dn` (fF/µm).
    #[must_use]
    pub fn dc_one_sided(&self, sink: Sink, walks: u64) -> Estimate {
        self.with_shell(lookup(&self.one_sided, sink, walks), sink)
    }

    /// Derivative of the total ground capacitance (positive convention, as
    /// [`NetResult::ground`]).
    #[must_use]
    pub fn d_ground(&self, walks: u64) -> Estimate {
        let e = self.ground.estimate(walks);
        match &self.shell {
            Some(s) => sum(e, s.ground.estimate(s.points)),
            None => e,
        }
        .scaled(-1.0)
    }
}

fn lookup(v: &[(Sink, Moments)], sink: Sink, walks: u64) -> Estimate {
    v.iter()
        .find(|(s, _)| *s == sink)
        .map_or(Moments::new().estimate(walks), |(_, m)| m.estimate(walks))
}

/// Sensitivities of one target net.
#[derive(Debug, Clone)]
pub struct SensNetResult {
    /// Nominal statistics, identical to [`solve`](crate::solve).
    pub nominal: NetResult,
    /// One entry per requested face, in request order.
    pub faces: Vec<FaceResult>,
    /// One entry per requested parameter, in request order.
    pub params: Vec<ParamResult>,
    /// One entry per requested deletion, in request order.
    pub deletions: Vec<DeletionResult>,
}

/// Result of [`solve_sensitivity`].
#[derive(Debug, Clone)]
pub struct SensSolution {
    /// One entry per target, in request order.
    pub nets: Vec<SensNetResult>,
    /// Stopping distance used (µm).
    pub eps_stop: f64,
}

/// One perturbed geometry (a face shrunk by `level · δ`).
struct Perturbed {
    /// The edit: the moved box plus fillers, replacing the nominal box.
    overlay: Overlay,
    /// The moved box in this geometry.
    moved: Aabb,
    /// For a box with a conformal shell: the medium `M₁` of design D3 §4.4c
    /// (`None` inside if it has no boxes). `None` keeps the nominal medium.
    boxes: Option<Option<BoxSteps>>,
}

struct FaceCtx {
    /// `(box, face)`, keys the continuation streams.
    handle: (BoxId, Face),
    /// The box's net.
    net: NetId,
    /// The box in the nominal geometry.
    nominal: Aabb,
    delta: f64,
    /// Levels 1 (`δ`) and, with Richardson, 2 (`2δ`).
    levels: Vec<Perturbed>,
    /// The moving shell face, for `T^Σ`.
    shell: Option<ShellFace>,
}

/// Per-face accumulators of one batch.
#[derive(Default)]
struct FaceAcc {
    sinks: BTreeMap<Sink, Moments>,
    one_sided: BTreeMap<Sink, Moments>,
    ground: Moments,
    continued: u64,
    aborted: u64,
    steps: u64,
}

impl FaceAcc {
    fn one_sided_add(&mut self, v: &[(Sink, f64, f64)]) {
        for &(k, one, _) in v {
            self.one_sided.entry(k).or_default().add(one);
        }
    }

    fn merge(&mut self, o: &Self) {
        for (k, m) in &o.sinks {
            self.sinks.entry(*k).or_default().merge(m);
        }
        for (k, m) in &o.one_sided {
            self.one_sided.entry(*k).or_default().merge(m);
        }
        self.ground.merge(&o.ground);
        self.continued += o.continued;
        self.aborted += o.aborted;
        self.steps += o.steps;
    }
}

/// A deletion: the edited geometry and the map of its net ids back to the
/// nominal ones.
struct DelCtx {
    nets: Vec<NetId>,
    /// The remaining geometry and its index; `None` if every conductor is
    /// deleted (then every target is deleted too and nothing is continued).
    edited: Option<(Geometry, UniformGrid)>,
    new_to_old: Vec<NetId>,
    key: u64,
}

impl DelCtx {
    fn new(geom: &Geometry, d: &Deletion) -> Result<Self, SolveError> {
        let mut nets = d.nets.clone();
        nets.sort_unstable();
        nets.dedup();
        if nets.is_empty() {
            return Err(SolveError::Config("a deletion names no net".into()));
        }
        for &n in &nets {
            if n.index() >= geom.num_nets() {
                return Err(SolveError::Config(format!("deletion: unknown {n}")));
            }
        }
        let err = |e: faradice_geom::GeomError| SolveError::Config(e.to_string());
        let mut gb = GeometryBuilder::new();
        let mut new_to_old = Vec::new();
        for n in geom.nets().filter(|n| !nets.contains(n)) {
            gb.net(geom.net_name(n)).map_err(err)?;
            new_to_old.push(n);
        }
        for (i, b) in geom.boxes().iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let n = geom.box_net(BoxId::new(i as u32));
            if let Some(new) = new_to_old.iter().position(|&o| o == n) {
                gb.add_box(
                    NetId::from_index(new).map_err(|e| SolveError::Config(e.to_string()))?,
                    *b,
                )
                .map_err(err)?;
            }
        }
        let edited = if new_to_old.is_empty() {
            None
        } else {
            let g = gb.build().map_err(err)?;
            let index = UniformGrid::new(&g, GridOptions::default());
            Some((g, index))
        };
        let mut key = DELETION_TAG;
        for n in &nets {
            key = splitmix64(&mut key) ^ u64::from(n.raw());
        }
        Ok(Self {
            nets,
            edited,
            new_to_old,
            key,
        })
    }

    fn remap(&self, s: Sink) -> Sink {
        match s {
            Sink::Net(n) => Sink::Net(self.new_to_old[n.index()]),
            other => other,
        }
    }
}

/// Per-deletion accumulators of one batch.
#[derive(Default)]
struct DelAcc {
    delta: BTreeMap<Sink, Moments>,
    delta_ground: Moments,
    edited: BTreeMap<Sink, Moments>,
    edited_ground: Moments,
    continued: u64,
    aborted: u64,
    steps: u64,
}

impl DelAcc {
    fn merge(&mut self, o: &Self) {
        for (k, m) in &o.delta {
            self.delta.entry(*k).or_default().merge(m);
        }
        for (k, m) in &o.edited {
            self.edited.entry(*k).or_default().merge(m);
        }
        self.delta_ground.merge(&o.delta_ground);
        self.edited_ground.merge(&o.edited_ground);
        self.continued += o.continued;
        self.aborted += o.aborted;
        self.steps += o.steps;
    }
}

/// Per-sink `(one-sided, extrapolated)` differences of one sample on one face.
type FaceDiffs = Vec<(Sink, f64, f64)>;

/// Faces by position: a walk end can only be continued for a face whose
/// nominal box lies within `ε_stop` of it (see `Hook::continue_end`), so
/// each sample visits only the faces hashed to the cells of its ends.
struct FaceLookup {
    inv: f64,
    cells: std::collections::HashMap<[i64; 3], Vec<u32>>,
    /// Faces whose box spans too many cells: always visited.
    always: Vec<u32>,
}

impl FaceLookup {
    const MAX_CELLS: i64 = 1 << 16;

    fn new(faces: &[FaceCtx], eps_stop: f64, cell: f64) -> Self {
        let inv = 1.0 / cell;
        let mut cells: std::collections::HashMap<[i64; 3], Vec<u32>> =
            std::collections::HashMap::new();
        let mut always = Vec::new();
        for (i, f) in faces.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let i = i as u32;
            #[allow(clippy::cast_possible_truncation)]
            let range = |a: usize| {
                (
                    ((f.nominal.min()[a] - eps_stop) * inv).floor() as i64,
                    ((f.nominal.max()[a] + eps_stop) * inv).floor() as i64,
                )
            };
            let r = [range(0), range(1), range(2)];
            let n = r.iter().map(|(lo, hi)| hi - lo + 1).product::<i64>();
            if n > Self::MAX_CELLS {
                always.push(i);
                continue;
            }
            for x in r[0].0..=r[0].1 {
                for y in r[1].0..=r[1].1 {
                    for z in r[2].0..=r[2].1 {
                        cells.entry([x, y, z]).or_default().push(i);
                    }
                }
            }
        }
        Self { inv, cells, always }
    }

    /// The faces that the ends of `s` may change, sorted and unique.
    fn candidates(&self, s: &HookSample, out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(&self.always);
        for end in s.ends.iter().flatten() {
            if !matches!(end.sink, Some(Sink::Net(_))) {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            let c = end.pos.map(|v| (v * self.inv).floor() as i64);
            if let Some(v) = self.cells.get(&c) {
                out.extend_from_slice(v);
            }
        }
        out.sort_unstable();
        out.dedup();
    }
}

struct Hook<'a> {
    faces: &'a [FaceCtx],
    lookup: &'a FaceLookup,
    deletions: &'a [DelCtx],
    params: &'a [Param],
    template: &'a Walker<'a>,
    seed: u64,
    net: NetId,
    eps_stop: f64,
    odd_image: bool,
    richardson: bool,
    ground_nets: &'a [NetId],
}

impl Hook<'_> {
    fn walker<'b>(&'b self, p: &'b Perturbed) -> Walker<'b> {
        Walker {
            overlay: Some(&p.overlay),
            boxes: match &p.boxes {
                Some(b) => b.as_ref(),
                None => self.template.boxes,
            },
            ..*self.template
        }
    }

    /// The continuation stream of one walk, keyed by the face *handle* (not
    /// its position in the request), so results do not depend on which or how
    /// many faces are requested together.
    fn stream(&self, s: &HookSample, slot: usize, f: &FaceCtx, level: usize) -> Rng {
        let mut h = self.seed ^ CONTINUATION_TAG;
        for v in [
            u64::from(self.net.raw()),
            s.batch,
            u64::from(s.index),
            slot as u64,
            u64::from(f.handle.0.raw()),
            f.handle.1.index() as u64,
            level as u64,
        ] {
            h = splitmix64(&mut h) ^ v;
        }
        Rng::from_seed(splitmix64(&mut h))
    }

    /// Re-terminates `end` in the geometry `p` whose moved box was `before`
    /// (as an [`Aabb`] in the geometry the end belongs to).
    fn continue_end(
        &self,
        end: WalkEnd,
        net: NetId,
        before: &Aabb,
        p: &Perturbed,
        rng: impl FnOnce() -> Rng,
        acc: &mut FaceAcc,
    ) -> Cont {
        // Only walks absorbed by the moved box can change: every other
        // termination keeps its distance (other conductors and boundaries are
        // unchanged, the moved box only gets farther).
        if end.sink != Some(Sink::Net(net)) || before.linf_distance(end.pos) >= self.eps_stop {
            return Cont::Same;
        }
        let walker = self.walker(p);
        let (h, sink) = walker.clearance(end.pos);
        if h < self.eps_stop {
            // Still absorbed, possibly by another box of a different net.
            return Cont::End(WalkEnd {
                sink: Some(sink),
                ..end
            });
        }
        acc.continued += 1;
        let cont = walker.run(end.pos, &mut rng());
        acc.steps += u64::from(cont.steps);
        if cont.sink.is_none() {
            acc.aborted += 1;
            return Cont::Aborted;
        }
        Cont::End(WalkEnd {
            sign: end.sign * cont.sign,
            steps: end.steps + cont.steps,
            ..cont
        })
    }

    /// Re-terminates the walks of sample `s` that a deletion removed the sink
    /// of, and accumulates the edited contributions and their differences.
    fn deletion_sample(&self, s: &HookSample, base: &[(Sink, f64)], d: &DelCtx, acc: &mut DelAcc) {
        let Some((geom, index)) = &d.edited else {
            // Every conductor deleted: no target survives to be sampled.
            return;
        };
        let mut ends = s.ends;
        let mut changed = false;
        for (slot, e) in ends.iter_mut().enumerate() {
            let Some(end) = *e else { continue };
            let Some(Sink::Net(n)) = end.sink else {
                continue;
            };
            if d.nets.binary_search(&n).is_err() {
                continue;
            }
            let walker = Walker {
                geom,
                index,
                far_step: 2.0 * index.cell_size(),
                ..*self.template
            };
            let (h, sink) = walker.clearance(end.pos);
            let new = if h < self.eps_stop {
                WalkEnd {
                    sink: Some(d.remap(sink)),
                    ..end
                }
            } else {
                acc.continued += 1;
                let mut key = self.seed ^ d.key;
                for v in [
                    u64::from(self.net.raw()),
                    s.batch,
                    u64::from(s.index),
                    slot as u64,
                ] {
                    key = splitmix64(&mut key) ^ v;
                }
                let cont = walker.run(end.pos, &mut Rng::from_seed(splitmix64(&mut key)));
                acc.steps += u64::from(cont.steps);
                if cont.sink.is_none() {
                    // Left unchanged (biased, counted).
                    acc.aborted += 1;
                    continue;
                }
                WalkEnd {
                    sink: cont.sink.map(|k| d.remap(k)),
                    sign: end.sign * cont.sign,
                    steps: end.steps + cont.steps,
                    pos: cont.pos,
                }
            };
            *e = Some(new);
            changed = true;
        }
        if !changed {
            add_sample(
                &mut acc.edited,
                &mut acc.edited_ground,
                base.iter().copied(),
                self.ground_nets,
            );
            return;
        }
        let edited = sample_contribs(hits_of(&ends, s.w_pair), self.odd_image, s.w_pair);
        add_sample(
            &mut acc.edited,
            &mut acc.edited_ground,
            edited.as_slice().iter().copied(),
            self.ground_nets,
        );
        let mut diff: BTreeMap<Sink, f64> = BTreeMap::new();
        for &(k, w) in base {
            *diff.entry(k).or_default() -= w;
        }
        for &(k, w) in edited.as_slice() {
            *diff.entry(k).or_default() += w;
        }
        diff.retain(|_, v| *v != 0.0);
        if !diff.is_empty() {
            add_sample(
                &mut acc.delta,
                &mut acc.delta_ground,
                diff.into_iter(),
                self.ground_nets,
            );
        }
    }

    /// Per-sink `(one-sided, selected)` derivative samples of face `f`, or
    /// `None` if the sample is unchanged.
    fn face_sample(
        &self,
        s: &HookSample,
        base: &[(Sink, f64)],
        f: &FaceCtx,
        acc: &mut FaceAcc,
    ) -> Option<Vec<(Sink, f64, f64)>> {
        let mut lvl1 = s.ends;
        let mut lvl2 = s.ends;
        let mut changed = false;
        for slot in 0..2 {
            let Some(end) = s.ends[slot] else { continue };
            if end.sink.is_none() {
                continue;
            }
            let e1 = match self.continue_end(
                end,
                f.net,
                &f.nominal,
                &f.levels[0],
                || self.stream(s, slot, f, 1),
                acc,
            ) {
                // An aborted continuation leaves the whole slot unchanged at
                // every level (counted in `aborted_continuations`).
                Cont::Aborted => continue,
                Cont::Same => end,
                Cont::End(e) => {
                    changed = true;
                    e
                }
            };
            let e2 = if self.richardson {
                match self.continue_end(
                    e1,
                    f.net,
                    &f.levels[0].moved,
                    &f.levels[1],
                    || self.stream(s, slot, f, 2),
                    acc,
                ) {
                    Cont::Aborted => continue,
                    Cont::Same => e1,
                    Cont::End(e) => {
                        changed = true;
                        e
                    }
                }
            } else {
                e1
            };
            lvl1[slot] = Some(e1);
            lvl2[slot] = Some(e2);
        }
        if !changed {
            return None;
        }
        let c1 = sample_contribs(hits_of(&lvl1, s.w_pair), self.odd_image, s.w_pair);
        let c2 = sample_contribs(hits_of(&lvl2, s.w_pair), self.odd_image, s.w_pair);
        // Per-sink differences of this sample.
        let mut d: BTreeMap<Sink, (f64, f64)> = BTreeMap::new();
        for &(k, w) in base {
            let e = d.entry(k).or_default();
            e.0 -= w;
            e.1 -= w;
        }
        for &(k, w) in c1.as_slice() {
            d.entry(k).or_default().0 += w;
        }
        for &(k, w) in c2.as_slice() {
            d.entry(k).or_default().1 += w;
        }
        let inv = 1.0 / f.delta;
        let out: Vec<(Sink, f64, f64)> = d
            .into_iter()
            .filter(|(_, (d1, d2))| *d1 != 0.0 || *d2 != 0.0)
            .map(|(k, (d1, d2))| {
                let one = -d1 * inv;
                let y = if self.richardson {
                    -(2.0 * d1 - 0.5 * d2) * inv
                } else {
                    one
                };
                (k, one, y)
            })
            .collect();
        (!out.is_empty()).then_some(out)
    }
}

/// Adds one sample's per-sink values (and their ground sum) to moments.
pub(crate) fn add_sample(
    sinks: &mut BTreeMap<Sink, Moments>,
    ground: &mut Moments,
    values: impl Iterator<Item = (Sink, f64)>,
    ground_nets: &[NetId],
) {
    let mut g = 0.0;
    let mut any_ground = false;
    for (k, y) in values {
        sinks.entry(k).or_default().add(y);
        if is_ground_sink(k, ground_nets) {
            g += y;
            any_ground = true;
        }
    }
    if any_ground {
        ground.add(g);
    }
}

/// Outcome of re-terminating one walk end in a perturbed geometry.
enum Cont {
    /// The termination cannot change.
    Same,
    /// The (possibly unchanged) new end.
    End(WalkEnd),
    /// The continuation exceeded `max_steps`.
    Aborted,
}

/// Per-batch accumulators.
#[derive(Default)]
struct SensAcc {
    faces: Vec<FaceAcc>,
    params: Vec<ParamAcc>,
    deletions: Vec<DelAcc>,
}

impl SensAcc {
    fn init(&mut self, faces: usize, params: usize, deletions: usize) {
        if self.faces.len() != faces {
            self.faces.resize_with(faces, FaceAcc::default);
        }
        if self.params.len() != params {
            self.params.resize_with(params, ParamAcc::default);
        }
        if self.deletions.len() != deletions {
            self.deletions.resize_with(deletions, DelAcc::default);
        }
    }
}

#[derive(Default)]
struct ParamAcc {
    sinks: BTreeMap<Sink, Moments>,
    ground: Moments,
}

impl SampleHook for Hook<'_> {
    type Stats = SensAcc;

    fn sample(&self, stats: &mut Self::Stats, s: &HookSample) {
        stats.init(self.faces.len(), self.params.len(), self.deletions.len());
        let base = sample_contribs(hits_of(&s.ends, s.w_pair), self.odd_image, s.w_pair);
        for (d, acc) in self.deletions.iter().zip(&mut stats.deletions) {
            // A deleted target has no row in the edited geometry.
            if d.nets.binary_search(&self.net).is_err() {
                self.deletion_sample(s, base.as_slice(), d, acc);
            }
        }
        let mut cand = Vec::new();
        self.lookup.candidates(s, &mut cand);
        // Changed faces, by increasing face index.
        let per_face: Vec<(usize, FaceDiffs)> = cand
            .iter()
            .filter_map(|&fi| {
                let fi = fi as usize;
                let (f, acc) = (&self.faces[fi], &mut stats.faces[fi]);
                let v = self.face_sample(s, base.as_slice(), f, acc)?;
                acc.one_sided_add(&v);
                add_sample(
                    &mut acc.sinks,
                    &mut acc.ground,
                    v.iter().map(|&(k, _, y)| (k, y)),
                    self.ground_nets,
                );
                Some((fi, v))
            })
            .collect();
        // Parameters: the per-sample combination Σ_f w_f·Y_f, so that their σ
        // includes the face–face covariances exactly.
        for (p, acc) in self.params.iter().zip(&mut stats.params) {
            let mut comb: BTreeMap<Sink, f64> = BTreeMap::new();
            for &(fi, w) in &p.faces {
                if let Ok(k) = per_face.binary_search_by_key(&fi, |e| e.0) {
                    let v = &per_face[k].1;
                    for &(k, _, y) in v {
                        *comb.entry(k).or_default() += w * y;
                    }
                }
            }
            if !comb.is_empty() {
                add_sample(
                    &mut acc.sinks,
                    &mut acc.ground,
                    comb.into_iter(),
                    self.ground_nets,
                );
            }
        }
    }

    fn merge(into: &mut Self::Stats, from: &Self::Stats) {
        into.init(from.faces.len(), from.params.len(), from.deletions.len());
        for (a, b) in into.deletions.iter_mut().zip(&from.deletions) {
            a.merge(b);
        }
        for (a, b) in into.faces.iter_mut().zip(&from.faces) {
            a.merge(b);
        }
        for (a, b) in into.params.iter_mut().zip(&from.params) {
            for (k, m) in &b.sinks {
                a.sinks.entry(*k).or_default().merge(m);
            }
            a.ground.merge(&b.ground);
        }
    }
}

/// `true` if `face` of `b` lies on a reflecting face of the domain box.
fn on_reflecting_wall(b: &Aabb, face: Face, domain: &BoundaryModel) -> bool {
    let Some(bounds) = domain.bounds() else {
        return false;
    };
    let a = face.axis.index();
    let (lo_reflect, hi_reflect) = domain.reflect()[a];
    let tol = 1e-12 * bounds.extent()[a].max(1.0);
    if face.side.sign() > 0.0 {
        hi_reflect && (b.max()[a] - bounds.max()[a]).abs() <= tol
    } else {
        lo_reflect && (b.min()[a] - bounds.min()[a]).abs() <= tol
    }
}

/// The two axes tangential to `axis`.
const fn tangential(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

/// A rectangle in the plane of a face: `(lo, hi)` along its two tangential
/// axes (see [`covered_parts`]).
pub type FaceRect = ([f64; 2], [f64; 2]);

/// The parts of `face` of box `id` covered by other boxes of the same net.
///
/// A box covers the part of the face it overlaps (with positive area) when
/// it contains the face plane and extends *beyond* it, outward: the conductor
/// continues there, so that part of the face is not a surface and does not
/// move. Rectangles are clipped to the face and may overlap each other.
///
/// # Errors
///
/// [`SolveError::Config`] if another box of the net is *flush* with the face
/// (ends exactly in its plane, on the same side) and overlaps it. The union's
/// surface there belongs to both boxes, and moving one face alone is not a
/// linear perturbation of it: merge the boxes, or move a face that is not
/// shared.
pub fn covered_parts(geom: &Geometry, id: BoxId, face: Face) -> Result<Vec<FaceRect>, SolveError> {
    let a = face.axis.index();
    let t = tangential(a);
    let b = geom.box_(id);
    let coord = b.face_coord(face);
    // Relative to the box size (not its position), plus the rounding of the
    // coordinates themselves.
    let size = b.extent().iter().copied().fold(0.0, f64::max);
    let tol = 1e-9 * size + 8.0 * f64::EPSILON * coord.abs();
    let out_sign = face.side.sign();
    let mut parts = Vec::new();
    for &other in geom.net_boxes(geom.box_net(id)) {
        if other == id {
            continue;
        }
        let c = geom.box_(other);
        let lo = [
            b.min()[t[0]].max(c.min()[t[0]]),
            b.min()[t[1]].max(c.min()[t[1]]),
        ];
        let hi = [
            b.max()[t[0]].min(c.max()[t[0]]),
            b.max()[t[1]].min(c.max()[t[1]]),
        ];
        let tol_t = |i: usize| {
            1e-9 * size + 8.0 * f64::EPSILON * b.min()[t[i]].abs().max(b.max()[t[i]].abs())
        };
        if !(hi[0] - lo[0] > tol_t(0) && hi[1] - lo[1] > tol_t(1)) {
            continue;
        }
        // Signed extents of `c` relative to the face plane, outward positive.
        let (near, far) = if out_sign > 0.0 {
            (c.min()[a] - coord, c.max()[a] - coord)
        } else {
            (coord - c.max()[a], coord - c.min()[a])
        };
        if near > tol || far < -tol {
            continue; // does not reach the face plane
        }
        if far <= tol {
            if near < -tol {
                return Err(SolveError::Config(format!(
                    "{id} face {} is flush with {other} of the same net; merge the boxes \
                     or move a face that is not shared",
                    face.as_str()
                )));
            }
            continue; // degenerate (zero thickness along the normal)
        }
        parts.push((lo, hi));
    }
    Ok(parts)
}

/// Area of the union of `rects` (coordinate compression; `rects` is small).
fn union_area(rects: &[FaceRect]) -> f64 {
    let mut xs: Vec<f64> = rects.iter().flat_map(|r| [r.0[0], r.1[0]]).collect();
    let mut ys: Vec<f64> = rects.iter().flat_map(|r| [r.0[1], r.1[1]]).collect();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    let mut area = 0.0;
    for x in xs.windows(2) {
        for y in ys.windows(2) {
            let (cx, cy) = (0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]));
            if rects
                .iter()
                .any(|r| r.0[0] <= cx && cx <= r.1[0] && r.0[1] <= cy && cy <= r.1[1])
            {
                area += (x[1] - x[0]) * (y[1] - y[0]);
            }
        }
    }
    area
}

/// Exposed area (µm²) of `face` of box `id`: the face minus the parts covered
/// by other boxes of the same net ([`covered_parts`]), or zero if the face
/// lies on a reflecting face of `domain` (mirror symmetry continues the
/// conductor there).
///
/// # Errors
///
/// As [`covered_parts`].
pub fn exposed_area(
    geom: &Geometry,
    domain: &BoundaryModel,
    id: BoxId,
    face: Face,
) -> Result<f64, SolveError> {
    let b = geom.box_(id);
    if on_reflecting_wall(b, face, domain) {
        return Ok(0.0);
    }
    let covered = union_area(&covered_parts(geom, id, face)?);
    Ok((b.face_area(face) - covered).max(0.0))
}

/// What a sensitivity driver needs to know before solving: the stopping
/// distance and the simulation domain of `solve` with the same inputs.
#[derive(Debug, Clone)]
pub struct SensSetup {
    /// Stopping distance `ε_stop` (µm). Steps must be at least
    /// [`MIN_STEP_OVER_EPS_STOP`] times this.
    pub eps_stop: f64,
    /// The simulation domain (grounded and reflecting faces, substrate).
    pub domain: BoundaryModel,
}

/// The [`SensSetup`] of `solve_sensitivity(geom, targets, cfg, …)`.
///
/// # Errors
///
/// As [`solve`](crate::solve).
pub fn sens_setup(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
) -> Result<SensSetup, SolveError> {
    let su = setup(geom, targets, cfg)?;
    Ok(SensSetup {
        eps_stop: su.eps_stop,
        domain: su.domain,
    })
}

/// Design M2c §7: deletions reuse the nominal walks, valid only if the
/// medium is unchanged. A conformal shell follows its conductor boxes, so
/// deleting a net with one is refused. Explicit dielectric boxes and untagged
/// conductors do not shape the medium. (Faces of tagged boxes are handled by
/// design D3 §4.4c: continuations in the medium `M₁`, plus the shell term.)
fn check_medium_unchanged(
    geom: &Geometry,
    cfg: &SolverConfig,
    deletions: &[Deletion],
) -> Result<(), SolveError> {
    let tagged = |b: BoxId| cfg.regions.conformal.iter().any(|r| r.boxes.contains(&b));
    for d in deletions {
        if let Some(&n) = d
            .nets
            .iter()
            .find(|&&n| n.index() < geom.num_nets() && geom.net_boxes(n).iter().any(|&b| tagged(b)))
        {
            return Err(SolveError::Config(format!(
                "net `{}` carries a conformal dielectric shell; deleting it changes the medium \
                 (solve the edited geometry instead)",
                geom.net_name(n)
            )));
        }
    }
    Ok(())
}

/// `b` with `face` moved inward by `d`.
fn shrunk(b: &Aabb, face: Face, d: f64) -> Result<Aabb, SolveError> {
    let (mut lo, mut hi) = (b.min(), b.max());
    let a = face.axis.index();
    if face.side.sign() > 0.0 {
        hi[a] -= d;
    } else {
        lo[a] += d;
    }
    Aabb::new(lo, hi).map_err(|e| SolveError::Config(e.to_string()))
}

/// `Σ_f w_f T_f^Σ` of a parameter, per sink and for the ground sum (faces
/// are sampled independently, so variances add).
fn param_shell(
    p: &Param,
    shells: &[Option<ShellResult>],
) -> (Vec<(Sink, Estimate)>, Option<Estimate>) {
    let mut per: BTreeMap<Sink, Estimate> = BTreeMap::new();
    let mut ground: Option<Estimate> = None;
    for &(fi, w) in &p.faces {
        let Some(s) = &shells[fi] else { continue };
        for (k, m) in &s.sinks {
            let e = m.estimate(s.points).scaled(w);
            per.entry(*k).and_modify(|v| *v = sum(*v, e)).or_insert(e);
        }
        let g = s.ground.estimate(s.points).scaled(w);
        ground = Some(ground.map_or(g, |v| sum(v, g)));
    }
    (per.into_iter().collect(), ground)
}

/// Smallest step relative to the stopping distance: below it, the stopping
/// rule masks the perturbation and the derivative is silently too small.
pub const MIN_STEP_OVER_EPS_STOP: f64 = 20.0;

/// The nominal solve's medium, for the edited media of shell faces.
struct MediumCtx<'a> {
    regions: &'a DielectricRegions,
    stack: &'a Stack,
    boxes: Option<&'a BoxSteps>,
    /// [`SensOptions::shell_conductor_term`].
    conductor_term_only: bool,
    /// [`SensOptions::shell_edge_margin`].
    edge_margin: f64,
}

fn face_ctx(
    geom: &Geometry,
    step: &FaceStep,
    richardson: bool,
    eps_stop: f64,
    domain: &BoundaryModel,
    reach: f64,
    medium: &MediumCtx<'_>,
) -> Result<FaceCtx, SolveError> {
    if step.box_id.index() >= geom.boxes().len() {
        return Err(SolveError::Config(format!("unknown {}", step.box_id)));
    }
    if !(step.delta.is_finite() && step.delta >= MIN_STEP_OVER_EPS_STOP * eps_stop) {
        return Err(SolveError::Config(format!(
            "sensitivity step {} µm must be at least {MIN_STEP_OVER_EPS_STOP} × ε_stop = {} µm",
            step.delta,
            MIN_STEP_OVER_EPS_STOP * eps_stop
        )));
    }
    let nominal = *geom.box_(step.box_id);
    if on_reflecting_wall(&nominal, step.face, domain) {
        return Err(SolveError::Config(format!(
            "{} face {} lies on a reflecting domain face; it has no exposed area",
            step.box_id,
            step.face.as_str()
        )));
    }
    let n_levels: u8 = if richardson { 2 } else { 1 };
    if nominal.extent()[step.face.axis.index()] <= 2.0 * f64::from(n_levels) * step.delta {
        return Err(SolveError::Config(format!(
            "{} face {} is too thin for a step of {} µm",
            step.box_id,
            step.face.as_str(),
            step.delta
        )));
    }
    // Parts of the face covered by other boxes of the net do not move: the
    // strip under them is filled back in, so only the exposed part shrinks
    // and no slit opens inside the conductor.
    let mut covered = covered_parts(geom, step.box_id, step.face)?;
    if let Some((klo, khi)) = step.keep {
        let t = tangential(step.face.axis.index());
        let lo = [
            klo[0].max(nominal.min()[t[0]]),
            klo[1].max(nominal.min()[t[1]]),
        ];
        let hi = [
            khi[0].min(nominal.max()[t[0]]),
            khi[1].min(nominal.max()[t[1]]),
        ];
        if lo[0] < hi[0] && lo[1] < hi[1] {
            covered.push((lo, hi));
        }
    }
    if (nominal.face_area(step.face) - union_area(&covered)) <= 0.0 {
        return Err(SolveError::Config(format!(
            "{} face {} is covered by other boxes of its net; it has no exposed area",
            step.box_id,
            step.face.as_str()
        )));
    }
    let rules: Vec<&ConformalRule> = medium
        .regions
        .conformal
        .iter()
        .filter(|r| r.boxes.contains(&step.box_id))
        .collect();
    check_shell_nesting(step, &rules, n_levels)?;
    let a = step.face.axis.index();
    let t = tangential(a);
    let coord = nominal.face_coord(step.face);
    let levels = (1..=n_levels)
        .map(|l| {
            let d = f64::from(l) * step.delta;
            let moved = shrunk(&nominal, step.face, d)?;
            let (n_lo, n_hi) = if step.face.side.sign() > 0.0 {
                (coord - d, coord)
            } else {
                (coord, coord + d)
            };
            let fillers = covered
                .iter()
                .map(|&(lo, hi)| {
                    let (mut min, mut max) = ([0.0; 3], [0.0; 3]);
                    (min[a], max[a]) = (n_lo, n_hi);
                    (min[t[0]], max[t[0]]) = (lo[0], hi[0]);
                    (min[t[1]], max[t[1]]) = (lo[1], hi[1]);
                    Aabb::new(min, max).map_err(|e| SolveError::Config(e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut boxes = vec![moved];
            boxes.extend(fillers);
            let edited = if rules.is_empty() {
                None
            } else {
                let m = BoxMedium::edited_inside(
                    medium.regions,
                    geom,
                    step.box_id,
                    &boxes,
                    medium.stack,
                    domain,
                );
                Some(match (m, medium.boxes) {
                    (None, _) => None,
                    (Some(m), Some(nominal)) => {
                        Some(nominal.with_medium(m).map_err(SolveError::Config)?)
                    }
                    // The shells lie outside the domain, and stay there.
                    (Some(_), None) => None,
                })
            };
            let overlay = Overlay::new(geom, step.box_id, boxes, reach);
            Ok(Perturbed {
                overlay,
                moved,
                boxes: edited,
            })
        })
        .collect::<Result<_, SolveError>>()?;
    let shell = if medium.conductor_term_only {
        None
    } else {
        shell_face(
            geom,
            medium.regions,
            domain,
            step.box_id,
            step.face,
            medium.edge_margin,
        )?
    };
    Ok(FaceCtx {
        handle: (step.box_id, step.face),
        net: geom.box_net(step.box_id),
        nominal,
        delta: step.delta,
        levels,
        shell,
    })
}

/// Richardson continues a level-1 walk at level 2, so the edited medium
/// `M₁` of both levels must agree where level-1 continuations run (the slab
/// vacated at level 1). The slab takes the shell's material if it lies within
/// the shell's thickness `g` of the moved box (`side` for lateral faces,
/// `top` for `z+`, nothing for `z−`), else the outside material; both levels
/// agree if there is one level, `g = 0`, or `levels · δ ≤ g`.
fn check_shell_nesting(
    step: &FaceStep,
    rules: &[&ConformalRule],
    levels: u8,
) -> Result<(), SolveError> {
    for r in rules {
        let g = match (step.face.axis, step.face.side) {
            (Axis::Z, Side::Max) => r.top,
            (Axis::Z, Side::Min) => 0.0,
            _ => r.side,
        };
        if levels > 1 && g > 0.0 && f64::from(levels) * step.delta > g {
            return Err(SolveError::Config(format!(
                "{} face {}: twice the step of {} µm (Richardson) exceeds the {} µm \
                 conformal shell of `{}`",
                step.box_id,
                step.face.as_str(),
                step.delta,
                g,
                r.layer
            )));
        }
    }
    Ok(())
}

/// Solves `targets` like [`solve`](crate::solve) and, from the same walks,
/// the derivatives of every reported `C_ik` with respect to the outward
/// displacement of each face in `faces`.
///
/// # Errors
///
/// As [`solve_sensitivity_with`].
pub fn solve_sensitivity(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    faces: &[FaceStep],
    params: &[Param],
    opts: SensOptions,
) -> Result<SensSolution, SolveError> {
    solve_sensitivity_with(
        geom,
        targets,
        cfg,
        &SensRequest {
            faces,
            params,
            deletions: &[],
            opts,
        },
    )
}

/// Solves `targets` like [`solve`](crate::solve) and, from the same walks,
/// face derivatives, parameter derivatives and deletions of `req`.
///
/// # Errors
///
/// Like [`solve`](crate::solve), and [`SolveError::Config`] for an unknown box,
/// a step below [`MIN_STEP_OVER_EPS_STOP`] × ε_stop, a box too thin for the
/// step, a face without exposed area, a parameter referring to an unknown
/// face, or a deletion of an unknown net.
pub fn solve_sensitivity_with(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    req: &SensRequest<'_>,
) -> Result<SensSolution, SolveError> {
    let SensRequest {
        faces,
        params,
        deletions,
        opts,
    } = *req;
    let su = setup(geom, targets, cfg)?;
    check_medium_unchanged(geom, cfg, deletions)?;
    if params.iter().any(|p| {
        p.faces
            .iter()
            .any(|&(i, w)| i >= faces.len() || !w.is_finite())
    }) {
        return Err(SolveError::Config(
            "a parameter refers to an unknown face or has a non-finite weight".into(),
        ));
    }
    // Overlay reach (see `Overlay`): the walker's far step, and at least
    // 4 ε_stop.
    let reach = (2.0 * su.index.cell_size()).max(4.0 * su.eps_stop);
    if !(opts.shell_edge_margin.is_finite() && opts.shell_edge_margin >= 0.0) {
        return Err(SolveError::Config(
            "the shell edge margin must be finite and non-negative".into(),
        ));
    }
    let ctxs: Vec<FaceCtx> = faces
        .iter()
        .map(|f| {
            face_ctx(
                geom,
                f,
                opts.richardson,
                su.eps_stop,
                &su.domain,
                reach,
                &MediumCtx {
                    regions: &cfg.regions,
                    stack: &su.stack,
                    boxes: su.boxes.as_ref(),
                    conductor_term_only: opts.shell_conductor_term,
                    edge_margin: opts.shell_edge_margin,
                },
            )
        })
        .collect::<Result<_, _>>()?;
    if opts.shell_points < 2 && ctxs.iter().any(|f| f.shell.is_some()) {
        return Err(SolveError::Config(
            "shell terms need at least two points per face".into(),
        ));
    }
    let lookup = FaceLookup::new(
        &ctxs,
        su.eps_stop,
        su.index.cell_size().max(4.0 * su.eps_stop),
    );
    let dels: Vec<DelCtx> = deletions
        .iter()
        .map(|d| DelCtx::new(geom, d))
        .collect::<Result<_, _>>()?;
    let walker = su.walker(geom, cfg);
    // Exactly as `solve_hooked`: layered launches carry ε_r(r) in their
    // factor. Continuations run the same (layered) walker from the walk ends.
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
    let nets = targets
        .par_iter()
        .zip(su.clearances.par_iter())
        .enumerate()
        .map(|(i, (&net, _))| {
            let surface = &surfaces[i];
            let ctx = NetCtx {
                target: i,
                net,
                eps,
                eps_stop: su.eps_stop,
            };
            let hook = Hook {
                faces: &ctxs,
                lookup: &lookup,
                deletions: &dels,
                params,
                template: &walker,
                seed: cfg.seed,
                net,
                eps_stop: su.eps_stop,
                odd_image: su.domain.odd_image(),
                richardson: opts.richardson,
                ground_nets: &cfg.ground_nets,
            };
            let (nominal, _, mut accs) = run_net(&walker, surface, ctx, cfg, &(), &hook);
            accs.init(ctxs.len(), params.len(), dels.len());
            let mut shells: Vec<Option<ShellResult>> = ctxs
                .iter()
                .map(|f| {
                    let sf = f.shell.as_ref()?;
                    let boxes = su.boxes.as_ref()?;
                    let sctx = ShellCtx {
                        walker: &walker,
                        boxes,
                        eps_stop: su.eps_stop,
                        seed: cfg.seed,
                        ground_nets: &cfg.ground_nets,
                        odd_image: su.domain.odd_image(),
                    };
                    Some(shell_term(&sctx, sf, net, opts.shell_points))
                })
                .collect();
            SensNetResult {
                nominal,
                deletions: accs
                    .deletions
                    .into_iter()
                    .map(|a| DeletionResult {
                        delta: a.delta.into_iter().collect(),
                        delta_ground: a.delta_ground,
                        edited: a.edited.into_iter().collect(),
                        edited_ground: a.edited_ground,
                        continued_walks: a.continued,
                        aborted_continuations: a.aborted,
                        continuation_steps: a.steps,
                    })
                    .collect(),
                params: accs
                    .params
                    .into_iter()
                    .zip(params)
                    .map(|(a, p)| {
                        let (shell, shell_ground) = param_shell(p, &shells);
                        ParamResult {
                            sinks: a.sinks.into_iter().collect(),
                            ground: a.ground,
                            shell,
                            shell_ground,
                        }
                    })
                    .collect(),
                faces: accs
                    .faces
                    .into_iter()
                    .zip(shells.iter_mut())
                    .map(|(a, shell)| FaceResult {
                        sinks: a.sinks.into_iter().collect(),
                        ground: a.ground,
                        one_sided: a.one_sided.into_iter().collect(),
                        continued_walks: a.continued,
                        aborted_continuations: a.aborted,
                        continuation_steps: a.steps,
                        shell: shell.take(),
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(SensSolution {
        nets,
        eps_stop: su.eps_stop,
    })
}
