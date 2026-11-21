//! The shell-interface term `T^Σ` of sensitivities with conformal shells
//! (design D3 §4.4c).
//!
//! Moving face f of a box tagged by a conformal rule also moves the face
//! `Σ_f` of its grown box, the outer surface of the shell. To first order that
//! adds the transmission-problem Hadamard term
//!
//! ```text
//! T_ik^Σ = ε₀ (ε_s − ε_o) ∫_{Σ_f} [ E_t^i · E_t^k + D_n^i D_n^k / (ε_s ε_o) ] dS
//! ```
//!
//! (relative ε, `D_n = ε_r ∂_n φ`), for an outward move of f. Points are drawn
//! uniformly on the grown face. A point counts only where the medium really
//! jumps there from the rule's shell (inside) to another material (outside):
//! that excludes parts inside another grown box of the rule, parts overridden
//! by explicit boxes or later rules, and parts masked by conductors. At each
//! point, two independent field estimates (axis cube centred on the point,
//! with the interface through its centre) give an unbiased product.

use std::collections::BTreeMap;

use faradice_core::{Axis, BoxId, EPS0_FF_PER_UM, Face, Moments, NetId, Side, Sink};
use faradice_geom::Geometry;
use rayon::prelude::*;

use crate::axis::BoxSteps;
use crate::boundary::BoundaryModel;
use crate::dielectric::grown;
use crate::regions::DielectricRegions;
use crate::rng::{Rng, splitmix64};
use crate::sens::add_sample;
use crate::solver::SolveError;
use crate::walk::Walker;

/// Domain-separation tag of the shell-term streams.
const SHELL_TAG: u64 = 0x5E11_7E2A_D3C4_4C00;

/// Offset (µm) of the material probes on either side of the face.
const PROBE: f64 = 1e-7;

/// The moving part of a shell's outer surface for one face.
#[derive(Debug, Clone)]
pub(crate) struct ShellFace {
    handle: (BoxId, Face),
    /// Normal axis and plane coordinate.
    axis: usize,
    coord: f64,
    /// `+1` if the outward normal of the face points to increasing `axis`.
    out: f64,
    /// Tangential axes and the grown face's extent along them.
    tangential: [usize; 2],
    lo: [f64; 2],
    hi: [f64; 2],
    /// Where the fields are evaluated: `[lo, hi]` shrunk by the edge margin
    /// along edges that are not on reflecting walls (see [`point`]).
    inner_lo: [f64; 2],
    inner_hi: [f64; 2],
    eps_shell: f64,
}

impl ShellFace {
    fn area(&self) -> f64 {
        (self.hi[0] - self.lo[0]) * (self.hi[1] - self.lo[1])
    }
}

/// The moving shell face of `face` of box `id`, or `None` if no conformal
/// rule tags the box.
///
/// # Errors
///
/// [`SolveError::Config`] for faces whose shell face touches the conductor
/// edge (`z−`, lateral faces with `side = 0`, `z+` with `top = 0`; design
/// §4.4c stage 3) and for boxes tagged by several rules.
pub(crate) fn shell_face(
    geom: &Geometry,
    regions: &DielectricRegions,
    domain: &BoundaryModel,
    id: BoxId,
    face: Face,
    margin: f64,
) -> Result<Option<ShellFace>, SolveError> {
    let mut rules = regions.conformal.iter().filter(|r| r.boxes.contains(&id));
    let Some(rule) = rules.next() else {
        return Ok(None);
    };
    if rules.next().is_some() {
        return Err(SolveError::Config(format!(
            "{id} is tagged by several conformal rules; its shell sensitivity is not supported"
        )));
    }
    let thickness = match (face.axis, face.side) {
        (Axis::Z, Side::Min) => 0.0,
        (Axis::Z, Side::Max) => rule.top,
        _ => rule.side,
    };
    if thickness <= 0.0 {
        return Err(SolveError::Config(format!(
            "{id} face {}: the conformal shell of `{}` has no thickness there, so its moving \
             face touches the conductor edge (not supported yet, design D3 §4.4c stage 3)",
            face.as_str(),
            rule.layer
        )));
    }
    let g = grown(geom.box_(id), rule.side, rule.top);
    let a = face.axis.index();
    let tangential = match a {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    // Only the part inside the domain (above the substrate) moves medium:
    // reflections continue it, and the medium is not defined outside.
    let (mut lo, mut hi) = (g.min(), g.max());
    if let Some(b) = domain.bounds() {
        for t in 0..3 {
            lo[t] = lo[t].max(b.min()[t]);
            hi[t] = hi[t].min(b.max()[t]);
        }
    }
    if let Some(z0) = domain.substrate_z() {
        lo[2] = lo[2].max(z0);
    }
    // Edge margin: a multiple of the shell's thickness there (design D3 §4.4c).
    let margin = margin * thickness;
    let reflect = |t: usize, hi_end: bool, v: f64| {
        let (Some(b), (r_lo, r_hi)) = (domain.bounds(), domain.reflect()[t]) else {
            return false;
        };
        if hi_end {
            r_hi && v >= b.max()[t]
        } else {
            r_lo && v <= b.min()[t]
        }
    };
    let (mut inner_lo, mut inner_hi) = ([0.0; 2], [0.0; 2]);
    for (k, &t) in tangential.iter().enumerate() {
        let (l, h) = (lo[t], hi[t].max(lo[t]));
        let il = if reflect(t, false, l) { l } else { l + margin };
        let ih = if reflect(t, true, h) { h } else { h - margin };
        (inner_lo[k], inner_hi[k]) = if il <= ih {
            (il, ih)
        } else {
            let m = 0.5 * (l + h);
            (m, m)
        };
    }
    Ok(Some(ShellFace {
        handle: (id, face),
        axis: a,
        coord: g.face_coord(face),
        out: face.side.sign(),
        tangential,
        lo: tangential.map(|t| lo[t]),
        hi: tangential.map(|t| hi[t].max(lo[t])),
        inner_lo,
        inner_hi,
        eps_shell: rule.eps_r,
    }))
}

/// Statistics of `T^Σ` for one target and one face.
#[derive(Debug, Clone, Default)]
pub struct ShellResult {
    /// Per-sink moments of the per-point estimate (fF/µm).
    pub sinks: Vec<(Sink, Moments)>,
    /// Moments of the per-point sum over ground-like sinks.
    pub ground: Moments,
    /// Points drawn on the grown face (the sample count of the moments).
    pub points: u64,
    /// Points on the face where the medium jumps (the others contribute 0).
    pub on_interface: u64,
    /// Interface points without a field cube (counted as 0: biased).
    pub skipped: u64,
    /// Field walks that exceeded `max_steps` (counted as 0: biased).
    pub aborted: u64,
}

#[derive(Default)]
struct Acc {
    sinks: BTreeMap<Sink, Moments>,
    ground: Moments,
    on_interface: u64,
    skipped: u64,
    aborted: u64,
}

impl Acc {
    fn merge(mut self, o: Self) -> Self {
        for (k, m) in &o.sinks {
            self.sinks.entry(*k).or_default().merge(m);
        }
        self.ground.merge(&o.ground);
        self.on_interface += o.on_interface;
        self.skipped += o.skipped;
        self.aborted += o.aborted;
        self
    }
}

/// What [`shell_term`] needs besides the face.
pub(crate) struct ShellCtx<'a> {
    pub(crate) walker: &'a Walker<'a>,
    pub(crate) boxes: &'a BoxSteps,
    pub(crate) eps_stop: f64,
    pub(crate) seed: u64,
    pub(crate) ground_nets: &'a [NetId],
    /// Open boundary with a substrate (odd image): see [`point`].
    pub(crate) odd_image: bool,
}

/// `T^Σ` of row `target` over `points` points on `f`.
pub(crate) fn shell_term(
    ctx: &ShellCtx<'_>,
    f: &ShellFace,
    target: NetId,
    points: u64,
) -> ShellResult {
    let mut key = ctx.seed ^ SHELL_TAG;
    for v in [
        u64::from(target.raw()),
        u64::from(f.handle.0.raw()),
        f.handle.1.index() as u64,
    ] {
        key = splitmix64(&mut key) ^ v;
    }
    let acc = (0..points)
        .into_par_iter()
        .fold(Acc::default, |mut acc, i| {
            let mut k = key ^ i;
            let mut rng = Rng::from_seed(splitmix64(&mut k));
            point(ctx, f, target, &mut rng, &mut acc);
            acc
        })
        .reduce(Acc::default, Acc::merge);
    ShellResult {
        sinks: acc.sinks.into_iter().collect(),
        ground: acc.ground,
        points,
        on_interface: acc.on_interface,
        skipped: acc.skipped,
        aborted: acc.aborted,
    }
}

/// One point: adds its per-sink estimate (if non-zero) to `acc`. With the
/// odd image, the fixed-ground sinks are reported together under
/// [`Sink::GroundPlane`], as in the solver.
fn point(ctx: &ShellCtx<'_>, f: &ShellFace, target: NetId, rng: &mut Rng, acc: &mut Acc) {
    let mut q = [0.0; 3];
    q[f.axis] = f.coord;
    for (k, &t) in f.tangential.iter().enumerate() {
        q[t] = (f.hi[k] - f.lo[k]).mul_add(rng.next_f64(), f.lo[k]);
    }
    // The material outside, if the point is part of Σ_f.
    let jump = |q: [f64; 3]| {
        if ctx.walker.clearance(q).0 < ctx.eps_stop {
            return None; // masked by a conductor (or within the stopping distance)
        }
        let probe = |s: f64| {
            let mut p = q;
            p[f.axis] += s * f.out * PROBE;
            ctx.boxes.medium().eps_at(p)
        };
        let (e_in, e_out) = (probe(-1.0), probe(1.0));
        (e_in.to_bits() == f.eps_shell.to_bits() && e_in.to_bits() != e_out.to_bits())
            .then_some(e_out)
    };
    let Some(e_out) = jump(q) else { return };
    acc.on_interface += 1;
    // Near an edge of Σ_f (where it meets another interface) the field cube
    // shrinks with the distance s to the edge, and the product of two walk
    // scores ∝ 1/s has an infinite second moment. The fields are therefore
    // taken at the nearest point of the face shrunk by the edge margin: a
    // zeroth-order extrapolation over the edge strip.
    let mut x = q;
    for (k, &t) in f.tangential.iter().enumerate() {
        x[t] = q[t].clamp(f.inner_lo[k], f.inner_hi[k]);
    }
    let q = if jump(x).is_some_and(|e| e.to_bits() == e_out.to_bits()) {
        x
    } else {
        q
    };
    let (h, _) = ctx.walker.clearance(q);
    let axis = Axis::ALL[f.axis];
    // Two independent field estimates per sink: [tangential…, flux] by world axis.
    let mut fields: [BTreeMap<Sink, [f64; 3]>; 2] = [BTreeMap::new(), BTreeMap::new()];
    for g in &mut fields {
        let Some((hops, _)) = ctx.boxes.interface_hops(q, axis, h, rng) else {
            acc.skipped += 1;
            return;
        };
        for (a, hop) in hops.iter().enumerate() {
            for (slot, w) in [(0, 0.5), (1, -0.5)] {
                let mut start = hop.starts[slot];
                ctx.walker.domain.fold(&mut start);
                let end = ctx.walker.run(start, rng);
                let Some(sink) = end.sink else {
                    acc.aborted += 1;
                    continue;
                };
                g.entry(sink).or_insert([0.0; 3])[a] += w * hop.factor * end.sign;
            }
        }
    }
    if ctx.odd_image {
        // As in the solver's `record`: the image flips the sign of walks that
        // cross the substrate, which is valid for conductor potentials only.
        // The fixed-ground part follows from the partition of unity
        // (the potentials of all sinks sum to 1, so their gradients to 0).
        for g in &mut fields {
            g.retain(|k, _| matches!(k, Sink::Net(_)));
            let rest = g
                .values()
                .fold([0.0; 3], |s, v| std::array::from_fn(|a| s[a] - v[a]));
            g.insert(Sink::GroundPlane, rest);
        }
    }
    let row = Sink::Net(target);
    let [g1, g2] = &fields;
    let zero = [0.0; 3];
    let (i1, i2) = (g1.get(&row).unwrap_or(&zero), g2.get(&row).unwrap_or(&zero));
    if !g1.contains_key(&row) && !g2.contains_key(&row) {
        return;
    }
    let scale = EPS0_FF_PER_UM * (f.eps_shell - e_out) * f.area();
    let normal_weight = 1.0 / (f.eps_shell * e_out);
    let mut sinks: Vec<Sink> = g1.keys().chain(g2.keys()).copied().collect();
    sinks.sort_unstable();
    sinks.dedup();
    let values = sinks.into_iter().filter_map(|k| {
        let (k1, k2) = (g1.get(&k).unwrap_or(&zero), g2.get(&k).unwrap_or(&zero));
        let y: f64 = (0..3)
            .map(|a| {
                let w = if a == f.axis { normal_weight } else { 1.0 };
                0.5 * w * i1[a].mul_add(k2[a], i2[a] * k1[a])
            })
            .sum();
        (y != 0.0).then_some((k, scale * y))
    });
    add_sample(&mut acc.sinks, &mut acc.ground, values, ctx.ground_nets);
}
