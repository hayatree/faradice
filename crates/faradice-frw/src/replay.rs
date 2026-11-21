//! Additions by replay (design D3 §8, D-add).
//!
//! A recorded extraction keeps, per Gaussian sample, the batch stream at the
//! start of the sample and the bounding box of every clearance cube the
//! sample used (first hop and walks). Adding conductors changes a sample only
//! if an added box cuts one of its cubes: elsewhere the clearance, and so the
//! whole path, is unchanged (the walker takes the added boxes only where they
//! are *strictly* nearer than the nominal step). [`replay_addition`] replays
//! just the samples whose box meets an added box, with the same stream, in
//! the nominal and the edited geometry. The paired difference is the exact
//! `ΔC` of the addition (no step, no bias), and the edited matrix is the
//! nominal one plus `ΔC`, statistically identical to a re-solve that reuses
//! the nominal Gaussian surfaces and stopping distance.

use std::collections::BTreeMap;

use faradice_core::{Estimate, Moments, NetId, Sink};
use faradice_geom::{Aabb, Geometry};
use rayon::prelude::*;

use crate::rng::{Rng, splitmix64};
use crate::solver::{
    HookSample, NetResult, OneSample, SampleHook, Solution, SolveError, SolverConfig,
    gauss_surface, hits_of, is_ground_sink, sample_contribs, sample_once, setup, solve_hooked,
};
use crate::walk::Walker;

/// Per-sink contributions of one sample.
type Contribs = Vec<(Sink, f64)>;

/// One recorded sample.
#[derive(Debug, Clone)]
struct SampleRecord {
    rng: Rng,
    /// Cube bounds, rounded outward to `f32`.
    bb: [[f32; 3]; 2],
}

/// The replay data of a recorded extraction (about 56 bytes per Gaussian
/// sample that produced launches).
#[derive(Debug, Clone)]
pub struct Recording {
    fingerprint: u64,
    nets: Vec<(NetId, Vec<SampleRecord>)>,
    /// Per target: walks and the bits of both moments of every sink and of
    /// the ground aggregate of the recorded solution (all of which a replay
    /// builds on), so the records cannot be paired with another solution.
    solution: Vec<SolutionKey>,
}

type SolutionKey = (u64, [u64; 2], Vec<(Sink, [u64; 2])>);

fn solution_key(r: &NetResult) -> SolutionKey {
    let bits = |m: &Moments| [m.sum().to_bits(), m.sum_sq().to_bits()];
    (
        r.walks,
        bits(&r.ground_moments),
        r.sinks.iter().map(|(s, m)| (*s, bits(m))).collect(),
    )
}

impl Recording {
    /// Recorded samples over all targets.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.nets.iter().map(|(_, v)| v.len()).sum()
    }
}

struct RecordHook;

impl SampleHook for RecordHook {
    type Stats = Vec<SampleRecord>;
    const RECORD: bool = true;

    fn sample(&self, stats: &mut Self::Stats, s: &HookSample) {
        if let Some((rng, bb)) = &s.record {
            stats.push(SampleRecord {
                rng: rng.clone(),
                bb: [bb[0].map(down), bb[1].map(up)],
            });
        }
    }

    fn merge(into: &mut Self::Stats, from: &Self::Stats) {
        into.extend_from_slice(from);
    }
}

/// `f32` at most `v` (`f32::next_down` needs Rust 1.86).
#[allow(clippy::cast_possible_truncation)]
fn down(v: f64) -> f32 {
    if v > f64::from(f32::MAX) {
        return f32::MAX;
    }
    if v < f64::from(f32::MIN) {
        return f32::NEG_INFINITY;
    }
    let f = v as f32;
    if f64::from(f) > v {
        f - (f.abs() * f32::EPSILON).max(f32::MIN_POSITIVE)
    } else {
        f
    }
}

/// `f32` at least `v`.
#[allow(clippy::cast_possible_truncation)]
fn up(v: f64) -> f32 {
    if v < f64::from(f32::MIN) {
        return f32::MIN;
    }
    if v > f64::from(f32::MAX) {
        return f32::INFINITY;
    }
    let f = v as f32;
    if f64::from(f) < v {
        f + (f.abs() * f32::EPSILON).max(f32::MIN_POSITIVE)
    } else {
        f
    }
}

/// What a recording is valid for: the boxes, nets and the configuration
/// fields that shape the walks.
fn fingerprint(geom: &Geometry, targets: &[NetId], cfg: &SolverConfig) -> u64 {
    let mut h = 0x005E_ED0F_D3A0_u64;
    let mut mix = |v: u64| {
        h ^= v;
        h = splitmix64(&mut h);
    };
    for (i, b) in geom.boxes().iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        mix(u64::from(
            geom.box_net(faradice_core::BoxId::new(i as u32)).raw(),
        ));
        for v in b.min().into_iter().chain(b.max()) {
            mix(v.to_bits());
        }
    }
    for t in targets {
        mix(u64::from(t.raw()));
    }
    mix(cfg.seed);
    // The whole configuration (boundary, dielectric, walk, accuracy and
    // ground nets all shape the recorded population).
    for byte in format!("{cfg:?}").bytes() {
        mix(u64::from(byte));
    }
    h
}

/// Solves `targets` like [`solve`](crate::solve) (bit-identical statistics)
/// and records the samples for [`replay_addition`].
///
/// # Errors
///
/// As [`solve`](crate::solve).
pub fn solve_recorded(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
) -> Result<(Solution, Recording), SolveError> {
    let (sol, stats) = solve_hooked(geom, targets, cfg, &(), &RecordHook)?;
    let nets = targets.iter().copied().zip(stats).collect();
    let solution = sol.nets.iter().map(solution_key).collect();
    Ok((
        sol,
        Recording {
            fingerprint: fingerprint(geom, targets, cfg),
            nets,
            solution,
        },
    ))
}

/// The net of added conductors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddedNet {
    /// Extend an existing net (e.g. widen or lengthen a wire by adding boxes
    /// that touch it).
    Existing(NetId),
    /// A new conductor at 0 V (a shield or a new wire), reported as net id
    /// `geom.num_nets()`.
    New,
}

/// Conductors to add.
#[derive(Debug, Clone, PartialEq)]
pub struct Addition {
    /// The boxes; they must not overlap existing conductors of other nets.
    pub boxes: Vec<Aabb>,
    /// Their net.
    pub net: AddedNet,
    /// Count a [`AddedNet::New`] net as ground in the ground aggregate (like
    /// `cfg.ground_nets`); its coupling is reported either way.
    pub ground: bool,
}

/// The addition seen from one target.
#[derive(Debug, Clone)]
pub struct AddedNetResult {
    /// The target.
    pub net: NetId,
    /// Walks of the nominal run (the denominator).
    pub walks: u64,
    /// Samples replayed (their cubes met an added box).
    pub replayed: u64,
    /// Per-sample moments of the change, per sink.
    pub delta: BTreeMap<Sink, Moments>,
    /// Per-sample moments of the change of the ground aggregate.
    pub delta_ground: Moments,
    /// Per-sample moments of the edited geometry, per sink.
    pub edited: BTreeMap<Sink, Moments>,
    /// Per-sample moments of the edited ground aggregate.
    pub edited_ground: Moments,
}

impl AddedNetResult {
    /// `ΔC_ik` (fF) of the addition (paired; zero if never changed).
    #[must_use]
    pub fn delta_c(&self, sink: Sink) -> Estimate {
        self.delta
            .get(&sink)
            .copied()
            .unwrap_or_default()
            .estimate(self.walks)
    }

    /// `C_ik` (fF) of the edited geometry.
    #[must_use]
    pub fn c(&self, sink: Sink) -> Estimate {
        self.edited
            .get(&sink)
            .copied()
            .unwrap_or_default()
            .estimate(self.walks)
    }

    /// Ground capacitance (fF, positive, as [`NetResult::ground`]) of the
    /// edited geometry.
    #[must_use]
    pub fn ground(&self) -> Estimate {
        self.edited_ground.estimate(self.walks).scaled(-1.0)
    }

    /// Change of the ground capacitance (fF, positive convention).
    #[must_use]
    pub fn delta_ground(&self) -> Estimate {
        self.delta_ground.estimate(self.walks).scaled(-1.0)
    }
}

/// Result of [`replay_addition`].
#[derive(Debug, Clone)]
pub struct AdditionResult {
    /// The id of the added net (`Existing` id, or `geom.num_nets()` for `New`).
    pub added_net: NetId,
    /// Per target, in recording order.
    pub nets: Vec<AddedNetResult>,
}

/// Closed intersection (sharing a face, edge or corner counts).
fn touches(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|i| a.min()[i] <= b.max()[i] && b.min()[i] <= a.max()[i])
}

fn bb_meets(bb: &[[f32; 3]; 2], b: &Aabb) -> bool {
    (0..3).all(|i| f64::from(bb[0][i]) < b.max()[i] && b.min()[i] < f64::from(bb[1][i]))
}

/// Replays the samples of `rec` that an [`Addition`] can change and returns
/// the paired `ΔC` and the edited capacitances of every recorded target.
///
/// `geom`, `cfg` and the targets must be those of the recording, and
/// `nominal` its solution.
///
/// # Errors
///
/// A recording of another geometry or configuration; added boxes that
/// overlap conductors of another net, leave the domain (box, substrate or
/// the escape sphere of an open domain), or cut a target's Gaussian surface;
/// an unknown existing net.
#[allow(clippy::too_many_lines)]
pub fn replay_addition(
    geom: &Geometry,
    cfg: &SolverConfig,
    nominal: &Solution,
    rec: &Recording,
    add: &Addition,
) -> Result<AdditionResult, SolveError> {
    let targets: Vec<NetId> = rec.nets.iter().map(|(n, _)| *n).collect();
    if fingerprint(geom, &targets, cfg) != rec.fingerprint {
        return Err(SolveError::Config(
            "the recording belongs to another geometry, target set or configuration".into(),
        ));
    }
    let keys: Vec<_> = nominal.nets.iter().map(solution_key).collect();
    if keys != rec.solution {
        return Err(SolveError::Config(
            "the solution is not the one the recording was made with".into(),
        ));
    }
    let bad = |m: String| Err(SolveError::Config(m));
    let added_net = match add.net {
        AddedNet::Existing(n) if targets.contains(&n) => {
            return bad(format!(
                "addition: `{}` is a target (its Gaussian surface would not enclose the new \
                 boxes); extract with the boxes present and delete them instead",
                geom.net_name(n)
            ));
        }
        AddedNet::Existing(n) if n.index() < geom.num_nets() => n,
        AddedNet::Existing(n) => return bad(format!("addition: unknown {n}")),
        AddedNet::New => {
            NetId::from_index(geom.num_nets()).map_err(|e| SolveError::Config(e.to_string()))?
        }
    };
    if add.boxes.is_empty() {
        return bad("an addition needs at least one box".into());
    }
    let su = setup(geom, &targets, cfg)?;
    for b in &add.boxes {
        for (i, c) in geom.boxes().iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let n = geom.box_net(faradice_core::BoxId::new(i as u32));
            if n != added_net && touches(b, c) {
                return bad(format!(
                    "addition overlaps or touches a conductor of net `{}`",
                    geom.net_name(n)
                ));
            }
        }
        if let Some(bounds) = su.domain.bounds() {
            if !(0..3).all(|i| bounds.min()[i] <= b.min()[i] && b.max()[i] <= bounds.max()[i]) {
                return bad("addition leaves the domain box".into());
            }
        }
        if su.domain.substrate_z.is_some_and(|z| b.min()[2] < z) {
            return bad("addition reaches below the substrate".into());
        }
        if let Some((c, r)) = su.domain.sphere {
            let far = (0..3)
                .map(|i| {
                    let d = (b.min()[i] - c[i]).abs().max((b.max()[i] - c[i]).abs());
                    d * d
                })
                .sum::<f64>()
                .sqrt();
            if far >= r {
                return bad("addition leaves the escape sphere of the open domain".into());
            }
        }
    }
    let added: Vec<(Aabb, NetId)> = add.boxes.iter().map(|b| (*b, added_net)).collect();
    let nominal_walker = su.walker(geom, cfg);
    let edited_walker = Walker {
        added: Some(&added),
        ..nominal_walker
    };
    let eps = su.eps_unit();
    let odd = su.domain.odd_image();
    let mut nets = Vec::with_capacity(targets.len());
    for (k, (net, samples)) in rec.nets.iter().enumerate() {
        let surface = gauss_surface(
            geom,
            *net,
            su.clearances[k],
            cfg,
            su.medium.as_ref(),
            su.boxes.as_ref(),
        )?;
        if add.boxes.iter().any(|b| surface.cuts(b)) {
            return bad(format!(
                "addition cuts the Gaussian surface of target `{}` (offset {:.4} µm); record \
                 with a smaller walk.gauss_alpha, or extract with the box present and delete it",
                geom.net_name(*net),
                surface.offset()
            ));
        }
        let a_cand = surface.candidate_area();
        let contribs = |walker: &Walker<'_>, rng: &Rng| -> Vec<(Sink, f64)> {
            let mut rng = rng.clone();
            sample_once(
                walker,
                &surface,
                a_cand,
                eps,
                cfg.walk.antithetic,
                &mut rng,
                &mut |_, _| {},
            )
            .map_or_else(Vec::new, |OneSample { ends, w_pair }| {
                sample_contribs(hits_of(&ends, w_pair), odd, w_pair)
                    .as_slice()
                    .to_vec()
            })
        };
        // Per replayed sample: (original, edited) contributions, in order.
        let pairs: Vec<(Contribs, Contribs)> = samples
            .par_iter()
            .filter(|s| add.boxes.iter().any(|b| bb_meets(&s.bb, b)))
            .map(|s| {
                (
                    contribs(&nominal_walker, &s.rng),
                    contribs(&edited_walker, &s.rng),
                )
            })
            .collect();
        let nom: &NetResult = nominal
            .nets
            .iter()
            .find(|r| r.net == *net)
            .ok_or_else(|| SolveError::Config("the solution lacks a recorded target".into()))?;
        let mut delta: BTreeMap<Sink, Moments> = BTreeMap::new();
        let mut delta_ground = Moments::new();
        // Σ over replayed samples of 2·o·d + d², per sink and for ground.
        let mut cross: BTreeMap<Sink, f64> = BTreeMap::new();
        let mut cross_ground = 0.0;
        let is_ground = |s: Sink| {
            is_ground_sink(s, &cfg.ground_nets)
                || (add.ground && matches!(add.net, AddedNet::New) && s == Sink::Net(added_net))
        };
        let ground_of = |v: &[(Sink, f64)]| -> Option<f64> {
            let g: Vec<f64> = v
                .iter()
                .filter(|(s, _)| is_ground(*s))
                .map(|(_, w)| *w)
                .collect();
            (!g.is_empty()).then(|| g.iter().sum())
        };
        for (orig, new) in &pairs {
            let mut d: BTreeMap<Sink, (f64, f64)> = BTreeMap::new();
            for &(s, w) in orig {
                d.entry(s).or_default().0 += w;
            }
            for &(s, w) in new {
                d.entry(s).or_default().1 += w;
            }
            for (s, (o, n)) in d {
                let dd = n - o;
                if dd != 0.0 {
                    delta.entry(s).or_default().add(dd);
                    *cross.entry(s).or_default() += 2.0 * o * dd + dd * dd;
                }
            }
            let (go, gn) = (ground_of(orig), ground_of(new));
            if go.is_some() || gn.is_some() {
                let (o, n) = (go.unwrap_or(0.0), gn.unwrap_or(0.0));
                let dd = n - o;
                delta_ground.add(dd);
                cross_ground += 2.0 * o * dd + dd * dd;
            }
        }
        let mut edited: BTreeMap<Sink, Moments> = nom.sinks.iter().copied().collect();
        for (s, m) in &delta {
            let e = edited.entry(*s).or_default();
            *e = Moments::from_sums(e.sum() + m.sum(), e.sum_sq() + cross[s]);
        }
        let g = &nom.ground_moments;
        let edited_ground =
            Moments::from_sums(g.sum() + delta_ground.sum(), g.sum_sq() + cross_ground);
        nets.push(AddedNetResult {
            net: *net,
            walks: nom.walks,
            replayed: pairs.len() as u64,
            delta,
            delta_ground,
            edited,
            edited_ground,
        });
    }
    Ok(AdditionResult { added_net, nets })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, solve};
    use faradice_geom::GeometryBuilder;

    #[test]
    fn outward_rounding_never_overflows() {
        for v in [1e39, -1e39, 3.4e38, -3.4e38, 0.1, -0.1, 0.0] {
            let (d, u) = (down(v), up(v));
            assert!(!d.is_nan() && !u.is_nan());
            assert!(f64::from(d) <= v && v <= f64::from(u), "{v}: {d} {u}");
        }
    }

    /// Replaying every recorded sample in the nominal geometry reproduces
    /// the nominal sums: the saved streams are the batch streams.
    #[test]
    fn records_reproduce_the_run() {
        let mut gb = GeometryBuilder::new();
        let a = gb.net("a").unwrap();
        let b = gb.net("b").unwrap();
        gb.add_box(a, Aabb::from_array([0.0, 0.0, 1.0, 3.0, 0.2, 1.3]).unwrap())
            .unwrap();
        gb.add_box(b, Aabb::from_array([0.0, 0.5, 1.0, 3.0, 0.7, 1.3]).unwrap())
            .unwrap();
        let g = gb.build().unwrap();
        let cfg = SolverConfig {
            eps_r: 1.0,
            boundary: Boundary::Open,
            substrate_z: Some(0.0),
            accuracy: crate::Accuracy {
                min_walks: 40_000,
                max_walks: 40_000,
                ..crate::Accuracy::default()
            },
            walk: crate::WalkOptions::default(),
            seed: 3,
            ground_nets: Vec::new(),
            layers: Vec::new(),
            regions: Default::default(),
        };
        let (sol, rec) = solve_recorded(&g, &[a], &cfg).unwrap();
        let plain = solve(&g, &[a], &cfg).unwrap();
        assert_eq!(
            sol.nets[0].total().value.to_bits(),
            plain.nets[0].total().value.to_bits()
        );
        let su = setup(&g, &[a], &cfg).unwrap();
        let walker = su.walker(&g, &cfg);
        let surface = gauss_surface(&g, a, su.clearances[0], &cfg, None, None).unwrap();
        let eps = su.stack.eps()[0] * faradice_core::EPS0_FF_PER_UM;
        let mut sums: BTreeMap<Sink, f64> = BTreeMap::new();
        for s in &rec.nets[0].1 {
            let mut rng = s.rng.clone();
            let out = sample_once(
                &walker,
                &surface,
                surface.candidate_area(),
                eps,
                cfg.walk.antithetic,
                &mut rng,
                &mut |_, _| {},
            )
            .unwrap();
            for &(k, w) in
                sample_contribs(hits_of(&out.ends, out.w_pair), true, out.w_pair).as_slice()
            {
                *sums.entry(k).or_default() += w;
            }
        }
        for (k, m) in &sol.nets[0].sinks {
            let r = sums[k];
            assert!(
                (r - m.sum()).abs() <= 1e-9 * m.sum().abs().max(1e-6),
                "{k:?}: {r} vs {}",
                m.sum()
            );
        }
    }
}
