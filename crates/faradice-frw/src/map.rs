//! Potential maps for insertion analysis (design D3 §7, pillar B).
//!
//! For a point `x` outside the conductors, `φ_k(x)` is the potential at `x`
//! when net `k` is at 1 V and every other conductor and ground is at 0 V. A
//! floating random walk from `x` ends on net `k` with probability `φ_k(x)`
//! (with the odd substrate image, weighted by the walk's sign), and the centre
//! gradient kernel of a cube around `x` gives `∇φ_k(x)` from the same kind of
//! walks. No global re-solve is needed.
//!
//! The leading-order change of the Maxwell matrix when a small object is
//! inserted at `x` (far from other conductors relative to its size):
//!
//! ```text
//! grounded, capacity c:          ΔC_ij ≈ +c · φ_i(x) φ_j(x)
//! floating, polarizability M:    ΔC_ij ≈ ∇φ_i(x) · M ∇φ_j(x)
//! ```

use std::collections::BTreeMap;

use faradice_core::{Axis, Estimate, Moments, NetId, Sink};
use faradice_geom::Geometry;
use rayon::prelude::*;

use crate::medium::homogeneous_hop;
use crate::rng::{Rng, splitmix64};
use crate::solver::{SolveError, SolverConfig, setup};

/// Domain-separation tag of the map streams.
const MAP_TAG: u64 = 0x3A9F_0E57_1A11_D3B0;

/// Options of [`potential_map`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapOptions {
    /// Samples per point (one potential walk, and with `gradient` one
    /// antithetic gradient pair per axis).
    pub samples: u64,
    /// Also estimate `∇φ_k`.
    pub gradient: bool,
}

impl Default for MapOptions {
    fn default() -> Self {
        Self {
            samples: 10_000,
            gradient: true,
        }
    }
}

/// Potentials (and gradients) at one point.
#[derive(Debug, Clone, PartialEq)]
pub struct PointPotentials {
    /// The point.
    pub point: [f64; 3],
    /// The net whose conductor contains the point (or lies within the
    /// stopping distance of it): then `φ` of that net is exactly 1 and the
    /// other potentials and all gradients are not estimated. Points within
    /// the stopping distance of a grounded boundary have all potentials zero.
    pub inside: Option<NetId>,
    /// `φ_k(x)` per conductor net hit by any walk (others are zero).
    pub phi: Vec<(NetId, Estimate)>,
    /// `∂φ_k/∂(x, y, z)` (1/µm) per conductor net, if requested.
    pub grad: Vec<(NetId, [Estimate; 3])>,
    /// Unbiased estimates of `φ_i φ_j` (`i ≤ j`) for pairs seen, from
    /// distinct samples (U-statistic): the product of two means from the same
    /// samples is biased by their covariance.
    pub phi_products: Vec<((NetId, NetId), f64)>,
    /// Unbiased estimates of `∇φ_i · ∇φ_j` (`i ≤ j`), likewise.
    pub grad_dots: Vec<((NetId, NetId), f64)>,
    /// The point lies outside the simulation domain (outside the box, or
    /// below the substrate plane): nothing is estimated.
    pub outside: bool,
    /// Walks that exceeded `max_steps` (counted as reaching no net).
    pub aborted_walks: u64,
}

impl PointPotentials {
    /// `φ_k(x)` (zero if never hit).
    #[must_use]
    pub fn phi(&self, net: NetId) -> Estimate {
        if self.inside == Some(net) {
            return Estimate {
                value: 1.0,
                std_err: 0.0,
            };
        }
        self.phi.iter().find(|(n, _)| *n == net).map_or(
            Estimate {
                value: 0.0,
                std_err: 0.0,
            },
            |(_, e)| *e,
        )
    }

    /// Unbiased `φ_i φ_j` (zero if the pair never occurred; 1 if the point
    /// is inside net `i = j`).
    #[must_use]
    pub fn phi_product(&self, i: NetId, j: NetId) -> f64 {
        if self.inside.is_some() {
            return self.phi(i).value * self.phi(j).value;
        }
        lookup_pair(&self.phi_products, i, j)
    }

    /// Unbiased `∇φ_i · ∇φ_j` (zero if never seen or not requested).
    #[must_use]
    pub fn grad_dot(&self, i: NetId, j: NetId) -> f64 {
        lookup_pair(&self.grad_dots, i, j)
    }

    /// `∇φ_k(x)` (zero if never hit or not requested).
    #[must_use]
    pub fn grad(&self, net: NetId) -> [Estimate; 3] {
        let zero = Estimate {
            value: 0.0,
            std_err: 0.0,
        };
        self.grad
            .iter()
            .find(|(n, _)| *n == net)
            .map_or([zero; 3], |(_, g)| *g)
    }
}

fn lookup_pair(v: &[((NetId, NetId), f64)], i: NetId, j: NetId) -> f64 {
    let key = if i <= j { (i, j) } else { (j, i) };
    v.iter().find(|(k, _)| *k == key).map_or(0.0, |(_, x)| *x)
}

/// `(S_i S_j − Σ_a x_ia x_ja) / (N (N − 1))` for every pair: the unbiased
/// estimate of the product of two means from `N` samples.
fn u_products(
    sums: &BTreeMap<NetId, f64>,
    within: &BTreeMap<(NetId, NetId), f64>,
    n: u64,
) -> Vec<((NetId, NetId), f64)> {
    #[allow(clippy::cast_precision_loss)]
    let norm = (n as f64) * (n as f64 - 1.0);
    let keys: Vec<NetId> = sums.keys().copied().collect();
    let mut out = Vec::new();
    for (a, &i) in keys.iter().enumerate() {
        for &j in &keys[a..] {
            let w = within.get(&(i, j)).copied().unwrap_or(0.0);
            out.push(((i, j), (sums[&i] * sums[&j] - w) / norm));
        }
    }
    out
}

/// Estimates `φ_k` (and `∇φ_k`) of every conductor net at `points`.
///
/// `targets` set the stopping distance exactly as in
/// [`solve`](crate::solve) with the same targets, so maps and extractions
/// share it. Results are reproducible for `cfg.seed` and independent of the
/// thread count.
///
/// Walks that exceed `max_steps` count as reaching no net (biasing the
/// potentials toward zero); [`PointPotentials::aborted_walks`] reports them.
///
/// # Errors
///
/// As [`solve`](crate::solve); fewer than two samples; gradients are not
/// available with stratified dielectrics yet.
pub fn potential_map(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    points: &[[f64; 3]],
    opts: MapOptions,
) -> Result<Vec<PointPotentials>, SolveError> {
    let su = setup(geom, targets, cfg)?;
    if opts.samples < 2 {
        return Err(SolveError::Config(
            "a potential map needs at least two samples per point".into(),
        ));
    }
    if opts.gradient && su.boxes.is_some() {
        return Err(SolveError::Config(
            "potential gradients do not support dielectric boxes yet".into(),
        ));
    }
    if opts.gradient && su.medium.is_some() {
        return Err(SolveError::Config(
            "potential gradients do not support stratified dielectrics yet".into(),
        ));
    }
    let walker = su.walker(geom, cfg);
    if let Some(b) = su.boxes.as_ref() {
        // Points on a dielectric edge have no transition domain (design M2c §4.2).
        for p in points {
            let (h, _) = walker.clearance(*p);
            if h >= su.eps_stop && !b.makes_progress(*p, h) {
                return Err(SolveError::Config(format!(
                    "map point {p:?} lies on a dielectric edge"
                )));
            }
        }
    }
    Ok(points
        .par_iter()
        .enumerate()
        .map(|(pi, &x)| {
            let empty = |inside, outside| PointPotentials {
                point: x,
                inside,
                phi: Vec::new(),
                grad: Vec::new(),
                phi_products: Vec::new(),
                grad_dots: Vec::new(),
                outside,
                aborted_walks: 0,
            };
            let out_of_box = su.domain.bounds().is_some_and(|b| !b.contains(x));
            let below = su.domain.substrate_z().is_some_and(|z| x[2] < z);
            if out_of_box || below {
                return empty(None, true);
            }
            let (h, sink) = walker.clearance(x);
            // Inside a box the clearance is zero.
            if h < su.eps_stop {
                let inside = match sink {
                    Sink::Net(n) => Some(n),
                    _ => None,
                };
                return empty(inside, false);
            }
            let mut key = cfg.seed ^ MAP_TAG;
            key = splitmix64(&mut key) ^ pi as u64;
            let mut rng = Rng::from_seed(splitmix64(&mut key));
            let mut phi: BTreeMap<NetId, Moments> = BTreeMap::new();
            let mut grad: BTreeMap<NetId, [Moments; 3]> = BTreeMap::new();
            let mut aborted = 0;
            let mut sample_phi: BTreeMap<NetId, f64> = BTreeMap::new();
            let mut sample_grad: BTreeMap<(NetId, usize), f64> = BTreeMap::new();
            // Sums and within-sample products for the unbiased pair products.
            let (mut phi_sum, mut grad_sum) = (BTreeMap::new(), BTreeMap::new());
            let (mut phi_within, mut grad_within) = (BTreeMap::new(), BTreeMap::new());
            let mut sample_vec: BTreeMap<NetId, [f64; 3]> = BTreeMap::new();
            for _ in 0..opts.samples {
                sample_phi.clear();
                sample_grad.clear();
                let end = walker.run(x, &mut rng);
                match end.sink {
                    Some(Sink::Net(n)) => *sample_phi.entry(n).or_default() += end.sign,
                    Some(_) => {}
                    None => aborted += 1,
                }
                if opts.gradient {
                    for (a, axis) in [Axis::X, Axis::Y, Axis::Z].into_iter().enumerate() {
                        let hop = homogeneous_hop(walker.green, x, axis, h, 1.0, true, &mut rng);
                        for (slot, w) in [(0, 0.5), (1, -0.5)] {
                            let mut start = hop.starts[slot];
                            walker.domain.fold(&mut start);
                            let end = walker.run(start, &mut rng);
                            match end.sink {
                                Some(Sink::Net(n)) => {
                                    *sample_grad.entry((n, a)).or_default() +=
                                        w * hop.factor * end.sign;
                                }
                                Some(_) => {}
                                None => aborted += 1,
                            }
                        }
                    }
                }
                for (&n, &v) in &sample_phi {
                    phi.entry(n).or_default().add(v);
                    *phi_sum.entry(n).or_insert(0.0) += v;
                }
                for (&i, &vi) in &sample_phi {
                    for (&j, &vj) in sample_phi.range(i..) {
                        *phi_within.entry((i, j)).or_insert(0.0) += vi * vj;
                    }
                }
                sample_vec.clear();
                for (&(n, a), &v) in &sample_grad {
                    grad.entry(n).or_insert_with(|| [Moments::new(); 3])[a].add(v);
                    sample_vec.entry(n).or_insert([0.0; 3])[a] += v;
                }
                for (&i, gi) in &sample_vec {
                    let s = grad_sum.entry(i).or_insert([0.0; 3]);
                    for a in 0..3 {
                        s[a] += gi[a];
                    }
                    for (&j, gj) in sample_vec.range(i..) {
                        let dot: f64 = (0..3).map(|a| gi[a] * gj[a]).sum();
                        *grad_within.entry((i, j)).or_insert(0.0) += dot;
                    }
                }
            }
            let grad_dots = {
                // Σ_a over axes of the per-axis U-statistics.
                #[allow(clippy::cast_precision_loss)]
                let norm = (opts.samples as f64) * (opts.samples as f64 - 1.0);
                let keys: Vec<NetId> = grad_sum.keys().copied().collect();
                let mut v = Vec::new();
                for (k, &i) in keys.iter().enumerate() {
                    for &j in &keys[k..] {
                        let (si, sj): (&[f64; 3], &[f64; 3]) = (&grad_sum[&i], &grad_sum[&j]);
                        let ss: f64 = (0..3).map(|a| si[a] * sj[a]).sum();
                        let w = grad_within.get(&(i, j)).copied().unwrap_or(0.0);
                        v.push(((i, j), (ss - w) / norm));
                    }
                }
                v
            };
            PointPotentials {
                point: x,
                inside: None,
                outside: false,
                phi_products: u_products(&phi_sum, &phi_within, opts.samples),
                grad_dots,
                phi: phi
                    .into_iter()
                    .map(|(n, m)| (n, m.estimate(opts.samples)))
                    .collect(),
                grad: grad
                    .into_iter()
                    .map(|(n, m)| (n, m.map(|m| m.estimate(opts.samples))))
                    .collect(),
                aborted_walks: aborted,
            }
        })
        .collect())
}
