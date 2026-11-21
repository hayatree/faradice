//! Per-target sink cross moments (design D3 §5.2).
//!
//! Entries of one row of the capacitance matrix come from the same samples,
//! so they are correlated. Non-linear functions of the matrix (e.g. the Schur
//! complement that floats a conductor) need the full per-row covariance. A
//! sample contributes to at most four sinks, so the cross moments
//! `Σ_samples c_k · c_l` (k < l) are cheap to accumulate through the
//! per-sample hook. [`solve`](crate::solve) itself is unchanged.

use std::collections::BTreeMap;

use faradice_core::stats::NeumaierSum;
use faradice_core::{NetId, Sink};
use faradice_geom::Geometry;

use crate::solver::{
    HookSample, Observer, SampleHook, Solution, SolveError, SolverConfig, hits_of, sample_contribs,
    solve_hooked,
};

/// Cross moments of one target: `Σ c_k · c_l` over samples, for sink pairs
/// `k < l` that occurred in the same sample (all other pairs are zero).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CrossMoments {
    /// `((k, l), Σ c_k c_l)` with `k < l`, sorted.
    pub pairs: Vec<((Sink, Sink), f64)>,
}

impl CrossMoments {
    /// `Σ c_k c_l` for the unordered pair `{k, l}` (`k ≠ l`).
    #[must_use]
    pub fn get(&self, k: Sink, l: Sink) -> f64 {
        let key = if k < l { (k, l) } else { (l, k) };
        self.pairs
            .binary_search_by(|(p, _)| p.cmp(&key))
            .map_or(0.0, |i| self.pairs[i].1)
    }
}

struct CrossHook {
    odd_image: bool,
}

impl SampleHook for CrossHook {
    type Stats = BTreeMap<(Sink, Sink), NeumaierSum>;

    fn sample(&self, stats: &mut Self::Stats, s: &HookSample) {
        let c = sample_contribs(hits_of(&s.ends, s.w_pair), self.odd_image, s.w_pair);
        let c = c.as_slice();
        for (i, &(a, wa)) in c.iter().enumerate() {
            for &(b, wb) in &c[i + 1..] {
                let key = if a < b { (a, b) } else { (b, a) };
                stats.entry(key).or_default().add(wa * wb);
            }
        }
    }

    fn merge(into: &mut Self::Stats, from: &Self::Stats) {
        for (k, v) in from {
            into.entry(*k).or_default().merge(v);
        }
    }
}

/// [`solve_observed`](crate::solve_observed) that also returns the
/// [`CrossMoments`] of every target (empty for targets that never started).
/// The [`Solution`] is bit-identical to `solve_observed`.
///
/// # Errors
///
/// As [`solve`](crate::solve).
pub fn solve_with_cross_moments(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    observer: &dyn Observer,
) -> Result<(Solution, Vec<CrossMoments>), SolveError> {
    let odd_image =
        crate::BoundaryModel::new(&cfg.boundary, cfg.substrate_z, geom.bounds()).odd_image();
    let (sol, stats) = solve_hooked(geom, targets, cfg, observer, &CrossHook { odd_image })?;
    // Keep only targets that started, aligned with `sol.nets`.
    let started: Vec<NetId> = sol.nets.iter().map(|n| n.net).collect();
    let cross = targets
        .iter()
        .zip(stats)
        .filter(|(t, _)| started.contains(t))
        .map(|(_, m)| CrossMoments {
            pairs: m.into_iter().map(|(k, v)| (k, v.value())).collect(),
        })
        .collect();
    Ok((sol, cross))
}
