//! Progress reports and cooperative cancellation.

#![allow(missing_docs)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use faradice_core::NetId;
use faradice_frw::{
    Accuracy, Boundary, CpuEngine, Observer, Progress, SolverConfig, WalkOptions, solve,
    solve_observed, solve_with_observed,
};
use faradice_geom::{Aabb, Geometry, GeometryBuilder};

fn three_wires() -> Geometry {
    let mut g = GeometryBuilder::new();
    for (i, name) in ["a", "b", "c"].iter().enumerate() {
        let id = g.net(name).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let x = i as f64 * 0.5;
        g.add_box(
            id,
            Aabb::from_array([x, 0.0, 1.0, x + 0.2, 4.0, 1.2]).unwrap(),
        )
        .unwrap();
    }
    g.build().unwrap()
}

fn cfg() -> SolverConfig {
    SolverConfig {
        eps_r: 3.9,
        layers: Vec::new(),
        boundary: Boundary::Open,
        substrate_z: Some(0.0),
        accuracy: Accuracy {
            total_rel_tol: 0.004,
            min_walks: 4_000,
            ..Accuracy::default()
        },
        walk: WalkOptions::default(),
        seed: 7,
        ground_nets: Vec::new(),
        regions: Default::default(),
    }
}

/// What the tests keep of a progress report.
#[derive(Debug, Clone, Copy)]
struct Seen {
    target: usize,
    walks: u64,
    finished: bool,
    converged: bool,
    snapshot_walks: u64,
}

#[derive(Default)]
struct Recorder {
    reports: Mutex<Vec<Seen>>,
}

impl Observer for Recorder {
    fn progress(&self, p: &Progress<'_>) {
        self.reports.lock().unwrap().push(Seen {
            target: p.target,
            walks: p.walks,
            finished: p.finished,
            converged: p.converged,
            snapshot_walks: p.snapshot().walks,
        });
    }
}

/// Every statistic of two solutions, bit for bit.
fn assert_same(a: &faradice_frw::Solution, b: &faradice_frw::Solution) {
    assert_eq!(a.nets.len(), b.nets.len());
    assert_eq!(a.cancelled, b.cancelled);
    assert_eq!(a.eps_stop.to_bits(), b.eps_stop.to_bits());
    for (x, y) in a.nets.iter().zip(&b.nets) {
        assert_eq!(
            (x.net, x.walks, x.aborted_walks, x.total_steps, x.converged),
            (y.net, y.walks, y.aborted_walks, y.total_steps, y.converged)
        );
        assert_eq!(x.sinks.len(), y.sinks.len());
        for ((s1, m1), (s2, m2)) in x.sinks.iter().zip(&y.sinks) {
            assert_eq!(s1, s2);
            assert_eq!(m1.sum().to_bits(), m2.sum().to_bits());
            assert_eq!(m1.sum_sq().to_bits(), m2.sum_sq().to_bits());
        }
        assert_eq!(
            x.ground_moments.sum().to_bits(),
            y.ground_moments.sum().to_bits()
        );
        assert_eq!(
            x.ground_moments.sum_sq().to_bits(),
            y.ground_moments.sum_sq().to_bits()
        );
    }
}

fn targets(g: &Geometry) -> Vec<NetId> {
    g.nets().collect()
}

#[test]
fn observation_does_not_change_results() {
    let g = three_wires();
    let t = targets(&g);
    let plain = solve(&g, &t, &cfg()).unwrap();
    let rec = Recorder::default();
    let seen = solve_observed(&g, &t, &cfg(), &rec).unwrap();
    assert!(!seen.cancelled);
    assert_same(&plain, &seen);
    // Also independent of the thread count.
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let other = pool
            .install(|| solve_observed(&g, &t, &cfg(), &Recorder::default()))
            .unwrap();
        assert_same(&plain, &other);
    }
    let reports = rec.reports.into_inner().unwrap();
    assert!(reports.iter().all(|p| p.snapshot_walks == p.walks));
    for (i, n) in seen.nets.iter().enumerate() {
        let mine: Vec<&Seen> = reports.iter().filter(|p| p.target == i).collect();
        let last = mine.last().expect("at least one report per net");
        assert!(last.finished, "last report of a net is final");
        assert_eq!(mine.iter().filter(|p| p.finished).count(), 1);
        assert_eq!(last.walks, n.walks);
        assert_eq!(last.converged, n.converged);
        assert!(mine.windows(2).all(|w| w[0].walks <= w[1].walks));
    }
}

/// Cancels after a given number of progress reports.
struct CancelAfter {
    after: usize,
    seen: AtomicUsize,
    stop: AtomicBool,
}

impl Observer for CancelAfter {
    fn progress(&self, _: &Progress<'_>) {
        if self.seen.fetch_add(1, Ordering::SeqCst) + 1 >= self.after {
            self.stop.store(true, Ordering::SeqCst);
        }
    }
    fn cancelled(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

fn cancel_after(after: usize) -> CancelAfter {
    CancelAfter {
        after,
        seen: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
    }
}

#[test]
fn cancellation_keeps_merged_batches() {
    let g = three_wires();
    let t = targets(&g);
    let full = solve(&g, &t, &cfg()).unwrap();
    let obs = cancel_after(1);
    let part = solve_observed(&g, &t, &cfg(), &obs).unwrap();
    assert!(part.cancelled);
    assert!(!part.nets.is_empty());
    let walks: u64 = part.nets.iter().map(|n| n.walks).sum();
    let full_walks: u64 = full.nets.iter().map(|n| n.walks).sum();
    assert!(walks < full_walks, "{walks} < {full_walks}");
    assert!(part.nets.iter().all(|n| n.walks > 0));
}

#[test]
fn cancelled_before_start_returns_nothing() {
    let g = three_wires();
    let t = targets(&g);
    let obs = cancel_after(0);
    obs.stop.store(true, Ordering::SeqCst);
    let sol = solve_observed(&g, &t, &cfg(), &obs).unwrap();
    assert!(sol.cancelled);
    assert!(sol.nets.is_empty());
}

#[test]
fn engine_path_reports_and_cancels() {
    let g = three_wires();
    let t = targets(&g);
    let rec = Recorder::default();
    let sol = solve_with_observed(&g, &t, &cfg(), |_| Ok(CpuEngine), &rec).unwrap();
    assert!(!sol.cancelled);
    let reports = rec.reports.into_inner().unwrap();
    assert_eq!(reports.iter().filter(|p| p.finished).count(), t.len());

    let obs = cancel_after(1);
    let part = solve_with_observed(&g, &t, &cfg(), |_| Ok(CpuEngine), &obs).unwrap();
    assert!(part.cancelled);
    assert!(part.nets.len() < t.len() || part.nets.iter().any(|n| !n.converged));
}

#[test]
fn pre_cancelled_engine_solve_never_builds_the_engine() {
    let g = three_wires();
    let t = targets(&g);
    let obs = cancel_after(0);
    obs.stop.store(true, Ordering::SeqCst);
    let sol = solve_with_observed(
        &g,
        &t,
        &cfg(),
        |_| -> Result<CpuEngine, String> { panic!("engine must not be created") },
        &obs,
    )
    .unwrap();
    assert!(sol.cancelled);
    assert!(sol.nets.is_empty());
}
