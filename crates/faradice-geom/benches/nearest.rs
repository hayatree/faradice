//! Nearest-box query throughput on a synthetic multi-layer bus.

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use faradice_geom::{Aabb, GridOptions, SpatialIndex, UniformGrid};

/// `layers` routing layers of `wires` parallel wires, alternating direction,
/// 0.14 µm wide on a 0.28 µm pitch, 100 µm long.
fn bus(layers: usize, wires: usize) -> Vec<Aabb> {
    let mut out = Vec::with_capacity(layers * wires);
    for l in 0..layers {
        let z0 = 1.0 + 0.8 * l as f64;
        for w in 0..wires {
            let t0 = 0.28 * w as f64;
            let c = if l % 2 == 0 {
                [0.0, t0, z0, 100.0, t0 + 0.14, z0 + 0.36]
            } else {
                [t0, 0.0, z0, t0 + 0.14, 100.0, z0 + 0.36]
            };
            out.push(Aabb::from_array(c).unwrap());
        }
    }
    out
}

/// Deterministic pseudo-random query points inside the bus volume (SplitMix64).
fn points(n: usize) -> Vec<[f64; 3]> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64
    };
    (0..n)
        .map(|_| [next() * 100.0, next() * 100.0, 0.5 + next() * 4.0])
        .collect()
}

fn bench(c: &mut Criterion) {
    let pts = points(10_000);
    let mut group = c.benchmark_group("nearest_linf");
    for &wires in &[100usize, 350] {
        let boxes = bus(4, wires);
        let grid = UniformGrid::from_boxes(&boxes, GridOptions::default());
        group.bench_with_input(BenchmarkId::new("exact", boxes.len()), &grid, |b, g| {
            b.iter(|| {
                let mut acc = 0.0;
                for &p in &pts {
                    acc += g
                        .nearest_linf(black_box(p), f64::INFINITY)
                        .map_or(0.0, |n| n.distance);
                }
                acc
            });
        });
        group.bench_with_input(
            BenchmarkId::new("clearance_lb", boxes.len()),
            &grid,
            |b, g| {
                b.iter(|| {
                    let mut acc = 0.0;
                    for &p in &pts {
                        acc += g.clearance_lower_bound(black_box(p));
                    }
                    acc
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
