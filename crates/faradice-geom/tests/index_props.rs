//! Property tests: the uniform grid must agree exactly with brute force.

#![allow(missing_docs)]

use faradice_geom::{Aabb, BruteForce, GridOptions, SpatialIndex, UniformGrid};
use proptest::prelude::*;

fn arb_box() -> impl Strategy<Value = Aabb> {
    (
        prop::array::uniform3(-10.0f64..10.0),
        prop::array::uniform3(0.01f64..4.0),
    )
        .prop_map(|(lo, ext)| {
            Aabb::new(lo, [lo[0] + ext[0], lo[1] + ext[1], lo[2] + ext[2]]).unwrap()
        })
}

fn arb_point() -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3(-30.0f64..30.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn grid_equals_brute_force(
        boxes in prop::collection::vec(arb_box(), 1..60),
        points in prop::collection::vec(arb_point(), 1..40),
        cell in prop::option::of(0.05f64..5.0),
        max_cells in prop_oneof![Just(1usize << 16), Just(64usize)],
        max_r in prop_oneof![Just(f64::INFINITY), 0.0f64..20.0],
    ) {
        let grid = UniformGrid::from_boxes(&boxes, GridOptions { cell_size: cell, max_cells, ..GridOptions::default() });
        let brute = BruteForce::from_boxes(&boxes);
        for p in points {
            prop_assert_eq!(grid.nearest_linf(p, max_r), brute.nearest_linf(p, max_r));
            let exact = brute.nearest_linf(p, f64::INFINITY).unwrap().distance;
            prop_assert!(grid.clearance_lower_bound(p) <= exact);
        }
    }

    #[test]
    fn linf_distance_is_the_largest_free_cube(b in arb_box(), p in arb_point()) {
        // A cube of half-side d centred at p touches the closed box but does not
        // penetrate it; a slightly larger cube intersects it.
        let d = b.linf_distance(p);
        prop_assume!(d > 1e-9);
        let cube = |h: f64| Aabb::new(p.map(|x| x - h), p.map(|x| x + h)).unwrap();
        prop_assert!(cube(d * 1.0001).intersects_closed(&b));
        prop_assert!(!cube(d * 0.9999).intersects_closed(&b));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Duplicated boxes and grid-aligned query points exercise tie-breaking and
    /// cell-boundary handling.
    #[test]
    fn ties_and_cell_boundaries(
        boxes in prop::collection::vec(arb_box(), 1..20),
        ijk in prop::collection::vec(prop::array::uniform3(-40i32..40), 1..40),
    ) {
        let mut doubled = boxes.clone();
        doubled.extend(boxes);
        let cell = 0.5;
        let grid = UniformGrid::from_boxes(&doubled, GridOptions { cell_size: Some(cell), ..GridOptions::default() });
        let brute = BruteForce::from_boxes(&doubled);
        let origin = doubled.iter().fold(doubled[0], |a, b| a.union(b)).min();
        for v in ijk {
            let p = [0, 1, 2].map(|a| origin[a] + f64::from(v[a]) * cell);
            prop_assert_eq!(grid.nearest_linf(p, f64::INFINITY), brute.nearest_linf(p, f64::INFINITY));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Far from the origin, with boxes, query points and radii on multiples of
    /// the cell size: the ring cutoffs must allow for rounding at the scale of
    /// the coordinates.
    #[test]
    fn grid_aligned_far_coordinates(
        lo in prop::collection::vec(prop::array::uniform3(-12i32..12), 1..12),
        ext in prop::collection::vec(prop::array::uniform3(0i32..4), 12),
        points in prop::collection::vec(prop::array::uniform3(-16i32..16), 1..20),
        offset in prop_oneof![Just(0.0f64), Just(1000.0), Just(-1000.0), Just(-3.7e4)],
        cell in prop_oneof![Just(0.1f64), Just(0.3), Just(0.003), Just(0.7)],
        rings in 0i32..12,
    ) {
        let at = |i: i32| f64::from(i).mul_add(cell, offset);
        let boxes: Vec<Aabb> = lo
            .iter()
            .zip(&ext)
            .map(|(l, e)| {
                Aabb::new(l.map(at), [0, 1, 2].map(|a| at(l[a] + e[a].max(1)))).unwrap()
            })
            .collect();
        let grid = UniformGrid::from_boxes(&boxes, GridOptions { cell_size: Some(cell), ..GridOptions::default() });
        let brute = BruteForce::from_boxes(&boxes);
        for q in points {
            let p = q.map(|i| f64::from(i).mul_add(cell, offset + 0.5 * cell * f64::from(i % 3)));
            for max_r in [f64::from(rings) * cell, f64::from(rings) * cell * (1.0 + 1e-15)] {
                prop_assert_eq!(grid.nearest_linf(p, max_r), brute.nearest_linf(p, max_r));
                prop_assert_eq!(grid.nearest_linf_where(p, max_r, |i| i % 2 == 0), brute_where(&boxes, p, max_r));
            }
        }
    }
}

fn brute_where(boxes: &[Aabb], p: [f64; 3], max_r: f64) -> Option<faradice_geom::Nearest> {
    let kept: Vec<Aabb> = boxes.iter().step_by(2).copied().collect();
    BruteForce::from_boxes(&kept)
        .nearest_linf(p, max_r)
        .map(|n| faradice_geom::Nearest {
            distance: n.distance,
            box_id: faradice_core::BoxId::new(n.box_id.index() as u32 * 2),
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Local lists (design M2d §3.6): wherever they cover a query they give
    /// the grid's answer exactly, and the second nearest on another key is
    /// the brute-force one.
    #[test]
    fn local_lists_equal_the_grid(
        boxes in prop::collection::vec(arb_box(), 1..60),
        centre in arb_point(),
        radius in 0.01f64..20.0,
        queries in prop::collection::vec((prop::array::uniform3(-1.0f64..1.0), 0.0f64..1.0), 1..30),
        cell in prop::option::of(0.05f64..5.0),
        modulus in 1u32..5,
    ) {
        let grid = UniformGrid::from_boxes(&boxes, GridOptions { cell_size: cell, ..GridOptions::default() });
        let mut local = faradice_geom::LocalBoxes::default();
        grid.local(centre, radius, &mut local);
        let key = |i: u32| u64::from(i % modulus);
        for (off, frac) in queries {
            let p = std::array::from_fn(|a| centre[a] + off[a] * radius);
            let max_r = frac * radius;
            if !local.covers(p, max_r) {
                continue;
            }
            let keep = |i: u32| i % 2 == 0;
            prop_assert_eq!(local.nearest_linf_where(p, max_r, |_| true), grid.nearest_linf(p, max_r));
            prop_assert_eq!(local.nearest_linf_where(p, max_r, keep), grid.nearest_linf_where(p, max_r, keep));
            let (first, second) = local.nearest_two(p, max_r, key);
            prop_assert_eq!(first, grid.nearest_linf(p, max_r));
            let want = first.and_then(|f| {
                let k = key(f.box_id.index() as u32);
                grid.nearest_linf_where(p, max_r, |i| key(i) != k)
            });
            prop_assert_eq!(second, want);
        }
    }
}
