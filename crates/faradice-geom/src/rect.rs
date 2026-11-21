//! Axis-aligned rectangles in 3D and an exact L∞ nearest-rectangle index.
//!
//! Dielectric interfaces (design M2c §3.2) are rectangles normal to one axis.
//! A cube centred at `p` with half-side `r` meets a closed rectangle iff the
//! rectangle's L∞ distance from `p` is below `r`, so the same nearest queries
//! as for conductor boxes apply.

use faradice_core::Axis;

use crate::{Aabb, GeomError, GridOptions, LocalBoxes, Nearest, Result, SpatialIndex, UniformGrid};

/// A closed axis-aligned rectangle: zero extent along its normal axis and a
/// positive extent along the two others.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    min: [f64; 3],
    max: [f64; 3],
    normal: Axis,
}

impl Rect {
    /// Creates a rectangle from its corners. Exactly one axis must have
    /// `min == max` (the normal); the others need `min < max`. All coordinates
    /// must be finite.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Result<Self> {
        let finite = (0..3).all(|a| min[a].is_finite() && max[a].is_finite());
        // With min ≤ max (checked below), `min >= max` means zero extent.
        let flat: Vec<usize> = (0..3).filter(|&a| min[a] >= max[a]).collect();
        let open = (0..3).all(|a| min[a] <= max[a]);
        match flat.as_slice() {
            [n] if finite && open => Ok(Self {
                min,
                max,
                normal: Axis::ALL[*n],
            }),
            _ => Err(GeomError::InvalidRect { min, max }),
        }
    }

    /// The rectangle in the plane `axis = coord` spanning `lo..hi` on the other
    /// axes (the entries of `lo`/`hi` at `axis` are ignored).
    pub fn in_plane(axis: Axis, coord: f64, lo: [f64; 3], hi: [f64; 3]) -> Result<Self> {
        let (mut min, mut max) = (lo, hi);
        min[axis.index()] = coord;
        max[axis.index()] = coord;
        Self::new(min, max)
    }

    /// Lower corner.
    #[must_use]
    pub const fn min(&self) -> [f64; 3] {
        self.min
    }

    /// Upper corner.
    #[must_use]
    pub const fn max(&self) -> [f64; 3] {
        self.max
    }

    /// The normal axis.
    #[must_use]
    pub const fn normal(&self) -> Axis {
        self.normal
    }

    /// Coordinate of the rectangle's plane along its normal.
    #[must_use]
    pub const fn coord(&self) -> f64 {
        self.min[self.normal.index()]
    }

    /// Area.
    #[must_use]
    pub fn area(&self) -> f64 {
        (0..3)
            .filter(|&a| a != self.normal.index())
            .map(|a| self.max[a] - self.min[a])
            .product()
    }

    /// L∞ distance from `p` to the closed rectangle (`0.0` on it).
    #[must_use]
    #[inline]
    pub fn linf_distance(&self, p: [f64; 3]) -> f64 {
        self.as_closed_box().linf_distance(p)
    }

    const fn as_closed_box(&self) -> Aabb {
        Aabb::closed(self.min, self.max)
    }
}

/// Cell size of a [`RectIndex`] in multiples of the median smallest in-plane
/// rectangle size. Interface faces of thin dielectric boxes (conformal shells,
/// 0.03 µm) are much smaller than the walk's query radii (the clearance), so
/// the conductor default (3) gives rings of hundreds of mostly empty cells;
/// 12 cut the CPU time of the SKY130 conformal bus (C5) by a third. Queries are
/// exact, so the cell size changes no result.
pub const RECT_CELL_FACTOR: f64 = 12.0;

/// Exact L∞ nearest-rectangle queries over a fixed set of rectangles.
///
/// Rectangle ids in the returned [`Nearest::box_id`] are indices into
/// [`rects`](Self::rects). Ties are broken by the smaller id.
#[derive(Debug, Clone)]
pub struct RectIndex {
    rects: Vec<Rect>,
    grid: Option<UniformGrid>,
}

impl RectIndex {
    /// Builds the index. An empty set is allowed (every query returns `None`).
    ///
    /// # Panics
    ///
    /// Panics if there are more than `u32::MAX` rectangles.
    #[must_use]
    pub fn new(rects: Vec<Rect>) -> Self {
        if rects.is_empty() {
            return Self { rects, grid: None };
        }
        let closed: Vec<Aabb> = rects.iter().map(Rect::as_closed_box).collect();
        // Cell size from the rectangles' in-plane sizes (their thickness is zero).
        let mut sizes: Vec<f64> = rects
            .iter()
            .map(|r| {
                (0..3)
                    .filter(|&a| a != r.normal.index())
                    .map(|a| r.max[a] - r.min[a])
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let mid = sizes.len() / 2;
        let (_, median, _) = sizes.select_nth_unstable_by(mid, f64::total_cmp);
        let opts = GridOptions {
            cell_size: Some(*median * RECT_CELL_FACTOR),
            ..GridOptions::default()
        };
        Self {
            grid: Some(UniformGrid::from_boxes(&closed, opts)),
            rects,
        }
    }

    /// The indexed rectangles.
    #[must_use]
    pub fn rects(&self) -> &[Rect] {
        &self.rects
    }

    /// Nearest rectangle to `p` within L∞ distance `max_r`, or `None`.
    #[must_use]
    pub fn nearest_linf(&self, p: [f64; 3], max_r: f64) -> Option<Nearest> {
        self.grid.as_ref()?.nearest_linf(p, max_r)
    }

    /// The rectangles within L∞ distance `radius` of `centre`, for repeated
    /// exact queries near it (see [`LocalBoxes`]).
    pub fn local(&self, centre: [f64; 3], radius: f64, out: &mut LocalBoxes) {
        match &self.grid {
            Some(g) => g.local(centre, radius, out),
            None => *out = LocalBoxes::default(),
        }
    }

    /// Nearest rectangle to `p` within `max_r` among those with `keep(id)`.
    #[must_use]
    pub fn nearest_linf_where(
        &self,
        p: [f64; 3],
        max_r: f64,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Nearest> {
        self.grid.as_ref()?.nearest_linf_where(p, max_r, keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faradice_core::BoxId;

    fn brute(
        rects: &[Rect],
        p: [f64; 3],
        max_r: f64,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Nearest> {
        let mut best: Option<Nearest> = None;
        for (i, r) in rects.iter().enumerate() {
            let i = u32::try_from(i).unwrap();
            let distance = r.linf_distance(p);
            if !keep(i) || distance > max_r {
                continue;
            }
            let cand = Nearest {
                distance,
                box_id: BoxId::new(i),
            };
            if best.is_none_or(|b| cand.better_than(&b)) {
                best = Some(cand);
            }
        }
        best
    }

    /// A face at exactly the query radius's ring cutoff (cell size
    /// 0.30000000000000004): the cutoff must allow for rounding in the cell
    /// assignment, as the block reach does.
    #[test]
    fn ring_cutoff_allows_for_rounding() {
        let r = 0.900_000_000_000_000_1;
        let boxes = [
            ([0.0, -3.0, -3.0], [0.6, 3.0, 3.0]),
            ([2.5, -1.0, 0.0], [3.5, 1.0, 0.1]),
            ([0.0, 6.0, 0.0], [1.0, 7.0, 0.1]),
            ([0.0, 8.0, 0.0], [1.0, 9.0, 0.1]),
            ([1.0, r, -3.0], [2.0, r + 1.0, 3.0]),
        ];
        let mut rects = Vec::new();
        for (lo, hi) in boxes {
            for x in [lo[0], hi[0]] {
                rects.push(Rect::new([x, lo[1], lo[2]], [x, hi[1], hi[2]]).unwrap());
            }
        }
        let index = RectIndex::new(rects.clone());
        let p = [1.5, 0.0, 0.05];
        for max_r in [r, 2.0, 0.9] {
            assert_eq!(
                index.nearest_linf_where(p, max_r, |_| true),
                brute(&rects, p, max_r, |_| true),
                "max_r {max_r}"
            );
            assert_eq!(
                index.nearest_linf(p, max_r),
                brute(&rects, p, max_r, |_| true)
            );
        }
    }

    /// The plane test of `nearest_two_normal` skips only faces the full scan
    /// skips: both give the same answers.
    #[test]
    fn nearest_two_normal_matches_nearest_two() {
        let mut x = 0.37_f64;
        let mut next = || {
            x = (x * 7.3 + 0.19).fract();
            x
        };
        for normal in Axis::ALL {
            let a = normal.index();
            let rects: Vec<Rect> = (0..60)
                .map(|_| {
                    let c = (4.0 * next()).round() / 4.0;
                    let mut lo = [next() * 3.0, next() * 3.0, next() * 3.0];
                    let mut hi = [lo[0] + next(), lo[1] + next(), lo[2] + next()];
                    (lo[a], hi[a]) = (c, c);
                    Rect::new(lo, hi).unwrap()
                })
                .collect();
            let index = RectIndex::new(rects.clone());
            let key = |id: u32| rects[id as usize].coord().to_bits();
            let mut set = LocalBoxes::default();
            for _ in 0..200 {
                let p = [next() * 4.0 - 0.5, next() * 4.0 - 0.5, next() * 4.0 - 0.5];
                let max_r = next() * 0.8;
                index.local(p, 4.0 * max_r, &mut set);
                assert!(set.covers(p, max_r));
                assert_eq!(
                    set.nearest_two_normal(p, max_r, a, key),
                    set.nearest_two(p, max_r, key),
                    "{normal:?} at {p:?} within {max_r}"
                );
            }
        }
    }

    #[test]
    fn validation() {
        assert!(Rect::new([0.0; 3], [1.0, 1.0, 0.0]).is_ok());
        assert_eq!(
            Rect::new([0.0; 3], [1.0, 1.0, 0.0]).unwrap().normal(),
            Axis::Z
        );
        assert!(Rect::new([0.0; 3], [1.0, 1.0, 1.0]).is_err()); // a box
        assert!(Rect::new([0.0; 3], [1.0, 0.0, 0.0]).is_err()); // a segment
        assert!(Rect::new([0.0; 3], [1.0, -1.0, 0.0]).is_err());
        assert!(Rect::new([0.0; 3], [1.0, f64::NAN, 0.0]).is_err());
        let r = Rect::in_plane(Axis::X, 2.0, [9.0, 0.0, 0.0], [9.0, 1.0, 3.0]).unwrap();
        assert_eq!((r.coord(), r.area()), (2.0, 3.0));
    }

    #[test]
    fn distances() {
        let r = Rect::in_plane(Axis::Z, 1.0, [0.0; 3], [2.0, 2.0, 0.0]).unwrap();
        assert_eq!(r.linf_distance([1.0, 1.0, 1.0]), 0.0);
        assert_eq!(r.linf_distance([1.0, 1.0, 1.5]), 0.5);
        assert_eq!(r.linf_distance([3.0, 1.0, 1.2]), 1.0);
        assert_eq!(r.linf_distance([-0.5, 1.0, -1.0]), 2.0);
    }

    #[test]
    fn empty_index() {
        let idx = RectIndex::new(Vec::new());
        assert!(idx.nearest_linf([0.0; 3], f64::INFINITY).is_none());
    }

    /// Random rectangles on all three axes against brute force, with and
    /// without a filter (here: keep only rectangles off a given plane).
    #[test]
    fn matches_brute_force() {
        let mut s = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            u
        };
        let mut rects = Vec::new();
        for i in 0..300 {
            let axis = Axis::ALL[i % 3];
            // Coordinates on a coarse lattice so that coplanar rectangles occur.
            let coord = (next() * 20.0).floor() * 0.5;
            let lo = [next() * 9.0, next() * 9.0, next() * 9.0];
            let hi = [
                lo[0] + 0.05 + next() * 2.0,
                lo[1] + 0.05 + next() * 2.0,
                lo[2] + 0.05 + next() * 2.0,
            ];
            rects.push(Rect::in_plane(axis, coord, lo, hi).unwrap());
        }
        let idx = RectIndex::new(rects.clone());
        for _ in 0..2000 {
            let p = [
                next() * 14.0 - 2.0,
                next() * 14.0 - 2.0,
                next() * 14.0 - 2.0,
            ];
            for max_r in [0.05, 0.7, f64::INFINITY] {
                assert_eq!(
                    idx.nearest_linf(p, max_r),
                    brute(&rects, p, max_r, |_| true),
                    "p={p:?}"
                );
                let Some(n) = idx.nearest_linf(p, max_r) else {
                    continue;
                };
                // Second-nearest distinct plane of the same normal (as in M2c §4.1).
                let first = rects[n.box_id.index()];
                let keep = |i: u32| {
                    let r = &rects[i as usize];
                    r.normal() == first.normal() && r.coord().to_bits() != first.coord().to_bits()
                };
                assert_eq!(
                    idx.nearest_linf_where(p, max_r, keep),
                    brute(&rects, p, max_r, keep),
                    "p={p:?}"
                );
            }
        }
    }
}
