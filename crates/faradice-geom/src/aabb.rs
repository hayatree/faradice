//! Axis-aligned boxes.

use faradice_core::{Axis, Face, Side};

use crate::{GeomError, Result};

/// A non-degenerate axis-aligned box with finite coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    min: [f64; 3],
    max: [f64; 3],
}

impl Aabb {
    /// Creates a box, validating that coordinates are finite and `min < max` on every axis.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Result<Self> {
        let ok = (0..3).all(|a| min[a].is_finite() && max[a].is_finite() && min[a] < max[a]);
        if ok {
            Ok(Self { min, max })
        } else {
            Err(GeomError::InvalidBox { min, max })
        }
    }

    /// A closed box that may have zero extent on some axes (a rectangle or a
    /// segment). Crate-private: such boxes only live inside indices that never
    /// hand them out (see [`RectIndex`](crate::RectIndex)). The caller
    /// guarantees finite coordinates with `min ≤ max`.
    pub(crate) const fn closed(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Creates a box from `[xmin, ymin, zmin, xmax, ymax, zmax]`.
    pub fn from_array(c: [f64; 6]) -> Result<Self> {
        Self::new([c[0], c[1], c[2]], [c[3], c[4], c[5]])
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

    /// Extent along each axis.
    #[must_use]
    pub fn extent(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Smallest extent over the three axes.
    #[must_use]
    pub fn min_extent(&self) -> f64 {
        let e = self.extent();
        e[0].min(e[1]).min(e[2])
    }

    /// Volume.
    #[must_use]
    pub fn volume(&self) -> f64 {
        let e = self.extent();
        e[0] * e[1] * e[2]
    }

    /// Total surface area.
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        let e = self.extent();
        2.0 * (e[0] * e[1] + e[1] * e[2] + e[0] * e[2])
    }

    /// Area of one face.
    #[must_use]
    pub fn face_area(&self, face: Face) -> f64 {
        let e = self.extent();
        match face.axis {
            Axis::X => e[1] * e[2],
            Axis::Y => e[0] * e[2],
            Axis::Z => e[0] * e[1],
        }
    }

    /// Coordinate of a face along its normal axis.
    #[must_use]
    pub const fn face_coord(&self, face: Face) -> f64 {
        let a = face.axis.index();
        match face.side {
            Side::Min => self.min[a],
            Side::Max => self.max[a],
        }
    }

    /// L∞ (Chebyshev) distance from `p` to the box; `0.0` if `p` is inside or on it.
    ///
    /// This is the half-side of the largest cube centred at `p` whose interior
    /// does not intersect the box.
    #[must_use]
    #[inline]
    pub fn linf_distance(&self, p: [f64; 3]) -> f64 {
        // Plain comparisons, not `f64::max`: no operand is NaN (finite
        // coordinates), and `maxnum` costs a compare-and-blend sequence per
        // call on baseline x86-64, a fifth of the walk time near dielectric
        // boxes. The result is never −0: `d` starts at +0 and only takes a
        // strictly larger value. (`f64::max` could give −0 for a point on a
        // face at coordinate −0; such a tie is now broken by box id.)
        let mut d = 0.0_f64;
        for a in 0..3 {
            let (lo, hi) = (self.min[a] - p[a], p[a] - self.max[a]);
            let da = if lo > hi { lo } else { hi };
            if da > d {
                d = da;
            }
        }
        d
    }

    /// `true` if `p` is inside the box or on its boundary.
    #[must_use]
    pub fn contains(&self, p: [f64; 3]) -> bool {
        (0..3).all(|a| self.min[a] <= p[a] && p[a] <= self.max[a])
    }

    /// `true` if the closed boxes intersect (including touching faces, edges or corners).
    #[must_use]
    pub fn intersects_closed(&self, other: &Self) -> bool {
        (0..3).all(|a| self.min[a] <= other.max[a] && other.min[a] <= self.max[a])
    }

    /// Smallest box containing both.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut min = self.min;
        let mut max = self.max;
        for a in 0..3 {
            min[a] = min[a].min(other.min[a]);
            max[a] = max[a].max(other.max[a]);
        }
        Self { min, max }
    }

    /// The box grown by `d ≥ 0` on every side.
    #[must_use]
    pub fn dilated(&self, d: f64) -> Self {
        debug_assert!(d >= 0.0 && d.is_finite());
        Self {
            min: [self.min[0] - d, self.min[1] - d, self.min[2] - d],
            max: [self.max[0] + d, self.max[1] + d, self.max[2] + d],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit() -> Aabb {
        Aabb::new([0.0; 3], [1.0; 3]).unwrap()
    }

    #[test]
    fn validation() {
        assert!(Aabb::new([0.0; 3], [1.0, 1.0, 0.0]).is_err());
        assert!(Aabb::new([0.0; 3], [1.0, f64::NAN, 1.0]).is_err());
        assert!(Aabb::new([0.0; 3], [1.0, 1.0, f64::INFINITY]).is_err());
        assert!(Aabb::from_array([0.0, 0.0, 0.0, 2.0, 1.0, 0.5]).is_ok());
    }

    #[test]
    fn measures() {
        let b = Aabb::from_array([0.0, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(b.volume(), 24.0);
        assert_eq!(b.surface_area(), 52.0);
        assert_eq!(b.min_extent(), 2.0);
        assert_eq!(b.face_area(Face::new(Axis::Z, Side::Max)), 6.0);
        assert_eq!(b.face_coord(Face::new(Axis::Y, Side::Max)), 3.0);
    }

    #[test]
    fn linf_distance_cases() {
        let b = unit();
        assert_eq!(b.linf_distance([0.5, 0.5, 0.5]), 0.0); // inside
        assert_eq!(b.linf_distance([1.0, 0.5, 0.5]), 0.0); // on face
        assert_eq!(b.linf_distance([3.0, 0.5, 0.5]), 2.0); // face region
        assert_eq!(b.linf_distance([3.0, 2.0, 0.5]), 2.0); // edge region: max(2, 1)
        assert_eq!(b.linf_distance([-1.0, -3.0, 4.0]), 3.0); // corner region
    }

    #[test]
    fn intersection_includes_touching() {
        let a = unit();
        let touching = Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]).unwrap();
        let apart = Aabb::new([1.5, 0.0, 0.0], [2.0, 1.0, 1.0]).unwrap();
        assert!(a.intersects_closed(&touching));
        assert!(!a.intersects_closed(&apart));
        assert_eq!(a.union(&apart).max(), [2.0, 1.0, 1.0]);
        assert_eq!(a.dilated(0.5).min(), [-0.5; 3]);
    }
}
