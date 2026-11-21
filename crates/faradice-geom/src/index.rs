//! The nearest-conductor query trait and a brute-force reference implementation.

use faradice_core::BoxId;

use crate::{Aabb, Geometry};

/// Result of a nearest-box query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nearest {
    /// L∞ distance from the query point to the box (`0.0` if inside or on it).
    pub distance: f64,
    /// The nearest box.
    pub box_id: BoxId,
}

impl Nearest {
    /// `true` if `self` should be preferred over `other`: smaller distance, ties
    /// broken by smaller box id so that every index returns identical results.
    #[must_use]
    #[inline]
    pub fn better_than(&self, other: &Self) -> bool {
        self.distance
            .total_cmp(&other.distance)
            .then(self.box_id.cmp(&other.box_id))
            .is_lt()
    }
}

/// Exact nearest-box queries in the L∞ metric.
pub trait SpatialIndex: Send + Sync {
    /// Returns the nearest box to `p` whose L∞ distance is `≤ max_r`, or `None`.
    ///
    /// Ties are broken by the smaller [`BoxId`], so all implementations return
    /// the same answer for the same input.
    fn nearest_linf(&self, p: [f64; 3], max_r: f64) -> Option<Nearest>;

    /// A cheap lower bound on the L∞ distance from `p` to the nearest box.
    ///
    /// Any cube centred at `p` with this half-side is conductor-free. A random
    /// walk can use it to take large steps in empty space without an exact
    /// query. The default implementation is exact (and therefore not cheap).
    fn clearance_lower_bound(&self, p: [f64; 3]) -> f64 {
        self.nearest_linf(p, f64::INFINITY)
            .map_or(f64::INFINITY, |n| n.distance)
    }
}

/// O(n) reference implementation, used for testing other indices.
#[derive(Debug, Clone)]
pub struct BruteForce {
    boxes: Vec<Aabb>,
}

impl BruteForce {
    /// Builds the index over the boxes of `geom`.
    #[must_use]
    pub fn new(geom: &Geometry) -> Self {
        Self::from_boxes(geom.boxes())
    }

    /// Builds the index over arbitrary boxes (box ids are slice indices).
    #[must_use]
    pub fn from_boxes(boxes: &[Aabb]) -> Self {
        Self {
            boxes: boxes.to_vec(),
        }
    }
}

impl SpatialIndex for BruteForce {
    fn nearest_linf(&self, p: [f64; 3], max_r: f64) -> Option<Nearest> {
        let mut best: Option<Nearest> = None;
        for (i, b) in self.boxes.iter().enumerate() {
            let distance = b.linf_distance(p);
            if distance > max_r {
                continue;
            }
            // Box counts are validated to fit into u32 by `GeometryBuilder`.
            #[allow(clippy::cast_possible_truncation)]
            let cand = Nearest {
                distance,
                box_id: BoxId::new(i as u32),
            };
            if best.is_none_or(|b| cand.better_than(&b)) {
                best = Some(cand);
            }
        }
        best
    }
}
