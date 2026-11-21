//! Gaussian surfaces around target nets.
//!
//! The Gaussian surface of a net is the outer surface of the union of its boxes
//! dilated by an offset `δ`. Candidate points are drawn uniformly (by area)
//! from all faces of all dilated boxes; a candidate is *rejected* if it is not
//! on the outer surface of the union (it lies inside another dilated box, or
//! the space just outside it is covered by one) or outside the domain.
//! Rejected candidates count as zero-weight samples, so the charge estimator
//! `A_cand · E[f · 1_accepted]` stays unbiased without knowing the exact
//! surface area.

use faradice_core::{Axis, Face, NetId};
use faradice_geom::{Aabb, Geometry, GridOptions, UniformGrid};

use crate::alias::AliasTable;
use crate::boundary::BoundaryModel;
use crate::rng::Rng;

/// A sampled Gaussian surface point.
#[derive(Debug, Clone, Copy)]
pub struct GaussPoint {
    /// Position.
    pub p: [f64; 3],
    /// Outward normal axis.
    pub axis: Axis,
    /// Outward normal sign (+1 for a max face).
    pub sign: f64,
    /// Dilated box and face the point was drawn from.
    pub(crate) owner: (usize, Face),
}

/// Gaussian surface of one net.
#[derive(Debug, Clone)]
pub struct GaussSurface {
    boxes: Vec<Aabb>,
    /// Occupancy grid over the dilated boxes, so rejection tests only look at
    /// boxes near the candidate (nets can have thousands of boxes).
    grid: UniformGrid,
    faces: AliasTable,
    face_ids: Vec<(usize, Face)>,
    area: f64,
    offset: f64,
    /// Distinguishes surfaces in caches (net, offset bits).
    key: (u32, u64),
}

impl GaussSurface {
    /// Builds the surface of `net` with offset `offset`.
    #[must_use]
    pub fn new(geom: &Geometry, net: NetId, offset: f64) -> Self {
        let boxes: Vec<Aabb> = geom
            .net_boxes(net)
            .iter()
            .map(|&b| geom.box_(b).dilated(offset))
            .collect();
        let mut weights = Vec::with_capacity(boxes.len() * 6);
        let mut face_ids = Vec::with_capacity(boxes.len() * 6);
        for (i, b) in boxes.iter().enumerate() {
            for face in Face::ALL {
                weights.push(b.face_area(face));
                face_ids.push((i, face));
            }
        }
        let area = weights.iter().sum();
        let opts = GridOptions {
            max_cells: boxes.len().saturating_mul(8).max(64),
            clearance_transform: false,
            ..GridOptions::default()
        };
        let grid = UniformGrid::from_boxes(&boxes, opts);
        Self {
            faces: AliasTable::new(&weights),
            grid,
            boxes,
            face_ids,
            area,
            offset,
            key: (net.raw(), offset.to_bits()),
        }
    }

    /// Cache key of this surface (net and offset).
    pub(crate) const fn key(&self) -> (u32, u64) {
        self.key
    }

    /// Dilated box `i`.
    pub(crate) fn dilated(&self, i: usize) -> &Aabb {
        &self.boxes[i]
    }

    /// `true` if no other dilated box meets the closed rectangle `[lo, hi]`
    /// on a face of box `bi`, or the space just outside it: every point of
    /// the rectangle is then on the outer surface (conservative).
    pub(crate) fn window_clear(&self, bi: usize, face: Face, lo: [f64; 3], hi: [f64; 3]) -> bool {
        let a = face.axis.index();
        let b = &self.boxes[bi];
        let scale = (0..3).map(|i| b.max()[i] - b.min()[i]).fold(0.0, f64::max);
        let e = 1e-9 * scale;
        let (mut lo, mut hi) = (lo, hi);
        if face.side.sign() > 0.0 {
            hi[a] += e;
        } else {
            lo[a] -= e;
        }
        let c: [f64; 3] = std::array::from_fn(|i| 0.5 * (lo[i] + hi[i]));
        let r = (0..3).map(|i| 0.5 * (hi[i] - lo[i])).fold(0.0, f64::max) + 2.0 * e;
        let mut seen: Vec<u32> = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        let own = bi as u32;
        while let Some(n) = self
            .grid
            .nearest_linf_where(c, r, |j| j != own && !seen.contains(&j))
        {
            let j = n.box_id.index();
            let o = &self.boxes[j];
            if (0..3).all(|i| o.min()[i] <= hi[i] + e && lo[i] - e <= o.max()[i]) {
                return false;
            }
            #[allow(clippy::cast_possible_truncation)]
            seen.push(j as u32);
        }
        true
    }

    /// Total candidate area `A_cand` (sum of all dilated-box face areas).
    #[must_use]
    pub const fn candidate_area(&self) -> f64 {
        self.area
    }

    /// The offset `δ`.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.offset
    }

    /// Whether `b` overlaps the region between the net and its surface (a
    /// conductor there would invalidate the surface).
    #[cfg_attr(not(feature = "sensitivity"), allow(dead_code))]
    pub(crate) fn cuts(&self, b: &Aabb) -> bool {
        self.boxes
            .iter()
            .any(|d| (0..3).all(|i| d.min()[i] < b.max()[i] && b.min()[i] < d.max()[i]))
    }

    /// Draws a candidate; returns `None` if it is rejected.
    pub(crate) fn sample(&self, rng: &mut Rng, domain: &BoundaryModel) -> Option<GaussPoint> {
        let (bi, face) = self.face_ids[self.faces.sample(rng)];
        let b = &self.boxes[bi];
        let (lo, hi) = (b.min(), b.max());
        let a = face.axis.index();
        let mut p = [0.0; 3];
        for t in 0..3 {
            p[t] = if t == a {
                b.face_coord(face)
            } else {
                lo[t] + (hi[t] - lo[t]) * rng.next_f64()
            };
        }
        if self.rejected(bi, face, p) || !domain.contains(p) {
            return None;
        }
        Some(GaussPoint {
            p,
            axis: face.axis,
            sign: face.side.sign(),
            owner: (bi, face),
        })
    }

    /// `true` if the point `p` on `face` of dilated box `bi` is not on the outer
    /// surface of the union (inside another dilated box, covered just outside,
    /// or a coincident face owned by a lower box index).
    fn rejected(&self, bi: usize, face: Face, p: [f64; 3]) -> bool {
        let b = &self.boxes[bi];
        let (lo, hi) = (b.min(), b.max());
        let a = face.axis.index();
        let sign = face.side.sign();
        // Probe just outside the face to detect internal (covered) faces.
        let scale = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
        let mut probe = p;
        probe[a] += sign * scale * 1e-9;
        // Boxes containing `p` or `probe` are registered in their cells.
        let near = self
            .grid
            .items_at(p)
            .iter()
            .chain(self.grid.items_at(probe));
        for &j in near {
            let j = j as usize;
            let other = &self.boxes[j];
            if j == bi {
                continue;
            }
            if strictly_inside(other, p) || strictly_inside(other, probe) {
                return true;
            }
            // Coincident exposed faces (duplicates, coplanar overlapping faces): the
            // lowest box index owns the patch, so it is counted exactly once.
            if j < bi && on_face(other, face, p) {
                return true;
            }
        }
        false
    }

    /// Length of the accepted run of the Gaussian surface from `p` (a point on
    /// the lateral face `owner` of the dilated box) along `dir · e_z`, capped at
    /// `limit`: every point of `[p, p + run·dir·e_z)` is accepted (on the outer
    /// surface and inside the domain). Returns 0 if `p` itself is not.
    ///
    /// Depends only on the face and the tangential position of `p`, which the
    /// half-band first hop relies on (design M2a §4).
    pub(crate) fn accepted_run(
        &self,
        owner: (usize, Face),
        p: [f64; 3],
        dir: f64,
        limit: f64,
        domain: &BoundaryModel,
    ) -> f64 {
        let (bi, face) = owner;
        let b = &self.boxes[bi];
        debug_assert!(face.axis != Axis::Z);
        let z = p[2];
        if z < b.min()[2] || z > b.max()[2] || self.rejected(bi, face, p) || !domain.contains(p) {
            return 0.0;
        }
        let mut run = limit.min(if dir > 0.0 {
            b.max()[2] - z
        } else {
            z - b.min()[2]
        });
        // Domain limits in z (points on the bound are outside).
        if let Some(bounds) = domain.bounds() {
            run = run.min(if dir > 0.0 {
                bounds.max()[2] - z
            } else {
                z - bounds.min()[2]
            });
        }
        if dir < 0.0 {
            if let Some(z0) = domain.substrate_z() {
                run = run.min(z - z0);
            }
        }
        if run <= 0.0 {
            return 0.0;
        }
        // Boxes that can reject a point of the run are registered in the cells the
        // run (and its probe) passes through.
        let a = face.axis.index();
        let (lo, hi) = (b.min(), b.max());
        let scale = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
        let mut probe = p;
        probe[a] += face.side.sign() * scale * 1e-9;
        let step = 0.5 * self.grid.cell_size();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = (run / step).ceil() as usize + 1;
        let mut ids: Vec<u32> = Vec::new();
        for k in 0..=n {
            #[allow(clippy::cast_precision_loss)]
            let dz = dir * run * (k as f64 / n as f64);
            for q in [p, probe] {
                ids.extend_from_slice(self.grid.items_at([q[0], q[1], q[2] + dz]));
            }
        }
        ids.sort_unstable();
        ids.dedup();
        // Distance along the run to where `[lo, hi]` (in z) starts, 0 if the
        // run starts inside it.
        let entry = |zlo: f64, zhi: f64, open: bool| -> f64 {
            let inside = if open {
                zlo < z && z < zhi
            } else {
                zlo <= z && z <= zhi
            };
            if inside {
                0.0
            } else if dir > 0.0 && zlo >= z {
                zlo - z
            } else if dir < 0.0 && zhi <= z {
                z - zhi
            } else {
                f64::INFINITY
            }
        };
        let lateral = |o: &Aabb, q: [f64; 3], strict: bool| {
            (0..2).all(|t| {
                if strict {
                    o.min()[t] < q[t] && q[t] < o.max()[t]
                } else {
                    o.min()[t] <= q[t] && q[t] <= o.max()[t]
                }
            })
        };
        for &j in &ids {
            let j = j as usize;
            if j == bi {
                continue;
            }
            let o = &self.boxes[j];
            let (ozlo, ozhi) = (o.min()[2], o.max()[2]);
            if lateral(o, p, true) || lateral(o, probe, true) {
                run = run.min(entry(ozlo, ozhi, true));
            }
            // Dilated coordinates are computed identically, so equality is exact.
            #[allow(clippy::float_cmp)]
            let coplanar = o.face_coord(face) == b.face_coord(face);
            if j < bi && coplanar && lateral(o, p, false) {
                run = run.min(entry(ozlo, ozhi, false));
            }
        }
        run.max(0.0)
    }
}

/// `true` if `p` lies on the closed face `face` of `b` (same orientation).
#[allow(clippy::float_cmp)] // dilated coordinates are computed identically, so equality is exact
fn on_face(b: &Aabb, face: Face, p: [f64; 3]) -> bool {
    let a = face.axis.index();
    let (lo, hi) = (b.min(), b.max());
    p[a] == b.face_coord(face) && (0..3).all(|t| t == a || (lo[t] <= p[t] && p[t] <= hi[t]))
}

fn strictly_inside(b: &Aabb, p: [f64; 3]) -> bool {
    let (lo, hi) = (b.min(), b.max());
    (0..3).all(|a| lo[a] < p[a] && p[a] < hi[a])
}

/// L∞ gap between two boxes (0 if they touch or overlap).
#[must_use]
pub fn box_gap(a: &Aabb, b: &Aabb) -> f64 {
    let mut d = 0.0_f64;
    for t in 0..3 {
        d = d.max(a.min()[t] - b.max()[t]).max(b.min()[t] - a.max()[t]);
    }
    d
}

/// Smallest L∞ gap between `net` and any other net or Dirichlet boundary.
#[must_use]
pub(crate) fn net_clearance(geom: &Geometry, net: NetId, domain: &BoundaryModel) -> f64 {
    let mut gap = f64::INFINITY;
    for &bi in geom.net_boxes(net) {
        let b = geom.box_(bi);
        for (j, other) in geom.boxes().iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            if geom.box_net(faradice_core::BoxId::new(j as u32)) != net {
                gap = gap.min(box_gap(b, other));
            }
        }
        if let Some(z0) = domain.substrate_z {
            gap = gap.min(b.min()[2] - z0);
        }
        for &(axis, coord, _) in &domain.ground_faces {
            gap = gap
                .min((b.min()[axis] - coord).abs())
                .min((b.max()[axis] - coord).abs());
        }
    }
    gap
}
