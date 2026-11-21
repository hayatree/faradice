//! Simulation domain and ground boundaries.

use faradice_core::{Face, Sink};
use faradice_geom::Aabb;

/// Boundary condition of one face of a box domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceKind {
    /// Dirichlet, 0 V. Walks reaching it terminate at [`Sink::Boundary`].
    Ground,
    /// Homogeneous Neumann (reflecting). Walks are mirrored back into the domain.
    Reflect,
}

/// The outer boundary of the simulation domain.
#[derive(Debug, Clone, PartialEq)]
pub enum Boundary {
    /// A finite box with a boundary condition per face, indexed by [`Face::index`].
    Box {
        /// Domain extent.
        bounds: Aabb,
        /// Per-face condition.
        faces: [FaceKind; 6],
    },
    /// Unbounded space; infinity is at 0 V ([`Sink::Infinity`]).
    Open,
}

impl Boundary {
    /// A box with every face grounded.
    #[must_use]
    pub const fn grounded_box(bounds: Aabb) -> Self {
        Self::Box {
            bounds,
            faces: [FaceKind::Ground; 6],
        }
    }
}

/// Precomputed boundary data used inside the walk loop.
///
/// Exposed read-only (see [`crate::engine::Scene`]) so that other walk engines
/// can reproduce the exact boundary semantics.
#[derive(Debug, Clone)]
pub struct BoundaryModel {
    pub(crate) bounds: Option<Aabb>,
    /// Grounded faces of the box: (axis index, coordinate, sink).
    pub(crate) ground_faces: Vec<(usize, f64, Sink)>,
    /// Reflecting faces: per axis `(min, max)` reflect flags.
    pub(crate) reflect: [(bool, bool); 3],
    pub(crate) substrate_z: Option<f64>,
    /// Open boundary: escape sphere `(centre, radius)`.
    pub(crate) sphere: Option<([f64; 3], f64)>,
}

impl BoundaryModel {
    /// Domain box (`None` for open space).
    #[must_use]
    pub const fn bounds(&self) -> Option<Aabb> {
        self.bounds
    }

    /// Grounded faces of the box as `(axis index, coordinate, sink)`.
    #[must_use]
    pub fn ground_faces(&self) -> &[(usize, f64, Sink)] {
        &self.ground_faces
    }

    /// Per-axis `(min, max)` reflecting flags.
    #[must_use]
    pub const fn reflect(&self) -> [(bool, bool); 3] {
        self.reflect
    }

    /// Substrate plane height.
    #[must_use]
    pub const fn substrate_z(&self) -> Option<f64> {
        self.substrate_z
    }

    /// Escape sphere `(centre, radius)` of the open boundary. Walks farther than
    /// `2 × radius` from the centre jump with the exterior harmonic measure.
    #[must_use]
    pub const fn sphere(&self) -> Option<([f64; 3], f64)> {
        self.sphere
    }

    /// `true` if walks may change sign (open boundary with substrate image):
    /// a walk reflected across the substrate plane flips its sign.
    #[must_use]
    pub const fn is_odd_image(&self) -> bool {
        self.odd_image()
    }

    pub(crate) fn new(boundary: &Boundary, substrate_z: Option<f64>, conductors: Aabb) -> Self {
        match boundary {
            Boundary::Box { bounds, faces } => {
                let mut ground_faces = Vec::new();
                let mut reflect = [(false, false); 3];
                for face in Face::ALL {
                    match faces[face.index()] {
                        FaceKind::Ground => {
                            ground_faces.push((
                                face.axis.index(),
                                bounds.face_coord(face),
                                Sink::Boundary(face),
                            ));
                        }
                        FaceKind::Reflect => {
                            let r = &mut reflect[face.axis.index()];
                            if face.side == faradice_core::Side::Min {
                                r.0 = true;
                            } else {
                                r.1 = true;
                            }
                        }
                    }
                }
                Self {
                    bounds: Some(*bounds),
                    ground_faces,
                    reflect,
                    substrate_z,
                    sphere: None,
                }
            }
            Boundary::Open => {
                let lo = conductors.min();
                let hi = conductors.max();
                let mut centre = [
                    0.5 * (lo[0] + hi[0]),
                    0.5 * (lo[1] + hi[1]),
                    0.5 * (lo[2] + hi[2]),
                ];
                if let Some(z0) = substrate_z {
                    // Centre on the plane so the odd image extension is symmetric.
                    centre[2] = z0;
                }
                let mut r2: f64 = 0.0;
                for &x in &[lo[0], hi[0]] {
                    for &y in &[lo[1], hi[1]] {
                        for &z in &[lo[2], hi[2]] {
                            let d = [x - centre[0], y - centre[1], z - centre[2]];
                            r2 = r2.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                        }
                    }
                }
                let radius = r2.sqrt() * 1.05 + 1e-9;
                Self {
                    bounds: None,
                    ground_faces: Vec::new(),
                    reflect: [(false, false); 3],
                    substrate_z,
                    sphere: Some((centre, radius)),
                }
            }
        }
    }

    /// Distance to the nearest Dirichlet boundary (substrate or grounded face) and its sink.
    #[inline]
    pub(crate) fn dirichlet_distance(&self, p: [f64; 3]) -> (f64, Sink) {
        let mut best = (f64::INFINITY, Sink::Infinity);
        if let Some(z0) = self.substrate_z {
            best = (p[2] - z0, Sink::GroundPlane);
        }
        for &(axis, coord, sink) in &self.ground_faces {
            let d = (p[axis] - coord).abs();
            if d < best.0 {
                best = (d, sink);
            }
        }
        best
    }

    /// `true` if walks may change sign (open boundary with substrate image).
    pub(crate) const fn odd_image(&self) -> bool {
        self.sphere.is_some() && self.substrate_z.is_some()
    }

    /// Mirrors `p` back into the box across reflecting faces.
    #[inline]
    pub(crate) fn fold(&self, p: &mut [f64; 3]) {
        let Some(b) = self.bounds else { return };
        let (lo, hi) = (b.min(), b.max());
        for a in 0..3 {
            let (rmin, rmax) = self.reflect[a];
            if !(rmin || rmax) {
                continue;
            }
            // At most a few iterations: a step never exceeds the domain diameter by much.
            for _ in 0..64 {
                if rmin && p[a] < lo[a] {
                    p[a] = 2.0 * lo[a] - p[a];
                } else if rmax && p[a] > hi[a] {
                    p[a] = 2.0 * hi[a] - p[a];
                } else {
                    break;
                }
            }
        }
    }

    /// `true` if `p` lies inside the domain (always true for open space above the substrate).
    pub(crate) fn contains(&self, p: [f64; 3]) -> bool {
        if self.substrate_z.is_some_and(|z0| p[2] <= z0) {
            return false;
        }
        self.bounds
            .is_none_or(|b| (0..3).all(|a| b.min()[a] < p[a] && p[a] < b.max()[a]))
    }
}
