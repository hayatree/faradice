//! Walk steps in a medium with dielectric boxes (design M2c §4).
//!
//! Inside a cube that contains no interface normal to two of the axes, ε
//! depends on the third coordinate only (the Lemma of design M2c §3.2), so
//! the one-interface cube of M2a applies along that axis, rotated. Every step
//! takes the largest of:
//!
//! - the **stack step** (M2a/M2b, unchanged) with its clearance capped by the
//!   distance to the nearest material box, so that its domain sees the stack
//!   only;
//! - an **axis cube** along x, y or z: the M2a selector with the nearest
//!   interface plane along that axis (box faces, and stack planes for z) and
//!   the room left by the interfaces normal to the two other axes.
//!
//! Near a dielectric edge (two interface planes, design M2d) these cubes
//! shrink with the distance to the edge. If the best of them is smaller than
//! `α·h`, the walk instead
//!
//! - lands exactly on the nearer edge plane with a homogeneous landing box
//!   (M2b §3.3, rotated), or
//! - from a point on one plane, takes an **edge cube** centred on itself
//!   with the other plane at a tabulated offset ([`crate::edge::EdgeCube`]).
//!
//! Any deterministic choice of exact kernels keeps the estimator exact.
//! Exits through a cube face that lies on an interface plane by construction
//! are moved onto the plane exactly (design M2c §4.4).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use faradice_core::Axis;
use faradice_geom::{LocalBoxes, Nearest};

use crate::dielectric::BoxMedium;
use crate::edge::EdgeCube;
use crate::green::GreenTables;
use crate::layered::{KAPPA_RANGE, LayeredCube, LayeredGradients};
use crate::medium::{Hop, Medium, homogeneous_hop};
use crate::multilayer::{LandingBox, NormalGradient, Profile};
use crate::rng::Rng;
use crate::stack::Stack;

/// A cube smaller than this fraction of the clearance is no progress: the
/// walk is at a dielectric edge (design M2c §4.2, §4.6).
const MIN_PROGRESS: f64 = 1e-9;

/// Interface planes within this fraction of the clearance are an edge the
/// walk is moved onto (design M2c §4.6).
const EDGE_SNAP: f64 = 1e-8;

/// Landing-box aspect ratios `2^{r/2}`, `r < LANDING_RATIOS` (as for the
/// stack, M2b §3.3).
const LANDING_RATIOS: usize = 9;
/// Landing-box start heights: the smallest one that fits is used.
const LANDING_TAUS: [f64; 4] = [0.125, 0.25, 0.375, 0.5];

/// An edge-cube table slot: `None` if the configuration failed the accuracy
/// gate (design M2d §3.4).
type EdgeSlot = OnceLock<Option<Box<EdgeCube>>>;

/// The edge-cube slots of one `[y side]` permittivity pair: `n² · EDGE_OFFSETS`.
type EdgeChunk = OnceLock<Box<[EdgeSlot]>>;

/// Material boxes with the M2a tables for their contrasts.
#[derive(Debug)]
pub struct BoxSteps {
    medium: BoxMedium,
    stack: Stack,
    /// Tabulated interface offsets `K` and cells per face edge `M` (as for
    /// the stack, [`crate::StackOptions`]).
    offsets: usize,
    resolution: usize,
    /// Distinct permittivities of the stack and the boxes.
    eps: Vec<f64>,
    /// Per (centre ε, other ε, offset): one-interface cubes (shared by
    /// [`BoxSteps::with_medium`]).
    cubes: Arc<[OnceLock<LayeredCube>]>,
    /// Per (centre ε, other ε, offset): their first-hop gradient kernels.
    grads: Arc<[OnceLock<LayeredGradients>]>,
    /// Design M2d: edge cubes and landing boxes are used when the best step
    /// is below `alpha · h` (`None`: never).
    alpha: Option<f64>,
    /// Per start height and aspect ratio: landing boxes.
    landing: Arc<[OnceLock<LandingBox>]>,
    /// Edge-cube tables per quadrant permittivity indices and tabulated
    /// offset, built on first use (no lock on lookups). Two levels, so that
    /// only the `n²`-slot chunks of the `[y side]` pairs met are allocated
    /// (not all `n⁴`).
    edges: Arc<[EdgeChunk]>,
    /// Per (lower ε, upper ε): gradient kernels at a centre on the interface
    /// (design D3 §4.4c).
    iface: Arc<[OnceLock<LayeredGradients>]>,
    /// Design M2e: normal first-hop kernels of layered cubes, keyed by the
    /// bits of their profile (interface positions, ε relative to the centre).
    normals: Arc<RwLock<HashMap<Vec<u64>, Arc<NormalGradient>>>>,
    /// Identifies this medium in the per-thread local query lists.
    id: u64,
    /// Design M2f: window decisions and kernels (not shared by
    /// [`BoxSteps::with_medium`]: they depend on the medium).
    windows: Arc<crate::windows::WindowCache>,
}

/// Local lists of the face indices and the box volume index around recent
/// walkers, per thread (design M2d §3.6): queries near a walker scan a short
/// list instead of searching grid rings, with exactly the same result.
struct Local {
    owner: u64,
    /// Faces normal to x, y, z, then the material boxes.
    sets: [LocalBoxes; 4],
}

thread_local! {
    static LOCAL: std::cell::RefCell<Local> = std::cell::RefCell::new(Local {
        owner: u64::MAX,
        sets: Default::default(),
    });
}

/// A local list is built `LOCAL_REACH` times the query radius around the
/// walker, so it serves the next few steps.
const LOCAL_REACH: f64 = 4.0;

fn next_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// At most this many layered first-hop kernels are cached (design M2e §3.3);
/// beyond it a kernel is built for each use (slower, but the results do not
/// depend on which profiles were met first).
const MAX_NORMAL_KERNELS: usize = 1024;

/// At most this many interface planes in a layered first-hop cube.
const MAX_NORMAL_PLANES: usize = 8;

/// The edge frame at a walker (design M2d §3.1): the nearest interface planes
/// normal to `b` and `c`, and the third axis `a`.
#[derive(Debug, Clone, Copy)]
struct EdgeFrame {
    a: usize,
    b: usize,
    c: usize,
    cb: f64,
    cc: f64,
}

/// An interface plane along an axis and the permittivities on its two sides.
#[derive(Debug, Clone, Copy)]
struct Plane {
    coord: f64,
    /// ε below and above the plane (along the axis).
    sides: [f64; 2],
}

/// What a step along an axis does.
#[derive(Debug, Clone, Copy)]
enum Kind {
    /// Homogeneous cube; an exit with `u[snap.0] == snap.1` lands exactly on
    /// the plane `snap.2` (the cube face lies on it).
    Homogeneous(Option<(usize, f64, f64)>),
    /// One-interface cube: centre and other ε indices, offset index, and
    /// whether the interface lies below the centre along the axis.
    Layered(usize, usize, usize, bool),
}

/// A ball centred on a dielectric edge along axis `a` (planes normal to `b`
/// and `c` through it), reached by moving the walk by `moved`.
#[derive(Debug, Clone, Copy)]
struct EdgeBall {
    a: usize,
    b: usize,
    c: usize,
    centre: [f64; 3],
    moved: f64,
    r: f64,
}

/// The nearest interface per axis at a point (see [`BoxSteps::axis_deltas`]).
struct Deltas {
    /// The clearance capped by the reflection reach.
    cap: f64,
    /// Per axis: L∞ distance to the nearest interface normal to it (at most
    /// `cap`), and its plane if its normal distance attains it.
    delta: [(f64, Option<f64>); 3],
    /// Per axis: the nearest face within `cap`.
    nearest: [Option<Nearest>; 3],
    /// Per axis: the nearest face within `cap` on another plane than
    /// `nearest` (so the nearest face off any plane is one of the two).
    second: [Option<Nearest>; 3],
}

impl Deltas {
    /// The nearest face normal to axis `a` within `r ≤ cap` whose plane is
    /// not `skip` (any plane for `None`), from the two known per axis.
    fn face_off(
        &self,
        faces: &faradice_geom::RectIndex,
        a: usize,
        r: f64,
        skip: Option<f64>,
    ) -> Option<Nearest> {
        let off = |n: &Nearest| {
            skip.is_none_or(|s| faces.rects()[n.box_id.index()].coord().to_bits() != s.to_bits())
        };
        match self.nearest[a] {
            Some(n) if off(&n) => Some(n),
            Some(_) => self.second[a],
            None => None,
        }
        .filter(|n| n.distance <= r)
    }
}

impl Deltas {
    /// Room of the axis cube along `a`: no interface normal to another axis
    /// within it.
    fn room(&self, a: usize) -> f64 {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        self.cap.min(self.delta[b].0).min(self.delta[c].0)
    }

    /// The largest room over the axes, an upper bound of every axis cube.
    fn max_room(&self) -> f64 {
        (0..3).map(|a| self.room(a)).fold(0.0, f64::max)
    }
}

/// Number of tabulated edge-cube offsets.
const EDGE_OFFSETS: usize = 3;

/// The tabulated offsets of edge cubes (design M2d §3.5): `K/8`, `K/4` and
/// `K/2` (rounded up; `K ≥ 8`), i.e. cubes about 8, 4 and 2 times the
/// distance to the edge.
fn edge_offsets(offsets: usize) -> [usize; EDGE_OFFSETS] {
    [
        offsets.div_ceil(8),
        offsets.div_ceil(4),
        offsets.div_ceil(2),
    ]
}

fn landing_ratio(r: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    (r as f64 / 2.0).exp2()
}

#[derive(Debug, Clone, Copy)]
struct Plan {
    axis: usize,
    half: f64,
    kind: Kind,
}

impl BoxSteps {
    /// Wraps `medium` for walks over `stack` (the effective stack) with `K`
    /// offsets and `M` cells per face edge.
    ///
    /// # Errors
    ///
    /// A message if two permittivities of the medium differ by more than
    /// [`KAPPA_RANGE`] allows.
    /// `alpha` enables edge cubes and landing boxes below `alpha · h`
    /// (design M2d §3.5).
    pub fn new(
        medium: BoxMedium,
        stack: &Stack,
        offsets: usize,
        resolution: usize,
        alpha: Option<f64>,
    ) -> Result<Self, String> {
        let mut eps: Vec<f64> = stack.eps().to_vec();
        eps.extend(medium.boxes().iter().map(|&(_, e)| e));
        eps.sort_by(f64::total_cmp);
        eps.dedup_by(|a, b| a.to_bits() == b.to_bits());
        let (lo, hi) = (eps[0], eps[eps.len() - 1]);
        if !KAPPA_RANGE.contains(&(hi / lo)) {
            return Err(format!(
                "dielectric contrast {} between boxes and layers is outside {KAPPA_RANGE:?}",
                hi / lo
            ));
        }
        let n = eps.len();
        Ok(Self {
            medium,
            stack: stack.clone(),
            offsets,
            resolution,
            cubes: (0..n * n * offsets).map(|_| OnceLock::new()).collect(),
            grads: (0..n * n * offsets).map(|_| OnceLock::new()).collect(),
            iface: (0..n * n).map(|_| OnceLock::new()).collect(),
            normals: Arc::default(),
            windows: Arc::default(),
            id: next_id(),
            eps,
            alpha,
            landing: (0..LANDING_RATIOS * LANDING_TAUS.len())
                .map(|_| OnceLock::new())
                .collect(),
            edges: (0..if alpha.is_some() { n * n } else { 0 })
                .map(|_| OnceLock::new())
                .collect(),
        })
    }

    /// The same tables (shared, not rebuilt) over another medium of the same
    /// stack, e.g. an edited medium of a sensitivity (design D3 §4.4c).
    ///
    /// # Errors
    ///
    /// A message if `medium` has a permittivity that `self` has no tables for.
    pub fn with_medium(&self, medium: BoxMedium) -> Result<Self, String> {
        if let Some(&(_, e)) = medium
            .boxes()
            .iter()
            .find(|&&(_, e)| !self.eps.iter().any(|v| v.to_bits() == e.to_bits()))
        {
            return Err(format!(
                "no tables for the permittivity {e} of an edited medium"
            ));
        }
        Ok(Self {
            medium,
            stack: self.stack.clone(),
            offsets: self.offsets,
            resolution: self.resolution,
            eps: self.eps.clone(),
            cubes: Arc::clone(&self.cubes),
            grads: Arc::clone(&self.grads),
            alpha: self.alpha,
            landing: Arc::clone(&self.landing),
            edges: Arc::clone(&self.edges),
            iface: Arc::clone(&self.iface),
            normals: Arc::clone(&self.normals),
            windows: Arc::default(),
            id: next_id(),
        })
    }

    /// The material boxes and their interfaces.
    #[must_use]
    pub const fn medium(&self) -> &BoxMedium {
        &self.medium
    }

    fn eps_index(&self, e: f64) -> usize {
        self.eps
            .iter()
            .position(|v| v.to_bits() == e.to_bits())
            .expect("every permittivity of the medium is listed")
    }

    fn cube(&self, c: usize, o: usize, k: usize) -> &LayeredCube {
        let n = self.eps.len();
        self.cubes[(c * n + o) * self.offsets + k].get_or_init(|| {
            LayeredCube::new(self.eps[o] / self.eps[c], k, self.offsets, self.resolution)
        })
    }

    fn gradients(&self, c: usize, o: usize, k: usize) -> &LayeredGradients {
        let n = self.eps.len();
        self.grads[(c * n + o) * self.offsets + k].get_or_init(|| {
            LayeredGradients::new(self.eps[o] / self.eps[c], k, self.offsets, self.resolution)
        })
    }

    /// Design D3 §4.4c: field hops at `p` on an interface plane normal to
    /// `normal`, through the axis cube centred on `p` (conductor clearance
    /// `h`). Per world axis, a hop whose estimate (as for first hops:
    /// `factor · φ(start₀)`, or `factor · (φ(start₀) − φ(start₁)) / 2` for a
    /// pair) is the tangential derivative of φ, or along `normal` the flux
    /// `ε_r ∂φ/∂n` (relative ε, towards increasing coordinate). Also returns
    /// the relative ε below and above the plane. `None` if `p` is not on an
    /// interface normal to `normal` with room for a cube.
    #[cfg(feature = "sensitivity")]
    pub(crate) fn interface_hops(
        &self,
        p: [f64; 3],
        normal: Axis,
        h: f64,
        rng: &mut Rng,
    ) -> Option<([Hop; 3], [f64; 2])> {
        let cap = h.min(self.medium.reach());
        let delta = self.axis_deltas(p, h).delta;
        let a = normal.index();
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let r = cap.min(delta[b].0).min(delta[c].0);
        if r <= 0.0 {
            return None;
        }
        let plan = self.axis_plan(p, normal, r, &delta, None);
        let Kind::Layered(ci, oi, 0, false) = plan.kind else {
            return None;
        };
        let s = plan.half;
        let n = self.eps.len();
        let (lo, hi) = (self.eps[ci], self.eps[oi]);
        let grads = self.iface[ci * n + oi]
            .get_or_init(|| LayeredGradients::on_interface(hi / lo, self.offsets, self.resolution));
        let world = |u: [f64; 3]| {
            let mut w = [0.0; 3];
            w[a] = u[2];
            w[b] = u[0];
            w[c] = u[1];
            [
                s.mul_add(w[0], p[0]),
                s.mul_add(w[1], p[1]),
                s.mul_add(w[2], p[2]),
            ]
        };
        let hops = std::array::from_fn(|axis| {
            if axis == a {
                let [pos, neg] = grads.z.sample_pair(rng);
                return Hop {
                    starts: [world(pos), world(neg)],
                    n: 2,
                    factor: lo * grads.z.l1() / (2.0 * s),
                };
            }
            // Canonical x maps onto b, canonical y onto c.
            let (mut u, sign) = grads.x.sample(rng);
            if axis == c {
                u.swap(0, 1);
            }
            let mut back = u;
            if axis == b {
                back[0] = -back[0];
            } else {
                back[1] = -back[1];
            }
            Hop {
                starts: [world(u), world(back)],
                n: 2,
                factor: sign * grads.x.l1() / (2.0 * s),
            }
        });
        Some((hops, [lo, hi]))
    }

    /// Design M2c §5: the first hop from the Gaussian point `p` (face normal
    /// `axis`, clearance `h`, ε_r at `p`) through the largest axis cube, if
    /// that is larger than `than`. The gradient along `axis` uses the rotated
    /// M2a normal kernel if the cube's interface is normal to `axis`, and the
    /// rotated tangential kernel otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn axis_first_hop(
        &self,
        green: &GreenTables,
        p: [f64; 3],
        axis: Axis,
        h: f64,
        eps_r: f64,
        than: f64,
        antithetic: bool,
        rng: &mut Rng,
    ) -> Option<Hop> {
        let plan = self.best_axis_cube(p, h);
        let best = plan.map_or(0.0, |q| q.half).max(than);
        if let Some(hop) = self.layered_first_hop(p, axis, h, eps_r, best, antithetic, rng) {
            return Some(hop);
        }
        let plan = plan.filter(|q| q.half > than)?;
        let s = plan.half;
        let (ci, oi, k, below) = match plan.kind {
            Kind::Homogeneous(_) => {
                return Some(homogeneous_hop(green, p, axis, s, eps_r, antithetic, rng));
            }
            // A centre on the interface has no gradient kernel.
            Kind::Layered(_, _, 0, _) => return None,
            Kind::Layered(ci, oi, k, below) => (ci, oi, k, below),
        };
        let a = plan.axis;
        let grads = self.gradients(ci, oi, k);
        // Canonical frame → world: z becomes axis a (mirrored if the interface
        // lies below), x and y follow cyclically.
        let world = |u: [f64; 3]| {
            let mut w = [0.0; 3];
            w[a] = if below { -u[2] } else { u[2] };
            w[(a + 1) % 3] = u[0];
            w[(a + 2) % 3] = u[1];
            [
                s.mul_add(w[0], p[0]),
                s.mul_add(w[1], p[1]),
                s.mul_add(w[2], p[2]),
            ]
        };
        let l1 = |kernel: &crate::layered::GradientKernel| eps_r * kernel.l1() / (2.0 * s);
        if axis.index() == a {
            // The normal kernel is ∂/∂z towards the interface: a mirrored cube
            // flips its sign (and swaps the parts of a pair).
            if antithetic {
                let [pos, neg] = grads.z.sample_pair(rng);
                let (pos, neg) = if below { (neg, pos) } else { (pos, neg) };
                return Some(Hop {
                    starts: [world(pos), world(neg)],
                    n: 2,
                    factor: l1(&grads.z),
                });
            }
            let (u, sign) = grads.z.sample(rng);
            let sign = if below { -sign } else { sign };
            return Some(Hop {
                starts: [world(u), p],
                n: 1,
                factor: sign * l1(&grads.z),
            });
        }
        // Tangential: the canonical x kernel, along canonical y by swapping.
        let (mut u, sign) = grads.x.sample(rng);
        if axis.index() == (a + 2) % 3 {
            u.swap(0, 1);
        }
        let mut back = u;
        if axis.index() == (a + 1) % 3 {
            back[0] = -back[0];
        } else {
            back[1] = -back[1];
        }
        Some(Hop {
            starts: [world(u), world(back)],
            n: if antithetic { 2 } else { 1 },
            factor: sign * l1(&grads.x),
        })
    }

    /// Design M2e §3.2: the first hop from the Gaussian point `p` (face normal
    /// `axis`, clearance `h`, ε_r at `p`) through the largest cube centred on
    /// `p` that only interfaces normal to `axis` cross, with the normal kernel
    /// of its layered profile, if that cube is larger than `than` and has an
    /// interface.
    #[allow(clippy::too_many_arguments)]
    fn layered_first_hop(
        &self,
        p: [f64; 3],
        axis: Axis,
        h: f64,
        eps_r: f64,
        than: f64,
        antithetic: bool,
        rng: &mut Rng,
    ) -> Option<Hop> {
        let a = axis.index();
        // Room: faces of the other two normals (and stack planes unless they
        // are normal to `axis`) stay off the open cube.
        let mut r = h.min(self.medium.reach());
        for b in Axis::ALL {
            if b != axis {
                if let Some(n) = self.medium.faces(b).nearest_linf(p, r) {
                    r = n.distance;
                }
            }
        }
        if axis != Axis::Z {
            if let Some((_, d1, _)) = self.stack.nearest(p[2]) {
                r = r.min(d1);
            }
        }
        // A ladder of half-sides (ratio 2^{1/8}) keeps the number of distinct
        // profiles small.
        let s = 2f64.powf((8.0 * r.log2()).floor() / 8.0).min(r);
        if s.is_nan() || s <= than {
            return None;
        }
        // The interface planes normal to `axis` inside the cube. A face with
        // L∞ distance below `s` spans the cube's cross-section: it could only
        // end at a face of another normal, and none is within `s`.
        let faces = self.medium.faces(axis);
        let mut planes: Vec<f64> = Vec::new();
        loop {
            let next = faces.nearest_linf_where(p, s, |id| {
                let c = faces.rects()[id as usize].coord();
                !planes.iter().any(|v| v.to_bits() == c.to_bits())
            });
            match next {
                Some(n) if n.distance < s => planes.push(faces.rects()[n.box_id.index()].coord()),
                _ => break,
            }
            if planes.len() > MAX_NORMAL_PLANES {
                return None;
            }
        }
        if axis == Axis::Z {
            let more: Vec<f64> = self
                .stack
                .interfaces()
                .iter()
                .copied()
                .filter(|&zi| (zi - p[2]).abs() < s && !planes.contains(&zi))
                .collect();
            planes.extend(more);
        }
        if planes.is_empty() || planes.len() > MAX_NORMAL_PLANES {
            return None;
        }
        planes.sort_by(f64::total_cmp);
        // Profile in [0, 1] along `axis`; the centre must lie inside a layer.
        let lo = p[a] - s;
        let z: Vec<f64> = planes.iter().map(|&c| (c - lo) / (2.0 * s)).collect();
        if z.iter()
            .any(|&v| !(v > 0.0 && v < 1.0) || (v - 0.5).abs() <= 1e-9)
            || z.windows(2).any(|w| w[1] - w[0] <= 1e-9)
        {
            return None;
        }
        let mut bounds = vec![0.0];
        bounds.extend_from_slice(&z);
        bounds.push(1.0);
        let eps: Vec<f64> = bounds
            .windows(2)
            .map(|w| {
                let mut q = p;
                q[a] = (w[0] + w[1]).mul_add(s, lo);
                self.eps_at(q) / eps_r
            })
            .collect();
        let key: Vec<u64> = z.iter().chain(&eps).map(|v| v.to_bits()).collect();
        let kernel = {
            let found = self.normals.read().ok()?.get(&key).cloned();
            match found {
                Some(k) => k,
                None => {
                    let k = Arc::new(NormalGradient::new(&Profile::new(z, eps), self.resolution));
                    let mut map = self.normals.write().ok()?;
                    if map.len() < MAX_NORMAL_KERNELS {
                        Arc::clone(map.entry(key).or_insert(k))
                    } else {
                        k
                    }
                }
            }
        };
        let world = |u: [f64; 3]| {
            let mut w = [0.0; 3];
            w[a] = u[2];
            w[(a + 1) % 3] = u[0];
            w[(a + 2) % 3] = u[1];
            [
                s.mul_add(w[0], p[0]),
                s.mul_add(w[1], p[1]),
                s.mul_add(w[2], p[2]),
            ]
        };
        let factor = eps_r * kernel.l1() / (2.0 * s);
        if antithetic {
            let [pos, neg] = kernel.sample_pair(rng);
            return Some(Hop {
                starts: [world(pos), world(neg)],
                n: 2,
                factor,
            });
        }
        let (u, sign) = kernel.sample(rng);
        Some(Hop {
            starts: [world(u), p],
            n: 1,
            factor: sign * factor,
        })
    }

    /// The nearest box of index `which` (faces normal to axis `which`, or
    /// the material boxes for 3) within `max_r` of `p` with `keep(id)`, from
    /// this thread's local list around a recent walker, rebuilt when it does
    /// not cover the query. Same result as the grid query.
    fn local_nearest(
        &self,
        which: usize,
        p: [f64; 3],
        max_r: f64,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Nearest> {
        if !max_r.is_finite() {
            return if which == 3 {
                // Only reached with an unbounded clearance.
                self.medium.volume_distance(p, max_r).map(|d| Nearest {
                    distance: d,
                    box_id: faradice_core::BoxId::new(0),
                })
            } else {
                self.medium
                    .faces(Axis::ALL[which])
                    .nearest_linf_where(p, max_r, keep)
            };
        }
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            if l.owner != self.id {
                l.owner = self.id;
                l.sets = Default::default();
            }
            let set = &mut l.sets[which];
            if !set.covers(p, max_r) {
                let radius = LOCAL_REACH * max_r;
                if which == 3 {
                    self.medium.volume_local(p, radius, set);
                } else {
                    self.medium.faces(Axis::ALL[which]).local(p, radius, set);
                }
            }
            if set.covers(p, max_r) {
                set.nearest_linf_where(p, max_r, keep)
            } else if which == 3 {
                self.medium.volume_distance(p, max_r).map(|d| Nearest {
                    distance: d,
                    box_id: faradice_core::BoxId::new(0),
                })
            } else {
                self.medium
                    .faces(Axis::ALL[which])
                    .nearest_linf_where(p, max_r, keep)
            }
        })
    }

    /// The nearest face normal to axis `a` within `max_r` of `p`, and the
    /// nearest on another plane (see [`Deltas::second`]), in one pass over
    /// this thread's local list (as [`Self::local_nearest`]).
    fn local_nearest_two(
        &self,
        a: usize,
        p: [f64; 3],
        max_r: f64,
    ) -> (Option<Nearest>, Option<Nearest>) {
        let faces = self.medium.faces(Axis::ALL[a]);
        let key = |id: u32| faces.rects()[id as usize].coord().to_bits();
        let direct = || {
            let first = faces.nearest_linf(p, max_r);
            let second = first.and_then(|f| {
                let k = key(f.box_id.index() as u32);
                faces.nearest_linf_where(p, max_r, |id| key(id) != k)
            });
            (first, second)
        };
        if !max_r.is_finite() {
            return direct();
        }
        LOCAL.with(|l| {
            let mut l = l.borrow_mut();
            if l.owner != self.id {
                l.owner = self.id;
                l.sets = Default::default();
            }
            let set = &mut l.sets[a];
            if !set.covers(p, max_r) {
                faces.local(p, LOCAL_REACH * max_r, set);
            }
            if set.covers(p, max_r) {
                set.nearest_two_normal(p, max_r, a, key)
            } else {
                direct()
            }
        })
    }

    /// L∞ distance from `p` to the nearest material box within `max_r`
    /// (`f64::INFINITY` beyond it; `0` inside a box).
    #[inline]
    pub(crate) fn box_distance(&self, p: [f64; 3], max_r: f64) -> f64 {
        self.local_nearest(3, p, max_r, |_| true)
            .map_or(f64::INFINITY, |n| n.distance)
    }

    /// L∞ distance from `p` to the nearest interface face of any orientation
    /// within `max_r` (`f64::INFINITY` beyond it): the largest homogeneous
    /// cube at `p` as far as the boxes are concerned.
    pub(crate) fn interface_distance(&self, p: [f64; 3], max_r: f64) -> f64 {
        Axis::ALL
            .iter()
            .filter_map(|&a| self.medium.faces(a).nearest_linf(p, max_r))
            .map(|n| n.distance)
            .fold(f64::INFINITY, f64::min)
    }

    /// Window decisions and kernels (design M2f).
    pub(crate) fn windows(&self) -> &crate::windows::WindowCache {
        &self.windows
    }

    /// The material boxes and their faces.
    pub(crate) const fn box_medium(&self) -> &BoxMedium {
        &self.medium
    }

    /// The stack outside the boxes.
    pub(crate) const fn stack_ref(&self) -> &Stack {
        &self.stack
    }

    /// See [`BoxMedium::inner_clearance`].
    pub(crate) fn inner_clearance(&self, p: [f64; 3]) -> f64 {
        self.medium.inner_clearance(p)
    }

    /// Relative permittivity at `p`.
    #[inline]
    pub(crate) fn eps_at(&self, p: [f64; 3]) -> f64 {
        self.medium.eps_at(p)
    }

    /// `true` if a walk from `x` (clearance `h`) can take a step: `x` is not
    /// a dielectric vertex (design M2c §4.2, §4.6).
    #[cfg_attr(not(feature = "sensitivity"), allow(dead_code))]
    pub(crate) fn makes_progress(&self, x: [f64; 3], h: f64) -> bool {
        self.box_distance(x, h) > 0.0
            || self
                .best_axis_cube(x, h)
                .is_some_and(|p| p.half >= MIN_PROGRESS * h)
            || self.edge_ball(x, h).is_some()
    }

    /// One transition step from `x` with geometric clearance `h`; returns the
    /// max-norm distance of the new point, or `None` if no cube makes
    /// progress (a point on a dielectric edge).
    pub(crate) fn step(
        &self,
        stack: Option<&Medium>,
        green: &GreenTables,
        x: &mut [f64; 3],
        h: f64,
        rng: &mut Rng,
    ) -> Option<f64> {
        // A walk within `EDGE_SNAP · h` of an interface plane moves onto it
        // (design M2c §4.6): rounding leaves walks a few ulps off planes, and
        // near a plane the offset cube shrinks with the distance (K·d₁) while
        // the cube centred on the plane does not. φ is Lipschitz near a
        // plane, so the move changes it by O(EDGE_SNAP) relative. The
        // clearance shrinks by the move, so every domain stays within the
        // conductor-free cube of the caller.
        // Faces lie on box boundaries and an image is never nearer than its
        // original (see `dielectric.rs`), so no face is within the tolerance
        // if no box is.
        let mut h = h;
        let mut v = self.box_distance(*x, h);
        let mut moved = false;
        let near = v <= EDGE_SNAP * h;
        for axis in Axis::ALL.into_iter().filter(|_| near) {
            let a = axis.index();
            let tol = EDGE_SNAP * h;
            if let Some(n) = self.medium.faces(axis).nearest_linf(*x, tol) {
                let c = self.medium.faces(axis).rects()[n.box_id.index()].coord();
                let d = (x[a] - c).abs();
                // Never across a nearer stack interface.
                let clear =
                    axis != Axis::Z || self.stack.nearest(x[2]).is_none_or(|(_, d1, _)| d1 >= d);
                if d <= tol && clear {
                    x[a] = c;
                    h -= d;
                    moved = true;
                }
            }
        }
        if moved {
            v = self.box_distance(*x, h);
        }
        let stack_step = |x: &mut [f64; 3], h: f64, rng: &mut Rng| match stack {
            Some(m) => m.step(green, x, h, rng),
            None => {
                let u = green.sample_transition(rng);
                *x = [x[0] + h * u[0], x[1] + h * u[1], x[2] + h * u[2]];
                h
            }
        };
        if v >= h {
            return Some(stack_step(x, h, rng));
        }
        let deltas = self.axis_deltas(*x, h);
        // Every axis cube is at most its room (see `best_from`) up to a few
        // ulps of rounding: if the box-free cube is no smaller, the plans need
        // not be made.
        if v > 0.0 && v >= deltas.max_room() * (1.0 + 1e-12) {
            return Some(stack_step(x, v, rng));
        }
        let best = self.best_from(*x, &deltas);
        let best_half = best.map_or(0.0, |p| p.half).max(v);
        if let Some(alpha) = self.alpha {
            if best_half < alpha * h {
                if let Some(r) = self.edge_or_landing(x, &deltas, best_half, rng) {
                    return Some(r);
                }
            }
        }
        if v > 0.0 && v >= best.map_or(0.0, |p| p.half) {
            // The stack step within the box-free cube is no smaller.
            return Some(stack_step(x, v, rng));
        }
        match best.filter(|p| p.half >= MIN_PROGRESS * h) {
            Some(plan) => {
                self.execute(plan, green, x, rng);
                Some(plan.half)
            }
            None => self.edge_step(x, h, rng),
        }
    }

    /// Design M2d §3.1: the nearest interface plane of each normal (within
    /// the cap of `d`), the two nearest of them forming the edge frame.
    fn edge_frame(&self, x: [f64; 3], d: &Deltas) -> Option<EdgeFrame> {
        let mut planes = [(f64::INFINITY, 0.0); 3];
        for (a, n) in d.nearest.iter().enumerate() {
            if let Some(n) = n {
                let faces = self.medium.faces(Axis::ALL[a]);
                planes[a] = (n.distance, faces.rects()[n.box_id.index()].coord());
            }
        }
        if let Some((i, d1, _)) = self.stack.nearest(x[2]) {
            if d1 <= d.cap && d1 < planes[2].0 {
                planes[2] = (d1, self.stack.interfaces()[i]);
            }
        }
        let mut axes = [0, 1, 2];
        axes.sort_by(|&p, &q| planes[p].0.total_cmp(&planes[q].0).then(p.cmp(&q)));
        let [b, c, a] = axes;
        if !planes[c].0.is_finite() {
            return None;
        }
        Some(EdgeFrame {
            a,
            b,
            c,
            cb: planes[b].1,
            cc: planes[c].1,
        })
    }

    /// The room of a domain at `x` in the edge frame `f` (design M2d §3.1):
    /// the L∞ distance to the nearest interface face off the two planes and
    /// to other stack planes, at most `cap ≤ d.cap` (a smaller cap shortens
    /// the searches).
    fn edge_room(&self, x: [f64; 3], f: &EdgeFrame, d: &Deltas, cap: f64) -> f64 {
        let off = |axis: usize, coord: Option<f64>, r: f64| {
            d.face_off(self.medium.faces(Axis::ALL[axis]), axis, r, coord)
                .map_or(r, |n| n.distance)
        };
        // Faces normal to `a`: the nearest one is known.
        let mut r = d.nearest[f.a].map_or(cap, |n| n.distance.min(cap));
        r = off(f.b, Some(f.cb), r);
        r = off(f.c, Some(f.cc), r);
        for zi in self.stack_planes(x[2], r) {
            let on = (f.b == 2 && zi.to_bits() == f.cb.to_bits())
                || (f.c == 2 && zi.to_bits() == f.cc.to_bits());
            if !on {
                r = r.min((x[2] - zi).abs());
            }
        }
        r
    }

    /// Design M2d §3.2, §3.3, §3.5: an edge cube from a point on one edge
    /// plane, or a landing box onto the nearer plane; `None` if neither is
    /// larger than `best` (the walk then takes its M2c step).
    fn edge_or_landing(
        &self,
        x: &mut [f64; 3],
        d: &Deltas,
        best: f64,
        rng: &mut Rng,
    ) -> Option<f64> {
        let f = self.edge_frame(*x, d)?;
        let (db, dc) = ((x[f.b] - f.cb).abs(), (x[f.c] - f.cc).abs());
        if db == 0.0 && dc == 0.0 {
            // On the edge line: the edge ball (M2c §4.6).
            return None;
        }
        // P: the plane the walker is on, or the nearer one; Q: the other.
        let (pn, qn, cp, cq, dp, dq) = if dc <= db {
            (f.c, f.b, f.cc, f.cb, dc, db)
        } else {
            (f.b, f.c, f.cb, f.cc, db, dc)
        };
        if dp == 0.0 {
            // Edge cubes have half-sides dq·K/k for the tabulated k: none fits
            // or beats `best` if the extremes do not.
            #[allow(clippy::cast_precision_loss)]
            let kf = self.offsets as f64;
            let ks = edge_offsets(self.offsets);
            #[allow(clippy::cast_precision_loss)]
            let (r_max, r_min) = (
                dq * kf / ks[0] as f64,
                dq * kf / ks[EDGE_OFFSETS - 1] as f64,
            );
            if r_max <= best || r_min > d.cap {
                return None;
            }
        }
        // No domain below reaches beyond `dq·K/k₀` (the largest edge cube;
        // landing boxes reach at most `max(dp/τ₀, dq) ≤ 8·dq`).
        #[allow(clippy::cast_precision_loss)]
        let reach = dq * (self.offsets as f64 / edge_offsets(self.offsets)[0] as f64).max(8.0);
        let room = self.edge_room(*x, &f, d, d.cap.min(reach));
        if dp == 0.0 {
            self.edge_cube_step(x, f.a, pn, qn, cq, dq, room, best, rng)
        } else {
            self.landing_step(x, f.a, pn, qn, cp, dp, dq, room, rng)
        }
    }

    /// An edge cube (design M2d §3.3) at a walker on the plane normal to
    /// `pn`, with the plane `x_qn = cq` at distance `dq`.
    #[allow(clippy::too_many_arguments)]
    fn edge_cube_step(
        &self,
        x: &mut [f64; 3],
        a: usize,
        pn: usize,
        qn: usize,
        cq: f64,
        dq: f64,
        room: f64,
        best: f64,
        rng: &mut Rng,
    ) -> Option<f64> {
        #[allow(clippy::cast_precision_loss)]
        let kf = self.offsets as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k_min = ((dq / room * kf).ceil() as usize).max(1);
        // Canonical y points from the walker towards Q.
        let sy = (cq - x[qn]).signum();
        // Quadrant permittivities (canonical `[y side][z side]`), read inside
        // every candidate cube (`r ≥ 2·dq`): ε is constant per quadrant there.
        let eps = |y: f64, z: f64| {
            let mut p = *x;
            p[qn] += sy * y * dq;
            p[pn] += z * dq;
            self.eps_index(self.eps_at(p))
        };
        let quads = [
            [eps(-0.5, -0.5), eps(-0.5, 0.5)],
            [eps(1.5, -0.5), eps(1.5, 0.5)],
        ];
        // A mirror in z shares the table: use the smaller key.
        let mirrored = quads.map(|[lo, hi]| [hi, lo]);
        let (quads, sz) = if mirrored < quads {
            (mirrored, -1.0)
        } else {
            (quads, 1.0)
        };
        let n = self.eps.len();
        let chunk = self.edges[quads[0][0] * n + quads[0][1]]
            .get_or_init(|| (0..n * n * EDGE_OFFSETS).map(|_| OnceLock::new()).collect());
        let base = quads[1][0] * n + quads[1][1];
        for (ki, k) in edge_offsets(self.offsets).into_iter().enumerate() {
            if k < k_min {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let r = dq * kf / k as f64;
            if r <= best {
                return None;
            }
            let slot = &chunk[base * EDGE_OFFSETS + ki];
            let cube = slot.get_or_init(|| {
                let e = quads.map(|row| row.map(|i| self.eps[i]));
                EdgeCube::new(e, k, self.offsets, self.resolution)
                    .ok()
                    .map(Box::new)
            });
            if let Some(cube) = cube {
                let u = cube.sample_transition(rng);
                x[a] = r.mul_add(u[0], x[a]);
                x[qn] = (sy * r).mul_add(u[1], x[qn]);
                x[pn] = (sz * r).mul_add(u[2], x[pn]);
                return Some(r);
            }
        }
        None
    }

    /// A landing box (M2b §3.3, rotated; design M2d §3.2) from a walker at
    /// distance `dp` of the plane `x_pn = cp`, with the other edge plane at
    /// distance `dq ≥ dp`.
    #[allow(clippy::too_many_arguments)]
    fn landing_step(
        &self,
        x: &mut [f64; 3],
        a: usize,
        pn: usize,
        qn: usize,
        cp: f64,
        dp: f64,
        dq: f64,
        room: f64,
        rng: &mut Rng,
    ) -> Option<f64> {
        for (ti, &tau) in LANDING_TAUS.iter().enumerate() {
            let height = dp / tau;
            if height - dp > room {
                continue;
            }
            // Lateral half-width L·D/2 within the room and short of Q.
            let width = room.min(dq);
            let Some(ri) = (0..LANDING_RATIOS)
                .rev()
                .find(|&ri| landing_ratio(ri) * height / 2.0 <= width)
            else {
                continue;
            };
            let ratio = landing_ratio(ri);
            let b = self.landing[ti * LANDING_RATIOS + ri]
                .get_or_init(|| LandingBox::new(ratio, tau, self.resolution));
            let u = b.sample_transition(rng);
            let s = height / 2.0;
            let dir = (x[pn] - cp).signum();
            // An exit through the bottom face lands exactly on the plane.
            #[allow(clippy::float_cmp)]
            let landed = u[2] == -1.0;
            x[pn] = if landed {
                cp
            } else {
                (dir * s).mul_add(u[2] + 1.0, cp)
            };
            x[qn] = s.mul_add(u[0], x[qn]);
            x[a] = s.mul_add(u[1], x[a]);
            return Some((ratio * s).max(height - dp).max(dp));
        }
        None
    }

    /// Design M2c §4.6: a walk at a dielectric edge (two interface planes,
    /// normal to axes b and c, within `EDGE_SNAP · h`) moves onto the edge
    /// line and takes a ball around it. In a ball centred on the line in
    /// which ε depends on the azimuth θ around it only, the ε-weighted
    /// azimuthal average of φ is harmonic (the θ-flux term integrates to zero
    /// over a period), so φ at the centre is the homogeneous ball average with
    /// θ drawn ∝ ε(θ): a uniform point on the sphere whose quadrant is drawn
    /// ∝ its ε. `None` at a vertex or if the ball makes no progress.
    fn edge_step(&self, x: &mut [f64; 3], h: f64, rng: &mut Rng) -> Option<f64> {
        let EdgeBall {
            a,
            b,
            c,
            centre,
            moved,
            r,
        } = self.edge_ball(*x, h)?;
        let mut y = centre;
        // Quadrant permittivities, at their centres in the ball.
        let q = 0.5 * r;
        let eps = |sb: f64, sc: f64| {
            let mut p = y;
            p[b] += sb * q;
            p[c] += sc * q;
            self.eps_at(p)
        };
        // Quadrants in θ order (θ from +b towards +c).
        let quads = [
            eps(1.0, 1.0),
            eps(-1.0, 1.0),
            eps(-1.0, -1.0),
            eps(1.0, -1.0),
        ];
        let total: f64 = quads.iter().sum();
        let mut pick = rng.next_f64() * total;
        let mut k = 3;
        for (i, &e) in quads.iter().enumerate() {
            if pick < e {
                k = i;
                break;
            }
            pick -= e;
        }
        #[allow(clippy::cast_precision_loss)]
        let theta = (k as f64 + rng.next_f64()) * std::f64::consts::FRAC_PI_2;
        let ua = 2.0f64.mul_add(rng.next_f64(), -1.0);
        let rho = (1.0 - ua * ua).max(0.0).sqrt();
        y[a] = r.mul_add(ua, y[a]);
        y[b] = (r * rho).mul_add(theta.cos(), y[b]);
        y[c] = (r * rho).mul_add(theta.sin(), y[c]);
        *x = y;
        Some(moved + r)
    }

    /// The edge ball of [`Self::edge_step`] at `x`, if there is one.
    fn edge_ball(&self, x: [f64; 3], h: f64) -> Option<EdgeBall> {
        let x = &x;
        let tol = EDGE_SNAP * h;
        // The nearest plane per axis within the tolerance.
        let mut plane: [Option<f64>; 3] = [None; 3];
        for axis in Axis::ALL {
            let a = axis.index();
            let faces = self.medium.faces(axis);
            if let Some(n) = faces.nearest_linf(*x, tol) {
                plane[a] = Some(faces.rects()[n.box_id.index()].coord());
            }
        }
        if let Some((i, d1, _)) = self.stack.nearest(x[2]) {
            let zi = self.stack.interfaces()[i];
            if d1 <= tol && plane[2].is_none_or(|c| (x[2] - zi).abs() < (x[2] - c).abs()) {
                plane[2] = Some(zi);
            }
        }
        let axes: Vec<usize> = (0..3).filter(|&a| plane[a].is_some()).collect();
        let [b, c] = axes[..] else { return None };
        let a = 3 - b - c;
        let (cb, cc) = (plane[b]?, plane[c]?);
        let mut y = *x;
        y[b] = cb;
        y[c] = cc;
        let moved = (x[b] - cb).abs().max((x[c] - cc).abs());
        // The ball: conductor-free, within the reflection reach, and clear of
        // every interface not in the planes b = cb, c = cc (inside it, ε is
        // constant per quadrant around the edge: an end of a face in one of
        // the planes needs an interface face off them).
        let mut r = (h - moved).min(self.medium.reach());
        let off = |axis: Axis, coord: Option<f64>, r: f64| {
            let faces = self.medium.faces(axis);
            faces
                .nearest_linf_where(y, r, |id| {
                    coord
                        .is_none_or(|k| faces.rects()[id as usize].coord().to_bits() != k.to_bits())
                })
                .map_or(r, |n| n.distance)
        };
        r = off(Axis::ALL[a], None, r);
        r = off(Axis::ALL[b], Some(cb), r);
        r = off(Axis::ALL[c], Some(cc), r);
        for zi in self.stack_planes(y[2], r) {
            let on = (a != 2)
                && ((b == 2 && zi.to_bits() == cb.to_bits())
                    || (c == 2 && zi.to_bits() == cc.to_bits()));
            if !on {
                r = r.min((y[2] - zi).abs());
            }
        }
        if r < MIN_PROGRESS * h {
            return None;
        }
        Some(EdgeBall {
            a,
            b,
            c,
            centre: y,
            moved,
            r,
        })
    }

    /// The largest axis cube at `x` (clearance `h`).
    fn best_axis_cube(&self, x: [f64; 3], h: f64) -> Option<Plan> {
        self.best_from(x, &self.axis_deltas(x, h))
    }

    /// The nearest interface per axis at `x` (clearance `h`), for
    /// [`Self::best_from`].
    fn axis_deltas(&self, x: [f64; 3], h: f64) -> Deltas {
        let cap = h.min(self.medium.reach());
        // Distance to the nearest interface normal to each axis (within the
        // cap), and whether its normal distance attains it (then a cube of
        // that size has a face on the interface's plane).
        let mut delta = [(cap, None::<f64>); 3];
        // The nearest face per axis within the cap, reused by `axis_plan`.
        let mut nearest: [Option<Nearest>; 3] = [None; 3];
        let mut second: [Option<Nearest>; 3] = [None; 3];
        for axis in Axis::ALL {
            let a = axis.index();
            (nearest[a], second[a]) = self.local_nearest_two(a, x, cap);
            if let Some(n) = nearest[a] {
                let f = &self.medium.faces(axis).rects()[n.box_id.index()];
                let on = (x[a] - f.coord()).abs().to_bits() == n.distance.to_bits();
                delta[a] = (n.distance, on.then(|| f.coord()));
            }
        }
        if let Some((i, d1, _)) = self.stack.nearest(x[2]) {
            if d1 < delta[2].0 {
                delta[2] = (d1, Some(self.stack.interfaces()[i]));
            }
        }
        Deltas {
            cap,
            delta,
            nearest,
            second,
        }
    }

    /// The largest axis cube from the nearest interfaces `d`. The cube along
    /// `a` is at most its room `cap ∧ δ_b ∧ δ_c` ([`Deltas::room`]).
    fn best_from(&self, x: [f64; 3], d: &Deltas) -> Option<Plan> {
        let mut best: Option<Plan> = None;
        for axis in Axis::ALL {
            let a = axis.index();
            let r = d.room(a);
            if r <= 0.0 {
                continue;
            }
            let plan = self.axis_plan(x, axis, r, &d.delta, Some(d));
            if best.is_none_or(|p| plan.half > p.half) {
                best = Some(plan);
            }
        }
        best
    }

    /// The M2a selector along `axis` with room `r` (no interface normal to the
    /// other axes within `r`).
    /// `first`, if known, is the nearest face normal to `axis` within a
    /// radius of at least `r` (`Some(None)`: there is none).
    fn axis_plan(
        &self,
        x: [f64; 3],
        axis: Axis,
        r: f64,
        delta: &[(f64, Option<f64>); 3],
        known: Option<&Deltas>,
    ) -> Plan {
        let a = axis.index();
        let Some(p1) = self.nearest_plane(x, axis, r, None, known) else {
            // Homogeneous within r: the cube touches the nearest interface
            // normal to another axis; snap if its face lies on that plane.
            let snap = (0..3).filter(|&b| b != a).find_map(|b| {
                let (d, coord) = delta[b];
                (d.to_bits() == r.to_bits())
                    .then_some(coord)
                    .flatten()
                    .map(|c| (b, (c - x[b]).signum(), c))
            });
            return Plan {
                axis: a,
                half: r,
                kind: Kind::Homogeneous(snap),
            };
        };
        let d1 = (x[a] - p1.coord).abs();
        let d2 = self
            .nearest_plane(x, axis, r, Some(p1.coord), known)
            .map_or(f64::INFINITY, |p| (x[a] - p.coord).abs());
        let h_max = r.min(d2);
        let below = x[a] > p1.coord; // the interface lies below the centre
        let (ec, eo) = if below {
            (p1.sides[1], p1.sides[0])
        } else {
            (p1.sides[0], p1.sides[1])
        };
        if ec.to_bits() == eo.to_bits() {
            // Not an interface here (an overridden stack plane).
            return Plan {
                axis: a,
                half: h_max,
                kind: Kind::Homogeneous(None),
            };
        }
        let (ci, oi) = (self.eps_index(ec), self.eps_index(eo));
        #[allow(clippy::cast_precision_loss)]
        let k_f = self.offsets as f64;
        if d1 <= 0.0 {
            return Plan {
                axis: a,
                half: h_max,
                kind: Kind::Layered(ci, oi, 0, false),
            };
        }
        let q = d1 / h_max * k_f;
        if q > k_f - 1.0 {
            // The interface touches the cube: its face lies on the plane.
            let sign = if below { -1.0 } else { 1.0 };
            return Plan {
                axis: a,
                half: d1,
                kind: Kind::Homogeneous(Some((a, sign, p1.coord))),
            };
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = (q.ceil() as usize).clamp(1, self.offsets - 1);
        #[allow(clippy::cast_precision_loss)]
        let half = d1 * k_f / k as f64;
        Plan {
            axis: a,
            half,
            kind: Kind::Layered(ci, oi, k, below),
        }
    }

    /// The nearest interface plane normal to `axis` within L∞ distance `r`
    /// of `x`, other than the plane `skip`, with the permittivities on its
    /// sides at `x`'s transverse position.
    /// With `skip = None`, a known `first` (see [`Self::axis_plan`]) replaces
    /// the query: the nearest face within `r` is the same face, because the
    /// order of [`Nearest`] is total.
    fn nearest_plane(
        &self,
        x: [f64; 3],
        axis: Axis,
        r: f64,
        skip: Option<f64>,
        known: Option<&Deltas>,
    ) -> Option<Plane> {
        let a = axis.index();
        let faces = self.medium.faces(axis);
        let keep = |id: u32| {
            skip.is_none_or(|s| faces.rects()[id as usize].coord().to_bits() != s.to_bits())
        };
        let found = match known {
            Some(d) => d.face_off(faces, a, r, skip),
            None => self.local_nearest(a, x, r, keep),
        };
        let mut best = found.filter(|n| n.distance < r).map(|n| {
            let id = n.box_id.index();
            Plane {
                coord: faces.rects()[id].coord(),
                #[allow(clippy::cast_possible_truncation)]
                sides: self.medium.face_sides(axis, id as u32),
            }
        });
        if axis == Axis::Z {
            // Stack planes, as full planes. Where a box covers the plane at
            // x's transverse position, both sides are the box (κ = 1); a box
            // face on the same plane takes precedence above.
            for zi in self.stack_planes(x[2], r) {
                if skip.is_some_and(|s| s.to_bits() == zi.to_bits()) {
                    continue;
                }
                let closer = best.is_none_or(|p| (x[2] - zi).abs() < (x[2] - p.coord).abs());
                if closer {
                    let on = [x[0], x[1], zi];
                    let sides = match self.medium.box_eps_closed(on) {
                        Some(e) => [e, e],
                        None => [self.stack.eps_below(zi), self.stack.eps_at(zi)],
                    };
                    best = Some(Plane { coord: zi, sides });
                }
            }
        }
        let _ = a;
        best
    }

    /// Stack interfaces within distance `r` of height `z`, nearest first
    /// (at most the two nearest on each side).
    fn stack_planes(&self, z: f64, r: f64) -> impl Iterator<Item = f64> {
        let ifaces = self.stack.interfaces();
        let i = ifaces.partition_point(|&zi| zi < z);
        let mut out = [0.0; 4];
        let mut n = 0;
        for j in [i.wrapping_sub(2), i.wrapping_sub(1), i, i + 1] {
            if let Some(&zi) = ifaces.get(j) {
                if (z - zi).abs() < r {
                    out[n] = zi;
                    n += 1;
                }
            }
        }
        // Stable insertion sort by distance (the order of `sort_by`).
        for k in 1..n {
            let mut m = k;
            while m > 0
                && (z - out[m - 1])
                    .abs()
                    .total_cmp(&(z - out[m]).abs())
                    .is_gt()
            {
                out.swap(m - 1, m);
                m -= 1;
            }
        }
        out.into_iter().take(n)
    }

    fn execute(&self, plan: Plan, green: &GreenTables, x: &mut [f64; 3], rng: &mut Rng) {
        let a = plan.axis;
        let s = plan.half;
        match plan.kind {
            Kind::Homogeneous(snap) => {
                let u = green.sample_transition(rng);
                for k in 0..3 {
                    x[k] = s.mul_add(u[k], x[k]);
                }
                if let Some((b, sign, coord)) = snap {
                    #[allow(clippy::float_cmp)] // face coordinates are exactly ±1
                    if u[b] == sign {
                        x[b] = coord;
                    }
                }
            }
            Kind::Layered(ci, oi, k, below) => {
                let u = self.cube(ci, oi, k).sample_transition(rng);
                // Canonical z (the interface normal) becomes axis a; the
                // tangential axes follow cyclically.
                let mut w = [0.0; 3];
                w[a] = if below { -u[2] } else { u[2] };
                w[(a + 1) % 3] = u[0];
                w[(a + 2) % 3] = u[1];
                for k in 0..3 {
                    x[k] = s.mul_add(w[k], x[k]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{Boundary, BoundaryModel, FaceKind};
    use crate::regions::{DielectricBox, DielectricRegions};
    use faradice_core::{Face, Side};
    use faradice_geom::{Aabb, GeometryBuilder};

    /// A slab x ∈ [1, 2] of ε = 4 in ε = 1 between grounded x faces at 0 and
    /// 3, reflecting y and z: the walk's hitting probability of x = 3 from x
    /// is the exact layered linear potential.
    #[test]
    fn walks_reproduce_the_layered_linear_potential_along_x() {
        let bounds = Aabb::new([0.0, -1.0, -1.0], [3.0, 1.0, 1.0]).unwrap();
        let mut faces = [FaceKind::Reflect; 6];
        faces[Face::new(Axis::X, Side::Min).index()] = FaceKind::Ground;
        faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Ground;
        let domain = BoundaryModel::new(&Boundary::Box { bounds, faces }, None, bounds);
        let mut g = GeometryBuilder::new();
        let n = g.net("far").unwrap();
        g.add_box(
            n,
            Aabb::from_array([10.0, 10.0, 10.0, 11.0, 11.0, 11.0]).unwrap(),
        )
        .unwrap();
        let geom = g.build().unwrap();
        let regions = DielectricRegions {
            boxes: vec![DielectricBox {
                region: Aabb::from_array([1.0, -1.0, -1.0, 2.0, 1.0, 1.0]).unwrap(),
                eps_r: 4.0,
            }],
            conformal: Vec::new(),
        };
        let stack = Stack::homogeneous(1.0);
        let medium = BoxMedium::build(&regions, &geom, &stack, &domain).unwrap();
        let steps = BoxSteps::new(medium, &stack, 16, 32, Some(0.7)).unwrap();
        let green = GreenTables::new(16);
        // Exact: potential 1 at x = 3, 0 at x = 0, εφ' constant: resistances
        // 1 (x ∈ [0,1], ε=1), 1/4 (ε=4), 1 (ε=1).
        let exact = |x: f64| {
            let r = |a: f64, b: f64| (b - a).max(0.0);
            let t = r(0.0, x.min(1.0)) + r(1.0, x.clamp(1.0, 2.0)) / 4.0 + r(2.0, x.max(2.0));
            t / 2.25
        };
        let mut rng = Rng::from_seed(7);
        for x0 in [0.5, 1.25, 1.5, 2.5] {
            let n_walks = 40_000;
            let mut hits = 0u32;
            for _ in 0..n_walks {
                let mut x = [x0, 0.3, -0.2];
                for _ in 0..100_000 {
                    let (d, _) = domain.dirichlet_distance(x);
                    if d < 1e-6 {
                        break;
                    }
                    let x_before = x;
                    if steps.step(None, &green, &mut x, d, &mut rng).is_none() {
                        let best = steps.best_axis_cube(x_before, d);
                        let faces: Vec<_> = Axis::ALL
                            .iter()
                            .map(|&a| steps.medium.faces(a).nearest_linf(x_before, d))
                            .collect();
                        panic!(
                            "no progress at {x_before:?} h={d}: {best:?} {faces:?} reach={}",
                            steps.medium.reach()
                        );
                    }
                    domain.fold(&mut x);
                }
                if x[0] > 1.5 {
                    hits += 1;
                }
            }
            let p = f64::from(hits) / f64::from(n_walks);
            let sigma = (exact(x0) * (1.0 - exact(x0)) / f64::from(n_walks)).sqrt();
            assert!(
                (p - exact(x0)).abs() < 4.0 * sigma + 1e-3,
                "x0={x0}: {p} vs {}",
                exact(x0)
            );
        }
    }

    /// Design M2c §4.6: four quadrants around the edge x-axis with
    /// ε = A(sign y)·B(sign z) (A = 1, 4; B = 1, 2), grounded y = ±1,
    /// reflecting x and z. Flux continuity holds for φ = φ(y), the layered
    /// linear potential (φ(0) = 0.8); walks start on, near and off the edge.
    /// Design M2e §3.2: between two facing shells the first hop takes the
    /// layered cube up to the clearance; a face of another normal inside it,
    /// or a centre on a plane, rejects it.
    #[test]
    fn layered_first_hops_are_selected_between_facing_planes() {
        let bounds = Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).unwrap();
        let mut faces = [FaceKind::Reflect; 6];
        faces[Face::new(Axis::X, Side::Min).index()] = FaceKind::Ground;
        faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Ground;
        let domain = BoundaryModel::new(&Boundary::Box { bounds, faces }, None, bounds);
        let mut g = GeometryBuilder::new();
        let n = g.net("far").unwrap();
        g.add_box(
            n,
            Aabb::from_array([0.9, 0.9, 0.9, 0.95, 0.95, 0.95]).unwrap(),
        )
        .unwrap();
        let geom = g.build().unwrap();
        let dbox = |c: [f64; 6], eps_r: f64| DielectricBox {
            region: Aabb::from_array(c).unwrap(),
            eps_r,
        };
        // Shells normal to x at ±0.04 around the origin, and a small box
        // normal to y further up.
        let regions = DielectricRegions {
            boxes: vec![
                dbox([-0.07, -1.0, -1.0, -0.04, 1.0, 1.0], 3.5),
                dbox([0.04, -1.0, -1.0, 0.07, 1.0, 1.0], 3.5),
                dbox([-0.02, 0.5, -1.0, 0.02, 0.6, 1.0], 2.0),
            ],
            conformal: Vec::new(),
        };
        let stack = Stack::homogeneous(4.4);
        let medium = BoxMedium::build(&regions, &geom, &stack, &domain).unwrap();
        let steps = BoxSteps::new(medium, &stack, 16, 32, Some(0.7)).unwrap();
        let mut rng = Rng::from_seed(3);
        let h = 0.07;
        let eps_r = 4.4;
        // M2c cubes are limited to 0.04 here.
        let hop = steps
            .layered_first_hop([0.0, 0.0, 0.0], Axis::X, h, eps_r, 0.04, false, &mut rng)
            .expect("a layered first hop");
        let s = 2f64.powf((8.0 * h.log2()).floor() / 8.0);
        assert_eq!(hop.n, 1);
        let reach = (0..3).map(|i| hop.starts[0][i].abs()).fold(0.0, f64::max);
        assert!((reach - s).abs() < 1e-12, "{reach} vs {s}");
        // ∂φ/∂x for φ = x is 1: E[sign · x(start)] · |factor| over many hops.
        let n_hops = 200_000;
        let mut mean = 0.0;
        for _ in 0..n_hops {
            let hop = steps
                .layered_first_hop([0.0, 0.0, 0.0], Axis::X, h, eps_r, 0.04, false, &mut rng)
                .unwrap();
            mean += hop.factor * layered_phi(hop.starts[0][0]);
        }
        mean /= f64::from(n_hops);
        // φ with ε φ′ constant (ε_r φ′ = ε_r at the centre).
        assert!((mean - eps_r).abs() < 0.02 * eps_r, "{mean}");
        // Not larger than the M2c cube: none.
        assert!(
            steps
                .layered_first_hop([0.0, 0.0, 0.0], Axis::X, h, eps_r, 0.07, false, &mut rng)
                .is_none()
        );
        // A face normal to y within the cube limits the room.
        assert!(
            steps
                .layered_first_hop([0.0, 0.47, 0.0], Axis::X, h, eps_r, 0.04, false, &mut rng)
                .is_none()
        );
        // A centre on a plane has two traces.
        assert!(
            steps
                .layered_first_hop([0.04, 0.0, 0.0], Axis::X, 0.03, eps_r, 0.0, false, &mut rng)
                .is_none()
        );
    }

    /// The layered potential of the test above: `ε φ′ = 4.4` across the
    /// shells at ±0.04 (ε 3.5) and the gap (ε 4.4), `φ(0) = 0`.
    fn layered_phi(x: f64) -> f64 {
        let r = |a: f64| {
            let a_abs = a.abs();
            let v = if a_abs <= 0.04 {
                a_abs
            } else {
                0.04 + (a_abs - 0.04) * 4.4 / 3.5
            };
            v.copysign(a)
        };
        r(x)
    }

    /// Design M2d §3.3, §3.5: a walker on one plane near the edge takes the
    /// largest tabulated edge cube (`r = d·K/2`) in every orientation of the
    /// frame, and one off both planes lands.
    #[test]
    fn edge_cubes_are_taken_near_an_edge() {
        let bounds = Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).unwrap();
        let mut faces = [FaceKind::Reflect; 6];
        faces[Face::new(Axis::Y, Side::Min).index()] = FaceKind::Ground;
        faces[Face::new(Axis::Y, Side::Max).index()] = FaceKind::Ground;
        let domain = BoundaryModel::new(&Boundary::Box { bounds, faces }, None, bounds);
        let mut g = GeometryBuilder::new();
        let n = g.net("far").unwrap();
        g.add_box(
            n,
            Aabb::from_array([10.0, 10.0, 10.0, 11.0, 11.0, 11.0]).unwrap(),
        )
        .unwrap();
        let geom = g.build().unwrap();
        let dbox = |c: [f64; 6], eps_r: f64| DielectricBox {
            region: Aabb::from_array(c).unwrap(),
            eps_r,
        };
        let regions = DielectricRegions {
            boxes: vec![
                dbox([-1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 4.0),
                dbox([-1.0, 0.0, -1.0, 1.0, 1.0, 0.0], 8.0),
                dbox([-1.0, -1.0, -1.0, 1.0, 0.0, 0.0], 2.0),
            ],
            conformal: Vec::new(),
        };
        let stack = Stack::homogeneous(1.0);
        let medium = BoxMedium::build(&regions, &geom, &stack, &domain).unwrap();
        let steps = BoxSteps::new(medium, &stack, 16, 32, Some(0.7)).unwrap();
        let mut rng = Rng::from_seed(9);
        let d = 0.01;
        for x0 in [
            [0.1, 0.0, d],
            [0.1, 0.0, -d],
            [-0.2, d, 0.0],
            [0.3, -d, 0.0],
        ] {
            let mut x = x0;
            let (h, _) = domain.dirichlet_distance(x);
            let deltas = steps.axis_deltas(x, h);
            let r = steps
                .edge_or_landing(&mut x, &deltas, 0.0, &mut rng)
                .expect("an edge cube");
            assert!((r - d * 8.0).abs() < 1e-12, "{x0:?}: r = {r}");
            let moved = (0..3).map(|i| (x[i] - x0[i]).abs()).fold(0.0, f64::max);
            assert!((moved - r).abs() < 1e-12, "{x0:?}: moved {moved}");
        }
        // Off both planes: a landing box onto the nearer one.
        let mut x = [0.0, 0.002, 0.01];
        let (h, _) = domain.dirichlet_distance(x);
        let deltas = steps.axis_deltas(x, h);
        assert!(
            steps
                .edge_or_landing(&mut x, &deltas, 0.0, &mut rng)
                .is_some()
        );
        // A best candidate at least as large as the largest edge cube wins.
        let mut x = [0.1, 0.0, d];
        let deltas = steps.axis_deltas(x, h);
        assert!(
            steps
                .edge_or_landing(&mut x, &deltas, 8.0 * d, &mut rng)
                .is_none()
        );
        assert_eq!(x, [0.1, 0.0, d]);
    }

    #[test]
    fn edge_balls_reproduce_a_quadrant_potential() {
        let bounds = Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).unwrap();
        let mut faces = [FaceKind::Reflect; 6];
        faces[Face::new(Axis::Y, Side::Min).index()] = FaceKind::Ground;
        faces[Face::new(Axis::Y, Side::Max).index()] = FaceKind::Ground;
        let domain = BoundaryModel::new(&Boundary::Box { bounds, faces }, None, bounds);
        let mut g = GeometryBuilder::new();
        let n = g.net("far").unwrap();
        g.add_box(
            n,
            Aabb::from_array([10.0, 10.0, 10.0, 11.0, 11.0, 11.0]).unwrap(),
        )
        .unwrap();
        let geom = g.build().unwrap();
        let dbox = |c: [f64; 6], eps_r: f64| DielectricBox {
            region: Aabb::from_array(c).unwrap(),
            eps_r,
        };
        let regions = DielectricRegions {
            boxes: vec![
                dbox([-1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 4.0),
                dbox([-1.0, 0.0, -1.0, 1.0, 1.0, 0.0], 8.0),
                dbox([-1.0, -1.0, -1.0, 1.0, 0.0, 0.0], 2.0),
            ],
            conformal: Vec::new(),
        };
        let stack = Stack::homogeneous(1.0);
        let medium = BoxMedium::build(&regions, &geom, &stack, &domain).unwrap();
        let steps = BoxSteps::new(medium, &stack, 16, 32, Some(0.7)).unwrap();
        let green = GreenTables::new(16);
        let exact = |y: f64| {
            if y < 0.0 {
                0.8 * (y + 1.0)
            } else {
                0.2f64.mul_add(y, 0.8)
            }
        };
        let mut rng = Rng::from_seed(5);
        // On the edge, a few ulps off it, near it (landing boxes and edge
        // cubes, design M2d) and away from it.
        for x0 in [
            [0.3, 0.0, 0.0],
            [0.0, 1e-12, -1e-12],
            [0.0, 0.01, 0.02],
            [0.4, -0.003, 0.001],
            [-0.2, 0.0, -0.05],
            [0.0, 0.2, -0.3],
            [0.0, -0.5, 0.4],
        ] {
            let n_walks = 40_000;
            let mut hits = 0u32;
            for _ in 0..n_walks {
                let mut x = x0;
                for _ in 0..100_000 {
                    let (d, _) = domain.dirichlet_distance(x);
                    if d < 1e-6 {
                        break;
                    }
                    steps
                        .step(None, &green, &mut x, d, &mut rng)
                        .expect("no dielectric vertex here");
                    domain.fold(&mut x);
                }
                if x[1] > 0.0 {
                    hits += 1;
                }
            }
            let p = f64::from(hits) / f64::from(n_walks);
            let e = exact(x0[1]);
            let sigma = (e * (1.0 - e) / f64::from(n_walks)).sqrt();
            assert!((p - e).abs() < 4.0 * sigma + 1e-3, "x0={x0:?}: {p} vs {e}");
        }
    }
}
