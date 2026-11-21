//! Walk steps and first hops in a stratified medium (design M2a §3.1, §4).
//!
//! [`Medium`] holds the effective stack and lazily built layered tables
//! (per contrast and offset; the content is deterministic, so results do not
//! depend on which thread builds a table first) and chooses the cube for
//! every step:
//!
//! - a homogeneous cube if it does not reach an interface;
//! - on an interface (exactly), a multi-layer cube of the full clearance
//!   (design M2b, [`crate::multilayer::MultiLayerCube`]);
//! - otherwise a cube containing exactly one interface, shrunk so that the
//!   interface lies at a tabulated offset (no interpolation, no bias), or, if
//!   that cube is much smaller than the clearance, a landing box whose face
//!   lies on the nearest interface ([`crate::multilayer::LandingBox`]).

use std::sync::OnceLock;

use faradice_core::Axis;

use crate::gauss::{GaussPoint, GaussSurface};
use crate::green::GreenTables;
use crate::layered::{
    BandSide, GradientKernel, KAPPA_RANGE, LayeredCube, LayeredGradients, half_band_gradient,
};
use crate::multilayer::{BandGradient, LandingBox, MultiLayerCube, Profile};
use crate::rng::Rng;
use crate::solver::StackOptions;
use crate::stack::Stack;
use crate::walk::Walker;

/// Half-band fractions `j / BAND_DEN`, `j = 1 … band_max` (η_max =
/// `band_max / BAND_DEN ≤ ½`).
const BAND_DEN: usize = 32;
/// Largest `band_max`.
const BAND_MAX: usize = 16;

/// Anchored cubes (design M2b §3.1): half-sides `2^{l/4}` for
/// `l ∈ [LADDER_MIN, LADDER_MAX]` (about 1.5e-5 to 4096 length units).
const LADDER_STEPS: f64 = 4.0;
const LADDER_MIN: i64 = -64;
const LADDER_MAX: i64 = 48;
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const LADDER_LEVELS: usize = (LADDER_MAX - LADDER_MIN + 1) as usize;
/// Landing-box aspect ratios `2^{r/2}`, `r < LANDING_RATIOS` (at most 16).
const LANDING_RATIOS: usize = 9;
/// Landing-box start heights (fractions of the box height above the face on
/// the interface): the smallest one that fits is used.
const LANDING_TAUS: [f64; 4] = [0.125, 0.25, 0.375, 0.5];
/// Interfaces this close to an anchored cube face (relative to its side) are
/// taken to lie on the face.
const FACE_TOLERANCE: f64 = 1e-12;

fn ladder(l: i64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    (l as f64 / LADDER_STEPS).exp2()
}

fn landing_ratio(r: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    (r as f64 / 2.0).exp2()
}

/// Distances to an interface below this are rounding: the walk is moved onto it.
fn snap_tolerance(z: f64) -> f64 {
    1e-12 * (1.0 + z.abs())
}

fn add(x: &mut [f64; 3], s: f64, u: [f64; 3]) {
    for k in 0..3 {
        x[k] = s.mul_add(u[k], x[k]);
    }
}

/// The effective stack and its layered tables.
#[derive(Debug)]
pub struct Medium {
    stack: Stack,
    /// Tabulated interface offsets `k / offsets` (`K`).
    offsets: usize,
    /// Cells per face edge of the layered tables (`M`, a multiple of `2K`).
    resolution: usize,
    /// Half-band fractions `j / BAND_DEN`, `j ≤ band_max`.
    band_max: usize,
    /// An M2a cube smaller than this fraction of the clearance is replaced
    /// by a landing box.
    alpha: f64,
    kappas: Vec<f64>,
    /// Per interface: contrast index with the cube centre below / above it.
    centre_below: Vec<usize>,
    centre_above: Vec<usize>,
    cubes: Vec<OnceLock<LayeredCube>>,
    grads: Vec<OnceLock<LayeredGradients>>,
    /// Per interface, fraction and side (`κ = ε_upper / ε_lower`).
    bands: Vec<OnceLock<GradientKernel>>,
    /// Per interface and ladder level: anchored multi-layer cubes (M2b).
    anchored: Vec<OnceLock<MultiLayerCube>>,
    /// Per start height and aspect ratio: landing boxes (M2b).
    landing: Vec<OnceLock<LandingBox>>,
    /// Per interface, ladder level, side and fraction (0: the full band):
    /// half-band kernels of anchored cubes (M2b §3.5), boxed because few are
    /// ever built.
    anchored_bands: Vec<OnceLock<Box<BandGradient>>>,
}

/// One first hop: up to two launches with weights `[½f, −½f]` (pair) or
/// `[f, 0]` (single), where the sample weight is `−A·ε₀·s·f`.
pub(crate) struct Hop {
    pub(crate) starts: [[f64; 3]; 2],
    pub(crate) n: usize,
    pub(crate) factor: f64,
}

/// Where a point lies relative to the nearest interface.
#[derive(Debug, Clone, Copy)]
struct Near {
    iface: usize,
    z: f64,
    d1: f64,
    d2: f64,
    /// The point lies below the interface (on it: `false`, the upper layer).
    below: bool,
}

/// The cube chosen at a point whose geometric clearance reaches an interface.
enum Cube {
    /// Homogeneous cube of this half-side (the interface touches its boundary).
    Homogeneous(f64),
    /// Layered cube: half-side, contrast index, offset index, interface below
    /// the centre (mirror `z`).
    Layered(f64, usize, usize, bool),
}

impl Medium {
    /// Builds the medium for an effective stack; `None` if it is homogeneous.
    ///
    /// # Errors
    ///
    /// A message if a contrast between adjacent layers is outside
    /// [`KAPPA_RANGE`].
    pub fn new(stack: Stack, opts: &StackOptions) -> Result<Option<Self>, String> {
        if stack.is_homogeneous() {
            return Ok(None);
        }
        let eps = stack.eps();
        let mut kappas: Vec<f64> = Vec::new();
        let mut index = |k: f64| -> Result<usize, String> {
            if !KAPPA_RANGE.contains(&k) {
                return Err(format!(
                    "dielectric contrast {k} between adjacent layers is outside {KAPPA_RANGE:?}"
                ));
            }
            // Equal contrasts come from equal permittivity pairs: exact equality.
            #[allow(clippy::float_cmp)]
            let found = kappas.iter().position(|&v| v == k);
            Ok(found.unwrap_or_else(|| {
                kappas.push(k);
                kappas.len() - 1
            }))
        };
        let n = stack.interfaces().len();
        let mut centre_below = Vec::with_capacity(n);
        let mut centre_above = Vec::with_capacity(n);
        for i in 0..n {
            centre_below.push(index(eps[i + 1] / eps[i])?);
            centre_above.push(index(eps[i] / eps[i + 1])?);
        }
        let slots = kappas.len() * opts.interface_offsets;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let band_max = ((opts.band_fraction * BAND_DEN as f64).round() as usize).clamp(1, BAND_MAX);
        Ok(Some(Self {
            offsets: opts.interface_offsets,
            resolution: opts.layered_resolution,
            band_max,
            alpha: opts.landing_alpha,
            cubes: (0..slots).map(|_| OnceLock::new()).collect(),
            grads: (0..slots).map(|_| OnceLock::new()).collect(),
            bands: (0..n * BAND_MAX * 2).map(|_| OnceLock::new()).collect(),
            anchored: (0..n * LADDER_LEVELS).map(|_| OnceLock::new()).collect(),
            landing: (0..LANDING_RATIOS * LANDING_TAUS.len())
                .map(|_| OnceLock::new())
                .collect(),
            anchored_bands: (0..n * LADDER_LEVELS * 2 * (BAND_MAX + 1))
                .map(|_| OnceLock::new())
                .collect(),
            stack,
            kappas,
            centre_below,
            centre_above,
        }))
    }

    /// The effective stack.
    #[must_use]
    pub const fn stack(&self) -> &Stack {
        &self.stack
    }

    fn near(&self, z: f64) -> Near {
        let (iface, d1, d2) = self
            .stack
            .nearest(z)
            .expect("a medium has at least one interface");
        let zi = self.stack.interfaces()[iface];
        Near {
            iface,
            z: zi,
            d1,
            d2,
            below: z < zi,
        }
    }

    fn cube(&self, kappa: usize, offset: usize) -> &LayeredCube {
        self.cubes[kappa * self.offsets + offset].get_or_init(|| {
            LayeredCube::new(self.kappas[kappa], offset, self.offsets, self.resolution)
        })
    }

    fn gradients(&self, kappa: usize, offset: usize) -> &LayeredGradients {
        self.grads[kappa * self.offsets + offset].get_or_init(|| {
            LayeredGradients::new(self.kappas[kappa], offset, self.offsets, self.resolution)
        })
    }

    fn band(&self, iface: usize, j: usize, side: BandSide) -> &GradientKernel {
        let s = usize::from(side == BandSide::Above);
        self.bands[(iface * BAND_MAX + (j - 1)) * 2 + s].get_or_init(|| {
            let eps = self.stack.eps();
            half_band_gradient(
                eps[iface + 1] / eps[iface],
                j,
                BAND_DEN,
                side,
                self.resolution,
            )
        })
    }

    /// Design §3.1 for a point at `near` with geometric clearance `h > d₁`.
    fn choose(&self, near: Near, h: f64) -> Cube {
        let h_max = h.min(near.d2);
        #[allow(clippy::cast_precision_loss)]
        let k_f = self.offsets as f64;
        let kappa = if near.below {
            self.centre_below[near.iface]
        } else {
            self.centre_above[near.iface]
        };
        if near.d1 <= 0.0 {
            return Cube::Layered(h_max, kappa, 0, !near.below);
        }
        let r = near.d1 / h_max * k_f;
        if r > k_f - 1.0 {
            return Cube::Homogeneous(near.d1);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = (r.ceil() as usize).clamp(1, self.offsets - 1);
        #[allow(clippy::cast_precision_loss)]
        let half = near.d1 * k_f / k as f64;
        Cube::Layered(half, kappa, k, !near.below)
    }

    /// Half-side of the first cube of a horizontal Gaussian face at height `z`
    /// with geometric clearance `h` (0 if the face lies on an interface).
    pub(crate) fn horizontal_hop_size(&self, z: f64, h: f64) -> f64 {
        let near = self.near(z);
        if h <= near.d1 {
            return h;
        }
        if near.d1 <= 0.0 {
            return 0.0;
        }
        match self.choose(near, h) {
            Cube::Homogeneous(s) | Cube::Layered(s, ..) => s,
        }
    }

    /// `true` if a clearance lower bound `bound` at height `z` is used as is
    /// (M2a §8, M2b §3.1): off an interface, with `bound` beyond the
    /// second-nearest interface, the M2a cube does not grow with the clearance.
    #[inline]
    pub(crate) fn bound_suffices(&self, z: f64, bound: f64, eps_stop: f64) -> bool {
        let near = self.near(z);
        near.d1 > snap_tolerance(z) && bound > near.d2.max(eps_stop)
    }

    /// One transition step from `x` with geometric clearance `h` (design M2b
    /// §3.1); returns the max-norm distance of the new point from `x`.
    #[inline]
    pub(crate) fn step(&self, green: &GreenTables, x: &mut [f64; 3], h: f64, rng: &mut Rng) -> f64 {
        let mut near = self.near(x[2]);
        if near.d1 > 0.0 && near.d1 <= snap_tolerance(x[2]) {
            // Rounding (e.g. folding) left the walk a few ulps off an interface.
            x[2] = near.z;
            near = self.near(x[2]);
        }
        if near.d1 == 0.0 && h > near.d2 {
            // Only cubes that reach a second interface need an anchored table;
            // smaller ones are the M2a cube through the centre (size min(h, d₂)).
            if let Some((s, cube)) = self
                .anchored_for(near.iface, h)
                .filter(|&(s, _)| s > near.d2)
            {
                let u = cube.sample_transition(rng);
                add(x, s, u);
                return s;
            }
        }
        if h <= near.d1 {
            add(x, h, green.sample_transition(rng));
            return h;
        }
        let cube = self.choose(near, h);
        let s = match cube {
            Cube::Homogeneous(s) | Cube::Layered(s, ..) => s,
        };
        if near.d1 > 0.0 && s < self.alpha * h {
            // Landing box: its bottom face on the nearest interface, the walk
            // at the fraction τ of its height; the box extends away from the
            // interface as far as the medium stays homogeneous and the
            // clearance allows.
            let a = near.d1;
            let room = self.far_distance(x[2], near).min(h);
            let k = LANDING_TAUS
                .iter()
                .position(|&t| t * (a + room) >= a)
                .unwrap_or(LANDING_TAUS.len() - 1);
            let half = 0.5 * a / LANDING_TAUS[k];
            let (ratio, b) = self.landing_for(k, h / half);
            let u = b.sample_transition(rng);
            let lat = half * ratio; // lateral half-width ≤ h
            x[0] = half.mul_add(u[0], x[0]);
            x[1] = half.mul_add(u[1], x[1]);
            // From the interface towards the walk.
            let dir = if near.below { -1.0 } else { 1.0 };
            #[allow(clippy::float_cmp)] // face coordinates are exactly ±1
            if u[2] == -1.0 {
                x[2] = near.z;
            } else {
                x[2] = (dir * half).mul_add(u[2] + 1.0, near.z);
            }
            return lat.max(2.0 * half - a);
        }
        match cube {
            Cube::Homogeneous(s) => add(x, s, green.sample_transition(rng)),
            Cube::Layered(s, kappa, k, mirror) => {
                let mut u = self.cube(kappa, k).sample_transition(rng);
                if mirror {
                    u[2] = -u[2];
                }
                add(x, s, u);
            }
        }
        s
    }

    /// The anchored cube on interface `i` for clearance `h`: the largest
    /// ladder half-side `2^{l/4} ≤ h` and its table, if in range.
    fn anchored_for(&self, i: usize, h: f64) -> Option<(f64, &MultiLayerCube)> {
        let (s, level) = self.anchored_level(h)?;
        let slot = i * LADDER_LEVELS + level;
        Some((
            s,
            self.anchored[slot]
                .get_or_init(|| MultiLayerCube::new(self.anchored_profile(i, s), self.resolution)),
        ))
    }

    /// The largest ladder half-side `2^{l/4} ≤ h` for an anchored cube and
    /// its level index, if in range.
    fn anchored_level(&self, h: f64) -> Option<(f64, usize)> {
        // Beyond the images of the effective stack the profile would be wrong.
        let h = h.min(self.stack.max_half());
        #[allow(clippy::cast_possible_truncation)]
        let mut l = (LADDER_STEPS * h.log2()).floor() as i64;
        if ladder(l) > h {
            l -= 1;
        }
        if !(LADDER_MIN..=LADDER_MAX).contains(&l) {
            return None;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        Some((ladder(l), (l - LADDER_MIN) as usize))
    }

    /// The half-band kernel of the anchored cube `(i, level)` of half-side
    /// `s` on side `side`, of length `len`: the full band (`j = 0`, the same
    /// `len` for every call with this cube and side) or the fraction
    /// `len = j·s / BAND_DEN`.
    fn anchored_band(
        &self,
        (i, level, s): (usize, usize, f64),
        side: BandSide,
        j: usize,
        len: f64,
    ) -> &BandGradient {
        let k = usize::from(side == BandSide::Above);
        let slot = ((i * LADDER_LEVELS + level) * 2 + k) * (BAND_MAX + 1) + j;
        self.anchored_bands[slot].get_or_init(|| {
            let half = 0.5 * len / s;
            let (z0, z1) = match side {
                BandSide::Below => (0.5 - half, 0.5),
                BandSide::Above => (0.5, 0.5 + half),
            };
            Box::new(BandGradient::new(
                &self.anchored_profile(i, s),
                z0,
                z1,
                self.resolution,
            ))
        })
    }

    /// The permittivity profile of the cube of half-side `s` centred on
    /// interface `i`.
    fn anchored_profile(&self, i: usize, s: f64) -> Profile {
        let (ifaces, eps) = (self.stack.interfaces(), self.stack.eps());
        let (zi, lo) = (ifaces[i], ifaces[i] - s);
        let mut zs = Vec::new();
        // Layer `k` lies below interface `k`.
        let first = ifaces.partition_point(|&z| z <= lo);
        let mut layer_eps = vec![eps[first]];
        for (j, &z) in ifaces.iter().enumerate().skip(first) {
            let zeta = (z - lo) / (2.0 * s);
            if zeta >= 1.0 - FACE_TOLERANCE || z >= zi + s {
                break;
            }
            if zeta <= FACE_TOLERANCE {
                // On the bottom face: the cube starts in the layer above it.
                layer_eps[0] = eps[j + 1];
                continue;
            }
            zs.push(zeta);
            layer_eps.push(eps[j + 1]);
        }
        Profile::new(zs, layer_eps)
    }

    /// Distance from `z` to the next interface on the side away from the
    /// nearest one (`near`), or infinity.
    fn far_distance(&self, z: f64, near: Near) -> f64 {
        let ifaces = self.stack.interfaces();
        if near.below {
            near.iface
                .checked_sub(1)
                .map_or(f64::INFINITY, |k| z - ifaces[k])
        } else {
            ifaces
                .get(near.iface + 1)
                .map_or(f64::INFINITY, |&zj| zj - z)
        }
    }

    /// The landing box with start height `LANDING_TAUS[k]` for a
    /// clearance-to-half-height ratio `q ≥ 1`: the largest ratio
    /// `2^{r/2} ≤ q` (at most `2^{LANDING_RATIOS/2}`) and its table.
    fn landing_for(&self, k: usize, q: f64) -> (f64, &LandingBox) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mut r = ((2.0 * q.log2()).floor().max(0.0) as usize).min(LANDING_RATIOS - 1);
        while r > 0 && landing_ratio(r) > q {
            r -= 1;
        }
        let ratio = landing_ratio(r);
        (
            ratio,
            self.landing[k * LANDING_RATIOS + r]
                .get_or_init(|| LandingBox::new(ratio, LANDING_TAUS[k], self.resolution)),
        )
    }

    /// Design §4: the first hop from the Gaussian point `g` (geometric
    /// clearance `h > 0` there). `trace` sees every extra clearance cube the
    /// hop queries (the band test), for replays.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn first_hop(
        &self,
        walker: &Walker<'_>,
        surface: &GaussSurface,
        g: &GaussPoint,
        h: f64,
        antithetic: bool,
        rng: &mut Rng,
        trace: &mut dyn FnMut([f64; 3], f64),
    ) -> Hop {
        // The half-band test comes first: every point of a band must use the
        // band estimator, whatever its own clearance, for the average to be exact.
        if let Some(hop) = self.band_first_hop(
            walker,
            surface,
            g,
            h,
            antithetic,
            rng,
            trace,
            &mut |_, _| true,
        ) {
            return hop;
        }
        self.cube_first_hop(walker.green, g, h, antithetic, rng)
    }

    /// The half-band part of [`Self::first_hop`]: `Some` for a lateral
    /// Gaussian point in the band of its nearest interface, if `allow`
    /// accepts the band's extent `[z_lo, z_hi]` (design M2f §3.1).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn band_first_hop(
        &self,
        walker: &Walker<'_>,
        surface: &GaussSurface,
        g: &GaussPoint,
        h: f64,
        antithetic: bool,
        rng: &mut Rng,
        trace: &mut dyn FnMut([f64; 3], f64),
        allow: &mut dyn FnMut(f64, f64) -> bool,
    ) -> Option<Hop> {
        if g.axis == Axis::Z {
            return None;
        }
        let eps_r = self.stack.eps_at(g.p[2]);
        let near = self.near(g.p[2]);
        self.band_hop(
            walker, surface, g, near, eps_r, h, antithetic, rng, trace, allow,
        )
    }

    /// Half-side of the cube [`Self::cube_first_hop`] uses at height `z` with
    /// clearance `h`.
    pub(crate) fn first_cube_size(&self, z: f64, h: f64) -> f64 {
        let near = self.near(z);
        if h <= near.d1 {
            return h;
        }
        match self.choose(near, h) {
            Cube::Homogeneous(s) | Cube::Layered(s, ..) => s,
        }
    }

    /// The centred-cube part of [`Self::first_hop`] (outside every band).
    pub(crate) fn cube_first_hop(
        &self,
        green: &GreenTables,
        g: &GaussPoint,
        h: f64,
        antithetic: bool,
        rng: &mut Rng,
    ) -> Hop {
        let eps_r = self.stack.eps_at(g.p[2]);
        let near = self.near(g.p[2]);
        if h <= near.d1 {
            return homogeneous_hop(green, g.p, g.axis, h, eps_r, antithetic, rng);
        }
        match self.choose(near, h) {
            Cube::Homogeneous(s) => homogeneous_hop(green, g.p, g.axis, s, eps_r, antithetic, rng),
            Cube::Layered(s, kappa, k, mirror) => {
                let grads = self.gradients(kappa, k);
                if g.axis == Axis::Z && antithetic {
                    // The normal kernel has no reflection symmetry: pair a
                    // point of its positive part with one of its negative part
                    // (a mirrored cube swaps the parts).
                    let [mut pos, mut neg] = grads.z.sample_pair(rng);
                    if mirror {
                        pos[2] = -pos[2];
                        neg[2] = -neg[2];
                        (pos, neg) = (neg, pos);
                    }
                    Hop {
                        starts: [offset(g.p, s, pos), offset(g.p, s, neg)],
                        n: 2,
                        factor: eps_r * grads.z.l1() / (2.0 * s),
                    }
                } else if g.axis == Axis::Z {
                    // The canonical normal kernel is ∂/∂z towards the interface.
                    let (mut u, mut sign) = grads.z.sample(rng);
                    if mirror {
                        u[2] = -u[2];
                        sign = -sign;
                    }
                    Hop {
                        starts: [offset(g.p, s, u), g.p],
                        n: 1,
                        factor: eps_r * sign * grads.z.l1() / (2.0 * s),
                    }
                } else {
                    let (mut u, sign) = grads.x.sample(rng);
                    if mirror {
                        u[2] = -u[2];
                    }
                    tangential_hop(
                        g.p,
                        g.axis,
                        s,
                        u,
                        eps_r * sign * grads.x.l1() / (2.0 * s),
                        antithetic,
                    )
                }
            }
        }
    }

    /// The half-band estimator for a lateral Gaussian point near an interface,
    /// or `None` if the point is outside the band of its nearest interface.
    #[allow(clippy::too_many_arguments)]
    fn band_hop(
        &self,
        walker: &Walker<'_>,
        surface: &GaussSurface,
        g: &GaussPoint,
        near: Near,
        eps_r: f64,
        h: f64,
        antithetic: bool,
        rng: &mut Rng,
        trace: &mut dyn FnMut([f64; 3], f64),
        allow: &mut dyn FnMut(f64, f64) -> bool,
    ) -> Option<Hop> {
        #[allow(clippy::cast_precision_loss)]
        let eta_max = self.band_max as f64 / BAND_DEN as f64;
        // Bands are at most η_max·h_p long, and h_p ≤ h + d₁ (the clearance
        // is 1-Lipschitz; `h` is exact below `far_step`): skip the clearance
        // query otherwise.
        if h < walker.far_step && near.d1 * (1.0 - eta_max) >= eta_max * h {
            return None;
        }
        let (_, _, d2) = self.stack.nearest(near.z)?;
        let p = [g.p[0], g.p[1], near.z];
        // Capped by the dielectric boxes: the band's cube sees the stack only.
        let (h_p, _) = walker.stack_clearance(p);
        trace(p, h_p);
        let (side, dir) = if near.below {
            (BandSide::Below, -1.0)
        } else {
            (BandSide::Above, 1.0)
        };
        // M2b §3.5: if a second interface limits the M2a cube, use the
        // anchored multi-layer cube on the interface. Its band stays within
        // the layer, up to half its thickness, and is cut to a tabulated
        // fraction of the cube; the rest of the run uses centred cubes with
        // d₁ ≥ the band length.
        if let Some((s, level)) = self.anchored_level(h_p).filter(|&(s, _)| s > d2) {
            let ifaces = self.stack.interfaces();
            let layer = match side {
                BandSide::Below => near.iface.checked_sub(1).map(|k| near.z - ifaces[k]),
                BandSide::Above => ifaces.get(near.iface + 1).map(|&z| z - near.z),
            }
            .unwrap_or(f64::INFINITY);
            // The full band reaches the middle of a thin layer, where the
            // band of the next interface starts.
            let cap = (eta_max * s).min(0.5 * layer);
            let run = surface.accepted_run(g.owner, p, dir, cap, walker.domain);
            #[allow(clippy::cast_precision_loss)]
            let den = BAND_DEN as f64;
            let (j, len) = if run >= cap {
                (0, cap)
            } else {
                // A shorter run (the Gaussian surface ends within the band):
                // the largest tabulated fraction.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let j = ((den * run / s).floor().max(0.0) as usize).min(self.band_max);
                #[allow(clippy::cast_precision_loss)]
                (j, s * j as f64 / den)
            };
            if len > 0.0 {
                if near.d1 >= len
                    || !allow(
                        near.z.min(dir.mul_add(len, near.z)),
                        near.z.max(dir.mul_add(len, near.z)),
                    )
                {
                    return None;
                }
                let kernel = self.anchored_band((near.iface, level, s), side, j, len);
                let (u, sign) = kernel.sample(rng);
                return Some(tangential_hop(
                    p,
                    g.axis,
                    s,
                    u,
                    eps_r * sign * kernel.l1() / (2.0 * s),
                    antithetic,
                ));
            }
            // A run shorter than one fraction: the M2a band below covers it.
        }
        let big_h = h_p.min(d2);
        if big_h <= 0.0 {
            return None;
        }
        let len = surface.accepted_run(g.owner, p, dir, eta_max * big_h, walker.domain);
        if len <= 0.0
            || near.d1 >= len
            || !allow(
                near.z.min(dir.mul_add(len, near.z)),
                near.z.max(dir.mul_add(len, near.z)),
            )
        {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let den = BAND_DEN as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let j = ((den * len / big_h).ceil() as usize).clamp(1, self.band_max);
        #[allow(clippy::cast_precision_loss)]
        let s = len * den / j as f64;
        let kernel = self.band(near.iface, j, side);
        let (u, sign) = kernel.sample(rng);
        Some(tangential_hop(
            p,
            g.axis,
            s,
            u,
            eps_r * sign * kernel.l1() / (2.0 * s),
            antithetic,
        ))
    }
}

fn offset(p: [f64; 3], s: f64, u: [f64; 3]) -> [f64; 3] {
    [p[0] + s * u[0], p[1] + s * u[1], p[2] + s * u[2]]
}

/// The M1 first hop: homogeneous cube of half-side `h` at `p`, antithetic
/// pair by point reflection.
pub(crate) fn homogeneous_hop(
    green: &GreenTables,
    p: [f64; 3],
    axis: Axis,
    h: f64,
    eps_r: f64,
    antithetic: bool,
    rng: &mut Rng,
) -> Hop {
    let (u, sign) = green.sample_gradient(axis, rng);
    let back = [-u[0], -u[1], -u[2]];
    Hop {
        starts: [offset(p, h, u), offset(p, h, back)],
        n: if antithetic { 2 } else { 1 },
        factor: eps_r * sign * green.grad_l1() / (2.0 * h),
    }
}

/// A layered tangential first hop from a cube centred at `c`: canonical
/// tangential axis `x` mapped onto `axis`; the antithetic partner reflects the
/// tangential coordinate (the kernel is odd under that reflection).
fn tangential_hop(
    c: [f64; 3],
    axis: Axis,
    s: f64,
    u: [f64; 3],
    factor: f64,
    antithetic: bool,
) -> Hop {
    let u = if axis == Axis::Y {
        [u[1], u[0], u[2]]
    } else {
        u
    };
    let a = axis.index();
    let mut back = u;
    back[a] = -back[a];
    Hop {
        starts: [offset(c, s, u), offset(c, s, back)],
        n: if antithetic { 2 } else { 1 },
        factor,
    }
}
