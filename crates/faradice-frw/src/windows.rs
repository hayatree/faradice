//! Window first hops (design M2f, `docs/design/M2f_window_first_hops.md`).
//!
//! The faces of a Gaussian surface are covered by smooth windows that add up
//! to one ([`crate::section::window`], spacing `P` along both tangential
//! axes). A sampled point picks one of the windows covering it with
//! probability its weight. Where the box around that window has a 2D
//! cross-section (every interface in it spans the box along the face's
//! longer tangential axis `w`, and planes of both other normals are present),
//! the window's flux `∫ ω q dA` is estimated with the kernel of
//! [`crate::section`], and the sample weight is `A · ∫ω q / ∫ω`. Other
//! windows leave the point to the point estimators. Which windows qualify
//! depends only on the geometry, so the mix is exact (design §3.1).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use faradice_core::{Axis, Face};

use crate::axis::BoxSteps;
use crate::gauss::{GaussPoint, GaussSurface};
use crate::medium::Hop;
use crate::rng::Rng;
use crate::section::{AxisCache, SectionKernel, SectionSpec, window};
use crate::walk::Walker;

/// Window spacing as a fraction of the Gaussian offset (design §3.2).
const SPACING: f64 = 0.5;

/// Interface planes per cross-section axis.
const MAX_PLANES: usize = 6;

/// Target table cells across the box in `(u, v)`, and along `w` (design
/// §4.3).
const CELLS: usize = 12;
const W_CELLS: usize = 8;

/// Faces examined per box before giving up (dense regions keep the point
/// estimators).
const MAX_FACES: usize = 256;

/// A qualified window: its kernel and the map to world coordinates.
#[derive(Debug)]
pub(crate) struct Plan {
    kernel: Arc<SectionKernel>,
    /// Axes `(w, u, v)`; `u` is the face normal.
    axes: [usize; 3],
    /// Outward sign of the normal.
    sign: f64,
    /// World coordinates of the canonical origin `(0, 0, 0)`.
    origin: [f64; 3],
    /// Canonical unit length (world).
    scale: f64,
    /// `∫ω` (world).
    area: f64,
    /// Centre and largest half-extent of the box (for traces).
    centre: [f64; 3],
    reach: f64,
}

impl Plan {
    fn world(&self, c: [f64; 3]) -> [f64; 3] {
        let mut x = [0.0; 3];
        let [w, u, v] = self.axes;
        x[w] = self.scale.mul_add(c[0], self.origin[w]);
        x[u] = (self.sign * self.scale).mul_add(c[1], self.origin[u]);
        x[v] = self.scale.mul_add(c[2], self.origin[v]);
        x
    }
}

type Slot<T> = Arc<OnceLock<Option<Arc<T>>>>;

/// Key of a window: surface, dilated box, face, window indices along `w`
/// and `v`.
type WindowKey = ((u32, u64), usize, usize, u64, u64);

/// Window decisions and kernels of one medium.
#[derive(Debug, Default)]
pub(crate) struct WindowCache {
    plans: RwLock<HashMap<WindowKey, Slot<Plan>>>,
    kernels: RwLock<HashMap<Vec<u64>, Slot<SectionKernel>>>,
    axes: AxisCache,
}

fn slot<K: std::hash::Hash + Eq + Clone, T>(map: &RwLock<HashMap<K, Slot<T>>>, key: &K) -> Slot<T> {
    if let Some(s) = map.read().expect("window cache lock").get(key) {
        return Arc::clone(s);
    }
    let mut w = map.write().expect("window cache lock");
    Arc::clone(w.entry(key.clone()).or_default())
}

/// The windows of one face: axes, counts and spacings.
#[derive(Debug, Clone, Copy)]
struct Tiling {
    w: usize,
    v: usize,
    /// Windows along `w` and `v` are centred at `lo + i p`, `i = 0..=n`.
    n: [u64; 2],
    p: [f64; 2],
}

impl Tiling {
    fn new(surface: &GaussSurface, bi: usize, face: Face) -> Self {
        let db = surface.dilated(bi);
        let a = face.axis.index();
        let len = |t: usize| db.max()[t] - db.min()[t];
        // `w` is the longer tangential axis.
        let (mut w, mut v) = ((a + 1) % 3, (a + 2) % 3);
        if len(v) > len(w) {
            std::mem::swap(&mut w, &mut v);
        }
        let target = SPACING * surface.offset();
        let spacing = |l: f64| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = (l / target).ceil().max(1.0) as u64;
            #[allow(clippy::cast_precision_loss)]
            let p = l / n as f64;
            (n, p)
        };
        let (nw, pw) = spacing(len(w));
        let (nv, pv) = spacing(len(v));
        Self {
            w,
            v,
            n: [nw, nv],
            p: [pw, pv],
        }
    }
}

/// Picks one of the two windows covering `x` with probability its weight.
fn pick(x: f64, lo: f64, n: u64, p: f64, rng: &mut Rng) -> u64 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i = (((x - lo) / p).floor().max(0.0) as u64).min(n - 1);
    #[allow(clippy::cast_precision_loss)]
    let wi = window(x - (i as f64).mul_add(p, lo), p);
    if rng.next_f64() < wi { i } else { i + 1 }
}

/// The window first hop of `g`, if the window it picks qualifies: `None`
/// leaves the point to the point estimators (with the probability of the
/// windows that do not qualify). `trace` sees the box.
#[allow(clippy::too_many_arguments)]
pub(crate) fn first_hop(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    g: &GaussPoint,
    b: &BoxSteps,
    antithetic: bool,
    rng: &mut Rng,
    trace: &mut dyn FnMut([f64; 3], f64),
) -> Option<Hop> {
    if walker.overlay.is_some() || walker.added.is_some() {
        return None;
    }
    let (bi, face) = g.owner;
    let db = surface.dilated(bi);
    let t = Tiling::new(surface, bi, face);
    let iw = pick(g.p[t.w], db.min()[t.w], t.n[0], t.p[0], rng);
    let iv = pick(g.p[t.v], db.min()[t.v], t.n[1], t.p[1], rng);
    let plan = plan_of(walker, surface, b, bi, face, &t, [iw, iv])?;
    trace(plan.centre, plan.reach);
    let k = &plan.kernel;
    let mag = k.l1() * plan.scale / plan.area;
    if antithetic {
        let [p, q] = k.sample_pair(rng);
        return Some(Hop {
            starts: [plan.world(p), plan.world(q)],
            n: 2,
            factor: -g.sign * mag,
        });
    }
    let (p, sgn) = k.sample(rng);
    Some(Hop {
        starts: [plan.world(p), g.p],
        n: 1,
        factor: -g.sign * sgn * mag,
    })
}

/// Whether the half-band estimator may be used at `g` after its window did
/// not qualify, for a band over `[z_lo, z_hi]` at the other coordinates of
/// `g` (design M2f §3.1): the fallback probability is constant along the
/// band if, for each of the two windows covering `g` along the other axis,
/// all windows meeting the band along `z` qualify alike (the windows along
/// `z` add up to one). Always on faces normal to z.
pub(crate) fn band_allowed(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    g: &GaussPoint,
    b: &BoxSteps,
    z_lo: f64,
    z_hi: f64,
) -> bool {
    let (bi, face) = g.owner;
    if face.axis == Axis::Z || walker.overlay.is_some() || walker.added.is_some() {
        return true;
    }
    let db = surface.dilated(bi);
    let t = Tiling::new(surface, bi, face);
    // Window indices whose support (open, half-width p) meets [a, b].
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let range = |a: f64, b: f64, lo: f64, n: u64, p: f64| {
        let i = ((a - lo) / p).floor().max(0.0) as u64;
        let j = ((b - lo) / p).ceil().max(0.0) as u64;
        i.min(n)..=j.min(n)
    };
    let (zi, oi) = if t.w == 2 { (0, 1) } else { (1, 0) };
    let o = [t.w, t.v][oi];
    let zs = range(z_lo, z_hi, db.min()[2], t.n[zi], t.p[zi]);
    let os = range(g.p[o], g.p[o], db.min()[o], t.n[oi], t.p[oi]);
    os.into_iter().all(|j| {
        let mut status = zs.clone().map(|i| {
            let mut ij = [0; 2];
            ij[zi] = i;
            ij[oi] = j;
            plan_of(walker, surface, b, bi, face, &t, ij).is_some()
        });
        let first = status.next();
        status.all(|s| Some(s) == first)
    })
}

/// The cached plan of window `ij` of a face.
fn plan_of(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    b: &BoxSteps,
    bi: usize,
    face: Face,
    t: &Tiling,
    ij: [u64; 2],
) -> Option<Arc<Plan>> {
    // Windows cut by the face's edges are not tabulated.
    if ij[0] == 0 || ij[0] == t.n[0] || ij[1] == 0 || ij[1] == t.n[1] {
        return None;
    }
    let key = (surface.key(), bi, face.index(), ij[0], ij[1]);
    let s = slot(&b.windows().plans, &key);
    s.get_or_init(|| plan(walker, surface, b, bi, face, t, ij))
        .clone()
}

/// Rounds a canonical coordinate so that equal configurations share keys.
fn q(x: f64) -> f64 {
    const S: f64 = (1u64 << 36) as f64;
    (x * S).round() / S
}

/// Distinct interface planes in the open box `[lo, hi]` per normal, if
/// every face there is normal to an axis other than `w` and spans the box
/// along `w`.
fn planes_in(
    b: &BoxSteps,
    c: [f64; 3],
    r: f64,
    lo: [f64; 3],
    hi: [f64; 3],
    w: usize,
) -> Option<[Vec<f64>; 3]> {
    let medium = b.box_medium();
    let mut planes: [Vec<f64>; 3] = Default::default();
    for axis in Axis::ALL {
        let a = axis.index();
        let faces = medium.faces(axis);
        let mut seen: Vec<u32> = Vec::new();
        while let Some(f) = faces.nearest_linf_where(c, r, |id| !seen.contains(&id)) {
            #[allow(clippy::cast_possible_truncation)]
            let id = f.box_id.index() as u32;
            seen.push(id);
            if seen.len() > MAX_FACES {
                return None;
            }
            let rect = &faces.rects()[id as usize];
            let k = rect.coord();
            let inside = lo[a] < k
                && k < hi[a]
                && (0..3)
                    .filter(|&e| e != a)
                    .all(|e| rect.min()[e] < hi[e] && lo[e] < rect.max()[e]);
            if !inside {
                continue;
            }
            if a == w || rect.min()[w] > lo[w] || rect.max()[w] < hi[w] {
                return None;
            }
            if !planes[a].iter().any(|x| x.to_bits() == k.to_bits()) {
                planes[a].push(k);
            }
        }
    }
    for &z in b.stack_ref().interfaces() {
        if lo[2] < z && z < hi[2] {
            if w == 2 {
                return None;
            }
            if !planes[2].iter().any(|x| x.to_bits() == z.to_bits()) {
                planes[2].push(z);
            }
        }
    }
    Some(planes)
}

/// A qualifying window before its kernel: the kernel's key and spec, and
/// the plan without the kernel.
struct Draft {
    key: Vec<u64>,
    spec: SectionSpec,
    axes: [usize; 3],
    sign: f64,
    origin: [f64; 3],
    scale: f64,
    area: f64,
    centre: [f64; 3],
    reach: f64,
}

/// The plan of window `ij`, if it qualifies; builds its kernel on first use.
fn plan(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    b: &BoxSteps,
    bi: usize,
    face: Face,
    t: &Tiling,
    ij: [u64; 2],
) -> Option<Arc<Plan>> {
    let d = draft(walker, surface, b, bi, face, t, ij)?;
    let kernel = slot(&b.windows().kernels, &d.key)
        .get_or_init(|| {
            SectionKernel::with_axes(&d.spec, &b.windows().axes)
                .ok()
                .map(Arc::new)
        })
        .clone()?;
    Some(Arc::new(Plan {
        kernel,
        axes: d.axes,
        sign: d.sign,
        origin: d.origin,
        scale: d.scale,
        area: d.area,
        centre: d.centre,
        reach: d.reach,
    }))
}

/// The geometry of window `ij`, if it qualifies (design §3.3).
#[allow(clippy::too_many_lines)]
fn draft(
    walker: &Walker<'_>,
    surface: &GaussSurface,
    b: &BoxSteps,
    bi: usize,
    face: Face,
    t: &Tiling,
    ij: [u64; 2],
) -> Option<Draft> {
    let db = surface.dilated(bi);
    let n = face.axis.index();
    let sign = face.side.sign();
    let (w, v) = (t.w, t.v);
    let [pw, pv] = t.p;
    let mut c = [0.0; 3];
    c[n] = db.face_coord(face);
    #[allow(clippy::cast_precision_loss)]
    {
        c[w] = (ij[0] as f64).mul_add(pw, db.min()[w]);
        c[v] = (ij[1] as f64).mul_add(pv, db.min()[v]);
    }
    // The window's support lies on the outer surface.
    let (mut slo, mut shi) = (c, c);
    slo[w] -= pw;
    shi[w] += pw;
    slo[v] -= pv;
    shi[v] += pv;
    if !surface.window_clear(bi, face, slo, shi) {
        return None;
    }
    // The box: the support plus one half-support along w and v, up to the
    // offset along the normal; conductor-free (the L∞ cube of its largest
    // half-extent).
    let (h, _) = walker.clearance(c);
    let hu = h.min(surface.offset());
    let (bw, bv) = (2.0 * pw, 2.0 * pv);
    if bw.max(bv) > h || hu <= 0.5 * bw.max(bv) {
        return None;
    }
    let half = [bw, hu, bv];
    let (mut lo, mut hi) = (c, c);
    for (k, &a) in [w, n, v].iter().enumerate() {
        lo[a] -= half[k];
        hi[a] += half[k];
    }
    for corner in 0..8 {
        let p: [f64; 3] = std::array::from_fn(|a| if corner >> a & 1 == 1 { hi[a] } else { lo[a] });
        if !walker.domain.contains(p) {
            return None;
        }
    }
    let r = bw.max(hu).max(bv);
    if r > b.box_medium().reach() {
        return None;
    }
    let planes = planes_in(b, c, r, lo, hi, w)?;
    // Truly 2D cross-sections only: where the box is layered along u or
    // homogeneous, the point cubes of M2e reach h and have less variance.
    if planes[v].is_empty() || planes[n].len() > MAX_PLANES || planes[v].len() > MAX_PLANES {
        return None;
    }
    // Canonical frame: u along the outward normal, unit `scale = 2 hu`.
    let scale = 2.0 * hu;
    let mut up: Vec<f64> = planes[n]
        .iter()
        .map(|&x| q((sign * (x - c[n]) + hu) / scale))
        .collect();
    let mut vp: Vec<f64> = planes[v].iter().map(|&x| q((x - lo[v]) / scale)).collect();
    up.sort_by(f64::total_cmp);
    vp.sort_by(f64::total_cmp);
    if up.iter().any(|&x| (x - 0.5).abs() < 1e-9) {
        return None;
    }
    let vlen = q(2.0 * bv / scale);
    let ub: Vec<f64> = std::iter::once(0.0)
        .chain(up.iter().copied())
        .chain([1.0])
        .collect();
    let vb: Vec<f64> = std::iter::once(0.0)
        .chain(vp.iter().copied())
        .chain([vlen])
        .collect();
    let mut eps = Vec::with_capacity((ub.len() - 1) * (vb.len() - 1));
    for uw in ub.windows(2) {
        for vw in vb.windows(2) {
            let mut x = c;
            x[n] = (sign * scale).mul_add(0.5 * (uw[0] + uw[1]), c[n] - sign * hu);
            x[v] = scale.mul_add(0.5 * (vw[0] + vw[1]), lo[v]);
            eps.push(b.eps_at(x));
        }
    }
    let spec = SectionSpec {
        half_w: q(bw / scale),
        u: 1.0,
        v: vlen,
        u_planes: up,
        v_planes: vp,
        eps,
        gauss: 0.5,
        window: [q(pw / scale), q(pv / scale)],
        w_cells: W_CELLS,
        cells: CELLS,
    };
    let key: Vec<u64> = [spec.half_w, spec.v, spec.window[0], spec.window[1]]
        .iter()
        .chain(&spec.u_planes)
        .chain(&[f64::NAN])
        .chain(&spec.v_planes)
        .chain(&[f64::NAN])
        .chain(&spec.eps)
        .map(|x| x.to_bits())
        .collect();
    let mut origin = c;
    origin[n] = c[n] - sign * hu;
    origin[v] = lo[v];
    Some(Draft {
        key,
        spec,
        axes: [w, n, v],
        sign,
        origin,
        scale,
        area: pw * pv,
        centre: c,
        reach: r,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_follow_the_window_weights() {
        let mut rng = Rng::from_seed(5);
        let (lo, n, p) = (0.0, 4, 0.25);
        let x = 0.3;
        let trials = 200_000u32;
        let mut ones = 0;
        for _ in 0..trials {
            match pick(x, lo, n, p, &mut rng) {
                1 => ones += 1,
                2 => {}
                other => panic!("window {other} does not cover {x}"),
            }
        }
        let want = window(x - p, p);
        let got = f64::from(ones) / f64::from(trials);
        assert!((got - want).abs() < 4e-3, "{got} vs {want}");
    }
}
