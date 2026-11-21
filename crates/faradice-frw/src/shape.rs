//! Shape derivatives with a face normal-derivative kernel (design D3 §6,
//! pillar A).
//!
//! For one homogeneous dielectric, the Hadamard formula gives the derivative
//! of the Maxwell matrix for a conductor face that grows by `V_n` along its
//! outward normal:
//!
//! ```text
//! dC_ij = ε ∮ (∂_n φ_i)(∂_n φ_j) V_n dS
//! ```
//!
//! `∂_n φ(x₀)` at a face point is estimated with a cube that *rests on the
//! face*: its bottom face `B` (centred at `x₀`) lies in the conductor face,
//! where every `φ_i` is constant (`V_k`). With `K_B` the normal derivative at
//! the centre of `B` of the Poisson kernel for data on the other five faces,
//!
//! ```text
//! ∂_n φ(x₀) = (1/L) ∫_{∂Q∖B} K_B(ŷ) (φ(y) − V_k) dŷ,     L = cube side.
//! ```
//!
//! `K_B` is bounded and has closed-form sine series, so [`FaceTables`] has no
//! quadrature error (like [`GreenTables`](crate::green::GreenTables)). The
//! subtraction of `V_k` is a control variate: walks that return to the face's
//! own conductor score zero.
//!
//! Products need independent walks: each face sample draws two independent
//! kernel hops and scores `½(g_i¹ g_j² + g_j¹ g_i²)`.
//!
//! Edges and corners are an open problem (§6.4): near a convex edge the
//! estimator has infinite variance. [`solve_shape_derivative`] therefore only
//! accepts faces whose edges all lie on reflecting walls (where the mirror
//! continues the conductor), or integrates over the face interior only, at a
//! caller-given distance from the edges.

use std::collections::BTreeMap;

use faradice_core::{BoxId, EPS0_FF_PER_UM, Estimate, Face, Moments, NetId, Sink};
use faradice_geom::{Aabb, Geometry};
use rayon::prelude::*;

use crate::alias::AliasTable;
use crate::boundary::BoundaryModel;
use crate::green::{NTERMS, bilinear, cell_integrals};
use crate::rng::{Rng, splitmix64};
use crate::sens::covered_parts;
use crate::solver::{SolveError, SolverConfig, setup};
use crate::walk::Walker;

/// Domain-separation tag of the shape-derivative streams.
const SHAPE_TAG: u64 = 0x5A9E_D0E5_7A11_D3A1;

/// Kernel tables of the face normal derivative on the unit cube `[0,1]³`
/// with `B` the bottom face `z = 0` and the derivative `∂/∂z` at `(½, ½, 0)`.
#[derive(Debug, Clone)]
pub struct FaceTables {
    m: usize,
    /// Signed kernel integrals per cell. Layout `face * m² + j * m + i`,
    /// faces `[x-, x+, y-, y+, z+]`; side faces carry `(tangential, z)`, the
    /// top face `(x, y)`.
    cells: Vec<f64>,
    alias: AliasTable,
    l1: f64,
}

impl FaceTables {
    /// Builds the tables with `resolution × resolution` cells per face.
    ///
    /// # Panics
    ///
    /// Panics if `resolution` is zero or odd.
    #[must_use]
    pub fn new(resolution: usize) -> Self {
        assert!(
            resolution >= 2 && resolution % 2 == 0,
            "resolution must be even and >= 2"
        );
        let m = resolution;
        let pi = std::f64::consts::PI;
        #[allow(clippy::cast_precision_loss)]
        let odd: Vec<f64> = (0..NTERMS).map(|t| (2 * t + 1) as f64).collect();
        #[allow(clippy::cast_precision_loss)]
        let all: Vec<f64> = (1..=2 * NTERMS).map(|t| t as f64).collect();
        let sin_half = |k: f64| (k * pi / 2.0).sin().round();
        let gamma = |a: f64, b: f64| pi * a.hypot(b);
        let a_odd = cell_integrals(&odd, sin_half, m);
        // Top face z = 1: ∂_z at (½,½,0) of Σ A sin sin sinh(γz)/sinh γ is
        // Σ A_mn s_m s_n γ / sinh γ, A_mn = 4 ∫∫ f sin sin.
        let top = bilinear(
            &a_odd,
            &a_odd,
            |s, t| {
                let g = gamma(odd[s], odd[t]);
                4.0 * g / g.sinh()
            },
            m,
        );
        // Side face x = 1 with data g(u, z): Σ B_nl sin(nπu) sin(lπz)
        // sinh(γx)/sinh γ; ∂_z at u = ½, z = 0, x = ½ is
        // Σ B_nl s_n (lπ) sinh(γ/2)/sinh γ = Σ B_nl s_n (lπ) / (2 cosh(γ/2)).
        let b_all = cell_integrals(&all, |l| l * pi, m);
        let side = bilinear(
            &a_odd,
            &b_all,
            |s, t| 2.0 / (gamma(odd[s], all[t]) / 2.0).cosh(),
            m,
        );
        let mut cells = Vec::with_capacity(5 * m * m);
        for _ in 0..4 {
            cells.extend_from_slice(&side);
        }
        cells.extend_from_slice(&top);
        let abs: Vec<f64> = cells.iter().map(|v| v.abs()).collect();
        let l1 = abs.iter().sum();
        Self {
            m,
            alias: AliasTable::new(&abs),
            cells,
            l1,
        }
    }

    /// Cells per face edge.
    #[must_use]
    pub const fn resolution(&self) -> usize {
        self.m
    }

    /// Signed kernel integrals (see the layout of the struct).
    #[must_use]
    pub fn cells(&self) -> &[f64] {
        &self.cells
    }

    /// `∫|K_B| dS` on the unit cube. For a cube of side `L` the score scale
    /// is `l1() / L`.
    #[must_use]
    pub const fn l1(&self) -> f64 {
        self.l1
    }

    /// The centre of cell `idx` on the unit cube (for tests and quadrature).
    #[must_use]
    pub fn cell_centre(&self, idx: usize) -> [f64; 3] {
        let mm = self.m * self.m;
        let (face, cell) = (idx / mm, idx % mm);
        #[allow(clippy::cast_precision_loss)]
        let mf = self.m as f64;
        #[allow(clippy::cast_precision_loss)]
        let (u, v) = (
            ((cell % self.m) as f64 + 0.5) / mf,
            ((cell / self.m) as f64 + 0.5) / mf,
        );
        Self::place(face, u, v)
    }

    /// The unit-cube point of face-local coordinates `(u, v)` on `face`.
    fn place(face: usize, u: f64, v: f64) -> [f64; 3] {
        match face {
            0 => [0.0, u, v],
            1 => [1.0, u, v],
            2 => [u, 0.0, v],
            3 => [u, 1.0, v],
            _ => [u, v, 1.0],
        }
    }

    /// Samples a point on `∂Q∖B` of the unit cube from `|K_B|`, returning it
    /// and the sign of the kernel there.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> ([f64; 3], f64) {
        let idx = self.alias.sample(rng);
        let sign = self.cells[idx].signum();
        let mm = self.m * self.m;
        let (face, cell) = (idx / mm, idx % mm);
        #[allow(clippy::cast_precision_loss)]
        let mf = self.m as f64;
        #[allow(clippy::cast_precision_loss)]
        let u = ((cell % self.m) as f64 + rng.next_f64()) / mf;
        #[allow(clippy::cast_precision_loss)]
        let v = ((cell / self.m) as f64 + rng.next_f64()) / mf;
        (Self::place(face, u, v), sign)
    }
}

impl Default for FaceTables {
    fn default() -> Self {
        Self::new(crate::green::DEFAULT_RESOLUTION)
    }
}

/// One face to differentiate with respect to (outward growth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeFace {
    /// The box.
    pub box_id: BoxId,
    /// The face.
    pub face: Face,
}

/// Options of [`solve_shape_derivative`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeOptions {
    /// Face samples per face (each runs two kernel walks).
    pub samples: u64,
    /// `None`: every face must have all its edges on reflecting walls (no
    /// geometric edge, design §6.4). `Some(r)`: integrate only over the part
    /// of the face at least `r` from its geometric edges (the result is then
    /// the interior part of the derivative, not the whole of it).
    pub edge_margin: Option<f64>,
}

impl Default for ShapeOptions {
    fn default() -> Self {
        Self {
            samples: 100_000,
            edge_margin: None,
        }
    }
}

/// Derivative statistics of one face: `dC_ij/dn` for every pair of
/// conductor nets seen, from one pass.
#[derive(Debug, Clone, Default)]
pub struct ShapeFaceResult {
    /// Per-sample moments of `dC_ij/dn` (fF/µm), `i ≤ j`.
    pub pairs: Vec<((NetId, NetId), Moments)>,
    /// Per-sample moments of `Σ_j dC_ij/dn` over non-ground conductors `j`
    /// (columns of `cfg.ground_nets` count as ground): the derivative of net
    /// `i`'s ground capacitance (positive convention).
    pub ground: Vec<(NetId, Moments)>,
    /// Face samples.
    pub samples: u64,
    /// Integrated area (µm²): the face, or its interior part with a margin.
    pub area: f64,
    /// Smallest cube half-side used (µm).
    pub min_half_side: f64,
    /// Walks that exceeded `max_steps` (scored as reaching no net).
    pub aborted_walks: u64,
}

impl ShapeFaceResult {
    /// `dC_ij/dn` (fF/µm), symmetric in `i, j`.
    #[must_use]
    pub fn dc(&self, i: NetId, j: NetId) -> Estimate {
        let key = if i <= j { (i, j) } else { (j, i) };
        self.pairs
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(Moments::new().estimate(self.samples), |(_, m)| {
                m.estimate(self.samples)
            })
    }

    /// Derivative of net `i`'s ground capacitance (fF/µm).
    #[must_use]
    pub fn d_ground(&self, i: NetId) -> Estimate {
        self.ground
            .iter()
            .find(|(k, _)| *k == i)
            .map_or(Moments::new().estimate(self.samples), |(_, m)| {
                m.estimate(self.samples)
            })
    }
}

/// Precomputed geometry of one face.
struct FaceGeom {
    net: NetId,
    axis: usize,
    sign: f64,
    t: [usize; 2],
    coord: f64,
    /// Sampling rectangle in the tangential axes.
    lo: [f64; 2],
    hi: [f64; 2],
    /// Geometric edges: `(tangential index 0/1, coordinate)`.
    edges: Vec<(usize, f64)>,
    /// Obstacles for the cube: boxes and mirror images.
    obstacles: Vec<Aabb>,
    /// Dirichlet planes `(axis, coordinate)` (grounded walls, substrate).
    planes: Vec<(usize, f64)>,
    /// Upper bound of the half-side.
    h_cap: f64,
    key: u64,
}

impl FaceGeom {
    fn new(
        geom: &Geometry,
        domain: &BoundaryModel,
        f: &ShapeFace,
        margin: Option<f64>,
    ) -> Result<Self, SolveError> {
        let err =
            |m: String| SolveError::Config(format!("{} face {}: {m}", f.box_id, f.face.as_str()));
        if f.box_id.index() >= geom.boxes().len() {
            return Err(SolveError::Config(format!("unknown {}", f.box_id)));
        }
        let b = *geom.box_(f.box_id);
        let net = geom.box_net(f.box_id);
        let axis = f.face.axis.index();
        let sign = f.face.side.sign();
        let t = match axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let coord = b.face_coord(f.face);
        if !covered_parts(geom, f.box_id, f.face)?.is_empty() {
            return Err(err(
                "partly covered by its own net; shape derivatives need an uncovered face".into(),
            ));
        }
        let bounds = domain.bounds();
        let reflect = domain.reflect();
        let tol = |x: f64| 1e-12 * (1.0 + x.abs());
        let on_wall = |ax: usize, hi: bool, x: f64| {
            bounds.is_some_and(|bb| {
                let (rl, rh) = reflect[ax];
                if hi {
                    rh && (x - bb.max()[ax]).abs() <= tol(x)
                } else {
                    rl && (x - bb.min()[ax]).abs() <= tol(x)
                }
            })
        };
        if on_wall(axis, sign > 0.0, coord) {
            return Err(err("lies on a reflecting wall".into()));
        }
        let mut lo = [b.min()[t[0]], b.min()[t[1]]];
        let mut hi = [b.max()[t[0]], b.max()[t[1]]];
        let mut edges = Vec::new();
        for k in 0..2 {
            if !on_wall(t[k], false, lo[k]) {
                edges.push((k, lo[k]));
            }
            if !on_wall(t[k], true, hi[k]) {
                edges.push((k, hi[k]));
            }
        }
        if !edges.is_empty() {
            let Some(r) = margin else {
                return Err(err(
                    "has geometric edges; their contribution is an open problem (D3 §6.4). \
                     Give an edge margin to integrate the interior only"
                        .into(),
                ));
            };
            if !(r.is_finite() && r > 0.0) {
                return Err(err("edge margin must be positive".into()));
            }
            for &(k, x) in &edges {
                if (x - lo[k]).abs() <= tol(x) {
                    lo[k] += r;
                } else {
                    hi[k] -= r;
                }
            }
            if !(lo[0] < hi[0] && lo[1] < hi[1]) {
                return Err(err(format!("no interior left at edge margin {r} µm")));
            }
        }
        // Obstacles: every other box, and the mirror images of all boxes
        // across the reflecting walls, except mirrors of this box across walls
        // tangential to the face (they continue the face itself).
        let mut obstacles = Vec::new();
        let mirrors = |x: &Aabb| -> Vec<(Aabb, [bool; 3])> {
            let Some(bb) = bounds else {
                return vec![(*x, [false; 3])];
            };
            let mut out = vec![(*x, [false; 3])];
            for ax in 0..3 {
                let (rl, rh) = reflect[ax];
                let cur = out.clone();
                for (c, flags) in cur {
                    for (on, w) in [(rl, bb.min()[ax]), (rh, bb.max()[ax])] {
                        if on {
                            let (mut mn, mut mx) = (c.min(), c.max());
                            (mn[ax], mx[ax]) = (2.0 * w - c.max()[ax], 2.0 * w - c.min()[ax]);
                            let mut fl = flags;
                            fl[ax] = true;
                            if let Ok(m) = Aabb::new(mn, mx) {
                                out.push((m, fl));
                            }
                        }
                    }
                }
            }
            out
        };
        for (i, x) in geom.boxes().iter().enumerate() {
            let own = i == f.box_id.index();
            for (m, flags) in mirrors(x) {
                if own && !flags[axis] {
                    continue; // the box itself or its continuation along the face
                }
                obstacles.push(m);
            }
        }
        let mut planes: Vec<(usize, f64)> = domain
            .ground_faces()
            .iter()
            .map(|&(a, c, _)| (a, c))
            .collect();
        if let Some(z) = domain.substrate_z() {
            planes.push((2, z));
        }
        let h_cap = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(b.extent()[axis]);
        let mut key = SHAPE_TAG;
        for v in [u64::from(f.box_id.raw()), f.face.index() as u64] {
            key = splitmix64(&mut key) ^ v;
        }
        Ok(Self {
            net,
            axis,
            sign,
            t,
            coord,
            lo,
            hi,
            edges,
            obstacles,
            planes,
            h_cap,
            key,
        })
    }

    fn area(&self) -> f64 {
        (self.hi[0] - self.lo[0]) * (self.hi[1] - self.lo[1])
    }

    /// The point of the face at tangential coordinates `(u, v)`.
    fn point(&self, u: f64, v: f64) -> [f64; 3] {
        let mut p = [0.0; 3];
        p[self.axis] = self.coord;
        p[self.t[0]] = u;
        p[self.t[1]] = v;
        p
    }

    /// Largest half-side of a cube resting on the face at `x0` that keeps its
    /// bottom inside the face and its volume free of conductors and Dirichlet
    /// planes.
    fn half_side(&self, x0: [f64; 3]) -> f64 {
        let mut h = self.h_cap;
        for &(k, e) in &self.edges {
            h = h.min((x0[self.t[k]] - e).abs());
        }
        for o in &self.obstacles {
            let ng = if self.sign > 0.0 {
                o.min()[self.axis] - self.coord
            } else {
                self.coord - o.max()[self.axis]
            };
            let behind = if self.sign > 0.0 {
                o.max()[self.axis] <= self.coord
            } else {
                o.min()[self.axis] >= self.coord
            };
            if behind {
                continue;
            }
            let tg = self
                .t
                .iter()
                .map(|&a| (o.min()[a] - x0[a]).max(x0[a] - o.max()[a]))
                .fold(f64::NEG_INFINITY, f64::max);
            h = h.min(tg.max(0.5 * ng));
        }
        for &(a, c) in &self.planes {
            if a == self.axis {
                let d = self.sign * (c - self.coord);
                if d > 0.0 {
                    h = h.min(0.5 * d);
                }
            } else {
                h = h.min((x0[a] - c).abs());
            }
        }
        h
    }
}

/// Whether [`solve_shape_derivative`] accepts each face of `faces` without an
/// edge margin: `Ok(())`, or the reason it does not (stratified dielectric,
/// geometric edges, partly covered by its own net, on a reflecting wall).
///
/// # Errors
///
/// As [`solve`](crate::solve) (invalid targets or configuration).
pub fn shape_support(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    faces: &[ShapeFace],
) -> Result<Vec<Result<(), String>>, SolveError> {
    let su = setup(geom, targets, cfg)?;
    if su.boxes.is_some() {
        return Ok(faces
            .iter()
            .map(|_| Err("dielectric boxes and conformal shells are not supported".into()))
            .collect());
    }
    if su.medium.is_some() {
        return Ok(faces
            .iter()
            .map(|_| Err("stratified dielectrics are not supported".into()))
            .collect());
    }
    Ok(faces
        .iter()
        .map(|f| {
            FaceGeom::new(geom, &su.domain, f, None)
                .map(|_| ())
                .map_err(|e| {
                    let m = match e {
                        SolveError::Config(m) => m,
                        other => other.to_string(),
                    };
                    let prefix = format!("{} face {}: ", f.box_id, f.face.as_str());
                    let m = m.strip_prefix(&prefix).unwrap_or(&m);
                    if m.starts_with("has geometric edges") {
                        "has geometric edges (the shape derivative needs every edge of the \
                         face on a reflecting wall)"
                            .to_owned()
                    } else {
                        m.to_owned()
                    }
                })
        })
        .collect())
}

/// Derivatives of the Maxwell matrix with respect to the outward growth of
/// each face in `faces`, for every pair of conductor nets at once.
///
/// `targets` set the stopping distance as in [`solve`](crate::solve) (use
/// the nets of interest).
///
/// # Errors
///
/// As [`solve`](crate::solve); stratified dielectrics; faces that are
/// partly covered by their own net, lie on a reflecting wall, or have
/// geometric edges without an edge margin.
pub fn solve_shape_derivative(
    geom: &Geometry,
    targets: &[NetId],
    cfg: &SolverConfig,
    faces: &[ShapeFace],
    opts: ShapeOptions,
) -> Result<Vec<ShapeFaceResult>, SolveError> {
    let su = setup(geom, targets, cfg)?;
    if su.boxes.is_some() {
        return Err(SolveError::Config(
            "shape derivatives do not support dielectric boxes or conformal shells yet (design M2c)".into(),
        ));
    }
    if su.medium.is_some() {
        return Err(SolveError::Config(
            "shape derivatives do not support stratified dielectrics yet".into(),
        ));
    }
    if opts.samples < 2 {
        return Err(SolveError::Config(
            "shape derivatives need at least two samples".into(),
        ));
    }
    let fgs: Vec<FaceGeom> = faces
        .iter()
        .map(|f| FaceGeom::new(geom, &su.domain, f, opts.edge_margin))
        .collect::<Result<_, _>>()?;
    let tables = FaceTables::new(cfg.walk.green_resolution);
    let walker = su.walker(geom, cfg);
    let eps = su.stack.eps()[0] * EPS0_FF_PER_UM;
    const BATCH: u64 = 4096;
    Ok(fgs
        .iter()
        .map(|fg| {
            let batches = opts.samples.div_ceil(BATCH);
            let parts: Vec<BatchAcc> = (0..batches)
                .into_par_iter()
                .map(|bi| {
                    let n = BATCH.min(opts.samples - bi * BATCH);
                    let mut key = cfg.seed ^ fg.key;
                    key = splitmix64(&mut key) ^ bi;
                    let mut rng = Rng::from_seed(splitmix64(&mut key));
                    run_face_batch(fg, &walker, &tables, eps, n, &cfg.ground_nets, &mut rng)
                })
                .collect();
            let mut acc = BatchAcc::default();
            for p in &parts {
                acc.merge(p);
            }
            ShapeFaceResult {
                pairs: acc.pairs.into_iter().collect(),
                ground: acc.ground.into_iter().collect(),
                samples: opts.samples,
                area: fg.area(),
                min_half_side: acc.min_h,
                aborted_walks: acc.aborted,
            }
        })
        .collect())
}

struct BatchAcc {
    pairs: BTreeMap<(NetId, NetId), Moments>,
    ground: BTreeMap<NetId, Moments>,
    min_h: f64,
    aborted: u64,
}

impl Default for BatchAcc {
    fn default() -> Self {
        Self {
            pairs: BTreeMap::new(),
            ground: BTreeMap::new(),
            min_h: f64::INFINITY,
            aborted: 0,
        }
    }
}

impl BatchAcc {
    fn merge(&mut self, o: &Self) {
        for (k, m) in &o.pairs {
            self.pairs.entry(*k).or_default().merge(m);
        }
        for (k, m) in &o.ground {
            self.ground.entry(*k).or_default().merge(m);
        }
        self.min_h = self.min_h.min(o.min_h);
        self.aborted += o.aborted;
    }
}

/// One kernel hop from the cube of half-side `h` resting on the face at
/// `x0`: the per-net scores `g_k` (sparse; the own net always present).
fn hop_scores(
    fg: &FaceGeom,
    walker: &Walker<'_>,
    tables: &FaceTables,
    x0: [f64; 3],
    h: f64,
    rng: &mut Rng,
    aborted: &mut u64,
) -> BTreeMap<NetId, f64> {
    let (u, sign) = tables.sample(rng);
    let side = 2.0 * h;
    let mut y = x0;
    y[fg.t[0]] += (u[0] - 0.5) * side;
    y[fg.t[1]] += (u[1] - 0.5) * side;
    y[fg.axis] += fg.sign * u[2] * side;
    walker.domain.fold(&mut y);
    let scale = sign * tables.l1() / side;
    let mut g = BTreeMap::new();
    g.insert(fg.net, -scale);
    let end = walker.run(y, rng);
    match end.sink {
        Some(Sink::Net(n)) => *g.entry(n).or_insert(0.0) += scale * end.sign,
        Some(_) => {}
        None => *aborted += 1,
    }
    g
}

fn run_face_batch(
    fg: &FaceGeom,
    walker: &Walker<'_>,
    tables: &FaceTables,
    eps: f64,
    n: u64,
    ground_nets: &[NetId],
    rng: &mut Rng,
) -> BatchAcc {
    let mut acc = BatchAcc::default();
    let w = fg.area() * eps;
    for _ in 0..n {
        let u = fg.lo[0] + rng.next_f64() * (fg.hi[0] - fg.lo[0]);
        let v = fg.lo[1] + rng.next_f64() * (fg.hi[1] - fg.lo[1]);
        let x0 = fg.point(u, v);
        let h = fg.half_side(x0);
        acc.min_h = acc.min_h.min(h);
        let g1 = hop_scores(fg, walker, tables, x0, h, rng, &mut acc.aborted);
        let g2 = hop_scores(fg, walker, tables, x0, h, rng, &mut acc.aborted);
        let mut nets: Vec<NetId> = g1.keys().chain(g2.keys()).copied().collect();
        nets.sort_unstable();
        nets.dedup();
        let get = |g: &BTreeMap<NetId, f64>, k: NetId| g.get(&k).copied().unwrap_or(0.0);
        let mut rows: BTreeMap<NetId, f64> = BTreeMap::new();
        for (a, &i) in nets.iter().enumerate() {
            for &j in &nets[a..] {
                let y = w * 0.5 * (get(&g1, i) * get(&g2, j) + get(&g1, j) * get(&g2, i));
                if y != 0.0 {
                    acc.pairs.entry((i, j)).or_default().add(y);
                }
                // A ground net's column is part of the ground, not a
                // coupling: it does not enter the row sum.
                if !ground_nets.contains(&j) {
                    *rows.entry(i).or_insert(0.0) += y;
                }
                if i != j && !ground_nets.contains(&i) {
                    *rows.entry(j).or_insert(0.0) += y;
                }
            }
        }
        for (i, y) in rows {
            if y != 0.0 {
                acc.ground.entry(i).or_default().add(y);
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Σ_cells K · f(centre)` for harmonic `f` reproduces `∂_z f(½,½,0)`
    /// (the midpoint rule over cells is the only error, O(1/m²)).
    #[test]
    fn reproduces_harmonic_polynomials() {
        let t = FaceTables::new(64);
        let n = t.cells().len();
        let quad = |f: &dyn Fn([f64; 3]) -> f64, at_b: f64| -> f64 {
            (0..n)
                .map(|i| t.cells()[i] * (f(t.cell_centre(i)) - at_b))
                .sum()
        };
        // f, ∂_z f at (½,½,0), f on B (must be constant there).
        type Case<'a> = (&'a str, &'a dyn Fn([f64; 3]) -> f64, f64, f64);
        let cases: [Case; 6] = [
            ("z", &|p| p[2], 1.0, 0.0),
            ("x z", &|p| p[0] * p[2], 0.5, 0.0),
            ("1 + z(y − ½)", &|p| 1.0 + p[2] * (p[1] - 0.5), 0.0, 1.0),
            (
                "1 + z(1 − 2y + x)",
                &|p| 1.0 + p[2] * (1.0 - 2.0 * p[1] + p[0]),
                0.5,
                1.0,
            ),
            ("x y z", &|p| p[0] * p[1] * p[2], 0.25, 0.0),
            (
                "z³ − 3 z x²",
                &|p| p[2].powi(3) - 3.0 * p[2] * p[0] * p[0],
                -0.75,
                0.0,
            ),
        ];
        for (name, f, exact, at_b) in cases {
            let v = quad(f, at_b);
            assert!((v - exact).abs() < 2e-3, "{name}: {v} vs {exact}");
        }
        // A constant has no normal derivative.
        assert!(quad(&|_| 1.0, 1.0).abs() < 1e-12);
    }

    /// `∫ K_B` over the five faces equals `∂_z` of the solution with data 1
    /// on them and 0 on `B`, which is positive; and the kernel is bounded
    /// (the largest cell density stays finite as the grid is refined).
    #[test]
    fn kernel_is_bounded() {
        for m in [16, 32, 64] {
            let t = FaceTables::new(m);
            #[allow(clippy::cast_precision_loss)]
            let area = 1.0 / (m * m) as f64;
            let max_density = t.cells().iter().fold(0.0_f64, |a, v| a.max(v.abs() / area));
            assert!(max_density < 10.0, "m = {m}: {max_density}");
            assert!(t.cells().iter().sum::<f64>() > 0.0);
        }
    }
}
