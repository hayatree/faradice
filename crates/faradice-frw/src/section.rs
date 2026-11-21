//! Window-flux kernels of boxes with a 2D cross-section (design M2f).
//!
//! Canonical frame: the box `[−W, W] × [0, U] × [0, V]` in `(w, u, v)`. ε
//! does not depend on `w` and is constant on the rectangles of a partition of
//! the cross-section by planes `u = const` and `v = const`. The window is the
//! smooth weight `ω(w) ω(v)` (see [`window`]) on the plane `u = g`, inside
//! the box.
//!
//! The kernel `F` on the box surface gives the weighted flux
//! `∫ ω ε (−∂φ/∂u) dA` over the plane of any `φ` harmonic in the box
//! (`∇·(ε∇φ) = 0`) as `∫ F φ`. It is computed from a node-based
//! finite-volume scheme on a tensor grid with nodes on every plane and the
//! box's walls (uniform along `w`): the flux is a linear functional of the
//! interior nodes, and `F` is the boundary coupling of the adjoint solution.
//! The discrete sine transform along `w` splits the system exactly into 2D
//! systems, one per odd mode (the window is even), each solved by conjugate
//! gradients, preconditioned by
//! the constant-coefficient operator on the same grid (inverted exactly by
//! the fast diagonalization method). `F` is the boundary coupling of the
//! adjoint solution. Three grid levels give an accuracy gate as for the edge
//! cubes. The window is smooth, so the functional needs no grid lines and
//! converges at second order.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, OnceLock, RwLock};

use crate::alias::AliasTable;
use crate::dense;
use crate::rng::Rng;

/// Largest accepted L1 error estimate of a table, relative to its L1 norm.
/// The estimate is that of the coarser extrapolation; the measured error of
/// the table is about a third of it (design M2f §4.3).
pub const MAX_ERROR: f64 = 3e-4;

/// Sub-cells per table cell and axis for sampling within a cell (the finest
/// level's grid cells).
const SUB: usize = 4;

/// Relative residual at which the conjugate gradients stop.
const CG_TOLERANCE: f64 = 1e-12;

/// Why a kernel was not built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectionReject {
    /// The observed convergence order is below 1.
    Order(f64),
    /// The error estimate exceeds [`MAX_ERROR`].
    Error(f64),
    /// The conjugate gradients did not converge.
    Solver,
}

/// The smooth step `S(t)` on `[0, 1]`: `S(t) + S(1 − t) = 1`, all
/// derivatives vanish at both ends.
fn smooth_step(t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let f = |x: f64| (-1.0 / x).exp();
    let a = f(t);
    a / (a + f(1.0 - t))
}

/// The window of half-support `p` centred at 0: `S(1 − |x|/p)`. Windows at
/// spacing `p` add up to one, and `∫ ω = p`.
#[must_use]
pub fn window(x: f64, p: f64) -> f64 {
    smooth_step(1.0 - x.abs() / p)
}

/// The configuration of a window kernel (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct SectionSpec {
    /// Half-length `W` of the box along the uniform axis `w`.
    pub half_w: f64,
    /// Extent `U` of the box along `u` (the window's normal).
    pub u: f64,
    /// Extent `V` of the box along `v`.
    pub v: f64,
    /// Interface planes `u = const` strictly inside the box, ascending.
    pub u_planes: Vec<f64>,
    /// Interface planes `v = const` strictly inside the box, ascending.
    pub v_planes: Vec<f64>,
    /// ε of the rectangles, `eps[iu * (v_planes.len() + 1) + iv]` with `iu`
    /// (`iv`) the number of `u` (`v`) planes below the rectangle.
    pub eps: Vec<f64>,
    /// Position `g` of the window's plane; not on a `u` plane.
    pub gauss: f64,
    /// Half-supports of the window along `w` (centred at 0) and `v`
    /// (centred at `V / 2`).
    pub window: [f64; 2],
    /// Number of table cells along `w`.
    pub w_cells: usize,
    /// Target number of table cells along `u` and `v`.
    pub cells: usize,
}

/// Cell bounds of the base grid (table cells) of one axis.
fn base_grid(len: f64, breaks: &[f64], cells: usize) -> Vec<f64> {
    let mut b: Vec<f64> = vec![0.0, len];
    b.extend(breaks.iter().copied().filter(|&x| x > 0.0 && x < len));
    b.sort_by(f64::total_cmp);
    b.dedup_by(|x, y| x.to_bits() == y.to_bits());
    #[allow(clippy::cast_precision_loss)]
    let target = len / cells as f64;
    let mut out = vec![0.0];
    for w in b.windows(2) {
        let d = w[1] - w[0];
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = ((d / target).ceil() as usize).max(1);
        for i in 1..=k {
            #[allow(clippy::cast_precision_loss)]
            let x = if i == k {
                w[1]
            } else {
                (d * i as f64).mul_add(1.0 / k as f64, w[0])
            };
            out.push(x);
        }
    }
    out
}

/// Each base cell split into `r` equal cells; also the base cell of each fine
/// cell.
fn refine(base: &[f64], r: usize) -> (Vec<f64>, Vec<usize>) {
    let mut nodes = vec![base[0]];
    let mut owner = Vec::new();
    for (c, w) in base.windows(2).enumerate() {
        for i in 1..=r {
            #[allow(clippy::cast_precision_loss)]
            let x = if i == r {
                w[1]
            } else {
                ((w[1] - w[0]) * i as f64).mul_add(1.0 / r as f64, w[0])
            };
            nodes.push(x);
            owner.push(c);
        }
    }
    (nodes, owner)
}

/// Cyclic Jacobi eigen decomposition of a dense symmetric matrix (row-major,
/// `n × n`): eigenvalues and eigenvectors as the rows of the returned `qᵀ`
/// (stored transposed so that each rotation updates two contiguous rows).
fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut qt = vec![0.0; n * n];
    for i in 0..n {
        qt[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        let diag: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum();
        if off <= 1e-30 * diag {
            break;
        }
        for p in 0..n {
            for r in p + 1..n {
                let apr = a[p * n + r];
                if apr.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[r * n + r] - a[p * n + p]) / (2.0 * apr);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akr = a[k * n + r];
                    a[k * n + p] = c * akp - s * akr;
                    a[k * n + r] = s * akp + c * akr;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let ark = a[r * n + k];
                    a[p * n + k] = c * apk - s * ark;
                    a[r * n + k] = s * apk + c * ark;
                }
                let (head, tail) = qt.split_at_mut(r * n);
                let (qp, qr) = (&mut head[p * n..(p + 1) * n], &mut tail[..n]);
                for (qkp, qkr) in qp.iter_mut().zip(qr.iter_mut()) {
                    let (vp, vr) = (*qkp, *qkr);
                    *qkp = c * vp - s * vr;
                    *qkr = s * vp + c * vr;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), qt)
}

/// The fast-diagonalization factors of one axis: `X` (row-major, interior
/// nodes × modes) with `Xᵀ M X = I`, `Xᵀ K X = Λ` for the 1D stiffness `K`
/// and lumped mass `M` of the node spacing `d`.
#[derive(Debug)]
struct Axis1 {
    x: Vec<f64>,
    xt: Vec<f64>,
    lambda: Vec<f64>,
}

impl Axis1 {
    fn new(d: &[f64]) -> Self {
        let n = d.len() - 1;
        let mass: Vec<f64> = (0..n).map(|i| 0.5 * (d[i] + d[i + 1])).collect();
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            t[i * n + i] = (1.0 / d[i] + 1.0 / d[i + 1]) / mass[i];
            if i + 1 < n {
                let o = -1.0 / d[i + 1] / (mass[i] * mass[i + 1]).sqrt();
                t[i * n + i + 1] = o;
                t[(i + 1) * n + i] = o;
            }
        }
        let (lambda, qt) = jacobi(t, n);
        let mut x = vec![0.0; n * n];
        for i in 0..n {
            let s = 1.0 / mass[i].sqrt();
            for j in 0..n {
                x[i * n + j] = qt[j * n + i] * s;
            }
        }
        let mut xt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                xt[j * n + i] = x[i * n + j];
            }
        }
        Self { x, xt, lambda }
    }
}

/// The [`Axis1`] factors of every node spacing seen so far. Kernels of one
/// medium share most of their grids, and the factors (a serial eigen
/// decomposition) are a large part of a kernel's wall time.
#[derive(Debug, Default)]
pub(crate) struct AxisCache {
    map: RwLock<HashMap<Vec<u64>, AxisSlot>>,
}

/// One spacing's factors, built by the first kernel that needs them.
type AxisSlot = Arc<OnceLock<Arc<Axis1>>>;

impl AxisCache {
    fn get(&self, d: &[f64]) -> Arc<Axis1> {
        let key: Vec<u64> = d.iter().map(|x| x.to_bits()).collect();
        let found = self.map.read().expect("axis cache lock").get(&key).cloned();
        let slot = found.unwrap_or_else(|| {
            let mut w = self.map.write().expect("axis cache lock");
            Arc::clone(w.entry(key).or_default())
        });
        Arc::clone(slot.get_or_init(|| Arc::new(Axis1::new(d))))
    }

    /// The factors of `du` and `dv`, the second on a thread of the budget
    /// when one is free.
    fn pair(&self, du: &[f64], dv: &[f64]) -> (Arc<Axis1>, Arc<Axis1>) {
        let extra = Budget::take(1);
        if extra.count == 0 {
            return (self.get(du), self.get(dv));
        }
        std::thread::scope(|sc| {
            let other = std::thread::Builder::new().spawn_scoped(sc, || self.get(dv));
            let au = self.get(du);
            let av = other
                .ok()
                .and_then(|h| h.join().ok())
                .unwrap_or_else(|| self.get(dv));
            (au, av)
        })
    }
}

/// One grid level.
struct Level {
    /// Interior node counts along `u` and `v` (`nu − 1`, `nv − 1`).
    iu: usize,
    iv: usize,
    du: Vec<f64>,
    dv: Vec<f64>,
    /// Fine cell ε, `eps[i * nv + k]`.
    eps: Vec<f64>,
    nv: usize,
    /// Flux coefficients on the `(nu + 1) × (nv + 1)` node array.
    cu: Vec<f64>,
    cv: Vec<f64>,
    e4: Vec<f64>,
}

impl Level {
    fn new(un: &[f64], vn: &[f64], spec: &SectionSpec) -> Self {
        let nu = un.len() - 1;
        let nv = vn.len() - 1;
        let du: Vec<f64> = un.windows(2).map(|w| w[1] - w[0]).collect();
        let dv: Vec<f64> = vn.windows(2).map(|w| w[1] - w[0]).collect();
        let nvp = spec.v_planes.len() + 1;
        let mut eps = vec![0.0; nu * nv];
        for i in 0..nu {
            let um = 0.5 * (un[i] + un[i + 1]);
            let a = spec.u_planes.iter().filter(|&&p| p < um).count();
            for k in 0..nv {
                let vm = 0.5 * (vn[k] + vn[k + 1]);
                let b = spec.v_planes.iter().filter(|&&p| p < vm).count();
                eps[i * nv + k] = spec.eps[a * nvp + b];
            }
        }
        let s = nv + 1;
        let cell = |i: usize, k: usize| eps[i * nv + k];
        let mut cu = vec![0.0; (nu + 1) * s];
        let mut cv = vec![0.0; (nu + 1) * s];
        let mut e4 = vec![0.0; (nu + 1) * s];
        for i in 0..nu {
            for k in 1..nv {
                cu[i * s + k] =
                    0.5f64.mul_add(cell(i, k - 1) * dv[k - 1], 0.5 * cell(i, k) * dv[k]) / du[i];
            }
        }
        for i in 1..nu {
            for k in 0..nv {
                cv[i * s + k] =
                    0.5f64.mul_add(cell(i - 1, k) * du[i - 1], 0.5 * cell(i, k) * du[i]) / dv[k];
            }
        }
        for i in 1..nu {
            for k in 1..nv {
                e4[i * s + k] = 0.25
                    * (cell(i - 1, k - 1) * du[i - 1] * dv[k - 1]
                        + cell(i, k - 1) * du[i] * dv[k - 1]
                        + cell(i - 1, k) * du[i - 1] * dv[k]
                        + cell(i, k) * du[i] * dv[k]);
            }
        }
        Self {
            iu: nu - 1,
            iv: nv - 1,
            du,
            dv,
            eps,
            nv,
            cu,
            cv,
            e4,
        }
    }

    /// `out = (A + μ E) x` on the node array (boundary entries stay 0).
    fn apply(&self, mu: f64, x: &[f64], out: &mut [f64]) {
        let s = self.nv + 1;
        for i in 1..=self.iu {
            for k in 1..=self.iv {
                let j = i * s + k;
                let v = x[j];
                out[j] = self.cu[j - s] * (v - x[j - s])
                    + self.cu[j] * (v - x[j + s])
                    + self.cv[j - 1] * (v - x[j - 1])
                    + self.cv[j] * (v - x[j + 1])
                    + self.e4[j] * mu * v;
            }
        }
    }
}

/// Preconditioned conjugate gradients for one level.
struct Solver<'a> {
    level: &'a Level,
    au: &'a Axis1,
    av: &'a Axis1,
    mean: f64,
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    b1: Vec<f64>,
    b2: Vec<f64>,
}

impl<'a> Solver<'a> {
    fn new(level: &'a Level, au: &'a Axis1, av: &'a Axis1, mean: f64) -> Self {
        let s = (level.iu + 2) * (level.iv + 2);
        let nn = level.iu * level.iv;
        Self {
            level,
            au,
            av,
            mean,
            x: vec![0.0; s],
            r: vec![0.0; s],
            z: vec![0.0; s],
            p: vec![0.0; s],
            ap: vec![0.0; s],
            b1: vec![0.0; nn],
            b2: vec![0.0; nn],
        }
    }

    /// `z = P⁻¹ r`, `P = mean · (K_u ⊗ M_v + M_u ⊗ K_v + μ M_u ⊗ M_v)`.
    fn precondition(&mut self, mu: f64) {
        let (iu, iv) = (self.level.iu, self.level.iv);
        let s = iv + 2;
        for i in 0..iu {
            for k in 0..iv {
                self.b1[i * iv + k] = self.r[(i + 1) * s + k + 1];
            }
        }
        // Y = Xuᵀ R Xv.
        dense::product(&self.au.xt, &self.b1, &mut self.b2, [iu, iu, iv], [1, 0]);
        dense::product(&self.b2, &self.av.x, &mut self.b1, [iu, iv, iv], [1, 0]);
        for a in 0..iu {
            for b in 0..iv {
                self.b1[a * iv + b] /= self.mean * (self.au.lambda[a] + self.av.lambda[b] + mu);
            }
        }
        // Z = Xu Y Xvᵀ.
        dense::product(&self.au.x, &self.b1, &mut self.b2, [iu, iu, iv], [1, 0]);
        dense::product(&self.b2, &self.av.xt, &mut self.b1, [iu, iv, iv], [1, 0]);
        for i in 0..iu {
            for k in 0..iv {
                self.z[(i + 1) * s + k + 1] = self.b1[i * iv + k];
            }
        }
    }

    fn solve(&mut self, mu: f64, rhs: &[f64]) -> Result<&[f64], SectionReject> {
        self.x.fill(0.0);
        self.r.copy_from_slice(rhs);
        let norm_b = dot(rhs, rhs).sqrt();
        if norm_b == 0.0 {
            return Ok(&self.x);
        }
        self.precondition(mu);
        self.p.copy_from_slice(&self.z);
        let mut rz = dot(&self.r, &self.z);
        for _ in 0..10_000 {
            self.level.apply(mu, &self.p, &mut self.ap);
            let alpha = rz / dot(&self.p, &self.ap);
            for k in 0..self.x.len() {
                self.x[k] += alpha * self.p[k];
                self.r[k] -= alpha * self.ap[k];
            }
            let res = dot(&self.r, &self.r).sqrt();
            if !res.is_finite() {
                break;
            }
            if res <= CG_TOLERANCE * norm_b {
                return Ok(&self.x);
            }
            self.precondition(mu);
            let rz_new = dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..self.p.len() {
                self.p[k] = beta * self.p[k] + self.z[k];
            }
        }
        Err(SectionReject::Solver)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Table cells: bounds per axis and the face layout.
#[derive(Debug, Clone)]
struct Cells {
    w: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
}

impl Cells {
    fn counts(&self) -> [usize; 3] {
        [self.w.len() - 1, self.u.len() - 1, self.v.len() - 1]
    }

    /// Offsets of the six faces `[w−, w+, u−, u+, v−, v+]` and the total.
    fn offsets(&self) -> [usize; 7] {
        let [nw, nu, nv] = self.counts();
        let sizes = [nu * nv, nu * nv, nw * nv, nw * nv, nw * nu, nw * nu];
        let mut o = [0; 7];
        for f in 0..6 {
            o[f + 1] = o[f] + sizes[f];
        }
        o
    }
}

/// A window kernel (see the module docs).
#[derive(Debug, Clone)]
pub struct SectionKernel {
    cells: Cells,
    /// Signed cell integrals of `F`.
    values: Vec<f64>,
    /// Per cell, the cumulative distribution of its `SUB²` sub-cells.
    sub: Vec<f64>,
    l1: f64,
    table: AliasTable,
    /// The positive and negative parts (for antithetic pairs).
    parts: [AliasTable; 2],
    error: f64,
    /// L1 differences of consecutive levels (for diagnostics).
    diffs: [f64; 2],
    modes: usize,
}

impl SectionKernel {
    /// Builds the kernel of `spec` (lengths in any unit; `F` scales with it).
    ///
    /// # Errors
    ///
    /// [`SectionReject`] if the levels fail the accuracy gate.
    ///
    /// # Panics
    ///
    /// Panics on an inconsistent spec.
    pub fn new(spec: &SectionSpec) -> Result<Self, SectionReject> {
        Self::with_axes(spec, &AxisCache::default())
    }

    /// [`Self::new`] with the axis factors taken from (and added to) `axes`.
    pub(crate) fn with_axes(spec: &SectionSpec, axes: &AxisCache) -> Result<Self, SectionReject> {
        let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert!(spec.w_cells >= 2);
        assert!(spec.window[0] > 0.0 && spec.window[0] < spec.half_w);
        assert!(spec.window[1] > 0.0 && 2.0 * spec.window[1] < spec.v);
        assert_eq!(
            spec.eps.len(),
            (spec.u_planes.len() + 1) * (spec.v_planes.len() + 1)
        );
        let mut ub = spec.u_planes.clone();
        ub.push(spec.gauss);
        // Table cells: the base grid split in two (the coarsest level's grid
        // cells), which keeps the bias of cell-uniform sampling small.
        let wc = 2 * spec.w_cells;
        #[allow(clippy::cast_precision_loss)]
        let w: Vec<f64> = (0..=wc)
            .map(|i| (2.0 * spec.half_w * i as f64).mul_add(1.0 / wc as f64, -spec.half_w))
            .collect();
        let cells = Cells {
            w,
            u: refine(&base_grid(spec.u, &ub, spec.cells), 2).0,
            v: refine(&base_grid(spec.v, &spec.v_planes, spec.cells), 2).0,
        };
        let sum =
            |v: Vec<f64>, rr: usize| -> Vec<f64> { v.chunks(rr).map(|c| c.iter().sum()).collect() };
        let p1 = sum(level(spec, &cells, 1, threads, axes)?, 1);
        let p2 = sum(level(spec, &cells, 2, threads, axes)?, 4);
        let fine = level(spec, &cells, SUB, threads, axes)?;
        let p4 = sum(fine.clone(), SUB * SUB);
        let d12 = l1(&p1, &p2);
        let d24 = l1(&p2, &p4);
        let norm: f64 = p4.iter().map(|v| v.abs()).sum();
        let order = (d12 / d24).log2();
        let (values, error) = if d24 <= 1e-14 * norm {
            (p4, d24)
        } else if (order - 2.0).abs() <= 0.25 {
            let rich = |a: &[f64], b: &[f64]| -> Vec<f64> {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| 4.0f64.mul_add(*y, -x) / 3.0)
                    .collect()
            };
            let r1 = rich(&p1, &p2);
            let r2 = rich(&p2, &p4);
            let e = l1(&r1, &r2);
            (r2, e)
        } else if order >= 1.0 {
            (p4, d24)
        } else {
            return Err(SectionReject::Order(order));
        };
        if error > MAX_ERROR * norm {
            return Err(SectionReject::Error(error / norm));
        }
        let abs: Vec<f64> = values.iter().map(|v| v.abs()).collect();
        let l1n = abs.iter().sum();
        // Within a cell, points follow the finest level's sub-cells where
        // they all have the cell's sign (uniform otherwise): this removes most
        // of the bias of cell-uniform sampling.
        let nsub = SUB * SUB;
        let mut sub = vec![0.0; values.len() * nsub];
        for (c, &v) in values.iter().enumerate() {
            let parts = &fine[c * nsub..(c + 1) * nsub];
            let same = parts.iter().all(|x| x * v >= 0.0);
            let total: f64 = parts.iter().map(|x| x.abs()).sum();
            let mut acc = 0.0;
            for (k, x) in parts.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let w = if same && total > 0.0 {
                    x.abs() / total
                } else {
                    1.0 / nsub as f64
                };
                acc += w;
                sub[c * nsub + k] = acc;
            }
        }
        let pos: Vec<f64> = values.iter().map(|v| v.max(0.0)).collect();
        let neg: Vec<f64> = values.iter().map(|v| (-v).max(0.0)).collect();
        Ok(Self {
            sub,
            table: AliasTable::new(&abs),
            parts: [AliasTable::new(&pos), AliasTable::new(&neg)],
            cells,
            values,
            l1: l1n,
            error: error / norm,
            diffs: [d12 / norm, d24 / norm],
            modes: 4 * spec.w_cells,
        })
    }

    /// `‖F‖₁`.
    #[must_use]
    pub const fn l1(&self) -> f64 {
        self.l1
    }

    /// Relative L1 error estimate.
    #[must_use]
    pub const fn error_estimate(&self) -> f64 {
        self.error
    }

    /// Relative L1 differences of levels 1–2 and 2–4.
    #[must_use]
    pub const fn level_differences(&self) -> [f64; 2] {
        self.diffs
    }

    /// Number of odd sine modes along `w` at the finest level.
    #[must_use]
    pub const fn modes(&self) -> usize {
        self.modes
    }

    /// `∫ F φ` as the sampled estimator sees it: `φ` averaged over each
    /// sub-cell by a midpoint rule with `samples²` points, weighted by the
    /// cell's sub-cell distribution. For tests.
    #[must_use]
    pub fn apply(&self, phi: &dyn Fn([f64; 3]) -> f64, samples: usize) -> f64 {
        let nsub = SUB * SUB;
        let mut total = 0.0;
        for (c, &v) in self.values.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let cdf = &self.sub[c * nsub..(c + 1) * nsub];
            for k in 0..nsub {
                let w = cdf[k] - if k == 0 { 0.0 } else { cdf[k - 1] };
                let mut acc = 0.0;
                for a in 0..samples {
                    for b in 0..samples {
                        #[allow(clippy::cast_precision_loss)]
                        let f = |i: usize, x: usize| {
                            ((i * samples + x) as f64 + 0.5) / (SUB * samples) as f64
                        };
                        acc += phi(self.point(c, f(k % SUB, a), f(k / SUB, b)));
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                let n = (samples * samples) as f64;
                total += v * w * acc / n;
            }
        }
        total
    }

    /// The point at fractions `(s, t)` of table cell `c`, as `(w, u, v)`.
    fn point(&self, c: usize, s: f64, t: f64) -> [f64; 3] {
        let off = self.cells.offsets();
        let [nw, nu, _] = self.cells.counts();
        let f = (0..6).rfind(|&f| off[f] <= c).expect("a face");
        let r = c - off[f];
        let lerp = |b: &[f64], i: usize, x: f64| (b[i + 1] - b[i]).mul_add(x, b[i]);
        let side = |b: &[f64], f: usize| if f % 2 == 0 { b[0] } else { b[b.len() - 1] };
        let cl = &self.cells;
        match f / 2 {
            0 => [
                side(&cl.w, f),
                lerp(&cl.u, r % nu, s),
                lerp(&cl.v, r / nu, t),
            ],
            1 => [
                lerp(&cl.w, r % nw, s),
                side(&cl.u, f),
                lerp(&cl.v, r / nw, t),
            ],
            _ => [
                lerp(&cl.w, r % nw, s),
                lerp(&cl.u, r / nw, t),
                side(&cl.v, f),
            ],
        }
    }

    /// Fractions `(s, t)` within cell `c`: a sub-cell from the cell's
    /// distribution, uniform within it.
    fn fractions(&self, c: usize, rng: &mut Rng) -> (f64, f64) {
        let nsub = SUB * SUB;
        let x = rng.next_f64();
        let cdf = &self.sub[c * nsub..(c + 1) * nsub];
        let k = cdf.iter().position(|&a| x < a).unwrap_or(nsub - 1);
        #[allow(clippy::cast_precision_loss)]
        let f = |i: usize, rng: &mut Rng| (i as f64 + rng.next_f64()) / SUB as f64;
        let s = f(k % SUB, rng);
        let t = f(k / SUB, rng);
        (s, t)
    }

    /// A pair of points from the positive and the negative part of `F`; as
    /// `∫ F = 0`, each part carries half of [`Self::l1`]:
    /// `∫ F φ = l1() / 2 · (E[φ(p₊)] − E[φ(p₋)])`.
    #[inline]
    pub fn sample_pair(&self, rng: &mut Rng) -> [[f64; 3]; 2] {
        let i = self.parts[0].sample(rng);
        let (s, t) = self.fractions(i, rng);
        let a = self.point(i, s, t);
        let j = self.parts[1].sample(rng);
        let (s, t) = self.fractions(j, rng);
        [a, self.point(j, s, t)]
    }

    /// Samples a point `(w, u, v)` on the box surface from `|F|` and the sign
    /// of `F` there: `∫ F φ = l1() · E[sign · φ(point)]`.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> ([f64; 3], f64) {
        let c = self.table.sample(rng);
        let (s, t) = self.fractions(c, rng);
        (self.point(c, s, t), self.values[c].signum())
    }
}

fn l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// Extra threads for kernel builds, shared by the whole process.
struct Budget {
    count: usize,
}

static IN_USE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Budget {
    /// Takes up to `want` threads of the budget (four times the available
    /// parallelism).
    fn take(want: usize) -> Self {
        use std::sync::atomic::Ordering;
        // Some oversubscription keeps concurrent builds busy; the bound
        // keeps the thread count finite.
        let cap = 4 * std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let mut cur = IN_USE.load(Ordering::Relaxed);
        loop {
            let count = want.min(cap.saturating_sub(cur));
            match IN_USE.compare_exchange_weak(
                cur,
                cur + count,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Self { count },
                Err(now) => cur = now,
            }
        }
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        IN_USE.fetch_sub(self.count, std::sync::atomic::Ordering::AcqRel);
    }
}

/// A mode's window coefficient and adjoint solution.
type ModeResult = Result<(f64, Vec<f64>), SectionReject>;

/// Integrals of `F` over the sub-cells of the table cells at `r` grid
/// cells per table cell along each axis (`r²` per table cell, see `at`).
#[allow(clippy::too_many_lines)]
fn level(
    spec: &SectionSpec,
    cells: &Cells,
    r: usize,
    threads: usize,
    axes: &AxisCache,
) -> Result<Vec<f64>, SectionReject> {
    let (un, uo) = refine(&cells.u, r);
    let (vn, vo) = refine(&cells.v, r);
    let lv = Level::new(&un, &vn, spec);
    let nu = un.len() - 1;
    let nv = vn.len() - 1;
    let s = nv + 1;
    let nw = (cells.w.len() - 1) * r;
    #[allow(clippy::cast_precision_loss)]
    let nwf = nw as f64;
    let dw = 2.0 * spec.half_w / nwf;
    // The window functional on (u, v): the flux across u = g, linearly
    // interpolated from the dual faces on either side of its node line, each
    // dual segment weighted by ω(v) at its node.
    let ig = un
        .iter()
        .position(|x| x.to_bits() == spec.gauss.to_bits())
        .expect("the window plane is a node line");
    let (h1, h2) = (lv.du[ig - 1], lv.du[ig]);
    let (wm, wp) = (h2 / (h1 + h2), h1 / (h1 + h2));
    let vc = 0.5 * spec.v;
    let mut g2 = vec![0.0; (nu + 1) * s];
    for k in 1..nv {
        let om = window(vn[k] - vc, spec.window[1]);
        if om == 0.0 {
            continue;
        }
        let c = |i: usize| lv.cu[i * s + k];
        let cm = wm * om * c(ig - 1);
        g2[(ig - 1) * s + k] += cm;
        g2[ig * s + k] -= cm;
        let cp = wp * om * c(ig);
        g2[ig * s + k] += cp;
        g2[(ig + 1) * s + k] -= cp;
    }
    // Along w: nodes m = 0..nw (w = −W + m Δw), weights Δw · ω(w_m), and
    // the discrete sine modes s_j(m) = sin(π j m / nw), odd j.
    #[allow(clippy::cast_precision_loss)]
    let t = |m: usize| dw * window((m as f64).mul_add(dw, -spec.half_w), spec.window[0]);
    let sj = |j: usize, m: usize| {
        #[allow(clippy::cast_precision_loss)]
        let v = (PI * (j * m) as f64 / nwf).sin();
        v
    };
    let (au, av) = axes.pair(&lv.du, &lv.dv);
    let total_area: f64 = (0..nu)
        .flat_map(|i| (0..nv).map(move |k| (i, k)))
        .map(|(i, k)| lv.du[i] * lv.dv[k])
        .sum();
    let mean = (0..nu)
        .flat_map(|i| (0..nv).map(move |k| (i, k)))
        .map(|(i, k)| lv.eps[i * nv + k] * lv.du[i] * lv.dv[k])
        .sum::<f64>()
        / total_area;
    let modes: Vec<usize> = (1..nw).step_by(2).collect();
    let solve_mode =
        |solver: &mut Solver<'_>, j: usize| -> Result<(f64, Vec<f64>), SectionReject> {
            let gj = (2.0 / nwf) * (1..nw).map(|m| t(m) * sj(j, m)).sum::<f64>();
            #[allow(clippy::cast_precision_loss)]
            let mu = 4.0 * (0.5 * PI * j as f64 / nwf).sin().powi(2) / (dw * dw);
            let psi = solver.solve(mu, &g2)?;
            Ok((gj, psi.to_vec()))
        };
    // Extra threads come from a process-wide budget (concurrent builds on
    // many walker threads must not multiply it); the calling thread always
    // works, and a failed spawn leaves its share to it.
    let extra = Budget::take(threads.clamp(1, modes.len()) - 1);
    let workers = extra.count + 1;
    let chunk = modes.len().div_ceil(workers);
    let mut solved: Vec<Option<ModeResult>> = vec![None; modes.len()];
    std::thread::scope(|sc| {
        let mut jobs = solved.chunks_mut(chunk).zip(modes.chunks(chunk));
        let own = jobs.next();
        let mut left = Vec::new();
        for (slots, js) in jobs {
            let solve_mode = &solve_mode;
            let (lv, au, av) = (&lv, &au, &av);
            let spawned = std::thread::Builder::new().spawn_scoped(sc, {
                let js = js.to_vec();
                move || {
                    let mut solver = Solver::new(lv, au, av, mean);
                    js.iter()
                        .map(|&j| solve_mode(&mut solver, j))
                        .collect::<Vec<_>>()
                }
            });
            left.push((slots, js, spawned.ok()));
        }
        let mut solver = Solver::new(&lv, &au, &av, mean);
        if let Some((slots, js)) = own {
            for (slot, &j) in slots.iter_mut().zip(js) {
                *slot = Some(solve_mode(&mut solver, j));
            }
        }
        for (slots, js, handle) in left {
            let done = handle.and_then(|h| h.join().ok());
            match done {
                Some(results) => {
                    for (slot, r) in slots.iter_mut().zip(results) {
                        *slot = Some(r);
                    }
                }
                None => {
                    for (slot, &j) in slots.iter_mut().zip(js) {
                        *slot = Some(solve_mode(&mut solver, j));
                    }
                }
            }
        }
    });
    drop(extra);
    // Ψ(m, ·) = Σ_j s_j(m) ĝ_j ψ_j on the rows next to the u and v faces,
    // and the full slice m = 1 (next to the w faces).
    let mut urow = vec![0.0; (nw + 1) * 2 * s]; // [m][i = 1 | i = nu − 1][k]
    let mut vrow = vec![0.0; (nw + 1) * 2 * (nu + 1)]; // [m][k = 1 | k = nv − 1][i]
    let mut wslice = vec![0.0; (nu + 1) * s];
    for (&j, sol) in modes.iter().zip(solved) {
        let (gj, psi) = sol.expect("every mode is solved")?;
        for m in 1..nw {
            let c = gj * sj(j, m);
            let row = &mut urow[m * 2 * s..(m + 1) * 2 * s];
            for k in 0..=nv {
                row[k] += c * psi[s + k];
                row[s + k] += c * psi[(nu - 1) * s + k];
            }
            let row = &mut vrow[m * 2 * (nu + 1)..(m + 1) * 2 * (nu + 1)];
            for i in 0..=nu {
                row[i] += c * psi[i * s + 1];
                row[nu + 1 + i] += c * psi[i * s + nv - 1];
            }
        }
        let c = gj * sj(j, 1);
        for (a, b) in wslice.iter_mut().zip(&psi) {
            *a += c * b;
        }
    }
    // Boundary couplings into table cells, quarter by quarter.
    let off = cells.offsets();
    let [cw, cu, _] = cells.counts();
    // Sub-cell `(ta, tb)` of a table cell: the fine cells along its two
    // tangential axes (in the order of [`SectionKernel::point`]).
    let rr = r * r;
    let mut out = vec![0.0; off[6] * rr];
    let at = |c: usize, ta: usize, tb: usize| c * rr + (tb % r) * r + ta % r;
    for m in 1..nw {
        for k in 1..nv {
            for (face, i_cell, idx) in [(2, 0, k), (3, nu - 1, s + k)] {
                let psi = urow[m * 2 * s + idx];
                for (kc, dvh) in [(k - 1, lv.dv[k - 1]), (k, lv.dv[k])] {
                    let q = 0.5 * lv.eps[i_cell * nv + kc] * dvh / lv.du[i_cell] * psi;
                    for mc in [m - 1, m] {
                        out[at(off[face] + vo[kc] * cw + mc / r, mc, kc)] += 0.5 * q;
                    }
                }
            }
        }
        for i in 1..nu {
            for (face, k_cell, idx) in [(4, 0, i), (5, nv - 1, nu + 1 + i)] {
                let psi = vrow[m * 2 * (nu + 1) + idx];
                for (ic, duh) in [(i - 1, lv.du[i - 1]), (i, lv.du[i])] {
                    let q = 0.5 * lv.eps[ic * nv + k_cell] * duh / lv.dv[k_cell] * psi;
                    for mc in [m - 1, m] {
                        out[at(off[face] + uo[ic] * cw + mc / r, mc, ic)] += 0.5 * q;
                    }
                }
            }
        }
    }
    for i in 1..nu {
        for k in 1..nv {
            let psi = wslice[i * s + k] / (dw * dw);
            for (ic, kc) in [(i - 1, k - 1), (i, k - 1), (i - 1, k), (i, k)] {
                let q = 0.25 * lv.eps[ic * nv + kc] * lv.du[ic] * lv.dv[kc] * psi;
                let c = vo[kc] * cu + uo[ic];
                out[at(off[0] + c, ic, kc)] += q;
                out[at(off[1] + c, ic, kc)] += q;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A C5-like box: the 0.14 gap between two met2 bits (u), 0.14 along the
    /// wire (w) and v; shells 0.03 thick; the conductor bottom (and a stack
    /// plane) at v = 0.07.
    pub(super) fn spec_c5(shells: bool) -> SectionSpec {
        let (u_planes, eps) = if shells {
            (vec![0.03, 0.11], vec![4.5, 3.5, 4.5, 4.2, 4.5, 3.5])
        } else {
            (vec![], vec![4.5, 4.2])
        };
        SectionSpec {
            half_w: 0.07,
            u: 0.14,
            v: 0.14,
            u_planes,
            v_planes: vec![0.07],
            eps,
            gauss: 0.07,
            window: [0.035, 0.035],
            w_cells: 8,
            cells: 12,
        }
    }

    /// `∫ ω(v) dv` by the nodes of the finest level is not exact, so the
    /// linear checks compare with the continuous window integral at a
    /// tolerance of the discretization.
    #[test]
    fn windows_add_up_to_one() {
        for x in [0.0, 0.1, 0.33, 0.5, 0.9] {
            assert!((window(x, 1.0) + window(x - 1.0, 1.0) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn homogeneous_linear_flux() {
        let spec = SectionSpec {
            u_planes: vec![],
            v_planes: vec![],
            eps: vec![1.0],
            ..spec_c5(false)
        };
        let k = SectionKernel::new(&spec).unwrap();
        let flux = k.apply(&|p| -p[1], 1);
        let want = 0.035 * 0.035;
        assert!((flux / want - 1.0).abs() < 1e-5, "{flux} vs {want}");
        let zero = k.apply(&|_| 1.0, 1);
        assert!(zero.abs() < 1e-9 * k.l1(), "{zero}");
    }

    /// Shells only (planes normal to u): φ with ε φ′ = −1 is harmonic.
    #[test]
    fn layered_linear_flux() {
        let spec = SectionSpec {
            v_planes: vec![],
            eps: vec![3.5, 4.2, 3.5],
            ..spec_c5(true)
        };
        let k = SectionKernel::new(&spec).unwrap();
        let phi = |p: [f64; 3]| {
            let u = p[1];
            -(u.min(0.03) / 3.5 + (u.min(0.11) - 0.03).max(0.0) / 4.2 + (u - 0.11).max(0.0) / 3.5)
        };
        let flux = k.apply(&phi, 1);
        let want = 0.035 * 0.035;
        assert!((flux / want - 1.0).abs() < 1e-4, "{flux} vs {want}");
    }

    #[test]
    fn plane_across_the_window() {
        for shells in [false, true] {
            let k = SectionKernel::new(&spec_c5(shells)).unwrap();
            if !shells {
                // φ = −u: ∫ω_w · ∫ω_v ε (half the window below v = 0.07).
                let flux = k.apply(&|p| -p[1], 1);
                let want = 0.035 * 0.035 * 0.5 * (4.5 + 4.2);
                assert!((flux / want - 1.0).abs() < 1e-5, "{flux} vs {want}");
            }
        }
    }
}

#[cfg(test)]
mod accuracy {
    use super::*;

    /// The production resolution against a reference at twice the
    /// resolution: `∫ F φ` for a few test functions (design M2f §4.3).
    #[test]
    fn tables_agree_with_a_finer_reference() {
        let base = tests::spec_c5(true);
        let fns: [fn([f64; 3]) -> f64; 4] = [
            |p| -p[1],
            |p| p[2] * 10.0,
            |p| (p[1] - 0.07) * (p[2] - 0.07) * 100.0,
            |p| (p[1] * 20.0).exp() * (p[2] * 20.0).cos() * 0.1,
        ];
        let fine = SectionKernel::new(&SectionSpec {
            cells: 24,
            w_cells: 16,
            ..base.clone()
        })
        .unwrap();
        let k = SectionKernel::new(&SectionSpec {
            cells: 12,
            w_cells: 8,
            ..base
        })
        .unwrap();
        assert!(k.error_estimate() < MAX_ERROR);
        for f in fns {
            let e = (k.apply(&f, 4) - fine.apply(&f, 4)).abs() / k.l1();
            assert!(e < 1e-4, "{e}");
        }
    }

    /// A unit box with a stack plane across the window and a cubic harmonic
    /// potential (bounded by one), against twice the resolution: sampling
    /// within the cells must not add a bias beyond the gate.
    #[test]
    fn sampling_within_cells_is_unbiased() {
        let base = SectionSpec {
            half_w: 0.5,
            u: 1.0,
            v: 1.0,
            u_planes: vec![],
            v_planes: vec![0.5],
            eps: vec![4.5, 4.2],
            gauss: 0.5,
            window: [0.25, 0.25],
            w_cells: 8,
            cells: 12,
        };
        let phi = |p: [f64; 3]| {
            let (x, y) = (p[1] - 0.5, p[2] - 0.5);
            4.0 * (x * x * x - 3.0 * x * y * y)
        };
        let k = SectionKernel::new(&base).unwrap();
        let fine = SectionKernel::new(&SectionSpec {
            cells: 24,
            w_cells: 16,
            ..base
        })
        .unwrap();
        let e = (k.apply(&phi, 4) - fine.apply(&phi, 4)).abs() / k.l1();
        assert!(e < 1e-4, "{e}");
    }
}
