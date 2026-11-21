//! Edge cubes (design M2d §3.3–§3.4, `docs/design/M2d_edge_cubes.md`).
//!
//! Canonical frame: the cube `[−1, 1]³` with the walker at the origin, the
//! plane `P: z = 0` through it, and the plane `Q: y = t`, `t = k/K`. ε is
//! constant in the four quadrants `(y < t | y > t) × (z < 0 | z > 0)` and does
//! not depend on `x`, the direction of the edge `P ∩ Q`.
//!
//! The exit probabilities are the boundary fluxes of the Green function of
//! `−∇·(ε∇G) = δ` with `G = 0` on the cube, from a node-based finite-volume
//! scheme on `n` subdivisions per axis (ε per cell, flux coefficients from the
//! cells along each dual face). ε does not depend on `x`, so the discrete sine
//! transform along `x` decouples the 3D system exactly into 2D systems, one per
//! odd mode. Every face flux of the 3D discrete solution is a finite sum over
//! the modes, and the fluxes add up to the unit source.
//!
//! Each 2D system is solved by conjugate gradients, preconditioned by the
//! constant-coefficient operator, which a 2D discrete sine transform inverts
//! exactly. Three grid levels give an accuracy gate (design §3.4): Richardson
//! extrapolation where the observed order is about 2, and an L1 error estimate
//! that must stay below [`MAX_ERROR`]. Configurations that fail the gate are
//! rejected, and the walk keeps its M2c steps there.

use std::f64::consts::PI;

use crate::alias::AliasTable;
use crate::dense;
use crate::rng::Rng;

/// Largest accepted L1 error estimate of a table's cell probabilities.
pub const MAX_ERROR: f64 = 1e-5;

/// Coarsest and finest grid levels (subdivisions per axis). Coarser grids
/// fail the accuracy gate even for a homogeneous cube (their Richardson
/// estimate is about 1e-4); finer ones cost too much (O(n⁴) per table).
const MIN_LEVEL: usize = 32;
const MAX_LEVEL: usize = 128;

/// Relative residual at which the conjugate gradients stop.
const CG_TOLERANCE: f64 = 1e-12;

/// Why an edge cube was not built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeReject {
    /// `lcm(2K, M)` is not a power of two, or the finest level would exceed
    /// the supported grid.
    Grid,
    /// The observed convergence order is below 1.
    Order(f64),
    /// The error estimate exceeds [`MAX_ERROR`].
    Error(f64),
    /// A cell probability is negative beyond rounding.
    Negative(f64),
    /// The fluxes of a level do not add up to the unit source.
    Conservation(f64),
    /// The conjugate gradients did not converge.
    Solver,
}

/// Transition tables of an edge cube.
#[derive(Debug, Clone)]
pub struct EdgeCube {
    m: usize,
    /// Per-cell probabilities, faces `[x-, x+, y-, y+, z-, z+]`, `m²` cells
    /// each, cell `j * m + i` with `(i, j)` the face's tangential axes in the
    /// order of [`crate::layered::LayeredCube`]: `(y, z)`, `(x, z)`, `(x, y)`.
    cells: Vec<f64>,
    transition: AliasTable,
    error: f64,
}

impl EdgeCube {
    /// Builds the tables for the quadrant permittivities `eps[y side][z side]`
    /// (side 0: `y < t` resp. `z < 0`), the offset `t = offset / offsets` and
    /// `resolution × resolution` cells per face.
    ///
    /// # Errors
    ///
    /// [`EdgeReject`] if the configuration fails the accuracy gate or the grid
    /// is unsupported.
    ///
    /// # Panics
    ///
    /// Panics if `offset` is not in `1..offsets`, `resolution` is not a
    /// positive multiple of `2 * offsets`, or a permittivity is not finite and
    /// positive.
    pub fn new(
        eps: [[f64; 2]; 2],
        offset: usize,
        offsets: usize,
        resolution: usize,
    ) -> Result<Self, EdgeReject> {
        assert!(
            0 < offset && offset < offsets,
            "offset must be in 1..offsets"
        );
        assert!(
            resolution > 0 && resolution % (2 * offsets) == 0,
            "resolution must be a multiple of 2 * offsets"
        );
        assert!(
            eps.iter().flatten().all(|e| e.is_finite() && *e > 0.0),
            "permittivities must be finite and positive"
        );
        // Every level must put y = t and the table cell boundaries on grid
        // lines: a power-of-two multiple of lcm(2K, M).
        let base = lcm(2 * offsets, resolution);
        if !base.is_power_of_two() {
            return Err(EdgeReject::Grid);
        }
        let n1 = base.max(MIN_LEVEL);
        if n1 > MAX_LEVEL / 4 {
            return Err(EdgeReject::Grid);
        }
        // Only ratios matter: scale the largest permittivity to 1.
        let top = eps.iter().flatten().copied().fold(0.0, f64::max);
        let eps = eps.map(|row| row.map(|e| e / top));
        let grid = Grid {
            eps,
            offset,
            offsets,
        };
        let p1 = grid.level(n1, resolution)?;
        let p2 = grid.level(2 * n1, resolution)?;
        let p4 = grid.level(4 * n1, resolution)?;
        let (mut cells, error) = gate(&p1, &p2, &p4)?;
        if error > MAX_ERROR {
            return Err(EdgeReject::Error(error));
        }
        let worst = cells.iter().copied().fold(0.0, f64::min);
        if worst < -1e-12 {
            return Err(EdgeReject::Negative(worst));
        }
        for c in &mut cells {
            *c = c.max(0.0);
        }
        let total: f64 = cells.iter().sum();
        for c in &mut cells {
            *c /= total;
        }
        let transition = AliasTable::new(&cells);
        Ok(Self {
            m: resolution,
            cells,
            transition,
            error,
        })
    }

    /// Per-cell probabilities of all six faces (see the field layout).
    #[must_use]
    pub fn cell_probabilities(&self) -> &[f64] {
        &self.cells
    }

    /// L1 error estimate of the cell probabilities (design §3.4).
    #[must_use]
    pub const fn error_estimate(&self) -> f64 {
        self.error
    }

    /// Samples the next walk position on the surface of the cube `[−1, 1]³`
    /// (walker at the origin); a coordinate on a face is exactly `±1`.
    #[inline]
    pub fn sample_transition(&self, rng: &mut Rng) -> [f64; 3] {
        let m = self.m;
        let r = self.transition.sample(rng);
        let (face, rem) = (r / (m * m), r % (m * m));
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        let coord = |c: usize, rng: &mut Rng| {
            #[allow(clippy::cast_precision_loss)]
            let v = 2.0 * (c as f64 + rng.next_f64()) / mf - 1.0;
            v
        };
        let u = coord(rem % m, rng);
        let v = coord(rem / m, rng);
        let w = if face % 2 == 0 { -1.0 } else { 1.0 };
        match face / 2 {
            0 => [w, u, v],
            1 => [u, w, v],
            _ => [u, v, w],
        }
    }
}

const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

const fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// The accuracy gate of design §3.4: the table and its error estimate.
fn gate(p1: &[f64], p2: &[f64], p4: &[f64]) -> Result<(Vec<f64>, f64), EdgeReject> {
    let d12 = l1(p1, p2);
    let d24 = l1(p2, p4);
    if d24 <= 1e-14 {
        // Converged to rounding (e.g. a configuration the grid represents
        // exactly).
        return Ok((p4.to_vec(), d24));
    }
    let order = (d12 / d24).log2();
    if (order - 2.0).abs() <= 0.2 {
        let rich = |a: &[f64], b: &[f64]| -> Vec<f64> {
            a.iter()
                .zip(b)
                .map(|(x, y)| (4.0f64).mul_add(*y, -x) / 3.0)
                .collect()
        };
        let r1 = rich(p1, p2);
        let r2 = rich(p2, p4);
        let error = l1(&r1, &r2);
        Ok((r2, error))
    } else if order >= 1.0 {
        Ok((p4.to_vec(), d24))
    } else {
        Err(EdgeReject::Order(order))
    }
}

/// The canonical configuration.
struct Grid {
    eps: [[f64; 2]; 2],
    offset: usize,
    offsets: usize,
}

impl Grid {
    /// Cell probabilities (not normalized) at `n` subdivisions per axis.
    #[allow(clippy::too_many_lines)]
    fn level(&self, n: usize, m: usize) -> Result<Vec<f64>, EdgeReject> {
        let stride = n + 1;
        // Grid index of the plane y = t; z = 0 is n / 2.
        let it = n / 2 + self.offset * n / (2 * self.offsets);
        // ε of cell (p, q), p along y, q along z (cells 0..n).
        let cell = |p: usize, q: usize| self.eps[usize::from(p >= it)][usize::from(q >= n / 2)];
        // Flux coefficients: cy between nodes (p, q) and (p + 1, q); cz
        // between (p, q) and (p, q + 1); e4 the reaction weight of node (p, q).
        let mut cy = vec![0.0; stride * stride];
        let mut cz = vec![0.0; stride * stride];
        let mut e4 = vec![0.0; stride * stride];
        for p in 0..n {
            for q in 1..n {
                cy[p * stride + q] = 0.5 * (cell(p, q - 1) + cell(p, q));
            }
        }
        for p in 1..n {
            for q in 0..n {
                cz[p * stride + q] = 0.5 * (cell(p - 1, q) + cell(p, q));
            }
        }
        for p in 1..n {
            for q in 1..n {
                e4[p * stride + q] =
                    0.25 * (cell(p - 1, q - 1) + cell(p, q - 1) + cell(p - 1, q) + cell(p, q));
            }
        }
        let mean = self.eps.iter().flatten().sum::<f64>() / 4.0;
        let dst = Dst::new(n);
        // Accumulated mode sums: boundary rows for the side faces, the full
        // field next to the x faces.
        let mut ym = vec![0.0; stride * stride]; // (i, q) at p = 1
        let mut yp = vec![0.0; stride * stride]; // (i, q) at p = n - 1
        let mut zm = vec![0.0; stride * stride]; // (i, p) at q = 1
        let mut zp = vec![0.0; stride * stride]; // (i, p) at q = n - 1
        let mut xf = vec![0.0; stride * stride]; // (p, q) at i = 1
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;
        // The odd modes are independent 2D solves: run them on scoped threads
        // (not the walk's rayon pool, whose work stealing could re-enter this
        // table's `OnceLock`), then accumulate in mode order, so the table is
        // bit-identical to a sequential build.
        let modes: Vec<usize> = (1..n).step_by(2).collect();
        let solve_mode = |solver: &mut Cg, j: usize| -> Result<Vec<f64>, EdgeReject> {
            #[allow(clippy::cast_precision_loss)]
            let jf = j as f64;
            let op = Op {
                n,
                cy: &cy,
                cz: &cz,
                e4: &e4,
                mu: 4.0 * (0.5 * PI * jf / nf).sin().powi(2),
            };
            let sign = if j % 4 == 1 { 1.0 } else { -1.0 };
            solver.solve(&op, &dst, mean, sign).map(<[f64]>::to_vec)
        };
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(modes.len());
        let mut solved: Vec<Option<Result<Vec<f64>, EdgeReject>>> = vec![None; modes.len()];
        if threads <= 1 {
            let mut solver = Cg::new(n);
            for (slot, &j) in solved.iter_mut().zip(&modes) {
                *slot = Some(solve_mode(&mut solver, j));
            }
        } else {
            let chunk = modes.len().div_ceil(threads);
            std::thread::scope(|sc| {
                for (slots, js) in solved.chunks_mut(chunk).zip(modes.chunks(chunk)) {
                    let solve_mode = &solve_mode;
                    sc.spawn(move || {
                        let mut solver = Cg::new(n);
                        for (slot, &j) in slots.iter_mut().zip(js) {
                            *slot = Some(solve_mode(&mut solver, j));
                        }
                    });
                }
            });
        }
        for (&j, phi) in modes.iter().zip(solved) {
            let phi = phi.expect("every mode is solved")?;
            #[allow(clippy::cast_precision_loss)]
            let jf = j as f64;
            let s = |i: usize| {
                #[allow(clippy::cast_precision_loss)]
                let v = (PI * jf * i as f64 / nf).sin();
                v
            };
            let s1 = s(1);
            for i in 1..n {
                let si = s(i);
                for q in 1..n {
                    ym[i * stride + q] += si * phi[stride + q];
                    yp[i * stride + q] += si * phi[(n - 1) * stride + q];
                }
                for p in 1..n {
                    zm[i * stride + p] += si * phi[p * stride + 1];
                    zp[i * stride + p] += si * phi[p * stride + n - 1];
                }
            }
            for p in 1..n {
                for q in 1..n {
                    xf[p * stride + q] += s1 * phi[p * stride + q];
                }
            }
        }
        // Node fluxes into table cells. The dual face of a boundary node has
        // four quarters, each in one grid cell next to the face; a quarter
        // carries `h · ε_cell / 4 · φ` (the node's flux coefficient is the
        // mean of the four), and belongs to the table cell of its grid cell.
        let h = 2.0 / nf;
        let w = n / m;
        let mut cells = vec![0.0; 6 * m * m];
        let mut add =
            |face: usize, a: usize, b: usize, phi: f64, eps: &dyn Fn(usize, usize) -> f64| {
                for da in 0..2 {
                    for db in 0..2 {
                        let (ga, gb) = (a - 1 + da, b - 1 + db);
                        cells[face * m * m + (gb / w) * m + ga / w] += 0.25 * h * eps(ga, gb) * phi;
                    }
                }
            };
        for p in 1..n {
            for q in 1..n {
                let phi = xf[p * stride + q];
                add(0, p, q, phi, &|gp, gq| cell(gp, gq));
                add(1, p, q, phi, &|gp, gq| cell(gp, gq));
            }
        }
        for i in 1..n {
            for q in 1..n {
                add(2, i, q, ym[i * stride + q], &|_, gq| cell(0, gq));
                add(3, i, q, yp[i * stride + q], &|_, gq| cell(n - 1, gq));
            }
            for p in 1..n {
                add(4, i, p, zm[i * stride + p], &|_, gp| cell(gp, 0));
                add(5, i, p, zp[i * stride + p], &|_, gp| cell(gp, n - 1));
            }
        }
        let total: f64 = cells.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(EdgeReject::Conservation(total));
        }
        Ok(cells)
    }
}

/// The 2D operator of one mode on the interior nodes (boundary values 0).
struct Op<'a> {
    n: usize,
    cy: &'a [f64],
    cz: &'a [f64],
    e4: &'a [f64],
    mu: f64,
}

impl Op<'_> {
    /// `out = A x` on the `(n + 1)²` node array (boundary entries stay 0).
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        let n = self.n;
        let s = n + 1;
        for p in 1..n {
            for q in 1..n {
                let k = p * s + q;
                let v = x[k];
                out[k] = self.cy[k - s] * (v - x[k - s])
                    + self.cy[k] * (v - x[k + s])
                    + self.cz[k - 1] * (v - x[k - 1])
                    + self.cz[k] * (v - x[k + 1])
                    + self.e4[k] * self.mu * v;
            }
        }
    }
}

/// The 2D discrete sine transform (DST-I) on the interior nodes, by dense
/// matrix products (`S` is symmetric and `S² = (n/2) I`). The symmetry
/// `s_j(n − i) = (−1)^{j+1} s_j(i)` splits each product into an odd-`j`
/// half acting on `x_i + x_{n−i}` and an even-`j` half acting on
/// `x_i − x_{n−i}`, which halves the work.
struct Dst {
    n: usize,
    /// Odd `j = 2o + 1` (`o < n/2`): `s_j(i)` for `i = 1 … n/2`.
    odd: Vec<f64>,
    /// Even `j = 2e + 2` (`e < n/2 − 1`): `s_j(i)` for `i = 1 … n/2 − 1`.
    even: Vec<f64>,
    /// Eigenvalues `4 sin²(jπ/2n)` of the 1D second difference.
    lambda: Vec<f64>,
}

impl Dst {
    fn new(n: usize) -> Self {
        // The odd/even fold pairs rows `i` and `n − i` around a middle row.
        assert!(n >= 4 && n % 2 == 0, "DST size {n} must be even");
        let h = n / 2;
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;
        #[allow(clippy::cast_precision_loss)]
        let s = |j: usize, i: usize| (PI * (j * i) as f64 / nf).sin();
        let mut odd = vec![0.0; h * h];
        for o in 0..h {
            for i in 1..=h {
                odd[o * h + i - 1] = s(2 * o + 1, i);
            }
        }
        let mut even = vec![0.0; (h - 1) * (h - 1)];
        for e in 0..h - 1 {
            for i in 1..h {
                even[e * (h - 1) + i - 1] = s(2 * e + 2, i);
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let lambda = (1..n)
            .map(|j| 4.0 * (0.5 * PI * j as f64 / nf).sin().powi(2))
            .collect();
        Self {
            n,
            odd,
            even,
            lambda,
        }
    }

    /// `dst = S src` along the row index of an `nn × nn` row-major block;
    /// `work` holds the folded rows.
    fn rows(&self, src: &[f64], dst: &mut [f64], work: &mut [f64]) {
        let nn = self.n - 1;
        let h = self.n / 2;
        // work rows 0..h: x_i + x_{n−i} (row h − 1: the middle row x_{n/2});
        // rows h..2h − 1: x_i − x_{n−i}.
        for r in 0..h - 1 {
            let (a, b) = (
                &src[r * nn..(r + 1) * nn],
                &src[(nn - 1 - r) * nn..(nn - r) * nn],
            );
            for c in 0..nn {
                work[r * nn + c] = a[c] + b[c];
                work[(h + r) * nn + c] = a[c] - b[c];
            }
        }
        work[(h - 1) * nn..h * nn].copy_from_slice(&src[(h - 1) * nn..h * nn]);
        // Odd j = 2o + 1 into rows 2o, even j = 2e + 2 into rows 2e + 1.
        let (odd, even) = work[..(2 * h - 1) * nn].split_at(h * nn);
        dense::product(&self.odd, odd, dst, [h, h, nn], [2, 0]);
        dense::product(&self.even, even, dst, [h - 1, h - 1, nn], [2, 1]);
    }

    /// `x ← S x S` on the dense interior block (`nn × nn`), as
    /// `(S (S x)ᵀ)ᵀ`.
    fn transform(&self, x: &mut [f64], tmp: &mut [f64], work: &mut [f64]) {
        let nn = self.n - 1;
        self.rows(x, tmp, work);
        transpose(tmp, x, nn);
        self.rows(x, tmp, work);
        transpose(tmp, x, nn);
    }
}

fn transpose(src: &[f64], dst: &mut [f64], nn: usize) {
    for r in 0..nn {
        for c in 0..nn {
            dst[c * nn + r] = src[r * nn + c];
        }
    }
}

/// Preconditioned conjugate gradients with reusable work arrays.
struct Cg {
    n: usize,
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    block: Vec<f64>,
    tmp: Vec<f64>,
    work: Vec<f64>,
}

impl Cg {
    fn new(n: usize) -> Self {
        let s = (n + 1) * (n + 1);
        let nn = (n - 1) * (n - 1);
        Self {
            n,
            x: vec![0.0; s],
            r: vec![0.0; s],
            z: vec![0.0; s],
            p: vec![0.0; s],
            ap: vec![0.0; s],
            block: vec![0.0; nn],
            tmp: vec![0.0; nn],
            work: vec![0.0; nn],
        }
    }

    /// `z = P⁻¹ r` with `P` the constant-coefficient operator `mean · (L + μ)`.
    fn precondition(&mut self, dst: &Dst, mean: f64, mu: f64) {
        let n = self.n;
        let s = n + 1;
        let nn = n - 1;
        for p in 1..n {
            for q in 1..n {
                self.block[(p - 1) * nn + (q - 1)] = self.r[p * s + q];
            }
        }
        dst.transform(&mut self.block, &mut self.tmp, &mut self.work);
        #[allow(clippy::cast_precision_loss)]
        let scale = (2.0 / n as f64).powi(2);
        for j in 0..nn {
            for k in 0..nn {
                self.block[j * nn + k] *= scale / (mean * (dst.lambda[j] + dst.lambda[k] + mu));
            }
        }
        dst.transform(&mut self.block, &mut self.tmp, &mut self.work);
        for p in 1..n {
            for q in 1..n {
                self.z[p * s + q] = self.block[(p - 1) * nn + (q - 1)];
            }
        }
    }

    /// Solves `A x = sign · δ_centre`; returns the node array.
    fn solve(
        &mut self,
        op: &Op<'_>,
        dst: &Dst,
        mean: f64,
        sign: f64,
    ) -> Result<&[f64], EdgeReject> {
        let n = self.n;
        let s = n + 1;
        let centre = (n / 2) * s + n / 2;
        self.x.fill(0.0);
        self.r.fill(0.0);
        self.r[centre] = sign;
        let norm_b = 1.0;
        self.precondition(dst, mean, op.mu);
        self.p.copy_from_slice(&self.z);
        let mut rz = dot(&self.r, &self.z);
        for _ in 0..10_000 {
            op.apply(&self.p, &mut self.ap);
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
            self.precondition(dst, mean, op.mu);
            let rz_new = dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;
            for k in 0..self.p.len() {
                self.p[k] = beta * self.p[k] + self.z[k];
            }
        }
        Err(EdgeReject::Solver)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered::LayeredCube;

    const K: usize = 16;
    const M: usize = 32;

    /// Cell boundaries of the face grid.
    fn bounds(c: usize) -> (f64, f64) {
        #[allow(clippy::cast_precision_loss)]
        let f = |i: usize| 2.0 * i as f64 / M as f64 - 1.0;
        (f(c), f(c + 1))
    }

    /// `E[φ(exit)]` with `φ` averaged over each exit cell (what cell-uniform
    /// sampling gives); `avg(axis, lo, hi)` is the cell average of a function
    /// of one coordinate, `fixed(axis, value)` its value on a face. `φ` is a
    /// sum of such one-coordinate terms.
    fn expectation(
        p: &[f64],
        term: &dyn Fn(usize, f64, f64) -> f64,
        point: &dyn Fn(usize, f64) -> f64,
    ) -> f64 {
        let mut e = 0.0;
        for face in 0..6 {
            let normal = face / 2;
            let w = if face % 2 == 0 { -1.0 } else { 1.0 };
            let tang: [usize; 2] = match normal {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            for j in 0..M {
                for i in 0..M {
                    let (a0, a1) = bounds(i);
                    let (b0, b1) = bounds(j);
                    let v = point(normal, w) + term(tang[0], a0, a1) + term(tang[1], b0, b1);
                    e += p[face * M * M + j * M + i] * v;
                }
            }
        }
        e
    }

    fn sq_avg(lo: f64, hi: f64) -> f64 {
        (hi.powi(3) - lo.powi(3)) / (3.0 * (hi - lo))
    }

    #[test]
    fn homogeneous_cube_matches_the_analytic_tables() {
        let c = EdgeCube::new([[1.0; 2]; 2], 1, K, M).unwrap();
        let exact = LayeredCube::new(1.0, 0, K, M);
        let d = l1(c.cell_probabilities(), exact.cell_probabilities());
        assert!(d < 2e-6, "L1 {d}");
        assert!(c.error_estimate() < MAX_ERROR);
        // Also for the smallest supported offsets (8) and resolution (16).
        let small = EdgeCube::new([[1.0; 2]; 2], 3, 8, 16).unwrap();
        let exact = LayeredCube::new(1.0, 0, 8, 16);
        assert!(l1(small.cell_probabilities(), exact.cell_probabilities()) < 2e-6);
    }

    /// `Q` not an interface: the M2a cube with the interface through the
    /// centre (ε = 1 below, κ above).
    #[test]
    fn a_plane_through_the_walker_matches_the_layered_cube() {
        let kappa = 3.0;
        let c = EdgeCube::new([[1.0, kappa], [1.0, kappa]], 5, K, M).unwrap();
        let exact = LayeredCube::new(kappa, 0, K, M);
        let d = l1(c.cell_probabilities(), exact.cell_probabilities());
        assert!(d < 1e-5, "L1 {d}, estimate {}", c.error_estimate());
    }

    /// `P` not an interface: the M2a cube with the interface at offset `t`,
    /// its normal `z` mapped to `y`.
    #[test]
    fn an_offset_plane_matches_the_layered_cube() {
        let (kappa, k) = (0.4, 3);
        let c = EdgeCube::new([[1.0, 1.0], [kappa, kappa]], k, K, M).unwrap();
        let exact = LayeredCube::new(kappa, k, K, M);
        let e = exact.cell_probabilities();
        let mut mapped = vec![0.0; 6 * M * M];
        for j in 0..M {
            for i in 0..M {
                // x faces: (y, z) here, (z, y) there.
                for f in 0..2 {
                    mapped[f * M * M + j * M + i] = e[f * M * M + i * M + j];
                }
                // y faces here are the z faces there, and the z faces the y faces.
                for (here, there) in [(2, 4), (3, 5), (4, 2), (5, 3)] {
                    mapped[here * M * M + j * M + i] = e[there * M * M + j * M + i];
                }
            }
        }
        let d = l1(c.cell_probabilities(), &mapped);
        assert!(d < 1e-5, "L1 {d}, estimate {}", c.error_estimate());
    }

    /// `x² − z²` and `x² − (y − t)²` are ε-harmonic for every quadrant tuple:
    /// harmonic in each quadrant, with zero normal derivative on both planes.
    #[test]
    fn curved_harmonic_functions_are_reproduced() {
        // The SKY130 strip-bottom edge (NILD2 below, NILD3_C / NILD3 above):
        // not a product medium.
        let k = 6;
        #[allow(clippy::cast_precision_loss)]
        let t = k as f64 / K as f64;
        let c = EdgeCube::new([[4.05, 4.5], [4.05, 3.5]], k, K, M).unwrap();
        let p = c.cell_probabilities();
        // x² − z²
        let term = |axis: usize, lo: f64, hi: f64| match axis {
            0 => sq_avg(lo, hi),
            2 => -sq_avg(lo, hi),
            _ => 0.0,
        };
        let point = |axis: usize, w: f64| match axis {
            0 => w * w,
            2 => -w * w,
            _ => 0.0,
        };
        let e = expectation(p, &term, &point);
        assert!(e.abs() < 1e-4, "x² − z²: {e}");
        // x² − (y − t)², value −t² at the walker.
        let term = |axis: usize, lo: f64, hi: f64| match axis {
            0 => sq_avg(lo, hi),
            1 => -sq_avg(lo - t, hi - t),
            _ => 0.0,
        };
        let point = |axis: usize, w: f64| match axis {
            0 => w * w,
            1 => -(w - t) * (w - t),
            _ => 0.0,
        };
        let e = expectation(p, &term, &point);
        assert!((e + t * t).abs() < 1e-4, "x² − (y − t)²: {e} vs {}", -t * t);
    }

    /// For ε = A(y)·B(z), `U(y)` with `(A U′)′ = 0` and `V(z)` with
    /// `(B V′)′ = 0` are ε-harmonic; they are linear on every cell, so
    /// cell-uniform sampling is exact and only the table error remains.
    #[test]
    fn product_media_reproduce_their_affine_potentials() {
        let k = 4;
        #[allow(clippy::cast_precision_loss)]
        let t = k as f64 / K as f64;
        let (a, b) = ([1.0, 3.0], [2.0, 0.5]);
        let eps = [[a[0] * b[0], a[0] * b[1]], [a[1] * b[0], a[1] * b[1]]];
        let c = EdgeCube::new(eps, k, K, M).unwrap();
        let p = c.cell_probabilities();
        let u = |y: f64| {
            if y < t {
                y / a[0]
            } else {
                t / a[0] + (y - t) / a[1]
            }
        };
        let v = |z: f64| if z < 0.0 { z / b[0] } else { z / b[1] };
        for (name, axis, f) in [("U", 1, &u as &dyn Fn(f64) -> f64), ("V", 2, &v)] {
            let term = |ax: usize, lo: f64, hi: f64| {
                if ax == axis {
                    0.5 * (f(lo) + f(hi))
                } else {
                    0.0
                }
            };
            let point = |ax: usize, w: f64| if ax == axis { f(w) } else { 0.0 };
            let e = expectation(p, &term, &point);
            assert!(e.abs() < 1e-5, "{name}: {e}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // face coordinates are exactly ±1
    fn sampling_matches_the_cells_and_lands_on_faces() {
        let c = EdgeCube::new([[4.5, 4.5], [3.5, 4.5]], 4, K, M).unwrap();
        let mut rng = Rng::from_seed(7);
        let n = 400_000;
        let mut counts = vec![0usize; 6 * M * M];
        #[allow(clippy::cast_precision_loss)]
        let mf = M as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let cell = |v: f64| (((v + 1.0) * 0.5 * mf) as usize).min(M - 1);
        for _ in 0..n {
            let x = c.sample_transition(&mut rng);
            let on: Vec<usize> = (0..3).filter(|&a| x[a].abs() == 1.0).collect();
            assert_eq!(on.len(), 1, "{x:?}");
            let a = on[0];
            assert!(x.iter().all(|v| v.abs() <= 1.0));
            let face = 2 * a + usize::from(x[a] > 0.0);
            let tang: [usize; 2] = match a {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            counts[face * M * M + cell(x[tang[1]]) * M + cell(x[tang[0]])] += 1;
        }
        let p = c.cell_probabilities();
        let mut checked = 0;
        for (k, &want) in p.iter().enumerate() {
            if want < 1e-4 {
                continue;
            }
            checked += 1;
            #[allow(clippy::cast_precision_loss)]
            let got = counts[k] as f64 / f64::from(n);
            let sd = (want * (1.0 - want) / f64::from(n)).sqrt();
            assert!((got - want).abs() < 5.0 * sd, "cell {k}: {got} vs {want}");
        }
        assert!(checked > 1000, "{checked}");
    }

    #[test]
    fn unsupported_grids_and_strong_singularities_are_rejected() {
        // lcm(2K, M) = 96 and 6 are not powers of two; 4·lcm(64, 64) > 128.
        for (k, m) in [(16, 96), (3, 6), (32, 64)] {
            assert_eq!(
                EdgeCube::new([[1.0; 2]; 2], 1, k, m).unwrap_err(),
                EdgeReject::Grid
            );
        }
        // Only ratios matter.
        let big = EdgeCube::new([[1e300; 2]; 2], 1, 8, 16).unwrap();
        let unit = EdgeCube::new([[1.0; 2]; 2], 1, 8, 16).unwrap();
        assert_eq!(big.cell_probabilities(), unit.cell_probabilities());
        // Strong edge singularities: below first order (a high-contrast
        // checkerboard), and second order not reached (contrast 6).
        assert!(matches!(
            EdgeCube::new([[1.0, 1000.0], [1000.0, 1.0]], 8, K, M),
            Err(EdgeReject::Order(_))
        ));
        assert!(matches!(
            EdgeCube::new([[1.0, 2.5], [4.0, 0.7]], 8, K, M),
            Err(EdgeReject::Error(_))
        ));
    }
}
