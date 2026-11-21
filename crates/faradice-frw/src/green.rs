//! Surface Green-function tables of the transition cube.
//!
//! For a cube with Dirichlet data on its surface, the potential at the centre
//! is a weighted average of the surface potential, with density `P` (the
//! surface Green function, or Poisson kernel), and the gradient at the centre
//! is a signed surface integral with kernel `K = ∂P/∂n_c`. The floating
//! random walk samples the next point from `P` and the first hop from a
//! Gaussian surface from `|K|` with a sign-carrying weight.
//!
//! For a single homogeneous dielectric both kernels have closed-form double
//! sine series. On the unit cube `[0,1]³` with data on the top face `z = 1`,
//!
//! ```text
//! φ(x,y,z) = Σ_{m,n≥1} A_mn sin(mπx) sin(nπy) sinh(γz)/sinh(γ),   γ = π√(m²+n²),
//! A_mn     = 4 ∫∫ f(x',y') sin(mπx') sin(nπy') dx'dy'.
//! ```
//!
//! Integrating `sin(mπx')` over a cell is analytic, so the per-cell
//! probabilities and kernel integrals are computed **without quadrature
//! error**; only the series is truncated, at a point where the remaining terms
//! are below `1e-30`. Within a cell the position is sampled uniformly, which is
//! the only discretization (its effect vanishes as `O(1/M²)` for smooth
//! potentials and is studied in the validation report).

use faradice_core::Axis;

use crate::alias::AliasTable;
use crate::rng::Rng;

/// Default number of cells per face edge.
pub const DEFAULT_RESOLUTION: usize = 64;

/// Number of odd (or even) series terms per dimension. With `γ/2 ≥ (π/2)·k`,
/// the first omitted term is below `e^{-π·NTERMS}`.
pub(crate) const NTERMS: usize = 48;

/// Precomputed transition and first-hop tables for a homogeneous cube.
#[derive(Debug, Clone)]
pub struct GreenTables {
    m: usize,
    /// Per-cell probabilities on one face (all six faces are identical).
    face_prob: Vec<f64>,
    transition: AliasTable,
    /// Gradient kernel integrals over every cell of every face, canonical
    /// normal +z. Layout: `face * m² + j * m + i` with faces ordered
    /// `[x-, x+, y-, y+, z-, z+]` and `(i, j)` the face's two tangential axes.
    grad_cells: Vec<f64>,
    grad: AliasTable,
    grad_l1: f64,
}

/// `∫_{a}^{b} sin(kπx) dx`.
pub(crate) fn sin_integral(k: f64, a: f64, b: f64) -> f64 {
    let kp = k * std::f64::consts::PI;
    ((kp * a).cos() - (kp * b).cos()) / kp
}

/// Matrix `I[t][i] = coef(t) · ∫_{cell i} sin(k_t π x) dx` for the given wavenumbers.
pub(crate) fn cell_integrals(ks: &[f64], coef: impl Fn(f64) -> f64, m: usize) -> Vec<Vec<f64>> {
    #[allow(clippy::cast_precision_loss)]
    let mf = m as f64;
    ks.iter()
        .map(|&k| {
            #[allow(clippy::cast_precision_loss)]
            (0..m)
                .map(|i| coef(k) * sin_integral(k, i as f64 / mf, (i + 1) as f64 / mf))
                .collect()
        })
        .collect()
}

/// `out[j*m+i] = Σ_{s,t} a[s][i] · c(s,t) · b[t][j]`.
pub(crate) fn bilinear(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: impl Fn(usize, usize) -> f64,
    m: usize,
) -> Vec<f64> {
    debug_assert!(a[0].len() == m && b[0].len() == m);
    bilinear_cells(a, b, c)
}

/// `out[j * a_cells + i] = Σ_{s,t} a[s][i] · c(s, t) · b[t][j]`, evaluating
/// every coefficient `c(s, t)` once.
pub(crate) fn bilinear_cells(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let (ni, nj) = (a[0].len(), b[0].len());
    // tmp[s][j] = Σ_t c(s, t) b[t][j], with contiguous inner loops.
    let mut tmp = vec![0.0; a.len() * nj];
    for (s, acc) in tmp.chunks_exact_mut(nj).enumerate() {
        for (t, bt) in b.iter().enumerate() {
            let cv = c(s, t);
            if cv == 0.0 {
                continue;
            }
            for (x, &y) in acc.iter_mut().zip(bt) {
                *x += cv * y;
            }
        }
    }
    let mut out = vec![0.0; ni * nj];
    for (av, row) in a.iter().zip(tmp.chunks_exact(nj)) {
        for (o, &tv) in out.chunks_exact_mut(ni).zip(row) {
            for (x, &y) in o.iter_mut().zip(av) {
                *x += tv * y;
            }
        }
    }
    out
}

impl GreenTables {
    /// Builds the tables with `resolution × resolution` cells per face.
    ///
    /// # Panics
    ///
    /// Panics if `resolution` is zero or odd (an even resolution keeps the
    /// sign change of the side-face gradient kernel on a cell boundary).
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
        let even: Vec<f64> = (1..=NTERMS).map(|t| (2 * t) as f64).collect();
        // sin(kπ/2) for odd k is ±1; cos(kπ/2) for even k is ±1.
        let sin_half = |k: f64| (k * pi / 2.0).sin().round();
        let cos_half = |k: f64| (k * pi / 2.0).cos().round();
        let gamma = |a: f64, b: f64| pi * a.hypot(b);

        // Top-face transition density at the centre: 4 s_m s_n / (2 cosh(γ/2)).
        let a_odd = cell_integrals(&odd, sin_half, m);
        let face_prob = bilinear(
            &a_odd,
            &a_odd,
            |s, t| 2.0 / (gamma(odd[s], odd[t]) / 2.0).cosh(),
            m,
        );

        // Gradient (∂/∂z at the centre) kernels.
        // Top face: 4 s_m s_n γ / (2 sinh(γ/2)); bottom face is its negative.
        let top = bilinear(
            &a_odd,
            &a_odd,
            |s, t| {
                let g = gamma(odd[s], odd[t]);
                2.0 * g / (g / 2.0).sinh()
            },
            m,
        );
        // Side face with tangential coordinate u (odd n) and z (even l):
        // 4 s_n (lπ cos(lπ/2)) / (2 cosh(γ/2)).
        let b_even = cell_integrals(&even, |l| l * pi * cos_half(l), m);
        let side = bilinear(
            &a_odd,
            &b_even,
            |s, t| 2.0 / (gamma(odd[s], even[t]) / 2.0).cosh(),
            m,
        );

        let mut grad_cells = Vec::with_capacity(6 * m * m);
        for face in 0..6 {
            match face {
                0..=3 => grad_cells.extend_from_slice(&side), // x-, x+, y-, y+ : (u, z)
                4 => grad_cells.extend(top.iter().map(|v| -v)), // z- : (x, y)
                _ => grad_cells.extend_from_slice(&top),      // z+ : (x, y)
            }
        }
        let abs: Vec<f64> = grad_cells.iter().map(|v| v.abs()).collect();
        let grad_l1 = abs.iter().sum();
        let transition = AliasTable::new(&face_prob);
        let grad = AliasTable::new(&abs);
        Self {
            m,
            face_prob,
            transition,
            grad_cells,
            grad,
            grad_l1,
        }
    }

    /// Cells per face edge.
    #[must_use]
    pub const fn resolution(&self) -> usize {
        self.m
    }

    /// Per-cell transition probabilities of one face (`j * m + i`), summing to 1/6.
    #[must_use]
    pub fn face_probabilities(&self) -> &[f64] {
        &self.face_prob
    }

    /// Signed first-hop kernel integrals for canonical normal +z (see field docs).
    #[must_use]
    pub fn gradient_cells(&self) -> &[f64] {
        &self.grad_cells
    }

    /// `∫|K| dS` on the unit cube (side 1). For a cube of side `2h` the
    /// gradient magnitude scale is `grad_l1() / (2h)`.
    #[must_use]
    pub const fn grad_l1(&self) -> f64 {
        self.grad_l1
    }

    #[inline]
    fn cell_point(&self, cell: usize, rng: &mut Rng) -> (f64, f64) {
        #[allow(clippy::cast_precision_loss)]
        let mf = self.m as f64;
        let (i, j) = (cell % self.m, cell / self.m);
        #[allow(clippy::cast_precision_loss)]
        let u = (i as f64 + rng.next_f64()) / mf;
        #[allow(clippy::cast_precision_loss)]
        let v = (j as f64 + rng.next_f64()) / mf;
        (2.0 * u - 1.0, 2.0 * v - 1.0)
    }

    /// Samples the next walk position on the surface of the cube `[-1, 1]³`
    /// (centre at the origin) from the transition density.
    #[inline]
    pub fn sample_transition(&self, rng: &mut Rng) -> [f64; 3] {
        let face = rng.below(6);
        let cell = self.transition.sample(rng);
        let (u, v) = self.cell_point(cell, rng);
        let w = if face & 1 == 0 { -1.0 } else { 1.0 };
        match face >> 1 {
            0 => [w, u, v],
            1 => [u, w, v],
            _ => [u, v, w],
        }
    }

    /// Samples a first-hop point on the surface of `[-1, 1]³` for the gradient
    /// along `+axis`, returning the point and the sign of the kernel there.
    ///
    /// For a cube of half-side `h` centred at `r`,
    /// `∂φ/∂axis(r) ≈ E[sign · φ(r + h·point)] · grad_l1() / (2h)`.
    #[inline]
    pub fn sample_gradient(&self, axis: Axis, rng: &mut Rng) -> ([f64; 3], f64) {
        let idx = self.grad.sample(rng);
        let sign = self.grad_cells[idx].signum();
        let mm = self.m * self.m;
        let (face, cell) = (idx / mm, idx % mm);
        let (u, v) = self.cell_point(cell, rng);
        let w = if face & 1 == 0 { -1.0 } else { 1.0 };
        // Canonical frame: normal z. Side faces carry (tangential, z).
        let c = match face >> 1 {
            0 => [w, u, v],
            1 => [u, w, v],
            _ => [u, v, w],
        };
        // Map canonical z onto `axis`; the kernel is symmetric under swapping x and y.
        let p = match axis {
            Axis::Z => c,
            Axis::X => [c[2], c[0], c[1]],
            Axis::Y => [c[0], c[2], c[1]],
        };
        (p, sign)
    }
}

impl Default for GreenTables {
    fn default() -> Self {
        Self::new(DEFAULT_RESOLUTION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_face_has_probability_one_sixth() {
        for m in [2, 8, 64] {
            let g = GreenTables::new(m);
            let s: f64 = g.face_probabilities().iter().sum();
            assert!((s - 1.0 / 6.0).abs() < 1e-13, "m={m} s={s}");
            assert!(g.face_probabilities().iter().all(|&p| p > 0.0));
        }
    }

    #[test]
    fn transition_reproduces_harmonic_mean_value() {
        // φ = z on [0,1]³ → φ(centre) = 1/2; φ = x² − z² → 0. Cell-centre quadrature.
        let m = 64;
        let g = GreenTables::new(m);
        let c = |i: usize| (i as f64 + 0.5) / m as f64;
        let p = g.face_probabilities();
        let mut lin = 0.0;
        let mut quad = 0.0;
        for j in 0..m {
            for i in 0..m {
                let w = p[j * m + i];
                // z on faces: top 1, bottom 0, 4 sides use v = z.
                lin += w * (1.0 + 4.0 * c(j));
                // x² − z²: top (x=c(i), z=1), bottom (z=0), x± faces (x∈{0,1}, z=c(j)), y± faces (x=c(i), z=c(j)).
                quad += w
                    * ((c(i).powi(2) - 1.0)
                        + c(i).powi(2)
                        + (0.0 - c(j).powi(2))
                        + (1.0 - c(j).powi(2))
                        + 2.0 * (c(i).powi(2) - c(j).powi(2)));
            }
        }
        assert!((lin - 0.5).abs() < 1e-12, "lin={lin}");
        assert!(quad.abs() < 1e-4, "quad={quad}");
    }

    #[test]
    fn gradient_kernel_sum_rules() {
        let m = 64;
        let g = GreenTables::new(m);
        let k = g.gradient_cells();
        let mm = m * m;
        let c = |i: usize| (i as f64 + 0.5) / m as f64;
        // Constant potential has zero gradient.
        let total: f64 = k.iter().sum();
        assert!(total.abs() < 1e-12, "total={total}");
        // φ = z has ∂φ/∂z = 1: top face φ=1, bottom 0, sides φ = v (second coord).
        let mut dz = k[5 * mm..].iter().sum::<f64>();
        for face in 0..4 {
            for j in 0..m {
                for i in 0..m {
                    dz += k[face * mm + j * m + i] * c(j);
                }
            }
        }
        assert!((dz - 1.0).abs() < 1e-3, "dz={dz}");
        // Top face kernel is positive; side kernels are antisymmetric in z.
        assert!(k[5 * mm..].iter().all(|&v| v > 0.0));
        for j in 0..m / 2 {
            for i in 0..m {
                let (lo, hi) = (k[j * m + i], k[(m - 1 - j) * m + i]);
                assert!((lo + hi).abs() < 1e-12 * lo.abs().max(1e-300) + 1e-15);
            }
        }
    }

    #[test]
    fn sampling_is_on_the_surface_and_unbiased() {
        let g = GreenTables::new(16);
        let mut rng = Rng::from_seed(3);
        let n = 200_000;
        let mut mean_z = 0.0;
        for _ in 0..n {
            let p = g.sample_transition(&mut rng);
            assert!(
                p.iter().any(|v| (v.abs() - 1.0).abs() < 1e-15) && p.iter().all(|v| v.abs() <= 1.0)
            );
            mean_z += p[2];
        }
        assert!((mean_z / f64::from(n)).abs() < 0.01);
        // Gradient estimator of φ = x (along X) on [-1,1]³: ∂φ/∂x = 1.
        // In unit-cube terms: E[sign·φ_unit]·grad_l1, with φ_unit = (x+1)/2.
        let mut acc = 0.0;
        for _ in 0..n {
            let (p, s) = g.sample_gradient(Axis::X, &mut rng);
            acc += s * (p[0] + 1.0) / 2.0;
        }
        let est = acc / f64::from(n) * g.grad_l1();
        assert!((est - 1.0).abs() < 0.03, "est={est}");
    }
}
