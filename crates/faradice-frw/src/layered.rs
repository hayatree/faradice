//! Kernels of a cube that contains one planar dielectric interface
//! (design M2a, `docs/design/M2_multi_dielectric.md` §3.2 and §4).
//!
//! Canonical frame: the unit cube `[0,1]³` with centre `(½, ½, ½)` and the
//! interface at `z = ζ = ½ + t/2` (at or above the centre). The centre's layer
//! has relative permittivity 1, the other layer `κ`. An interface below the
//! centre uses the same tables with `z` mirrored.
//!
//! - **Top / bottom faces**: double sine series in `(x, y)`; per mode the
//!   two-layer profile is propagated from one end with continuity of the
//!   function and of `ε ×` its derivative.
//! - **Side faces**: sine series in the tangential coordinate times the
//!   eigenfunctions `ψ_k(z)` of `−(εψ′)′ = μ² ε ψ`, `ψ(0) = ψ(1) = 0`, which are
//!   orthogonal with weight `ε`. The eigenvalues are the solutions of
//!   `θ(1; μ) = kπ` for the Prüfer angle `θ` of the propagated solution, which
//!   is strictly increasing in `μ`, so every root is bracketed exactly once.
//!
//! Three kinds of kernels are built from the same series:
//!
//! - [`LayeredCube`]: transition probabilities at the centre;
//! - [`LayeredGradients`]: first-hop gradient kernels at the centre, along a
//!   tangential axis (`x`) and the normal axis (`z`);
//! - [`half_band_gradient`]: the tangential gradient kernel averaged over a
//!   segment of the vertical line through the centre on one side of an
//!   interface through the centre (§4, lateral Gaussian faces).
//!
//! The interface lies on a cell boundary of the `M × M` face grid, so every
//! cell integral is a closed-form integral of sines and hyperbolic functions.
//! As for the homogeneous tables, only the series is truncated.

use std::f64::consts::PI;

use crate::alias::AliasTable;
use crate::green::{NTERMS, bilinear, cell_integrals};
use crate::rng::Rng;

/// Number of Sturm–Liouville modes on the side faces (their terms decay like
/// `e^{−μ d}` at distance `d` from the face; `μ_k ≈ kπ`).
const NMODES: usize = 2 * NTERMS;

/// Supported contrasts `ε_other / ε_centre` (relative permittivities of real
/// stacks are between 1 and a few tens).
pub const KAPPA_RANGE: std::ops::RangeInclusive<f64> = 1e-3..=1e3;

/// One side-face eigenmode.
#[derive(Debug, Clone, Copy)]
struct Mode {
    mu: f64,
    /// `ψ` above the interface is `a·cos(μ(z−ζ)) + b·sin(μ(z−ζ))`.
    a: f64,
    b: f64,
    /// `∫ ε ψ² dz`.
    norm: f64,
}

/// Prüfer angle at `z = 1` of the solution with `ψ = sin(μz)` below `ζ`.
fn prufer_end(mu: f64, zeta: f64, kappa: f64) -> f64 {
    let below = mu * zeta;
    let half_periods = (below / PI).floor();
    let phi = below - half_periods * PI;
    // tan θ⁺ = κ tan θ⁻ in the same half period (ψ continuous, ψ′ divided by κ).
    let g = (kappa * phi.sin()).atan2(phi.cos());
    half_periods.mul_add(PI, g) + mu * (1.0 - zeta)
}

/// The first `n` eigenmodes for interface height `zeta` and contrast `kappa`.
fn modes(n: usize, zeta: f64, kappa: f64) -> Vec<Mode> {
    (1..=n)
        .map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let target = k as f64 * PI;
            // θ(1; μ) ∈ (μ − π, μ + π), so the root lies in ((k−1)π, (k+1)π).
            let (mut lo, mut hi) = (target - PI, target + PI);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if prufer_end(mid, zeta, kappa) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
                if hi - lo <= 4.0 * f64::EPSILON * hi {
                    break;
                }
            }
            let mu = 0.5 * (lo + hi);
            let (a, b) = ((mu * zeta).sin(), (mu * zeta).cos() / kappa);
            let len = 1.0 - zeta;
            let s2 = (2.0 * mu * len).sin() / (4.0 * mu);
            let below = 0.5 * zeta - (2.0 * mu * zeta).sin() / (4.0 * mu);
            let above = a * a * (0.5 * len + s2)
                + b * b * (0.5 * len - s2)
                + a * b * (1.0 - (2.0 * mu * len).cos()) / (2.0 * mu);
            Mode {
                mu,
                a,
                b,
                norm: below + kappa * above,
            }
        })
        .collect()
}

/// Where a kernel is evaluated along the vertical line `x = y = ½`.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Value at height `z`.
    Value(f64),
    /// `z`-derivative at height `z`.
    Slope(f64),
    /// `z`-derivative at height `z`, the lower trace at `z = ζ` (in the
    /// centre's layer, ε = 1: the normal flux).
    SlopeBelow(f64),
    /// Average over `[a, b]`, which lies on one side of the interface.
    Mean(f64, f64),
}

impl Mode {
    /// `∫_{lo}^{hi} ψ dz` over a range on one side of `zeta`.
    fn integral(&self, zeta: f64, lo: f64, hi: f64) -> f64 {
        let mu = self.mu;
        if hi <= zeta {
            ((mu * lo).cos() - (mu * hi).cos()) / mu
        } else {
            let (u0, u1) = (mu * (lo - zeta), mu * (hi - zeta));
            (self.a * (u1.sin() - u0.sin()) + self.b * (u0.cos() - u1.cos())) / mu
        }
    }

    fn eval(&self, zeta: f64, e: Eval) -> f64 {
        let mu = self.mu;
        match e {
            Eval::Value(z) if z <= zeta => (mu * z).sin(),
            Eval::Value(z) => {
                let u = mu * (z - zeta);
                self.a * u.cos() + self.b * u.sin()
            }
            Eval::Slope(z) if z < zeta => mu * (mu * z).cos(),
            Eval::SlopeBelow(z) if z <= zeta => mu * (mu * z).cos(),
            Eval::Slope(z) | Eval::SlopeBelow(z) => {
                let u = mu * (z - zeta);
                mu * (self.b * u.cos() - self.a * u.sin())
            }
            Eval::Mean(a, b) => self.integral(zeta, a, b) / (b - a),
        }
    }
}

/// Profile for data on the top face, `Z(z)/Z(1)`: `sinh(γz)` below `ζ`,
/// propagated above.
fn top_profile(g: f64, zeta: f64, kappa: f64, e: Eval) -> f64 {
    let (gz, gl) = (g * zeta, g * (1.0 - zeta));
    let z1 = gz.sinh() * gl.cosh() + gz.cosh() * gl.sinh() / kappa;
    let (sz, cz) = (gz.sinh(), gz.cosh());
    let v = match e {
        Eval::Value(z) if z <= zeta => (g * z).sinh(),
        Eval::Value(z) => {
            let u = g * (z - zeta);
            sz * u.cosh() + cz * u.sinh() / kappa
        }
        Eval::Slope(z) if z < zeta => g * (g * z).cosh(),
        Eval::SlopeBelow(z) if z <= zeta => g * (g * z).cosh(),
        Eval::Slope(z) | Eval::SlopeBelow(z) => {
            let u = g * (z - zeta);
            g * (sz * u.sinh() + cz * u.cosh() / kappa)
        }
        Eval::Mean(a, b) if b <= zeta => ((g * b).cosh() - (g * a).cosh()) / (g * (b - a)),
        Eval::Mean(a, b) => {
            let (u0, u1) = (g * (a - zeta), g * (b - zeta));
            (sz * (u1.sinh() - u0.sinh()) + cz * (u1.cosh() - u0.cosh()) / kappa) / (g * (b - a))
        }
    };
    v / z1
}

/// Profile for data on the bottom face, `W(z)/W(0)`: `sinh(γ(1−z))` above `ζ`,
/// propagated below.
fn bottom_profile(g: f64, zeta: f64, kappa: f64, e: Eval) -> f64 {
    let (gz, gl) = (g * zeta, g * (1.0 - zeta));
    let w0 = gl.sinh() * gz.cosh() + kappa * gl.cosh() * gz.sinh();
    let (sl, cl) = (gl.sinh(), gl.cosh());
    let v = match e {
        Eval::Value(z) if z >= zeta => (g * (1.0 - z)).sinh(),
        Eval::Value(z) => {
            let u = g * (zeta - z);
            sl * u.cosh() + kappa * cl * u.sinh()
        }
        Eval::Slope(z) | Eval::SlopeBelow(z) if z > zeta => -g * (g * (1.0 - z)).cosh(),
        Eval::Slope(z) | Eval::SlopeBelow(z) => {
            let u = g * (zeta - z);
            -g * (sl * u.sinh() + kappa * cl * u.cosh())
        }
        Eval::Mean(a, b) if a >= zeta => {
            ((g * (1.0 - a)).cosh() - (g * (1.0 - b)).cosh()) / (g * (b - a))
        }
        Eval::Mean(a, b) => {
            let (u0, u1) = (g * (zeta - b), g * (zeta - a));
            (sl * (u1.sinh() - u0.sinh()) + kappa * cl * (u1.cosh() - u0.cosh())) / (g * (b - a))
        }
    };
    v / w0
}

/// Series data shared by all kernels of one `(κ, ζ, M)`.
struct Series {
    m: usize,
    zeta: f64,
    kappa: f64,
    odd: Vec<f64>,
    even: Vec<f64>,
    /// `s_k(½) ∫_cell sin(kπu)` for odd `k`.
    a_odd: Vec<Vec<f64>>,
    /// `kπ cos(kπ/2) ∫_cell sin(kπu)` for even `k` (∂/∂u at the centre).
    d_even: Vec<Vec<f64>>,
    modes: Vec<Mode>,
    /// `∫_cell ε ψ_k / N_k` per mode and z-cell.
    e_modes: Vec<Vec<f64>>,
}

impl Series {
    fn new(kappa: f64, zeta: f64, m: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let odd: Vec<f64> = (0..NTERMS).map(|s| (2 * s + 1) as f64).collect();
        #[allow(clippy::cast_precision_loss)]
        let even: Vec<f64> = (1..=NTERMS).map(|s| (2 * s) as f64).collect();
        let a_odd = cell_integrals(&odd, |k| (k * PI / 2.0).sin().round(), m);
        let d_even = cell_integrals(&even, |k| k * PI * (k * PI / 2.0).cos().round(), m);
        let modes = modes(NMODES, zeta, kappa);
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        let e_modes = modes
            .iter()
            .map(|mo| {
                #[allow(clippy::cast_precision_loss)]
                (0..m)
                    .map(|l| {
                        let (lo, hi) = (l as f64 / mf, (l + 1) as f64 / mf);
                        let eps = if hi <= zeta { 1.0 } else { kappa };
                        eps * mo.integral(zeta, lo, hi) / mo.norm
                    })
                    .collect()
            })
            .collect();
        Self {
            m,
            zeta,
            kappa,
            odd,
            even,
            a_odd,
            d_even,
            modes,
            e_modes,
        }
    }

    /// Top or bottom face: `Σ 4 A[s][i] B[t][j] profile(γ_st)`.
    fn horizontal(&self, top: bool, derivative_x: bool, e: Eval) -> Vec<f64> {
        let (ax, kx) = if derivative_x {
            (&self.d_even, &self.even)
        } else {
            (&self.a_odd, &self.odd)
        };
        let prof = |g: f64| {
            if top {
                top_profile(g, self.zeta, self.kappa, e)
            } else {
                bottom_profile(g, self.zeta, self.kappa, e)
            }
        };
        bilinear(
            ax,
            &self.a_odd,
            |s, t| 4.0 * prof(PI * kx[s].hypot(self.odd[t])),
            self.m,
        )
    }

    /// A side face with tangential coordinate `u` and `z`:
    /// `Σ 2 A[s][i] E_k(j) factor(β, ψ_k)`.
    fn side(&self, derivative_u: bool, factor: impl Fn(f64, &Mode) -> f64) -> Vec<f64> {
        let (au, ku) = if derivative_u {
            (&self.d_even, &self.even)
        } else {
            (&self.a_odd, &self.odd)
        };
        bilinear(
            au,
            &self.e_modes,
            |s, k| {
                let mo = &self.modes[k];
                2.0 * factor((ku[s] * PI).hypot(mo.mu), mo)
            },
            self.m,
        )
    }

    /// Gradient along `x` of the solution evaluated by `e` (a value or a mean
    /// on the line `x = y = ½`), faces `[x-, x+, y-, y+, z-, z+]`.
    fn gradient_x(&self, e: Eval) -> Vec<f64> {
        let z = self.zeta;
        // x = 1 face: ∂_x [sinh(βx)/sinh β] at ½ = β / (2 sinh(β/2)).
        let xp = self.side(false, |beta, mo| {
            mo.eval(z, e) * beta / (2.0 * (beta / 2.0).sinh())
        });
        // y faces: ∂_x of the tangential sine, sinh(β/2)/sinh β = 1/(2 cosh(β/2)).
        let yf = self.side(true, |beta, mo| mo.eval(z, e) / (2.0 * (beta / 2.0).cosh()));
        let mut out = Vec::with_capacity(6 * self.m * self.m);
        out.extend(xp.iter().map(|v| -v));
        out.extend_from_slice(&xp);
        out.extend_from_slice(&yf);
        out.extend_from_slice(&yf);
        out.extend(self.horizontal(false, true, e));
        out.extend(self.horizontal(true, true, e));
        out
    }

    /// Gradient along `z` (evaluated by `e`, a slope), faces
    /// `[x-, x+, y-, y+, z-, z+]`.
    fn gradient_z(&self, e: Eval) -> Vec<f64> {
        let z = self.zeta;
        let side = self.side(false, |beta, mo| {
            mo.eval(z, e) / (2.0 * (beta / 2.0).cosh())
        });
        let mut out = Vec::with_capacity(6 * self.m * self.m);
        for _ in 0..4 {
            out.extend_from_slice(&side);
        }
        out.extend(self.horizontal(false, false, e));
        out.extend(self.horizontal(true, false, e));
        out
    }

    /// Transition probabilities at the centre.
    fn transition(&self) -> Vec<f64> {
        let z = self.zeta;
        let e = Eval::Value(0.5);
        let side = self.side(false, |beta, mo| {
            mo.eval(z, e) / (2.0 * (beta / 2.0).cosh())
        });
        let mut out = Vec::with_capacity(6 * self.m * self.m);
        for _ in 0..4 {
            out.extend_from_slice(&side);
        }
        out.extend(self.horizontal(false, false, e));
        out.extend(self.horizontal(true, false, e));
        out
    }
}

/// Checks the common arguments and returns `ζ`.
fn canonical_zeta(kappa: f64, offset: usize, offsets: usize, resolution: usize) -> f64 {
    assert!(
        KAPPA_RANGE.contains(&kappa),
        "kappa must lie in {KAPPA_RANGE:?}"
    );
    assert!(offset < offsets, "offset must be below offsets");
    assert!(
        resolution > 0 && resolution % (2 * offsets) == 0,
        "resolution must be a multiple of 2 * offsets"
    );
    let m = resolution;
    // ζ·m is an integer: the interface is a cell boundary.
    let zeta_cells = m / 2 + offset * m / (2 * offsets);
    #[allow(clippy::cast_precision_loss)]
    let zeta = zeta_cells as f64 / m as f64;
    zeta
}

/// Samples a point of the cube `[-1, 1]³` from a cell index of the six-face
/// layout, uniformly within the cell.
#[inline]
fn cell_point(m: usize, idx: usize, rng: &mut Rng) -> [f64; 3] {
    let mm = m * m;
    let (face, cell) = (idx / mm, idx % mm);
    #[allow(clippy::cast_precision_loss)]
    let mf = m as f64;
    let (i, j) = (cell % m, cell / m);
    #[allow(clippy::cast_precision_loss)]
    let u = 2.0 * (i as f64 + rng.next_f64()) / mf - 1.0;
    #[allow(clippy::cast_precision_loss)]
    let v = 2.0 * (j as f64 + rng.next_f64()) / mf - 1.0;
    let w = if face & 1 == 0 { -1.0 } else { 1.0 };
    match face >> 1 {
        0 => [w, u, v],
        1 => [u, w, v],
        _ => [u, v, w],
    }
}

/// Transition tables of the canonical layered cube.
#[derive(Debug, Clone)]
pub struct LayeredCube {
    m: usize,
    kappa: f64,
    t: f64,
    /// Per-cell probabilities, faces `[x-, x+, y-, y+, z-, z+]`, `m²` cells
    /// each, cell `j * m + i` with `(i, j)` the face's tangential axes in the
    /// order of [`crate::green::GreenTables`] (side faces: `(tangential, z)`).
    cells: Vec<f64>,
    transition: AliasTable,
}

impl LayeredCube {
    /// Builds the tables for contrast `kappa = ε_other / ε_centre`, interface
    /// offset `t = offset / offsets` (in units of the half-side) and
    /// `resolution × resolution` cells per face.
    ///
    /// # Panics
    ///
    /// Panics if `kappa` is outside [`KAPPA_RANGE`], `offset >= offsets`, or
    /// `resolution` is not a positive multiple of `2 * offsets` (the interface
    /// must lie on a cell boundary).
    #[must_use]
    pub fn new(kappa: f64, offset: usize, offsets: usize, resolution: usize) -> Self {
        let zeta = canonical_zeta(kappa, offset, offsets, resolution);
        let mut cells = Series::new(kappa, zeta, resolution).transition();
        // Truncation can leave values of order 1e-17 below zero in far cells.
        for c in &mut cells {
            assert!(
                c.is_finite() && *c > -1e-12,
                "layered table out of range: {c}"
            );
            *c = c.max(0.0);
        }
        let transition = AliasTable::new(&symmetry_classes(&cells, resolution));
        #[allow(clippy::cast_precision_loss)]
        let t = offset as f64 / offsets as f64;
        Self {
            m: resolution,
            kappa,
            t,
            cells,
            transition,
        }
    }

    /// Cells per face edge.
    #[must_use]
    pub const fn resolution(&self) -> usize {
        self.m
    }

    /// Contrast `ε_other / ε_centre`.
    #[must_use]
    pub const fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Interface offset above the centre in units of the half-side.
    #[must_use]
    pub const fn offset(&self) -> f64 {
        self.t
    }

    /// Per-cell probabilities of all six faces (see the field layout).
    #[must_use]
    pub fn cell_probabilities(&self) -> &[f64] {
        &self.cells
    }

    /// Samples the next walk position on the surface of the cube `[-1, 1]³`
    /// (centre at the origin, interface at `z = t`).
    #[inline]
    pub fn sample_transition(&self, rng: &mut Rng) -> [f64; 3] {
        let m = self.m;
        let h = m / 2;
        let r = self.transition.sample(rng);
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        #[allow(clippy::cast_precision_loss)]
        let coord = |c: usize, rng: &mut Rng| 2.0 * (c as f64 + rng.next_f64()) / mf - 1.0;
        if r < 2 * h * h {
            // Horizontal faces: a cell of the positive quadrant, then a random
            // reflection in x and in y.
            let rem = r % (h * h);
            let (i, j) = (h + rem % h, h + rem / h);
            let (u, v) = (coord(i, rng), coord(j, rng));
            let bits = rng.below(4);
            let u = if bits & 1 == 0 { u } else { -u };
            let v = if bits & 2 == 0 { v } else { -v };
            [u, v, if r < h * h { -1.0 } else { 1.0 }]
        } else {
            // Side faces: a cell of the positive tangential half, then one of
            // the four faces and a random tangential reflection.
            let rem = r - 2 * h * h;
            let (i, j) = (h + rem % h, rem / h);
            let (u, v) = (coord(i, rng), coord(j, rng));
            let bits = rng.below(8);
            let u = if bits & 4 == 0 { u } else { -u };
            let w = if bits & 1 == 0 { -1.0 } else { 1.0 };
            if bits & 2 == 0 { [w, u, v] } else { [u, w, v] }
        }
    }
}

/// Folds the six-face cell probabilities of a layered cube into its symmetry
/// classes (the cube is invariant under `x ↔ y`, `x → −x` and `y → −y`):
/// the positive quadrant of the bottom and the top face (`(m/2)²` cells each),
/// then the positive tangential half of one side face (`m/2 × m` cells), each
/// class weighted by the sum of its members. Sampling a class and a uniform
/// member reproduces the six-face distribution with a sixth of the table.
fn symmetry_classes(cells: &[f64], m: usize) -> Vec<f64> {
    let (h, mm) = (m / 2, m * m);
    let mut out = Vec::with_capacity(mm);
    for face in [4, 5] {
        for j in h..m {
            for i in h..m {
                out.push(
                    [
                        (i, j),
                        (m - 1 - i, j),
                        (i, m - 1 - j),
                        (m - 1 - i, m - 1 - j),
                    ]
                    .iter()
                    .map(|&(a, b)| cells[face * mm + b * m + a])
                    .sum(),
                );
            }
        }
    }
    for j in 0..m {
        for i in h..m {
            out.push(
                (0..4)
                    .flat_map(|f| [cells[f * mm + j * m + i], cells[f * mm + j * m + m - 1 - i]])
                    .sum(),
            );
        }
    }
    out
}

/// A signed first-hop kernel over the six faces, sampled from `|K|`.
///
/// For a cube of half-side `h`, the derivative is
/// `E[sign · φ(centre + h·point)] · l1() / (2h)`, as for
/// [`crate::green::GreenTables::sample_gradient`].
#[derive(Debug, Clone)]
pub struct GradientKernel {
    m: usize,
    cells: Vec<f64>,
    alias: AliasTable,
    l1: f64,
    /// Samplers of the positive and negative parts.
    parts: [AliasTable; 2],
}

impl GradientKernel {
    fn new(m: usize, cells: Vec<f64>) -> Self {
        assert!(
            cells.iter().all(|c| c.is_finite()),
            "gradient kernel is not finite"
        );
        let abs: Vec<f64> = cells.iter().map(|c| c.abs()).collect();
        let l1 = abs.iter().sum();
        let part = |positive: bool| {
            let w: Vec<f64> = cells
                .iter()
                .map(|&c| if (c > 0.0) == positive { c.abs() } else { 0.0 })
                .collect();
            AliasTable::new(&w)
        };
        Self {
            m,
            alias: AliasTable::new(&abs),
            parts: [part(true), part(false)],
            cells,
            l1,
        }
    }

    /// Signed cell integrals (unit cube), faces `[x-, x+, y-, y+, z-, z+]`.
    #[must_use]
    pub fn cells(&self) -> &[f64] {
        &self.cells
    }

    /// `∫|K| dS` on the unit cube.
    #[must_use]
    pub const fn l1(&self) -> f64 {
        self.l1
    }

    /// Samples a point on the surface of `[-1, 1]³` and the kernel's sign
    /// there, in the canonical frame of the cube.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> ([f64; 3], f64) {
        let idx = self.alias.sample(rng);
        let sign = self.cells[idx].signum();
        (cell_point(self.m, idx, rng), sign)
    }

    /// Samples one point from the positive part and one from the negative
    /// part. The kernel of a gradient integrates to zero, so both parts
    /// carry half of [`Self::l1`] and the derivative is
    /// `(E[φ(c + h·p₊)] − E[φ(c + h·p₋)]) · l1() / (4h)`.
    #[inline]
    pub fn sample_pair(&self, rng: &mut Rng) -> [[f64; 3]; 2] {
        let mut point = |k: usize| {
            let idx = self.parts[k].sample(rng);
            cell_point(self.m, idx, rng)
        };
        [point(0), point(1)]
    }
}

/// First-hop gradient kernels at the centre of a layered cube.
#[derive(Debug, Clone)]
pub struct LayeredGradients {
    /// Along the tangential axis `x` (`y` by symmetry).
    pub x: GradientKernel,
    /// Along the normal axis `z` (towards the interface).
    pub z: GradientKernel,
}

impl LayeredGradients {
    /// Builds the kernels for the same parameters as [`LayeredCube::new`].
    ///
    /// # Panics
    ///
    /// As [`LayeredCube::new`]; additionally `offset` must be positive (the
    /// centre must not lie on the interface, where `∂φ/∂z` has two traces).
    #[must_use]
    pub fn new(kappa: f64, offset: usize, offsets: usize, resolution: usize) -> Self {
        assert!(offset > 0, "gradients need the centre off the interface");
        let zeta = canonical_zeta(kappa, offset, offsets, resolution);
        let s = Series::new(kappa, zeta, resolution);
        Self {
            x: GradientKernel::new(resolution, s.gradient_x(Eval::Value(0.5))),
            z: GradientKernel::new(resolution, s.gradient_z(Eval::Slope(0.5))),
        }
    }

    /// Kernels at a centre **on** the interface (`t = 0`, design D3 §4.4c):
    /// `x` is the tangential gradient (continuous across the interface), and
    /// `z` the lower trace of `∂φ/∂z`, i.e. the normal flux `ε ∂φ/∂z` in units
    /// of the centre layer's ε (the layer below the interface, ε = 1).
    ///
    /// # Panics
    ///
    /// As [`LayeredCube::new`] with offset 0.
    #[must_use]
    pub fn on_interface(kappa: f64, offsets: usize, resolution: usize) -> Self {
        let zeta = canonical_zeta(kappa, 0, offsets, resolution);
        let s = Series::new(kappa, zeta, resolution);
        Self {
            x: GradientKernel::new(resolution, s.gradient_x(Eval::Value(0.5))),
            z: GradientKernel::new(resolution, s.gradient_z(Eval::SlopeBelow(0.5))),
        }
    }
}

/// Side of the interface for [`half_band_gradient`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandSide {
    /// Below the interface (the centre's layer, relative permittivity 1).
    Below,
    /// Above the interface (relative permittivity `κ`).
    Above,
}

/// Tangential (`x`) gradient kernel of the cube whose centre lies on the
/// interface (`t = 0`), averaged over the vertical segment from the centre
/// to `±η·h` on side `side`, with `η = eta_num / eta_den ≤ ½` (design §4,
/// half-band estimator). The permittivity of the band is not included.
///
/// # Panics
///
/// Panics if `kappa` is outside [`KAPPA_RANGE`], `resolution` is not a
/// positive even number, or `η` is not in `(0, ½]`.
#[must_use]
pub fn half_band_gradient(
    kappa: f64,
    eta_num: usize,
    eta_den: usize,
    side: BandSide,
    resolution: usize,
) -> GradientKernel {
    assert!(
        eta_num > 0 && 2 * eta_num <= eta_den,
        "eta must lie in (0, 1/2]"
    );
    let zeta = canonical_zeta(kappa, 0, 1, resolution);
    #[allow(clippy::cast_precision_loss)]
    let half = 0.5 * eta_num as f64 / eta_den as f64; // η·h in unit-cube length
    let e = match side {
        BandSide::Below => Eval::Mean(zeta - half, zeta),
        BandSide::Above => Eval::Mean(zeta, zeta + half),
    };
    let s = Series::new(kappa, zeta, resolution);
    GradientKernel::new(resolution, s.gradient_x(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::green::GreenTables;

    const KAPPAS: [f64; 5] = [0.1, 0.5, 1.0, 2.0, 10.0];

    /// `Σ K · ⟨φ⟩_cell` for a potential `φ(x, y, z)` that is bilinear within
    /// every cell (cell average = value at the cell centre).
    fn apply(k: &[f64], m: usize, phi: impl Fn(f64, f64, f64) -> f64) -> f64 {
        let mm = m * m;
        #[allow(clippy::cast_precision_loss)]
        let c = |i: usize| (i as f64 + 0.5) / m as f64;
        k.iter()
            .enumerate()
            .map(|(idx, w)| {
                let (face, cell) = (idx / mm, idx % mm);
                let (u, v) = (c(cell % m), c(cell / m));
                let p = match face {
                    0 => phi(0.0, u, v),
                    1 => phi(1.0, u, v),
                    2 => phi(u, 0.0, v),
                    3 => phi(u, 1.0, v),
                    4 => phi(u, v, 0.0),
                    _ => phi(u, v, 1.0),
                };
                w * p
            })
            .sum()
    }

    /// Layered linear profile: `z` below `ζ`, slope `1/κ` above.
    fn layered_linear(zeta: f64, kappa: f64) -> impl Fn(f64) -> f64 {
        move |z| {
            if z <= zeta {
                z
            } else {
                zeta + (z - zeta) / kappa
            }
        }
    }

    #[test]
    fn normal_kernel_parts_carry_half_of_the_mass_each() {
        // ∫K_z = 0 (the gradient of a constant): the paired first hop gives
        // each part ½·l1. Truncation leaves only a tiny imbalance.
        for kappa in KAPPAS {
            for offset in [1, 4, 8, 15] {
                let g = LayeredGradients::new(kappa, offset, 16, 32);
                let sum: f64 = g.z.cells().iter().sum();
                assert!(sum.abs() < 1e-9 * g.z.l1(), "κ={kappa} t={offset}: {sum}");
            }
        }
    }

    #[test]
    fn unit_contrast_gradient_matches_the_homogeneous_kernel() {
        let m = 32;
        let h = GreenTables::new(m);
        for offset in [1, 5] {
            let g = LayeredGradients::new(1.0, offset, 8, m);
            for (a, b) in g.z.cells().iter().zip(h.gradient_cells()) {
                assert!((a - b).abs() < 1e-12, "offset={offset}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn gradients_reproduce_exact_layered_potentials() {
        let m = 32;
        for &kappa in &KAPPAS {
            for offset in [1, 4, 7] {
                let g = LayeredGradients::new(kappa, offset, 8, m);
                #[allow(clippy::cast_precision_loss)]
                let zeta = 0.5 + 0.5 * offset as f64 / 8.0;
                let f = layered_linear(zeta, kappa);
                let tag = format!("κ={kappa} offset={offset}");
                // Constants have no gradient.
                assert!(g.x.cells().iter().sum::<f64>().abs() < 1e-10, "{tag}");
                assert!(g.z.cells().iter().sum::<f64>().abs() < 1e-10, "{tag}");
                // φ = f(z): ∂z = 1 at the centre (below ζ), ∂x = 0.
                let dz = apply(g.z.cells(), m, |_, _, z| f(z));
                assert!((dz - 1.0).abs() < 1e-9, "{tag} dz={dz}");
                assert!(apply(g.x.cells(), m, |_, _, z| f(z)).abs() < 1e-9, "{tag}");
                // φ = x: ∂x = 1. φ = (x − ½) f(z): ∂x = f(½) = ½, ∂z = 0.
                let dx = apply(g.x.cells(), m, |x, _, _| x);
                assert!((dx - 1.0).abs() < 1e-9, "{tag} dx={dx}");
                let dx = apply(g.x.cells(), m, |x, _, z| (x - 0.5) * f(z));
                assert!((dx - 0.5).abs() < 1e-9, "{tag} dx={dx}");
                let dz = apply(g.z.cells(), m, |x, _, z| (x - 0.5) * f(z));
                assert!(dz.abs() < 1e-9, "{tag} dz={dz}");
            }
        }
    }

    #[test]
    fn interface_kernels_give_the_tangential_field_and_the_flux() {
        let m = 32;
        for &kappa in &KAPPAS {
            let g = LayeredGradients::on_interface(kappa, 8, m);
            let f = layered_linear(0.5, kappa);
            let tag = format!("κ={kappa}");
            assert!(g.x.cells().iter().sum::<f64>().abs() < 1e-10, "{tag}");
            assert!(g.z.cells().iter().sum::<f64>().abs() < 1e-10, "{tag}");
            // φ = f(z): flux 1 (slope 1 below, 1/κ above), no tangential field.
            let dz = apply(g.z.cells(), m, |_, _, z| f(z));
            assert!((dz - 1.0).abs() < 1e-9, "{tag} dz={dz}");
            assert!(apply(g.x.cells(), m, |_, _, z| f(z)).abs() < 1e-9, "{tag}");
            // φ = x: ∂x = 1, no flux. φ = (x − ½) f(z): ∂x = f(½) = ½, flux 0.
            let dx = apply(g.x.cells(), m, |x, _, _| x);
            assert!((dx - 1.0).abs() < 1e-9, "{tag} dx={dx}");
            assert!(apply(g.z.cells(), m, |x, _, _| x).abs() < 1e-9, "{tag}");
            let dx = apply(g.x.cells(), m, |x, _, z| (x - 0.5) * f(z));
            assert!((dx - 0.5).abs() < 1e-9, "{tag} dx={dx}");
            let dz = apply(g.z.cells(), m, |x, _, z| (x - 0.5) * f(z));
            assert!(dz.abs() < 1e-9, "{tag} dz={dz}");
            // φ = (x − ½)² − (y − ½)² (harmonic, no z dependence): ∂x = 0.
            let d = apply(g.x.cells(), m, |x, y, _| {
                (x - 0.5).powi(2) - (y - 0.5).powi(2)
            });
            assert!(d.abs() < 1e-9, "{tag} quad dx={d}");
        }
    }

    #[test]
    fn half_band_kernels_average_the_tangential_gradient() {
        let m = 32;
        for &kappa in &KAPPAS {
            let f = layered_linear(0.5, kappa);
            for (num, den) in [(1, 2), (1, 8), (3, 16)] {
                #[allow(clippy::cast_precision_loss)]
                let half = 0.5 * num as f64 / den as f64;
                for side in [BandSide::Below, BandSide::Above] {
                    let k = half_band_gradient(kappa, num, den, side, m);
                    let tag = format!("κ={kappa} η={num}/{den} {side:?}");
                    assert!(k.cells().iter().sum::<f64>().abs() < 1e-10, "{tag}");
                    let dx = apply(k.cells(), m, |x, _, _| x);
                    assert!((dx - 1.0).abs() < 1e-9, "{tag} dx={dx}");
                    // ∂x[(x − ½) f(z)] = f(z), averaged over the band.
                    let mean = match side {
                        BandSide::Below => 0.5 - 0.5 * half,
                        BandSide::Above => 0.5 + 0.5 * half / kappa,
                    };
                    let dx = apply(k.cells(), m, |x, _, z| (x - 0.5) * f(z));
                    assert!((dx - mean).abs() < 1e-9, "{tag} dx={dx} mean={mean}");
                }
            }
        }
    }

    #[test]
    fn narrow_half_bands_approach_the_centre_kernel() {
        // κ = 1: the centre kernel is the homogeneous ∂x kernel. One half band
        // differs from it by O(η); the mean of both half bands by O(η²).
        let m = 32;
        let centre = LayeredGradients::new(1.0, 1, 8, m).x; // κ = 1: offset irrelevant
        let rel = |num: usize, den: usize| {
            let up = half_band_gradient(1.0, num, den, BandSide::Above, m);
            let down = half_band_gradient(1.0, num, den, BandSide::Below, m);
            let diff: f64 = centre
                .cells()
                .iter()
                .zip(up.cells().iter().zip(down.cells()))
                .map(|(c, (u, d))| (c - 0.5 * (u + d)).abs())
                .sum();
            diff / centre.l1()
        };
        let (wide, narrow) = (rel(1, 16), rel(1, 64));
        assert!(narrow < 1e-3, "narrow={narrow}");
        // Quadratic convergence: 4× narrower → ≈ 16× smaller.
        assert!(wide / narrow > 10.0, "wide={wide} narrow={narrow}");
    }

    #[test]
    fn eigenvalues_are_roots_and_orthogonal() {
        for &kappa in &KAPPAS {
            for zeta in [0.5, 0.5 + 1.0 / 32.0, 0.75, 0.5 + 15.0 / 32.0] {
                let md = modes(40, zeta, kappa);
                for (k, mo) in md.iter().enumerate() {
                    // ψ(1) = 0.
                    let len = 1.0 - zeta;
                    let end = mo.a * (mo.mu * len).cos() + mo.b * (mo.mu * len).sin();
                    assert!(end.abs() < 1e-9, "κ={kappa} ζ={zeta} k={k} ψ(1)={end}");
                    if k > 0 {
                        assert!(mo.mu > md[k - 1].mu);
                    }
                }
                // Weighted orthogonality by midpoint quadrature on a fine grid.
                let psi = |mo: &Mode, z: f64| {
                    if z < zeta {
                        (mo.mu * z).sin()
                    } else {
                        mo.a * (mo.mu * (z - zeta)).cos() + mo.b * (mo.mu * (z - zeta)).sin()
                    }
                };
                let n = 20_000;
                for (p, q) in [(0, 1), (2, 5), (7, 8)] {
                    let mut acc = 0.0;
                    for s in 0..n {
                        let z = (f64::from(s) + 0.5) / f64::from(n);
                        let e = if z < zeta { 1.0 } else { kappa };
                        acc += e * psi(&md[p], z) * psi(&md[q], z) / f64::from(n);
                    }
                    assert!(
                        acc.abs() < 1e-6 * md[p].norm.max(md[q].norm),
                        "κ={kappa} ζ={zeta} ({p},{q}): {acc}"
                    );
                }
            }
        }
    }

    #[test]
    fn roots_match_an_independent_scan() {
        // ψ(1; μ) for ψ = sin(μz) below ζ, propagated across the interface.
        let end = |mu: f64, zeta: f64, kappa: f64| {
            (mu * zeta).sin() * (mu * (1.0 - zeta)).cos()
                + (mu * zeta).cos() * (mu * (1.0 - zeta)).sin() / kappa
        };
        for &kappa in &[0.01, 0.1, 2.0, 10.0, 100.0] {
            for zeta in [0.5, 0.5 + 1.0 / 32.0, 0.75, 0.5 + 15.0 / 32.0] {
                let md = modes(30, zeta, kappa);
                // Sign changes on a fine grid, refined by bisection.
                let mut scanned = Vec::new();
                let n = 400_000;
                let top = md.last().unwrap().mu + 0.5;
                let mut prev = (1e-9, end(1e-9, zeta, kappa));
                for s in 1..=n {
                    let mu = top * f64::from(s) / f64::from(n);
                    let v = end(mu, zeta, kappa);
                    if v == 0.0 || v.signum() != prev.1.signum() {
                        let (mut lo, mut hi) = (prev.0, mu);
                        for _ in 0..100 {
                            let mid = 0.5 * (lo + hi);
                            if end(mid, zeta, kappa).signum() == end(lo, zeta, kappa).signum() {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        scanned.push(0.5 * (lo + hi));
                    }
                    prev = (mu, v);
                }
                assert_eq!(scanned.len(), md.len(), "κ={kappa} ζ={zeta}");
                for (a, b) in scanned.iter().zip(&md) {
                    assert!(
                        (a - b.mu).abs() < 1e-8,
                        "κ={kappa} ζ={zeta}: {a} vs {}",
                        b.mu
                    );
                }
            }
        }
        // High contrast near the centre: two roots inside one π-wide interval.
        let md = modes(4, 0.5 + 1.0 / 32.0, 10.0);
        let counted = md.iter().filter(|m| m.mu > PI && m.mu < 2.0 * PI).count();
        assert_eq!(counted, 2);
    }

    #[test]
    fn unit_contrast_matches_the_homogeneous_tables() {
        for m in [16, 64] {
            let h = GreenTables::new(m);
            for offset in [0, 3, 7] {
                let c = LayeredCube::new(1.0, offset, 8, m);
                for face in 0..6 {
                    for (a, b) in c.cells[face * m * m..(face + 1) * m * m]
                        .iter()
                        .zip(h.face_probabilities())
                    {
                        assert!((a - b).abs() < 1e-13, "m={m} offset={offset} face={face}");
                    }
                }
            }
        }
    }

    #[test]
    fn probabilities_sum_to_one_and_reproduce_the_layered_linear_potential() {
        let m = 32;
        for &kappa in &KAPPAS {
            for offset in [0, 1, 4, 7] {
                let c = LayeredCube::new(kappa, offset, 8, m);
                let zeta = 0.5 + 0.5 * c.offset();
                let p = c.cell_probabilities();
                let sum: f64 = p.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "κ={kappa} t={} sum={sum}",
                    c.offset()
                );
                // φ(z) = z below ζ, ζ + (z − ζ)/κ above (ε φ′ continuous); φ(½) = ½.
                // φ is linear within every cell, so its cell average is its centre value.
                let phi = |z: f64| {
                    if z <= zeta {
                        z
                    } else {
                        zeta + (z - zeta) / kappa
                    }
                };
                let mm = m * m;
                let mut mean = 0.0;
                for (idx, w) in p.iter().enumerate() {
                    let (face, cell) = (idx / mm, idx % mm);
                    #[allow(clippy::cast_precision_loss)]
                    let zc = ((cell / m) as f64 + 0.5) / m as f64;
                    let z = match face {
                        4 => 0.0,
                        5 => 1.0,
                        _ => zc,
                    };
                    mean += w * phi(z);
                }
                assert!(
                    (mean - 0.5).abs() < 1e-12,
                    "κ={kappa} t={} mean={mean}",
                    c.offset()
                );
            }
        }
    }

    #[test]
    fn interface_through_the_centre_mirrors_with_the_inverse_contrast() {
        let m = 16;
        for &kappa in &KAPPAS {
            let a = LayeredCube::new(kappa, 0, 8, m);
            let b = LayeredCube::new(1.0 / kappa, 0, 8, m);
            let mm = m * m;
            // Mirroring z swaps top and bottom and flips the side faces' z rows.
            for i in 0..mm {
                assert!((a.cells[5 * mm + i] - b.cells[4 * mm + i]).abs() < 1e-13);
            }
            for j in 0..m {
                for i in 0..m {
                    let (x, y) = (a.cells[j * m + i], b.cells[(m - 1 - j) * m + i]);
                    assert!((x - y).abs() < 1e-13, "κ={kappa}");
                }
            }
        }
    }

    #[test]
    fn higher_permittivity_above_pulls_walks_up() {
        // With κ > 1 the upper layer "conducts" better: more mass on the top face.
        let m = 16;
        let mm = m * m;
        let top = |kappa: f64| {
            LayeredCube::new(kappa, 4, 8, m).cells[5 * mm..]
                .iter()
                .sum::<f64>()
        };
        assert!(top(10.0) > top(2.0) && top(2.0) > top(1.0) && top(1.0) > top(0.5));
    }

    #[test]
    #[allow(clippy::float_cmp)] // face coordinates are exactly ±1
    fn sampling_matches_the_cell_probabilities() {
        // Every one of the 6·m² cells, not only the faces: the symmetry-class
        // table must reproduce the full distribution (chi-square test).
        let m = 16;
        let c = LayeredCube::new(4.0, 2, 8, m);
        let mut rng = Rng::from_seed(5);
        let n = 2_000_000_u32;
        let mm = m * m;
        let mut counts = vec![0u32; 6 * mm];
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = |x: f64| (((x + 1.0) * 0.5 * mf) as usize).min(m - 1);
        for _ in 0..n {
            let p = c.sample_transition(&mut rng);
            assert!(p.iter().all(|v| v.abs() <= 1.0));
            let (f, u, v) = if p[0].abs() == 1.0 {
                (usize::from(p[0] > 0.0), p[1], p[2])
            } else if p[1].abs() == 1.0 {
                (2 + usize::from(p[1] > 0.0), p[0], p[2])
            } else {
                (4 + usize::from(p[2] > 0.0), p[0], p[1])
            };
            counts[f * mm + bin(v) * m + bin(u)] += 1;
        }
        let (mut chi2, mut dof) = (0.0_f64, 0.0_f64);
        for (&k, &p) in counts.iter().zip(&c.cells) {
            let e = p * f64::from(n);
            if e > 20.0 {
                chi2 += (f64::from(k) - e).powi(2) / e;
                dof += 1.0;
            }
        }
        assert!(dof > 1000.0, "dof {dof}");
        assert!(
            chi2 < dof + 5.0 * (2.0 * dof).sqrt(),
            "chi2 {chi2} for {dof} cells"
        );
    }
}
