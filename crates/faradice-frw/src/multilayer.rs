//! Transition domains of interface-anchored walks (design M2b,
//! `docs/design/M2b_interface_anchored.md` §3.2–§3.3).
//!
//! - [`MultiLayerCube`]: a cube whose centre lies on a dielectric interface
//!   and which contains any number of further planar interfaces at their true
//!   heights. The series of [`crate::layered`] carry over layer by layer:
//!   profiles of the top and bottom faces are propagated through the layers
//!   with continuity of `Z` and `ε Z′`, and the side faces use the
//!   eigenfunctions of `−(εψ′)′ = μ² ε ψ`, `ψ(0) = ψ(1) = 0`, whose Prüfer
//!   angle at `z = 1` is strictly increasing in `μ`.
//! - [`LandingBox`]: a homogeneous box of half-sizes `(L, L, 1)` (in units of
//!   its half-height) centred on the walker, used to place walks exactly on
//!   an interface.
//!
//! Every cell probability is a closed-form integral of the truncated series;
//! side-face cells are split at the interfaces so that none straddles one.

use std::f64::consts::PI;

use crate::alias::AliasTable;
use crate::green::{NTERMS, bilinear_cells, cell_integrals};
use crate::rng::Rng;

/// Side-face `z` cells per face cell. Interfaces off the uniform grid make
/// cell-uniform sampling inexact even for the layered linear potential (error
/// `≈ 1e-5` of the cube's potential range at one cell per face cell, falling
/// like `Δz³`); four sub-cells bring it below `1e-6`.
const Z_REFINE: usize = 4;

/// Piecewise-constant relative permittivity on `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    /// Interface heights, strictly increasing, in `(0, 1)`.
    z: Vec<f64>,
    /// Permittivity of each layer, bottom-up (`z.len() + 1` values).
    eps: Vec<f64>,
}

impl Profile {
    /// A profile with interfaces `z` and layer permittivities `eps`.
    ///
    /// # Panics
    ///
    /// Panics unless `eps.len() == z.len() + 1`, the heights are strictly
    /// increasing in `(0, 1)` and every permittivity is finite and positive.
    #[must_use]
    pub fn new(z: Vec<f64>, eps: Vec<f64>) -> Self {
        assert_eq!(eps.len(), z.len() + 1, "one permittivity per layer");
        assert!(
            z.iter().all(|&v| v > 0.0 && v < 1.0) && z.windows(2).all(|w| w[0] < w[1]),
            "interfaces must be increasing in (0, 1)"
        );
        assert!(
            eps.iter().all(|&e| e.is_finite() && e > 0.0),
            "permittivities must be positive"
        );
        Self { z, eps }
    }

    /// Layer boundaries `[0, z…, 1]`.
    fn bounds(&self) -> Vec<f64> {
        let mut b = Vec::with_capacity(self.z.len() + 2);
        b.push(0.0);
        b.extend_from_slice(&self.z);
        b.push(1.0);
        b
    }

    /// The layer containing `z` (an interface belongs to the layer above).
    fn layer(&self, z: f64) -> usize {
        self.z.partition_point(|&v| v <= z)
    }

    fn mirrored(&self) -> Self {
        Self {
            z: self.z.iter().rev().map(|v| 1.0 - v).collect(),
            eps: self.eps.iter().rev().copied().collect(),
        }
    }
}

/// `Z(at)/Z(1)` for `Z″ = γ² Z` in every layer, `Z(0) = 0`, `Z` and `ε Z′`
/// continuous: the top-face profile of one `(x, y)` mode.
fn top_profile(p: &Profile, g: f64, at: f64) -> f64 {
    let b = p.bounds();
    let (mut zv, mut q) = (0.0_f64, 1.0_f64);
    let mut z_at = None;
    for l in 0..p.eps.len() {
        let (za, zb, e) = (b[l], b[l + 1], p.eps[l]);
        let advance = |d: f64| {
            let (s, c) = ((g * d).sinh(), (g * d).cosh());
            (zv * c + q / (e * g) * s, zv * e * g * s + q * c)
        };
        if z_at.is_none() && at <= zb {
            z_at = Some(advance(at - za).0);
        }
        (zv, q) = advance(zb - za);
    }
    z_at.expect("evaluation point inside [0, 1]") / zv
}

/// `Z′(at)/Z(1)` for the profile of [`top_profile`] (`at` inside a layer).
fn top_profile_slope(p: &Profile, g: f64, at: f64) -> f64 {
    let b = p.bounds();
    let (mut zv, mut q) = (0.0_f64, 1.0_f64);
    let mut slope_at = None;
    for l in 0..p.eps.len() {
        let (za, zb, e) = (b[l], b[l + 1], p.eps[l]);
        let advance = |d: f64| {
            let (s, c) = ((g * d).sinh(), (g * d).cosh());
            (zv * c + q / (e * g) * s, zv * e * g * s + q * c)
        };
        if slope_at.is_none() && at <= zb {
            // Z′ = (ε Z′)/ε within the layer.
            slope_at = Some(advance(at - za).1 / e);
        }
        (zv, q) = advance(zb - za);
    }
    slope_at.expect("evaluation point inside [0, 1]") / zv
}

/// Mean of `Z(z)/Z(1)` over `[z0, z1]` for the profile of [`top_profile`].
fn top_profile_mean(p: &Profile, g: f64, z0: f64, z1: f64) -> f64 {
    let b = p.bounds();
    let (mut zv, mut q) = (0.0_f64, 1.0_f64);
    let mut integral = 0.0;
    for l in 0..p.eps.len() {
        let (za, zb, e) = (b[l], b[l + 1], p.eps[l]);
        // ∫_0^d Z = zv sinh(gd)/g + q (cosh(gd) − 1)/(ε g²) from the layer start.
        let part = |d: f64| {
            let gd = g * d;
            (zv * gd.sinh() + q / e * (gd.cosh() - 1.0) / g) / g
        };
        let (lo, hi) = (z0.max(za), z1.min(zb));
        if hi > lo {
            integral += part(hi - za) - part(lo - za);
        }
        let (s, c) = ((g * (zb - za)).sinh(), (g * (zb - za)).cosh());
        (zv, q) = (zv * c + q / (e * g) * s, zv * e * g * s + q * c);
    }
    integral / ((z1 - z0) * zv)
}

/// One side-face eigenmode: `ψ = A_l cos(μ(z − z_l)) + B_l sin(μ(z − z_l))` in
/// layer `l` starting at `z_l`.
#[derive(Debug, Clone)]
struct Mode {
    mu: f64,
    coef: Vec<(f64, f64)>,
    /// `∫ ε ψ² dz`.
    norm: f64,
}

/// Prüfer angle at `z = 1` of the solution with `ψ = sin(μz)` in the bottom layer.
fn prufer_end(p: &Profile, mu: f64) -> f64 {
    let b = p.bounds();
    let mut theta = 0.0_f64;
    for l in 0..p.eps.len() {
        theta += mu * (b[l + 1] - b[l]);
        if l + 1 < p.eps.len() {
            // ψ continuous, ψ′ scaled by ε_l/ε_{l+1}: same half period.
            let r = p.eps[l] / p.eps[l + 1];
            let hp = (theta / PI).floor();
            let phi = theta - hp * PI;
            theta = hp.mul_add(PI, phi.sin().atan2(r * phi.cos()));
        }
    }
    theta
}

impl Mode {
    fn new(p: &Profile, mu: f64) -> Self {
        let b = p.bounds();
        let n = p.eps.len();
        let mut coef = Vec::with_capacity(n);
        let (mut a, mut bb) = (0.0, 1.0);
        let mut norm = 0.0;
        for l in 0..n {
            coef.push((a, bb));
            let d = b[l + 1] - b[l];
            let (s2, c2) = ((2.0 * mu * d).sin(), (2.0 * mu * d).cos());
            norm += p.eps[l]
                * (a * a * (0.5 * d + s2 / (4.0 * mu))
                    + bb * bb * (0.5 * d - s2 / (4.0 * mu))
                    + a * bb * (1.0 - c2) / (2.0 * mu));
            if l + 1 < n {
                let (s, c) = ((mu * d).sin(), (mu * d).cos());
                let v = a * c + bb * s;
                let slope = -a * s + bb * c;
                (a, bb) = (v, p.eps[l] / p.eps[l + 1] * slope);
            }
        }
        Self { mu, coef, norm }
    }

    fn value(&self, p: &Profile, z: f64) -> f64 {
        let l = p.layer(z).min(p.eps.len() - 1);
        let z0 = if l == 0 { 0.0 } else { p.z[l - 1] };
        let (a, b) = self.coef[l];
        let u = self.mu * (z - z0);
        a * u.cos() + b * u.sin()
    }

    fn slope(&self, p: &Profile, z: f64) -> f64 {
        let l = p.layer(z).min(p.eps.len() - 1);
        let z0 = if l == 0 { 0.0 } else { p.z[l - 1] };
        let (a, b) = self.coef[l];
        let u = self.mu * (z - z0);
        self.mu * (b * u.cos() - a * u.sin())
    }

    /// `∫_{lo}^{hi} ψ dz` within layer `l`.
    fn integral(&self, p: &Profile, l: usize, lo: f64, hi: f64) -> f64 {
        let z0 = if l == 0 { 0.0 } else { p.z[l - 1] };
        let (a, b) = self.coef[l];
        let (u0, u1) = (self.mu * (lo - z0), self.mu * (hi - z0));
        (a * (u1.sin() - u0.sin()) + b * (u0.cos() - u1.cos())) / self.mu
    }
}

impl Mode {
    /// Mean of `ψ` over `[z0, z1]`.
    fn mean(&self, p: &Profile, z0: f64, z1: f64) -> f64 {
        let b = p.bounds();
        let mut total = 0.0;
        for l in 0..p.eps.len() {
            let (lo, hi) = (z0.max(b[l]), z1.min(b[l + 1]));
            if hi > lo {
                total += self.integral(p, l, lo, hi);
            }
        }
        total / (z1 - z0)
    }
}

/// The first `n` eigenmodes.
fn modes(p: &Profile, n: usize) -> Vec<Mode> {
    #[allow(clippy::cast_precision_loss)]
    let shift = p.z.len() as f64 * PI;
    (1..=n)
        .map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let target = k as f64 * PI;
            // Every interface moves θ within one half period: |θ(1; μ) − μ| < nπ.
            let (mut lo, mut hi) = ((target - shift).max(0.0), target + shift);
            for _ in 0..300 {
                let mid = 0.5 * (lo + hi);
                if prufer_end(p, mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
                if hi - lo <= 4.0 * f64::EPSILON * hi {
                    break;
                }
            }
            Mode::new(p, 0.5 * (lo + hi))
        })
        .collect()
}

/// `Σ_{s,t} a[s][i] · c[s][t] · b[t][j]` into `out[j * a_cells + i]`.
fn bilinear(a: &[Vec<f64>], b: &[Vec<f64>], c: &[Vec<f64>]) -> Vec<f64> {
    bilinear_cells(a, b, |s, t| c[s][t])
}

#[allow(clippy::cast_precision_loss)]
fn odd(n: usize) -> Vec<f64> {
    (0..n).map(|t| (2 * t + 1) as f64).collect()
}

/// `sin(kπ/2)` for odd `k`.
fn sin_half(k: f64) -> f64 {
    (k * PI / 2.0).sin().round()
}

/// Six-face cell probabilities folded into the symmetry classes of a domain
/// invariant under `x ↔ y`, `x → −x`, `y → −y`, and their sampling.
#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))] // the face tables are kept for tests
struct Folded {
    m: usize,
    /// Lateral cell boundaries in `[0, 1]` (symmetric about ½), shared by
    /// both lateral axes.
    xb: Vec<f64>,
    /// Side-face cell boundaries in `z ∈ [0, 1]`.
    zb: Vec<f64>,
    bottom: Vec<f64>,
    top: Vec<f64>,
    /// One side face, `j * m + i` with `i` tangential and `j` the `z` cell.
    side: Vec<f64>,
    alias: AliasTable,
}

impl Folded {
    fn new(xb: Vec<f64>, zb: Vec<f64>, bottom: Vec<f64>, top: Vec<f64>, side: Vec<f64>) -> Self {
        let m = xb.len() - 1;
        let h = m / 2;
        let mz = zb.len() - 1;
        let mut w = Vec::with_capacity(2 * h * h + h * mz);
        for face in [&bottom, &top] {
            for j in h..m {
                for i in h..m {
                    w.push(
                        face[j * m + i]
                            + face[j * m + m - 1 - i]
                            + face[(m - 1 - j) * m + i]
                            + face[(m - 1 - j) * m + m - 1 - i],
                    );
                }
            }
        }
        for j in 0..mz {
            for i in h..m {
                w.push(4.0 * (side[j * m + i] + side[j * m + m - 1 - i]));
            }
        }
        for v in &mut w {
            assert!(
                v.is_finite() && *v > -1e-12,
                "table value out of range: {v}"
            );
            *v = v.max(0.0);
        }
        Self {
            m,
            alias: AliasTable::new(&w),
            xb,
            zb,
            bottom,
            top,
            side,
        }
    }

    #[cfg(test)]
    fn total(&self) -> f64 {
        self.bottom.iter().sum::<f64>()
            + self.top.iter().sum::<f64>()
            + 4.0 * self.side.iter().sum::<f64>()
    }

    /// A point on the surface of `[−lat, lat]² × [−1, 1]`.
    #[inline]
    fn sample(&self, lat: f64, rng: &mut Rng) -> [f64; 3] {
        let h = self.m / 2;
        let xb = &self.xb;
        let coord = |c: usize, rng: &mut Rng| {
            lat * (2.0 * rng.next_f64().mul_add(xb[c + 1] - xb[c], xb[c]) - 1.0)
        };
        let r = self.alias.sample(rng);
        if r < 2 * h * h {
            let rem = r % (h * h);
            let (u, v) = (coord(h + rem % h, rng), coord(h + rem / h, rng));
            let bits = rng.below(4);
            let u = if bits & 1 == 0 { u } else { -u };
            let v = if bits & 2 == 0 { v } else { -v };
            [u, v, if r < h * h { -1.0 } else { 1.0 }]
        } else {
            let rem = r - 2 * h * h;
            let (i, j) = (h + rem % h, rem / h);
            let u = coord(i, rng);
            let (z0, z1) = (self.zb[j], self.zb[j + 1]);
            let v = 2.0 * rng.next_f64().mul_add(z1 - z0, z0) - 1.0;
            let bits = rng.below(8);
            let u = if bits & 4 == 0 { u } else { -u };
            let w = if bits & 1 == 0 { -lat } else { lat };
            if bits & 2 == 0 { [w, u, v] } else { [u, w, v] }
        }
    }
}

/// Transition tables of a cube whose centre lies on an interface of a
/// stratified medium (§3.2).
#[derive(Debug, Clone)]
pub struct MultiLayerCube {
    profile: Profile,
    folded: Folded,
}

impl MultiLayerCube {
    /// Builds the tables of the unit cube with permittivity `profile` (the
    /// centre `z = ½` is expected to lie on one of its interfaces, but any
    /// profile is valid) and `resolution × resolution` cells per face, side
    /// faces split at the interfaces.
    ///
    /// # Panics
    ///
    /// Panics if `resolution` is odd or zero.
    #[must_use]
    pub fn new(profile: Profile, resolution: usize) -> Self {
        assert!(
            resolution >= 2 && resolution % 2 == 0,
            "resolution must be even"
        );
        let m = resolution;
        let odd_k = odd(NTERMS);
        let a_odd = cell_integrals(&odd_k, sin_half, m);
        let coef = |prof: &Profile| -> Vec<Vec<f64>> {
            odd_k
                .iter()
                .map(|&s| {
                    odd_k
                        .iter()
                        .map(|&t| 4.0 * top_profile(prof, PI * s.hypot(t), 0.5))
                        .collect()
                })
                .collect()
        };
        let top = bilinear(&a_odd, &a_odd, &coef(&profile));
        let bottom = bilinear(&a_odd, &a_odd, &coef(&profile.mirrored()));

        // Side faces: Z_REFINE·M uniform z cells, split at the interfaces.
        let zb = side_cells(&profile, m);
        let md = modes(&profile, 2 * NTERMS + profile.z.len());
        let e_modes = mode_cells(&profile, &md, &zb);
        let c: Vec<Vec<f64>> = odd_k
            .iter()
            .map(|&s| {
                md.iter()
                    .map(|mo| {
                        let beta = (s * PI).hypot(mo.mu);
                        2.0 * mo.value(&profile, 0.5) / (2.0 * (beta / 2.0).cosh())
                    })
                    .collect()
            })
            .collect();
        let side = bilinear(&a_odd, &e_modes, &c);
        Self {
            folded: Folded::new(uniform(m), zb, bottom, top, side),
            profile,
        }
    }

    /// The permittivity profile.
    #[must_use]
    pub const fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Samples the next position on the surface of `[−1, 1]³`.
    #[inline]
    pub fn sample_transition(&self, rng: &mut Rng) -> [f64; 3] {
        self.folded.sample(1.0, rng)
    }
}

/// Side-face `z` cell boundaries: `Z_REFINE · m` uniform cells split at the
/// interfaces of `p`.
fn side_cells(p: &Profile, m: usize) -> Vec<f64> {
    let zm = Z_REFINE * m;
    #[allow(clippy::cast_precision_loss)]
    let mut zb: Vec<f64> = (0..=zm).map(|l| l as f64 / zm as f64).collect();
    zb.extend_from_slice(&p.z);
    zb.sort_by(f64::total_cmp);
    zb.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    zb
}

/// `∫_cell ε ψ_k / N_k` per mode and side-face `z` cell.
fn mode_cells(p: &Profile, md: &[Mode], zb: &[f64]) -> Vec<Vec<f64>> {
    md.iter()
        .map(|mo| {
            zb.windows(2)
                .map(|w| {
                    let l = p.layer(0.5 * (w[0] + w[1]));
                    p.eps[l] * mo.integral(p, l, w[0], w[1]) / mo.norm
                })
                .collect()
        })
        .collect()
}

/// Tangential (`x`) gradient kernel of a multi-layer cube, averaged over the
/// vertical segment `z ∈ [z0, z1]` of the line `x = y = ½` (design M2b §3.5,
/// the half-band estimator of M2a §4 on an anchored cube). The permittivity
/// of the segment is not included.
///
/// For a cube of half-side `h`, the mean derivative is
/// `E[sign · φ(centre + h·point)] · l1() / (2h)`.
#[derive(Debug, Clone)]
pub struct BandGradient {
    m: usize,
    /// Side-face cell boundaries in `z ∈ [0, 1]`.
    zb: Vec<f64>,
    /// Signed cell integrals: four side faces `[x-, x+, y-, y+]` with
    /// `j * m + i` (`i` tangential, `j` the `z` cell), then `z-` and `z+`
    /// (`j * m + i`, `i` along `x`).
    cells: Vec<f64>,
    alias: AliasTable,
    l1: f64,
}

impl BandGradient {
    /// Builds the kernel for the unit cube with permittivity `profile`,
    /// averaged over `[z0, z1]`, with `resolution × resolution` cells per face
    /// (side faces split at the interfaces).
    ///
    /// # Panics
    ///
    /// Panics if `resolution` is odd or zero or `0 ≤ z0 < z1 ≤ 1` fails.
    #[must_use]
    pub fn new(profile: &Profile, z0: f64, z1: f64, resolution: usize) -> Self {
        assert!(
            resolution >= 2 && resolution % 2 == 0,
            "resolution must be even"
        );
        assert!(
            0.0 <= z0 && z0 < z1 && z1 <= 1.0,
            "segment must lie in [0, 1]"
        );
        let m = resolution;
        let odd_k = odd(NTERMS);
        #[allow(clippy::cast_precision_loss)]
        let even_k: Vec<f64> = (1..=NTERMS).map(|s| (2 * s) as f64).collect();
        let a_odd = cell_integrals(&odd_k, sin_half, m);
        // ∂/∂u of sin(kπu) at ½ for even k.
        let d_even = cell_integrals(&even_k, |k| k * PI * (k * PI / 2.0).cos().round(), m);

        let zb = side_cells(profile, m);
        let md = modes(profile, 2 * NTERMS + profile.z.len());
        let e_modes = mode_cells(profile, &md, &zb);
        let psi: Vec<f64> = md.iter().map(|mo| mo.mean(profile, z0, z1)).collect();
        let side = |ks: &[f64], factor: &dyn Fn(f64) -> f64| -> Vec<Vec<f64>> {
            ks.iter()
                .map(|&s| {
                    md.iter()
                        .zip(&psi)
                        .map(|(mo, &v)| 2.0 * v * factor((s * PI).hypot(mo.mu)))
                        .collect()
                })
                .collect()
        };
        // x = 1 face: ∂_x [sinh(βx)/sinh β] at ½ = β / (2 sinh(β/2)).
        let xp = bilinear(
            &a_odd,
            &e_modes,
            &side(&odd_k, &|beta| beta / (2.0 * (beta / 2.0).sinh())),
        );
        // y faces: ∂_x of the tangential sine, sinh(β/2)/sinh β = 1/(2 cosh(β/2)).
        let yf = bilinear(
            &d_even,
            &e_modes,
            &side(&even_k, &|beta| 1.0 / (2.0 * (beta / 2.0).cosh())),
        );
        let horizontal = |prof: &Profile, lo: f64, hi: f64| {
            let c: Vec<Vec<f64>> = even_k
                .iter()
                .map(|&s| {
                    odd_k
                        .iter()
                        .map(|&t| 4.0 * top_profile_mean(prof, PI * s.hypot(t), lo, hi))
                        .collect()
                })
                .collect();
            bilinear(&d_even, &a_odd, &c)
        };
        let top = horizontal(profile, z0, z1);
        let bottom = horizontal(&profile.mirrored(), 1.0 - z1, 1.0 - z0);

        let mut cells = Vec::with_capacity(4 * xp.len() + 2 * m * m);
        cells.extend(xp.iter().map(|v| -v));
        cells.extend_from_slice(&xp);
        cells.extend_from_slice(&yf);
        cells.extend_from_slice(&yf);
        cells.extend(bottom);
        cells.extend(top);
        assert!(
            cells.iter().all(|c| c.is_finite()),
            "band kernel is not finite"
        );
        let abs: Vec<f64> = cells.iter().map(|c| c.abs()).collect();
        Self {
            m,
            zb,
            alias: AliasTable::new(&abs),
            l1: abs.iter().sum(),
            cells,
        }
    }

    /// `∫|K| dS` on the unit cube.
    #[must_use]
    pub const fn l1(&self) -> f64 {
        self.l1
    }

    /// Samples a point on the surface of `[−1, 1]³` and the kernel's sign
    /// there, in the canonical frame of the cube.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> ([f64; 3], f64) {
        let idx = self.alias.sample(rng);
        let sign = self.cells[idx].signum();
        let m = self.m;
        #[allow(clippy::cast_precision_loss)]
        let coord = |c: usize, rng: &mut Rng| 2.0 * (c as f64 + rng.next_f64()) / m as f64 - 1.0;
        let per_side = m * (self.zb.len() - 1);
        let p = if idx < 4 * per_side {
            let (face, cell) = (idx / per_side, idx % per_side);
            let (i, j) = (cell % m, cell / m);
            let u = coord(i, rng);
            let (z0, z1) = (self.zb[j], self.zb[j + 1]);
            let v = 2.0 * rng.next_f64().mul_add(z1 - z0, z0) - 1.0;
            let w = if face & 1 == 0 { -1.0 } else { 1.0 };
            if face < 2 { [w, u, v] } else { [u, w, v] }
        } else {
            let rem = idx - 4 * per_side;
            let (face, cell) = (rem / (m * m), rem % (m * m));
            let (u, v) = (coord(cell % m, rng), coord(cell / m, rng));
            [u, v, if face == 0 { -1.0 } else { 1.0 }]
        };
        (p, sign)
    }

    /// Signed cell integrals (see the field layout).
    #[cfg(test)]
    fn cells(&self) -> &[f64] {
        &self.cells
    }
}

/// Normal (`z`) gradient kernel at the centre of a multi-layer cube whose
/// centre lies **inside** a layer (design M2e §3.1): the first hop from a
/// Gaussian point between facing interfaces parallel to its face. The
/// permittivity of the centre's layer is not included.
///
/// For a cube of half-side `h`, the derivative is
/// `E[sign · φ(centre + h·point)] · l1() / (2h)`.
#[derive(Debug, Clone)]
pub struct NormalGradient {
    m: usize,
    /// Side-face cell boundaries in `z ∈ [0, 1]`.
    zb: Vec<f64>,
    /// Signed cell integrals, laid out as for [`BandGradient`]: four side
    /// faces `[x-, x+, y-, y+]` (`j * m + i`), then `z-` and `z+`.
    cells: Vec<f64>,
    alias: AliasTable,
    /// Samplers of the positive and negative parts.
    parts: [AliasTable; 2],
    l1: f64,
}

impl NormalGradient {
    /// Builds the kernel of the unit cube with permittivity `profile`, with
    /// `resolution × resolution` cells per face (side faces split at the
    /// interfaces).
    ///
    /// # Panics
    ///
    /// Panics if `resolution` is odd or zero, or the centre `z = ½` lies on
    /// an interface (where `∂φ/∂z` has two traces).
    #[must_use]
    pub fn new(profile: &Profile, resolution: usize) -> Self {
        assert!(
            resolution >= 2 && resolution % 2 == 0,
            "resolution must be even"
        );
        assert!(
            profile.z.iter().all(|&z| (z - 0.5).abs() > 1e-12),
            "the centre must not lie on an interface"
        );
        let m = resolution;
        let odd_k = odd(NTERMS);
        let a_odd = cell_integrals(&odd_k, sin_half, m);
        // Top and bottom faces: ∂/∂z at ½ of Z(z)/Z(1); the bottom face is the
        // top face of the mirrored profile, whose z runs the other way.
        let coef = |prof: &Profile| -> Vec<Vec<f64>> {
            odd_k
                .iter()
                .map(|&s| {
                    odd_k
                        .iter()
                        .map(|&t| 4.0 * top_profile_slope(prof, PI * s.hypot(t), 0.5))
                        .collect()
                })
                .collect()
        };
        let top = bilinear(&a_odd, &a_odd, &coef(profile));
        let bottom = bilinear(&a_odd, &a_odd, &coef(&profile.mirrored()));
        // Side faces: ψ′(½) in place of ψ(½) (the four faces are alike).
        let zb = side_cells(profile, m);
        let md = modes(profile, 2 * NTERMS + profile.z.len());
        let e_modes = mode_cells(profile, &md, &zb);
        let c: Vec<Vec<f64>> = odd_k
            .iter()
            .map(|&s| {
                md.iter()
                    .map(|mo| {
                        let beta = (s * PI).hypot(mo.mu);
                        2.0 * mo.slope(profile, 0.5) / (2.0 * (beta / 2.0).cosh())
                    })
                    .collect()
            })
            .collect();
        let side = bilinear(&a_odd, &e_modes, &c);
        let mut cells = Vec::with_capacity(4 * side.len() + 2 * m * m);
        for _ in 0..4 {
            cells.extend_from_slice(&side);
        }
        cells.extend(bottom.iter().map(|v| -v));
        cells.extend(top);
        assert!(
            cells.iter().all(|c| c.is_finite()),
            "normal kernel is not finite"
        );
        let abs: Vec<f64> = cells.iter().map(|c| c.abs()).collect();
        let part = |positive: bool| {
            let w: Vec<f64> = cells
                .iter()
                .map(|&c| if (c > 0.0) == positive { c.abs() } else { 0.0 })
                .collect();
            AliasTable::new(&w)
        };
        Self {
            m,
            zb,
            alias: AliasTable::new(&abs),
            parts: [part(true), part(false)],
            l1: abs.iter().sum(),
            cells,
        }
    }

    /// `∫|K| dS` on the unit cube.
    #[must_use]
    pub const fn l1(&self) -> f64 {
        self.l1
    }

    /// A point of cell `idx` on the surface of `[−1, 1]³`.
    fn point(&self, idx: usize, rng: &mut Rng) -> [f64; 3] {
        let m = self.m;
        #[allow(clippy::cast_precision_loss)]
        let coord = |c: usize, rng: &mut Rng| 2.0 * (c as f64 + rng.next_f64()) / m as f64 - 1.0;
        let per_side = m * (self.zb.len() - 1);
        if idx < 4 * per_side {
            let (face, cell) = (idx / per_side, idx % per_side);
            let (i, j) = (cell % m, cell / m);
            let u = coord(i, rng);
            let (z0, z1) = (self.zb[j], self.zb[j + 1]);
            let v = 2.0 * rng.next_f64().mul_add(z1 - z0, z0) - 1.0;
            let w = if face & 1 == 0 { -1.0 } else { 1.0 };
            if face < 2 { [w, u, v] } else { [u, w, v] }
        } else {
            let rem = idx - 4 * per_side;
            let (face, cell) = (rem / (m * m), rem % (m * m));
            let (u, v) = (coord(cell % m, rng), coord(cell / m, rng));
            [u, v, if face == 0 { -1.0 } else { 1.0 }]
        }
    }

    /// Samples a point on the surface of `[−1, 1]³` and the kernel's sign
    /// there, in the canonical frame of the cube.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> ([f64; 3], f64) {
        let idx = self.alias.sample(rng);
        (self.point(idx, rng), self.cells[idx].signum())
    }

    /// Samples one point from the positive part and one from the negative
    /// part (each carries half of [`Self::l1`], as the kernel integrates to
    /// zero); the derivative is `(E[φ(c + h·p₊)] − E[φ(c + h·p₋)]) · l1() / (4h)`.
    #[inline]
    pub fn sample_pair(&self, rng: &mut Rng) -> [[f64; 3]; 2] {
        let i = self.parts[0].sample(rng);
        let j = self.parts[1].sample(rng);
        [self.point(i, rng), self.point(j, rng)]
    }

    /// Signed cell integrals (see the field layout).
    #[cfg(test)]
    fn cells(&self) -> &[f64] {
        &self.cells
    }
}

/// `m + 1` uniform cell boundaries on `[0, 1]`.
fn uniform(m: usize) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    (0..=m).map(|i| i as f64 / m as f64).collect()
}

/// `m + 1` cell boundaries on `[0, 1]`, symmetric about ½ and graded
/// towards it so that the two central cells are at most `first` wide
/// (uniform if that is wider than `1/m`): `½ ± ½·sinh(c·s)/sinh(c)` for
/// `s = 2i/m`.
fn graded(m: usize, first: f64) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    let h = (m / 2) as f64;
    // sinh(c/h)/sinh(c) = 2·first: solve for c ≥ 0 by bisection.
    let target = 2.0 * first;
    if target >= 1.0 / h {
        return uniform(m);
    }
    let width = |c: f64| (c / h).sinh() / c.sinh();
    let (mut lo, mut hi) = (1e-9_f64, 1.0_f64);
    while width(hi) > target {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if width(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let c = 0.5 * (lo + hi);
    let half = m / 2;
    #[allow(clippy::cast_precision_loss)]
    let pos = |i: usize| 0.5 * (c * i as f64 / h).sinh() / c.sinh();
    let mut xb = vec![0.0; m + 1];
    for i in 0..=half {
        xb[half + i] = 0.5 + pos(i);
        xb[half - i] = 0.5 - pos(i);
    }
    xb[0] = 0.0;
    xb[m] = 1.0;
    xb
}

/// [`cell_integrals`] over the cells between consecutive boundaries `xb`.
fn cell_integrals_at(ks: &[f64], coef: impl Fn(f64) -> f64, xb: &[f64]) -> Vec<Vec<f64>> {
    ks.iter()
        .map(|&k| {
            let c = coef(k);
            xb.windows(2)
                .map(|w| c * crate::green::sin_integral(k, w[0], w[1]))
                .collect()
        })
        .collect()
}

/// `sinh(a·f) / sinh(a)` for `a ≥ 0`, `f ∈ [0, 1]`, without overflow.
fn sinh_ratio(a: f64, f: f64) -> f64 {
    if a < 1e-12 {
        return f;
    }
    if a * f.min(1.0 - f) > 20.0 {
        // Both correction factors are 1 to within e^{−40}.
        return (-a * (1.0 - f)).exp();
    }
    // e^{−a(1−f)} (1 − e^{−2af}) / (1 − e^{−2a})
    (-a * (1.0 - f)).exp() * (-(-2.0 * a * f).exp_m1()) / (-(-2.0 * a).exp_m1())
}

/// Transition tables of the homogeneous box `[−L, L]² × [−1, 1]` from the
/// point `(0, 0, 2τ − 1)`, at the fraction `τ ∈ (0, ½]` of its height above
/// the bottom face (§3.3). The bottom face is the one placed on an interface.
#[derive(Debug, Clone)]
pub struct LandingBox {
    ratio: f64,
    tau: f64,
    folded: Folded,
}

impl LandingBox {
    /// Builds the tables for the aspect ratio `ratio = L ≥ 1` and the height
    /// fraction `tau` with `resolution × resolution` cells per face.
    ///
    /// # Panics
    ///
    /// Panics if `ratio < 1`, it is not finite, `tau` is not in `(0, ½]`, or
    /// `resolution` is odd or zero.
    #[must_use]
    pub fn new(ratio: f64, tau: f64, resolution: usize) -> Self {
        assert!(ratio.is_finite() && ratio >= 1.0, "ratio must be >= 1");
        assert!(tau > 0.0 && tau <= 0.5, "tau must lie in (0, 1/2]");
        assert!(
            resolution >= 2 && resolution % 2 == 0,
            "resolution must be even"
        );
        let m = resolution;
        let xb = graded(m, tau / (2.0 * ratio));
        // Lateral coordinates are scaled to [0, 1]; the height is then 1/L.
        // A mode decays like e^{−γτ/L} towards the source: L/(2τ) times more
        // terms than the centred cube for the same accuracy.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let nh = (NTERMS as f64 * ratio / (2.0 * tau)).ceil() as usize;
        let kh = odd(nh);
        let ah = cell_integrals_at(&kh, sin_half, &xb);
        // The coefficients are evaluated on the fly: nh² of them would not
        // fit comfortably in memory for flat boxes and small τ.
        let horizontal = |f: f64| {
            bilinear_cells(&ah, &ah, |s, t| {
                let a = PI * kh[s].hypot(kh[t]) / ratio;
                // Terms below e^{−40} of the leading one are dropped.
                if a * (1.0 - f) > 40.0 {
                    0.0
                } else {
                    4.0 * sinh_ratio(a, f)
                }
            })
        };
        let top = horizontal(tau);
        let bottom = horizontal(1.0 - tau);
        // Side faces: tangential odd modes, all z modes sin(kπz), z ∈ [0, 1].
        let ks = odd(NTERMS);
        let a_odd = cell_integrals_at(&ks, sin_half, &xb);
        #[allow(clippy::cast_precision_loss)]
        let kz: Vec<f64> = (1..=2 * NTERMS).map(|k| k as f64).collect();
        let az = cell_integrals(&kz, |k| (k * PI * tau).sin(), m);
        let cs: Vec<Vec<f64>> = ks
            .iter()
            .map(|&t| {
                kz.iter()
                    .map(|&k| 2.0 / (0.5 * PI * t.hypot(k * ratio)).cosh())
                    .collect()
            })
            .collect();
        let side = bilinear(&a_odd, &az, &cs);
        #[allow(clippy::cast_precision_loss)]
        let zb: Vec<f64> = (0..=m).map(|l| l as f64 / m as f64).collect();
        Self {
            ratio,
            tau,
            folded: Folded::new(xb, zb, bottom, top, side),
        }
    }

    /// The aspect ratio `L`.
    #[must_use]
    pub const fn ratio(&self) -> f64 {
        self.ratio
    }

    /// The height fraction `τ` of the start point.
    #[must_use]
    pub const fn tau(&self) -> f64 {
        self.tau
    }

    /// Probability of leaving through the bottom face.
    #[must_use]
    pub fn bottom_probability(&self) -> f64 {
        self.folded.bottom.iter().sum()
    }

    /// Samples the next position on the surface of `[−L, L]² × [−1, 1]`;
    /// the `z` coordinate of a point on a horizontal face is exactly `±1`.
    #[inline]
    pub fn sample_transition(&self, rng: &mut Rng) -> [f64; 3] {
        self.folded.sample(self.ratio, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::green::GreenTables;
    use crate::layered::LayeredCube;

    /// Side-face cells merged into the `m` uniform rows of a face.
    fn merged_rows(f: &Folded) -> Vec<f64> {
        let m = f.m;
        let mut merged = vec![0.0; m * m];
        for (j, w) in f.zb.windows(2).enumerate() {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let row = (0.5 * (w[0] + w[1]) * m as f64) as usize;
            for i in 0..m {
                merged[row * m + i] += f.side[j * m + i];
            }
        }
        merged
    }

    fn stack() -> Profile {
        // A thin high-k layer below the centre, the centre on an interface,
        // and a low-k layer above.
        Profile::new(vec![0.2, 0.26, 0.5, 0.8], vec![3.9, 7.3, 4.05, 4.5, 2.0])
    }

    #[test]
    fn unit_contrast_matches_the_homogeneous_cube() {
        let m = 16;
        let h = GreenTables::new(m);
        let c = MultiLayerCube::new(Profile::new(vec![0.3, 0.5], vec![2.0, 2.0, 2.0]), m);
        let f = &c.folded;
        for (face, name) in [(&f.bottom, "bottom"), (&f.top, "top")] {
            for (a, b) in face.iter().zip(h.face_probabilities()) {
                assert!((a - b).abs() < 1e-13, "{name}");
            }
        }
        for (a, b) in merged_rows(f).iter().zip(h.face_probabilities()) {
            assert!((a - b).abs() < 1e-13, "side");
        }
    }

    #[test]
    fn one_interface_through_the_centre_matches_the_layered_cube() {
        let m = 16;
        for kappa in [0.1, 0.5, 3.0, 10.0] {
            let a = MultiLayerCube::new(Profile::new(vec![0.5], vec![1.0, kappa]), m);
            let b = LayeredCube::new(kappa, 0, 8, m);
            let p = b.cell_probabilities();
            let mm = m * m;
            let f = &a.folded;
            let side = merged_rows(f);
            for i in 0..mm {
                assert!((side[i] - p[i]).abs() < 1e-12, "κ={kappa} side");
                assert!(
                    (f.bottom[i] - p[4 * mm + i]).abs() < 1e-12,
                    "κ={kappa} bottom"
                );
                assert!((f.top[i] - p[5 * mm + i]).abs() < 1e-12, "κ={kappa} top");
            }
        }
    }

    #[test]
    fn eigenvalues_are_increasing_roots() {
        let p = stack();
        let md = modes(&p, 60);
        for (k, mo) in md.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let target = (k + 1) as f64 * PI;
            assert!((prufer_end(&p, mo.mu) - target).abs() < 1e-9);
            // ψ(1) = 0.
            assert!(mo.value(&p, 1.0).abs() < 1e-8 * mo.norm.sqrt().max(1.0));
        }
        assert!(md.windows(2).all(|w| w[0].mu < w[1].mu));
        // ε-orthogonality of two modes (numerical quadrature).
        let n = 200_000;
        #[allow(clippy::cast_precision_loss)]
        let dot: f64 = (0..n)
            .map(|i| {
                let z = (f64::from(i) + 0.5) / f64::from(n);
                p.eps[p.layer(z)] * md[2].value(&p, z) * md[5].value(&p, z)
            })
            .sum::<f64>()
            / f64::from(n);
        assert!(dot.abs() < 1e-6 * (md[2].norm * md[5].norm).sqrt(), "{dot}");
    }

    #[test]
    fn probabilities_sum_to_one_and_reproduce_the_layered_linear_potential() {
        for m in [16, 32] {
            let p = stack();
            let c = MultiLayerCube::new(p.clone(), m);
            let f = &c.folded;
            assert!((f.total() - 1.0).abs() < 1e-12, "sum {}", f.total());
            // φ(0) = 0, φ′ = 1/ε: ε-harmonic and linear within every side cell.
            let b = p.bounds();
            let phi = |z: f64| {
                let mut v = 0.0;
                for l in 0..p.eps.len() {
                    let hi = b[l + 1].min(z);
                    if hi > b[l] {
                        v += (hi - b[l]) / p.eps[l];
                    }
                }
                v
            };
            // φ is linear within every side cell (cells never straddle an
            // interface), so its cell average is its value at the midpoint.
            let mut mean = f.top.iter().sum::<f64>() * phi(1.0);
            for (j, w) in f.zb.windows(2).enumerate() {
                let row: f64 = f.side[j * m..(j + 1) * m].iter().sum();
                mean += 4.0 * row * phi(0.5 * (w[0] + w[1]));
            }
            // Exact for the kernel; the residual is that of cell-uniform
            // sampling with interfaces off the uniform grid (Z_REFINE).
            assert!(
                (mean - phi(0.5)).abs() < 1e-6,
                "m={m}: {mean} vs {}",
                phi(0.5)
            );
            // With every interface on the uniform grid it is exact.
            let g = Profile::new(vec![0.25, 0.5, 0.875], vec![2.0, 7.0, 1.0, 3.0]);
            let c = MultiLayerCube::new(g.clone(), m);
            let gb = g.bounds();
            let phi_g = |z: f64| {
                (0..g.eps.len())
                    .map(|l| (gb[l + 1].min(z) - gb[l]).max(0.0) / g.eps[l])
                    .sum::<f64>()
            };
            let f = &c.folded;
            let mut mean = f.top.iter().sum::<f64>() * phi_g(1.0);
            for (j, w) in f.zb.windows(2).enumerate() {
                let row: f64 = f.side[j * m..(j + 1) * m].iter().sum();
                mean += 4.0 * row * phi_g(0.5 * (w[0] + w[1]));
            }
            assert!((mean - phi_g(0.5)).abs() < 1e-12, "grid-aligned m={m}");
        }
    }

    #[test]
    fn landing_box_of_unit_ratio_is_the_cube() {
        let m = 16;
        let h = GreenTables::new(m);
        let b = LandingBox::new(1.0, 0.5, m);
        for face in [&b.folded.top, &b.folded.bottom, &b.folded.side] {
            for (x, y) in face.iter().zip(h.face_probabilities()) {
                assert!((x - y).abs() < 1e-13);
            }
        }
    }

    const LANDING_TAUS_TEST: [f64; 4] = [0.125, 0.25, 0.375, 0.5];

    #[test]
    fn off_centre_landing_boxes_reproduce_harmonic_polynomials() {
        // Cell averages of φ = z and φ = x² − z² (harmonic) against their
        // values at the start point (0, 0, 2τ − 1).
        let m = 32;
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        let sq = |a: f64, b: f64| (a * a + a * b + b * b) / 3.0; // mean of u² on [a, b]
        for ratio in [1.0, 4.0, 16.0] {
            for tau in LANDING_TAUS_TEST {
                let b = LandingBox::new(ratio, tau, m);
                let f = &b.folded;
                assert!((f.total() - 1.0).abs() < 1e-12, "L={ratio} τ={tau}");
                let edge = |i: usize| ratio * (2.0 * f.xb[i] - 1.0);
                let zc = |j: usize| 2.0 * j as f64 / mf - 1.0;
                let (mut lin, mut quad) = (0.0, 0.0);
                for j in 0..m {
                    for i in 0..m {
                        let k = j * m + i;
                        let x2 = sq(edge(i), edge(i + 1));
                        // Horizontal faces: z = ±1.
                        lin += f.top[k] - f.bottom[k];
                        quad += (f.top[k] + f.bottom[k]) * (x2 - 1.0);
                        // Four side faces: half with x = ±L, half with x tangential.
                        let (z, z2) = (0.5 * (zc(j) + zc(j + 1)), sq(zc(j), zc(j + 1)));
                        lin += 4.0 * f.side[k] * z;
                        quad +=
                            2.0 * f.side[k] * (ratio * ratio - z2) + 2.0 * f.side[k] * (x2 - z2);
                    }
                }
                let z0 = 2.0 * tau - 1.0;
                assert!((lin - z0).abs() < 1e-10, "L={ratio} τ={tau}: {lin} vs {z0}");
                // φ = z is exact (linear in every cell). For x² − z², cells
                // carry their probability uniformly: the error is a fraction
                // of the box's lateral extent squared, about that of the
                // cube tables.
                let err = (quad + z0 * z0).abs();
                assert!(err < 1e-3 * ratio * ratio, "L={ratio} τ={tau}: {err}");
                // The start height sets the landing probability, about 1 − τ
                // for flat boxes.
                if ratio >= 16.0 {
                    assert!((b.bottom_probability() - (1.0 - tau)).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn flat_landing_boxes_exit_mostly_through_the_faces() {
        let mut last = 1.0 / 6.0 - 1e-12;
        for r in 0..=8 {
            let ratio = 2f64.powf(f64::from(r) / 2.0);
            let b = LandingBox::new(ratio, 0.5, 16);
            assert!((b.folded.total() - 1.0).abs() < 1e-12, "L={ratio}");
            let p = b.bottom_probability();
            assert!(p > last && p < 0.5, "L={ratio}: {p}");
            last = p;
        }
        assert!(last > 0.49, "{last}");
    }

    #[test]
    #[allow(clippy::float_cmp)] // face coordinates are exactly ±lat or ±1
    fn sampling_matches_the_cells() {
        let m = 16;
        let c = MultiLayerCube::new(stack(), m);
        let f = &c.folded;
        let mz = f.zb.len() - 1;
        let mut rng = Rng::from_seed(9);
        let n = 2_000_000_u32;
        let (mut bottom, mut top) = (vec![0u32; m * m], vec![0u32; m * m]);
        let mut side = vec![0u32; m * mz];
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = |x: f64| (((x + 1.0) * 0.5 * mf) as usize).min(m - 1);
        for _ in 0..n {
            let p = c.sample_transition(&mut rng);
            if p[2] == -1.0 {
                bottom[bin(p[1]) * m + bin(p[0])] += 1;
            } else if p[2] == 1.0 {
                top[bin(p[1]) * m + bin(p[0])] += 1;
            } else {
                let u = if p[0].abs() == 1.0 { p[1] } else { p[0] };
                let z = 0.5 * (p[2] + 1.0);
                let j = f.zb.partition_point(|&b| b <= z) - 1;
                side[j * m + bin(u)] += 1;
            }
        }
        let (mut chi2, mut dof) = (0.0_f64, 0.0_f64);
        let mut add = |k: u32, p: f64| {
            let e = p * f64::from(n);
            if e > 20.0 {
                chi2 += (f64::from(k) - e).powi(2) / e;
                dof += 1.0;
            }
        };
        for i in 0..m * m {
            add(bottom[i], f.bottom[i]);
            add(top[i], f.top[i]);
        }
        for i in 0..m * mz {
            add(side[i], 4.0 * f.side[i]);
        }
        assert!(dof > 500.0, "dof {dof}");
        assert!(
            chi2 < dof + 5.0 * (2.0 * dof).sqrt(),
            "chi2 {chi2} for {dof}"
        );
    }

    /// Side-face cells of a band kernel merged into `m` uniform `z` rows,
    /// in the six-face `m × m` layout of [`crate::layered`].
    fn band_uniform(k: &BandGradient) -> Vec<f64> {
        let m = k.m;
        let mz = k.zb.len() - 1;
        let mut out = vec![0.0; 6 * m * m];
        for face in 0..4 {
            for (j, w) in k.zb.windows(2).enumerate() {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                let row = (0.5 * (w[0] + w[1]) * m as f64) as usize;
                for i in 0..m {
                    out[face * m * m + row * m + i] += k.cells()[face * m * mz + j * m + i];
                }
            }
        }
        out[4 * m * m..].copy_from_slice(&k.cells()[4 * m * mz..]);
        out
    }

    #[test]
    fn band_kernel_with_one_interface_matches_the_half_band_kernel() {
        use crate::layered::{BandSide, half_band_gradient};
        let m = 16;
        for kappa in [0.1, 0.5, 3.0, 10.0] {
            let p = Profile::new(vec![0.5], vec![1.0, kappa]);
            for (side, j) in [(BandSide::Below, 3), (BandSide::Above, 16)] {
                let half = 0.5 * f64::from(j) / 32.0;
                let (z0, z1) = match side {
                    BandSide::Below => (0.5 - half, 0.5),
                    BandSide::Above => (0.5, 0.5 + half),
                };
                let a = band_uniform(&BandGradient::new(&p, z0, z1, m));
                let b = half_band_gradient(kappa, usize::try_from(j).unwrap(), 32, side, m);
                for (i, (x, y)) in a.iter().zip(b.cells()).enumerate() {
                    assert!(
                        (x - y).abs() < 1e-11,
                        "κ={kappa} {side:?} cell {i}: {x} vs {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn band_kernel_reproduces_a_lateral_linear_potential() {
        // φ = x is a solution in every stratified medium; its x-derivative is 1.
        let m = 16;
        let p = stack();
        let k = BandGradient::new(&p, 0.5, 0.56, m);
        let mz = k.zb.len() - 1;
        #[allow(clippy::cast_precision_loss)]
        let centre = |i: usize| (i as f64 + 0.5) / m as f64;
        let mut sum = 0.0;
        for (idx, w) in k.cells().iter().enumerate() {
            let x = if idx < 4 * m * mz {
                let (face, cell) = (idx / (m * mz), idx % (m * mz));
                match face {
                    0 => 0.0,
                    1 => 1.0,
                    _ => centre(cell % m),
                }
            } else {
                centre((idx - 4 * m * mz) % (m * m) % m)
            };
            sum += w * x;
        }
        assert!((sum - 1.0).abs() < 1e-10, "{sum}");
    }

    #[test]
    #[allow(clippy::float_cmp)] // face coordinates are exactly ±1
    fn band_sampling_matches_the_cells() {
        let m = 8;
        let k = BandGradient::new(&stack(), 0.45, 0.5, m);
        let mz = k.zb.len() - 1;
        let per_side = m * mz;
        let mut rng = Rng::from_seed(5);
        let n = 2_000_000_u32;
        let mut counts = vec![0u32; k.cells().len()];
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = |x: f64| (((x + 1.0) * 0.5 * mf) as usize).min(m - 1);
        for _ in 0..n {
            let (p, sign) = k.sample(&mut rng);
            let idx = if p[2].abs() == 1.0 && p[0].abs() != 1.0 && p[1].abs() != 1.0 {
                let face = usize::from(p[2] == 1.0);
                4 * per_side + face * m * m + bin(p[1]) * m + bin(p[0])
            } else {
                let (face, u) = if p[0].abs() == 1.0 {
                    (usize::from(p[0] == 1.0), p[1])
                } else {
                    (2 + usize::from(p[1] == 1.0), p[0])
                };
                let z = 0.5 * (p[2] + 1.0);
                let j = k.zb.partition_point(|&b| b <= z) - 1;
                face * per_side + j * m + bin(u)
            };
            assert_eq!(sign, k.cells()[idx].signum());
            counts[idx] += 1;
        }
        let (mut chi2, mut dof) = (0.0_f64, 0.0_f64);
        for (c, w) in counts.iter().zip(k.cells()) {
            let e = w.abs() / k.l1() * f64::from(n);
            if e > 20.0 {
                chi2 += (f64::from(*c) - e).powi(2) / e;
                dof += 1.0;
            }
        }
        assert!(dof > 300.0, "dof {dof}");
        assert!(
            chi2 < dof + 5.0 * (2.0 * dof).sqrt(),
            "chi2 {chi2} for {dof}"
        );
    }

    /// Side cells of a [`NormalGradient`] (one face) merged into `m` uniform rows.
    fn merged_normal_side(k: &NormalGradient) -> Vec<f64> {
        let m = k.m;
        let mut merged = vec![0.0; m * m];
        for (j, w) in k.zb.windows(2).enumerate() {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let row = (0.5 * (w[0] + w[1]) * m as f64) as usize;
            for i in 0..m {
                merged[row * m + i] += k.cells()[j * m + i];
            }
        }
        merged
    }

    #[test]
    fn normal_kernel_with_one_interface_matches_the_layered_kernel() {
        let (m, offsets) = (16, 8);
        for kappa in [1.0, 0.25, 3.0, 10.0] {
            for k in [1, 3, 7] {
                #[allow(clippy::cast_precision_loss)]
                let zi = 0.5 + k as f64 / (2.0 * offsets as f64);
                let a = NormalGradient::new(&Profile::new(vec![zi], vec![1.0, kappa]), m);
                let b = crate::layered::LayeredGradients::new(kappa, k, offsets, m);
                let c = b.z.cells();
                let mm = m * m;
                let side = merged_normal_side(&a);
                let per_side = m * (a.zb.len() - 1);
                let tol = 1e-11 * b.z.l1();
                for i in 0..mm {
                    assert!((side[i] - c[i]).abs() < tol, "κ={kappa} k={k} side");
                    let (ab, at) = (
                        a.cells()[4 * per_side + i],
                        a.cells()[4 * per_side + mm + i],
                    );
                    assert!((ab - c[4 * mm + i]).abs() < tol, "κ={kappa} k={k} bottom");
                    assert!((at - c[5 * mm + i]).abs() < tol, "κ={kappa} k={k} top");
                }
                // Finer side cells separate signs a uniform row mixes.
                assert!(a.l1() >= b.z.l1() * (1.0 - 1e-10));
            }
        }
    }

    #[test]
    fn normal_kernel_reproduces_the_layered_linear_potential() {
        // φ(0) = 0, φ′ = 1/ε: ∂φ/∂z at the centre is 1/ε of its layer. Two
        // facing interfaces around the centre (a gap between two shells) and
        // a profile with interfaces off the grid.
        for (z, eps) in [
            (vec![0.2, 0.8], vec![3.5, 4.5, 3.5]),
            (vec![0.137, 0.31, 0.71], vec![2.0, 3.5, 4.4, 7.0]),
        ] {
            let p = Profile::new(z, eps);
            let m = 32;
            let k = NormalGradient::new(&p, m);
            let b = p.bounds();
            let phi = |z: f64| {
                (0..p.eps.len())
                    .map(|l| (b[l + 1].min(z) - b[l]).max(0.0) / p.eps[l])
                    .sum::<f64>()
            };
            let per_side = m * (k.zb.len() - 1);
            let mut sum = 0.0;
            for (j, w) in k.zb.windows(2).enumerate() {
                let row: f64 = k.cells()[j * m..(j + 1) * m].iter().sum();
                sum += 4.0 * row * phi(0.5 * (w[0] + w[1]));
            }
            sum += k.cells()[4 * per_side + m * m..].iter().sum::<f64>() * phi(1.0);
            let exact = 1.0 / p.eps[p.layer(0.5)];
            assert!((sum - exact).abs() < 1e-6 * exact, "{sum} vs {exact}");
            // ∫K = 0: constants have no gradient.
            let total: f64 = k.cells().iter().sum();
            assert!(total.abs() < 1e-10 * k.l1(), "{total}");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // face coordinates are exactly ±1
    fn normal_kernel_sampling_matches_the_cells() {
        let m = 8;
        let k = NormalGradient::new(&Profile::new(vec![0.2, 0.8], vec![3.5, 4.5, 3.5]), m);
        let per_side = m * (k.zb.len() - 1);
        let mut rng = Rng::from_seed(7);
        let n = 1_000_000_u32;
        let mut counts = vec![0u32; k.cells().len()];
        let mut pair_sign = [0.0_f64; 2];
        #[allow(clippy::cast_precision_loss)]
        let mf = m as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = |x: f64| (((x + 1.0) * 0.5 * mf) as usize).min(m - 1);
        let index = |p: [f64; 3]| {
            if p[2].abs() == 1.0 && p[0].abs() != 1.0 && p[1].abs() != 1.0 {
                let face = usize::from(p[2] == 1.0);
                4 * per_side + face * m * m + bin(p[1]) * m + bin(p[0])
            } else {
                let (face, u) = if p[0].abs() == 1.0 {
                    (usize::from(p[0] == 1.0), p[1])
                } else {
                    (2 + usize::from(p[1] == 1.0), p[0])
                };
                let z = 0.5 * (p[2] + 1.0);
                let j = k.zb.partition_point(|&b| b <= z) - 1;
                face * per_side + j * m + bin(u)
            }
        };
        for _ in 0..n {
            let (p, sign) = k.sample(&mut rng);
            let idx = index(p);
            assert_eq!(sign, k.cells()[idx].signum());
            counts[idx] += 1;
            let [a, b] = k.sample_pair(&mut rng);
            pair_sign[0] += k.cells()[index(a)].signum();
            pair_sign[1] += k.cells()[index(b)].signum();
        }
        assert_eq!(pair_sign, [f64::from(n), -f64::from(n)]);
        let (mut chi2, mut dof) = (0.0_f64, 0.0_f64);
        for (c, w) in counts.iter().zip(k.cells()) {
            let e = w.abs() / k.l1() * f64::from(n);
            if e > 20.0 {
                chi2 += (f64::from(*c) - e).powi(2) / e;
                dof += 1.0;
            }
        }
        assert!(
            chi2 < dof + 5.0 * (2.0 * dof).sqrt(),
            "chi² {chi2} for {dof} cells"
        );
    }
}
