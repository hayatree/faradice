//! Insertion maps (design D3 §7, pillar B): potentials of two nets on a grid
//! of points and the leading-order change of their coupling if a small
//! grounded or floating object were inserted there.

use std::fmt::Write as _;

use faradice_core::{EPS0_FF_PER_UM, NetId};
use faradice_engine::{Project, RunRequest};
use faradice_frw::map::{MapOptions, PointPotentials, potential_map};

use crate::{Result, SensError};

/// A regular grid: `lo..=hi` with `n` points per axis (`n = 1` uses `lo`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// `(lo, hi, n)` per axis (µm).
    pub axes: [(f64, f64, usize); 3],
}

impl Grid {
    /// Parses `x0:x1:nx,y0:y1:ny,z0:z1:nz`; an axis may also be a single
    /// value `z`.
    ///
    /// # Errors
    ///
    /// Malformed text, non-finite values or zero counts.
    pub fn parse(s: &str) -> Result<Self> {
        let bad = || SensError::Param(format!("grid `{s}`: expected x0:x1:nx,y0:y1:ny,z0:z1:nz"));
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(bad());
        }
        let mut axes = [(0.0, 0.0, 1); 3];
        for (a, p) in parts.iter().enumerate() {
            let f: Vec<&str> = p.split(':').collect();
            let num = |t: &str| t.trim().parse::<f64>().ok().filter(|v| v.is_finite());
            axes[a] = match f.as_slice() {
                [v] => {
                    let v = num(v).ok_or_else(bad)?;
                    (v, v, 1)
                }
                [lo, hi, n] => {
                    let n: usize = n.trim().parse().map_err(|_| bad())?;
                    if n == 0 {
                        return Err(bad());
                    }
                    (num(lo).ok_or_else(bad)?, num(hi).ok_or_else(bad)?, n)
                }
                _ => return Err(bad()),
            };
        }
        Ok(Self { axes })
    }

    /// The points, x fastest.
    #[must_use]
    pub fn points(&self) -> Vec<[f64; 3]> {
        let coord = |(lo, hi, n): (f64, f64, usize), i: usize| {
            if n == 1 {
                lo
            } else {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / (n - 1) as f64;
                lo + t * (hi - lo)
            }
        };
        let [ax, ay, az] = self.axes;
        let mut v = Vec::with_capacity(ax.2 * ay.2 * az.2);
        for k in 0..az.2 {
            for j in 0..ay.2 {
                for i in 0..ax.2 {
                    v.push([coord(ax, i), coord(ay, j), coord(az, k)]);
                }
            }
        }
        v
    }
}

/// An insertion-map request.
#[derive(Debug, Clone)]
pub struct MapRequest {
    /// The configuration (accuracy is ignored; the seed and walk options apply).
    pub run: RunRequest,
    /// The two nets whose coupling is mapped (may be equal for C_ii).
    pub nets: (String, String),
    /// Points.
    pub grid: Grid,
    /// Samples per point.
    pub samples: u64,
    /// Radius (µm) of the sphere-equivalent object for the predictions.
    pub radius: f64,
}

/// One row of the map.
#[derive(Debug, Clone, PartialEq)]
pub struct MapRow {
    /// The point and its potentials.
    pub p: PointPotentials,
    /// `ΔC_ij` (fF) for a grounded sphere of the radius: `4πεr · φ_i φ_j`
    /// (NaN inside conductors, outside the domain, or with aborted walks).
    pub grounded: f64,
    /// `ΔC_ij` (fF) for a floating sphere: `4πεr³ · ∇φ_i·∇φ_j`.
    pub floating: f64,
}

/// Computes an insertion map.
///
/// # Errors
///
/// Unknown nets, stratified dielectrics (predictions need one ε), invalid
/// configuration or solver errors.
pub fn insertion_map(project: &Project, req: &MapRequest) -> Result<(Vec<MapRow>, [NetId; 2])> {
    let p = faradice_engine::prepare(project, &req.run)?;
    if !p.solver.layers.is_empty() {
        return Err(SensError::Param(
            "insertion maps need a homogeneous dielectric for now".into(),
        ));
    }
    let g = &p.geometry;
    let id = |n: &str| {
        g.net_id(n)
            .ok_or_else(|| SensError::Param(format!("unknown net `{n}`")))
    };
    let (i, j) = (id(&req.nets.0)?, id(&req.nets.1)?);
    let mut targets = vec![i];
    if j != i {
        targets.push(j);
    }
    let pool = faradice_engine::thread_pool(&p.config)?;
    let pts = req.grid.points();
    let map = pool.install(|| {
        potential_map(
            g,
            &targets,
            &p.solver,
            &pts,
            MapOptions {
                samples: req.samples,
                gradient: true,
            },
        )
    })?;
    let eps = p.solver.eps_r * EPS0_FF_PER_UM;
    let four_pi = 4.0 * std::f64::consts::PI;
    let rows = map
        .into_iter()
        .map(|pp| {
            // Unbiased products (not products of the two means).
            let ff = pp.phi_product(i, j);
            let dot = pp.grad_dot(i, j);
            let inside = pp.inside.is_some() || pp.outside || pp.aborted_walks > 0;
            MapRow {
                grounded: if inside {
                    f64::NAN
                } else {
                    four_pi * eps * req.radius * ff
                },
                floating: if inside {
                    f64::NAN
                } else {
                    four_pi * eps * req.radius.powi(3) * dot
                },
                p: pp,
            }
        })
        .collect();
    Ok((rows, [i, j]))
}

/// The map as CSV: point, potentials and gradients of both nets with their
/// standard errors, and the two predictions. Points inside conductors have
/// empty predictions.
#[must_use]
pub fn to_csv(rows: &[MapRow], nets: [NetId; 2], names: (&str, &str)) -> String {
    let mut s = String::new();
    let (a, b) = names;
    let _ = writeln!(
        s,
        "x_um,y_um,z_um,inside,phi_{a},phi_{a}_err,phi_{b},phi_{b}_err,\
         dx_phi_{a},dy_phi_{a},dz_phi_{a},dx_phi_{b},dy_phi_{b},dz_phi_{b},\
         dC_grounded_fF,dC_floating_fF,status"
    );
    for r in rows {
        let p = &r.p;
        let (fa, fb) = (p.phi(nets[0]), p.phi(nets[1]));
        let (ga, gb) = (p.grad(nets[0]), p.grad(nets[1]));
        let num = |v: f64| {
            if v.is_finite() {
                format!("{v:.6e}")
            } else {
                String::new()
            }
        };
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            p.point[0],
            p.point[1],
            p.point[2],
            u8::from(p.inside.is_some()),
            num(fa.value),
            num(fa.std_err),
            num(fb.value),
            num(fb.std_err),
            num(ga[0].value),
            num(ga[1].value),
            num(ga[2].value),
            num(gb[0].value),
            num(gb[1].value),
            num(gb[2].value),
            num(r.grounded),
            num(r.floating),
            if p.outside {
                "outside".to_owned()
            } else if p.inside.is_some() {
                "inside".to_owned()
            } else if p.aborted_walks > 0 {
                format!("aborted:{}", p.aborted_walks)
            } else {
                "ok".to_owned()
            },
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_parsing() {
        let g = Grid::parse("0:1:3, 2 ,0:4:2").unwrap();
        let p = g.points();
        assert_eq!(p.len(), 6);
        assert_eq!(p[0], [0.0, 2.0, 0.0]);
        assert_eq!(p[2], [1.0, 2.0, 0.0]);
        assert_eq!(p[5], [1.0, 2.0, 4.0]);
        for bad in ["0:1:3,2", "0:1:0,1,1", "a,1,1", "0:1,1,1"] {
            assert!(Grid::parse(bad).is_err(), "{bad}");
        }
    }
}
