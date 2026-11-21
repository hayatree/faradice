//! Exact what-if edits on the conductor-level Maxwell matrix (design D3 §5,
//! pillar C): swap, merge, ground and float conductors without re-solving.
//!
//! The matrix is built from the raw `result.json`: row `i` comes from the
//! walks of target `i`, entry `(i, j)` from its sink `j`, symmetrized as
//! `(C_ij + C_ji)/2`. Nets that are not targets are grounded conductors, as in
//! the extraction. Standard errors use the delta method with a numerical
//! Jacobian and the full within-row covariance (sink cross moments, written
//! with `output.cross_moments = true`); rows are independent.

use faradice_core::{Estimate, Moments};
use faradice_io::result::{Completeness, ResultFile};
use serde::{Deserialize, Serialize};

use crate::{Result, SensError};

/// One edit. [`apply`] runs edits in the order swap → merge → ground →
/// float, whatever order they are given in (ground and float commute).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum Edit {
    /// Exchange the names (net assignments) of two conductors.
    Swap {
        /// First conductor.
        a: String,
        /// Second conductor.
        b: String,
    },
    /// Connect two conductors into one net, named `a+b`.
    Merge {
        /// First conductor.
        a: String,
        /// Second conductor.
        b: String,
    },
    /// Tie a conductor to ground: its couplings become ground capacitance.
    Ground {
        /// The conductor.
        net: String,
    },
    /// Leave a conductor floating (zero net charge, e.g. dummy fill).
    Float {
        /// The conductor.
        net: String,
    },
}

impl Edit {
    const fn rank(&self) -> u8 {
        match self {
            Self::Swap { .. } => 0,
            Self::Merge { .. } => 1,
            Self::Ground { .. } => 2,
            Self::Float { .. } => 3,
        }
    }
}

/// A symmetric matrix over named conductors (row-major, `n × n`).
#[derive(Debug, Clone, PartialEq)]
struct Sym {
    names: Vec<String>,
    c: Vec<f64>,
}

impl Sym {
    fn n(&self) -> usize {
        self.names.len()
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.c[i * self.n() + j]
    }

    fn index(&self, name: &str) -> Result<usize> {
        self.names.iter().position(|n| n == name).ok_or_else(|| {
            SensError::Param(format!(
                "what-if: `{name}` is not a target of this result (every edited net must be \
                     a target)"
            ))
        })
    }

    fn remove(&self, k: usize) -> Self {
        let n = self.n();
        let keep: Vec<usize> = (0..n).filter(|&i| i != k).collect();
        Self {
            names: keep.iter().map(|&i| self.names[i].clone()).collect(),
            c: keep
                .iter()
                .flat_map(|&i| keep.iter().map(move |&j| (i, j)))
                .map(|(i, j)| self.at(i, j))
                .collect(),
        }
    }

    fn apply(&self, e: &Edit) -> Result<Self> {
        let n = self.n();
        Ok(match e {
            Edit::Swap { a, b } => {
                let (i, j) = (self.index(a)?, self.index(b)?);
                let mut m = self.clone();
                m.names.swap(i, j);
                m
            }
            Edit::Merge { a, b } => {
                let (i, j) = (self.index(a)?, self.index(b)?);
                if i == j {
                    return Err(SensError::Param(format!(
                        "what-if: cannot merge `{a}` with itself"
                    )));
                }
                // Mᵀ C M: row/column j is added into i, then j is removed.
                let mut m = self.clone();
                for k in 0..n {
                    m.c[i * n + k] += self.at(j, k);
                }
                for k in 0..n {
                    m.c[k * n + i] = m.c[i * n + k];
                }
                m.c[i * n + i] = self.at(i, i) + self.at(j, j) + 2.0 * self.at(i, j);
                m.names[i] = format!("{a}+{b}");
                m.remove(j)
            }
            Edit::Ground { net } => self.remove(self.index(net)?),
            Edit::Float { net } => {
                let k = self.index(net)?;
                let ckk = self.at(k, k);
                if ckk.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return Err(SensError::Param(format!(
                        "what-if: `{net}` has no positive total capacitance"
                    )));
                }
                let mut m = self.clone();
                for i in 0..n {
                    for j in 0..n {
                        m.c[i * n + j] -= self.at(i, k) * self.at(k, j) / ckk;
                    }
                }
                m.remove(k)
            }
        })
    }

    /// Applies `e` to `self` and propagates the tangent `t` (a direction in
    /// matrix space) exactly: every edit is linear except float, whose
    /// derivative is `dC − (dc cᵀ + c dcᵀ)/c_kk + c cᵀ dc_kk/c_kk²`.
    fn apply_tangent(&self, t: &Self, e: &Edit) -> Result<(Self, Self)> {
        let Edit::Float { net } = e else {
            let mut tt = t.apply(e)?;
            let m = self.apply(e)?;
            tt.names.clone_from(&m.names);
            return Ok((m, tt));
        };
        let n = self.n();
        let k = self.index(net)?;
        let m = self.apply(e)?;
        let ckk = self.at(k, k);
        let dkk = t.at(k, k);
        let mut tt = t.clone();
        for i in 0..n {
            for j in 0..n {
                let (ci, cj) = (self.at(i, k), self.at(k, j));
                let (di, dj) = (t.at(i, k), t.at(k, j));
                tt.c[i * n + j] -= (di * cj + ci * dj) / ckk - ci * cj * dkk / (ckk * ckk);
            }
        }
        tt.names.clone_from(&self.names);
        let mut tt = tt.remove(k);
        tt.names.clone_from(&m.names);
        Ok((m, tt))
    }

    /// The quantities reported: per conductor `[total, ground (row sum),
    /// −C_ij for j > i]`, flattened.
    fn outputs(&self) -> Vec<f64> {
        let n = self.n();
        let mut v = Vec::new();
        for i in 0..n {
            v.push(self.at(i, i));
            v.push((0..n).map(|j| self.at(i, j)).sum());
            v.extend((i + 1..n).map(|j| -self.at(i, j)));
        }
        v
    }
}

fn ordered(edits: &[Edit]) -> Vec<&Edit> {
    let mut order: Vec<&Edit> = edits.iter().collect();
    order.sort_by_key(|e| e.rank());
    order
}

fn run(edits: &[Edit], base: &Sym) -> Result<Sym> {
    ordered(edits)
        .into_iter()
        .try_fold(base.clone(), |m, e| m.apply(e))
}

/// The edited matrix and the derivative of its outputs along `tangent`.
fn run_tangent(edits: &[Edit], base: &Sym, tangent: &Sym) -> Result<Vec<f64>> {
    let (_, t) = ordered(edits)
        .into_iter()
        .try_fold((base.clone(), tangent.clone()), |(m, t), e| {
            m.apply_tangent(&t, e)
        })?;
    Ok(t.outputs())
}

/// The raw row statistics of the targets of a result file.
struct Rows {
    names: Vec<String>,
    /// `mean[i][j]`: raw C_ij from row i.
    mean: Vec<Vec<f64>>,
    /// `cov[i][j][l]`: covariance of the means of row i's entries j and l.
    cov: Vec<Vec<Vec<f64>>>,
}

fn rows(r: &ResultFile, warnings: &mut Vec<String>) -> Result<Rows> {
    if r.completeness != Completeness::Complete {
        return Err(SensError::Param(
            "what-if needs a complete result (this run was cancelled or partial)".into(),
        ));
    }
    let names: Vec<String> = r.targets.iter().map(|t| t.net.clone()).collect();
    let n = names.len();
    let have_cross = faradice_io::config::Config::parse(&r.config_toml, "result.json").map_or_else(
        |_| r.targets.iter().any(|t| !t.cross.is_empty()),
        |c| c.output.cross_moments,
    );
    let mut mean = vec![vec![0.0; n]; n];
    let mut cov = vec![vec![vec![0.0; n]; n]; n];
    for (i, t) in r.targets.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let w = t.walks as f64;
        let sums: Vec<(f64, f64)> = names
            .iter()
            .map(|j| {
                t.sinks
                    .iter()
                    .find(|s| &s.sink == j)
                    .map_or((0.0, 0.0), |s| (s.sum, s.sum_sq))
            })
            .collect();
        for (j, &(s, sq)) in sums.iter().enumerate() {
            let e: Estimate = Moments::from_sums(s, sq).estimate(t.walks);
            mean[i][j] = e.value;
            cov[i][j][j] = e.std_err * e.std_err;
        }
        if !have_cross {
            continue;
        }
        for j in 0..n {
            for l in j + 1..n {
                let s = t
                    .cross
                    .iter()
                    .find(|c| {
                        (c.a == names[j] && c.b == names[l]) || (c.a == names[l] && c.b == names[j])
                    })
                    .map_or(0.0, |c| c.sum);
                let c = (s - w * mean[i][j] * mean[i][l]) / ((w - 1.0) * w);
                cov[i][j][l] = c;
                cov[i][l][j] = c;
            }
        }
    }
    if !have_cross {
        warnings.push(
            "result.json has no cross moments: σ ignores within-row covariances (rerun with \
             output.cross_moments = true)"
                .into(),
        );
    }
    Ok(Rows { names, mean, cov })
}

/// A value with its standard error (fF).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Value {
    /// Value.
    pub value: f64,
    /// Standard error (1σ).
    pub std_err: f64,
}

/// One conductor of the edited matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhatIfNet {
    /// Conductor name (`a+b` for merged conductors).
    pub net: String,
    /// Total capacitance `C_ii` (fF).
    pub total: Value,
    /// Capacitance to ground and to non-target nets: `Σ_j C_ij` (fF).
    pub ground: Value,
    /// Couplings `−C_ij` (fF) to the other conductors, largest first.
    pub couplings: Vec<WhatIfCoupling>,
}

/// One coupling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhatIfCoupling {
    /// The other conductor.
    pub net: String,
    /// `−C_ij` (fF).
    pub c: Value,
}

/// Result of [`apply`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhatIf {
    /// The edits, in the order applied.
    pub edits: Vec<Edit>,
    /// The edited conductors.
    pub nets: Vec<WhatIfNet>,
    /// Warnings.
    pub warnings: Vec<String>,
}

/// Applies `edits` to the Maxwell matrix of `result`'s targets.
///
/// # Errors
///
/// Incomplete results, edits naming non-target conductors, or floating a
/// conductor without positive total capacitance.
pub fn apply(result: &ResultFile, edits: &[Edit]) -> Result<WhatIf> {
    let mut warnings = Vec::new();
    let rows = rows(result, &mut warnings)?;
    let n = rows.names.len();
    let mut base = Sym {
        names: rows.names.clone(),
        c: vec![0.0; n * n],
    };
    for i in 0..n {
        for j in 0..n {
            base.c[i * n + j] = 0.5 * (rows.mean[i][j] + rows.mean[j][i]);
        }
    }
    let out = run(edits, &base)?;
    let f0 = out.outputs();
    // Exact Jacobian with respect to each symmetric entry s_jl (j ≤ l), by
    // forward-mode propagation of the unit direction.
    let mut grad = vec![vec![vec![0.0; n]; n]; f0.len()]; // [output][j][l]
    for j in 0..n {
        for l in j..n {
            let mut dir = Sym {
                names: base.names.clone(),
                c: vec![0.0; n * n],
            };
            dir.c[j * n + l] = 1.0;
            dir.c[l * n + j] = 1.0;
            let d = run_tangent(edits, &base, &dir)?;
            for (o, g) in grad.iter_mut().enumerate() {
                g[j][l] = d[o];
                g[l][j] = d[o];
            }
        }
    }
    // ∂f/∂x_ij = ½ ∂f/∂s_ij off the diagonal, ∂f/∂s_ii on it.
    let var = |o: usize| -> f64 {
        let g = &grad[o];
        let dx = |j: usize, k: usize| if j == k { g[j][k] } else { 0.5 * g[j][k] };
        (0..n)
            .map(|i| {
                let mut v = 0.0;
                for (j, cj) in rows.cov[i].iter().enumerate() {
                    for (l, c) in cj.iter().enumerate() {
                        v += dx(i, j) * dx(i, l) * c;
                    }
                }
                v
            })
            .sum::<f64>()
            .max(0.0)
    };
    let est = |o: usize| Value {
        value: f0[o],
        std_err: var(o).sqrt(),
    };
    let m = out.n();
    let mut o = 0;
    let mut nets: Vec<WhatIfNet> = out
        .names
        .iter()
        .map(|name| WhatIfNet {
            net: name.clone(),
            total: Value::default(),
            ground: Value::default(),
            couplings: Vec::new(),
        })
        .collect();
    for i in 0..m {
        nets[i].total = est(o);
        nets[i].ground = est(o + 1);
        o += 2;
        for j in i + 1..m {
            let c = est(o);
            o += 1;
            nets[i].couplings.push(WhatIfCoupling {
                net: out.names[j].clone(),
                c,
            });
            nets[j].couplings.push(WhatIfCoupling {
                net: out.names[i].clone(),
                c,
            });
        }
    }
    for net in &mut nets {
        net.couplings
            .sort_by(|a, b| b.c.value.abs().total_cmp(&a.c.value.abs()));
    }
    let mut edits = edits.to_vec();
    edits.sort_by_key(Edit::rank);
    Ok(WhatIf {
        edits,
        nets,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(names: &[&str], c: &[f64]) -> Sym {
        Sym {
            names: names.iter().map(|s| (*s).to_owned()).collect(),
            c: c.to_vec(),
        }
    }

    /// Three plates in series: ground | c0 | P1 | c1 | M | c2 | P2 | c3 | ground.
    fn chain(c: [f64; 4]) -> Sym {
        sym(
            &["p1", "m", "p2"],
            &[
                c[0] + c[1],
                -c[1],
                0.0, //
                -c[1],
                c[1] + c[2],
                -c[2], //
                0.0,
                -c[2],
                c[2] + c[3],
            ],
        )
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn float_is_series_capacitance() {
        let c = [1.0, 2.0, 3.0, 4.0];
        let m = run(&[Edit::Float { net: "m".into() }], &chain(c)).unwrap();
        assert_eq!(m.names, ["p1", "p2"]);
        let series = c[1] * c[2] / (c[1] + c[2]);
        assert!(close(m.at(0, 1), -series));
        assert!(close(m.at(0, 0), c[0] + series));
        assert!(close(m.at(1, 1), c[3] + series));
    }

    #[test]
    fn merge_ground_swap() {
        let c = [1.0, 2.0, 3.0, 4.0];
        let m = run(
            &[Edit::Merge {
                a: "p1".into(),
                b: "p2".into(),
            }],
            &chain(c),
        )
        .unwrap();
        assert_eq!(m.names, ["p1+p2", "m"]);
        assert!(close(m.at(0, 0), c[0] + c[1] + c[2] + c[3]));
        assert!(close(m.at(0, 1), -(c[1] + c[2])));
        // Grounding m: p1 and p2 keep their totals and lose their coupling to it.
        let g = run(&[Edit::Ground { net: "m".into() }], &chain(c)).unwrap();
        assert_eq!(g.c, vec![c[0] + c[1], 0.0, 0.0, c[2] + c[3]]);
        let s = run(
            &[Edit::Swap {
                a: "p1".into(),
                b: "p2".into(),
            }],
            &chain(c),
        )
        .unwrap();
        assert_eq!(s.names, ["p2", "m", "p1"]);
        assert_eq!(s.c, chain(c).c);
    }

    #[test]
    fn ground_and_float_commute() {
        let base = sym(
            &["a", "b", "c", "d"],
            &[
                5.0, -1.0, -0.5, -0.2, //
                -1.0, 6.0, -2.0, -0.3, //
                -0.5, -2.0, 7.0, -1.5, //
                -0.2, -0.3, -1.5, 4.0,
            ],
        );
        let f = Edit::Float { net: "b".into() };
        let g = Edit::Ground { net: "d".into() };
        let x = base.apply(&f).unwrap().apply(&g).unwrap();
        let y = base.apply(&g).unwrap().apply(&f).unwrap();
        assert_eq!(x.names, y.names);
        for (a, b) in x.c.iter().zip(&y.c) {
            assert!(close(*a, *b));
        }
        assert!(base.apply(&Edit::Float { net: "zz".into() }).is_err());
    }

    #[test]
    fn tangents_match_finite_differences() {
        let base = sym(
            &["a", "b", "c", "d"],
            &[
                5.0, -1.0, -0.5, -0.2, //
                -1.0, 6.0, -2.0, -0.3, //
                -0.5, -2.0, 7.0, -1.5, //
                -0.2, -0.3, -1.5, 4.0,
            ],
        );
        let edits = [
            Edit::Merge {
                a: "a".into(),
                b: "d".into(),
            },
            Edit::Float { net: "b".into() },
            Edit::Float { net: "c".into() },
        ];
        let n = 4;
        for (j, l) in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3), (3, 3)] {
            let mut dir = sym(&["a", "b", "c", "d"], &[0.0; 16]);
            dir.c[j * n + l] = 1.0;
            dir.c[l * n + j] = 1.0;
            let t = run_tangent(&edits, &base, &dir).unwrap();
            let h = 1e-6;
            let mut p = base.clone();
            let mut q = base.clone();
            for (m, d) in [(&mut p, h), (&mut q, -h)] {
                m.c[j * n + l] += d;
                if j != l {
                    m.c[l * n + j] += d;
                }
            }
            let (fp, fq) = (
                run(&edits, &p).unwrap().outputs(),
                run(&edits, &q).unwrap().outputs(),
            );
            for (o, &to) in t.iter().enumerate() {
                let fd = (fp[o] - fq[o]) / (2.0 * h);
                assert!(
                    (to - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                    "({j},{l}) output {o}: {to} vs {fd}"
                );
            }
        }
    }

    /// Floating a conductor whose total is tiny next to another's must not
    /// fail in the σ computation (no finite-difference step).
    #[test]
    fn float_with_disparate_scales() {
        let base = sym(&["a", "b"], &[2e6, -0.1, -0.1, 1.0]);
        let dir = sym(&["a", "b"], &[0.0, 0.0, 0.0, 1.0]);
        let t = run_tangent(&[Edit::Float { net: "b".into() }], &base, &dir).unwrap();
        // d(C_aa − 0.01/C_bb)/dC_bb = 0.01.
        assert!((t[0] - 0.01).abs() < 1e-12, "{t:?}");
    }

    /// A result file with one walk per unit, so means equal the sums.
    fn result(cross: bool) -> ResultFile {
        let cfg = if cross {
            r"[output]\ncross_moments = true\n"
        } else {
            ""
        };
        let x = if cross {
            r#", "cross": [ { "a": "a", "b": "b", "sum": -150.0 } ]"#
        } else {
            ""
        };
        let text = format!(
            r#"{{ "schema_version": 2, "faradice_version": "t", "seed": 1, "eps_stop": 0.001,
            "elapsed_s": 0, "nets_all": ["a", "b"], "ground_nets": [], "warnings": [],
            "config_toml": "{cfg}", "targets": [
              {{ "net": "a", "walks": 100, "aborted_walks": 0, "total_steps": 0, "converged": true,
                "gauss_offset": 0.1, "ground": {{ "sink": "<GND>", "sum": 0, "sum_sq": 0 }},
                "sinks": [ {{ "sink": "a", "sum": 300.0, "sum_sq": 1000.0 }},
                           {{ "sink": "b", "sum": -100.0, "sum_sq": 200.0 }} ] {x} }},
              {{ "net": "b", "walks": 100, "aborted_walks": 0, "total_steps": 0, "converged": true,
                "gauss_offset": 0.1, "ground": {{ "sink": "<GND>", "sum": 0, "sum_sq": 0 }},
                "sinks": [ {{ "sink": "b", "sum": 400.0, "sum_sq": 1700.0 }},
                           {{ "sink": "a", "sum": -120.0, "sum_sq": 250.0 }} ] }} ] }}"#
        );
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn delta_method_matches_linear_propagation() {
        let r = result(true);
        let w = apply(&r, &[]).unwrap();
        let est = |s: f64, sq: f64| Moments::from_sums(s, sq).estimate(100);
        let (ab, ba) = (est(-100.0, 200.0), est(-120.0, 250.0));
        // Coupling: −(C_ab + C_ba)/2 with independent rows.
        let c = &w.nets[0].couplings[0].c;
        assert!(close(c.value, 110.0 / 100.0));
        assert!((c.std_err - 0.5 * ab.std_err.hypot(ba.std_err)).abs() < 1e-9);
        // Ground of a = C_aa + (C_ab + C_ba)/2: the within-row covariance of
        // (C_aa, C_ab) enters through the cross moment.
        let aa = est(300.0, 1000.0);
        // Σ c_aa c_ab = −150, means 3 and −1 over 100 walks.
        let cov = (-150.0 + 100.0 * 3.0) / (99.0 * 100.0);
        let var = aa.std_err.powi(2) + 0.25 * ab.std_err.powi(2) + cov + 0.25 * ba.std_err.powi(2);
        let g = &w.nets[0].ground;
        assert!(
            (g.std_err - var.sqrt()).abs() < 1e-9,
            "{g:?} vs {}",
            var.sqrt()
        );
        assert!(w.warnings.is_empty());
        assert!(!apply(&result(false), &[]).unwrap().warnings.is_empty());
    }
}
