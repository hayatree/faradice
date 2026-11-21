//! Post-processing of raw statistics into capacitance reports (design §4.5).
//!
//! Decisions about a coupling `(i, j)` use the symmetrized value and both
//! totals, so the same coupling is treated identically from both sides.
//! Moving a coupling to ground adds `multiplier × c_ij` (the row's own
//! estimate) to the ground capacitance of the net; with a multiplier of 1 the
//! total capacitance is unchanged. The standard error of a ground value that
//! received moved couplings is combined in quadrature (covariances between
//! sinks are neglected for the moved part; the original ground aggregate uses
//! the exact per-sample variance).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use faradice_core::{Estimate, Moments, is_reserved_label};
use serde::Serialize;

use crate::config::{
    CouplingConfig, CouplingMode, GroundReport, NonTarget, RelRef, Symmetrize, ThresholdOp,
};
use crate::result::ResultFile;

/// Boundary capacitance fraction above which a warning is emitted.
pub const BOUNDARY_WARN_RATIO: f64 = 0.01;

/// A value with its standard error (fF).
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct Value {
    /// Value (fF).
    pub value: f64,
    /// Standard error (fF, 1σ).
    pub sigma: f64,
}

impl From<Estimate> for Value {
    fn from(e: Estimate) -> Self {
        Self {
            value: e.value,
            sigma: e.std_err,
        }
    }
}

/// One coupling entry.
#[derive(Debug, Clone, Serialize)]
pub struct Coupling {
    /// The other net.
    pub net: String,
    /// Coupling capacitance (positive).
    pub c: Value,
}

/// Report of one target net.
#[derive(Debug, Clone, Serialize)]
pub struct NetReport {
    /// Net name.
    pub net: String,
    /// Total capacitance.
    pub total: Value,
    /// Ground capacitance after post-processing (`None` if dropped).
    pub ground: Option<Value>,
    /// Per-source ground breakdown (only with `by_source`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ground_by_source: Vec<Coupling>,
    /// Remaining couplings, largest first.
    pub couplings: Vec<Coupling>,
    /// Sum of couplings moved to ground (before the multiplier).
    pub moved_to_ground: f64,
    /// Samples used.
    pub walks: u64,
    /// Converged to the requested accuracy.
    pub converged: bool,
    /// Aborted walks (biases results if non-zero).
    pub aborted_walks: u64,
}

/// Node naming in SPEF output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpefNodes {
    /// Internal node `net:1` (default).
    #[default]
    Internal,
    /// The bare net name (matches top-level ports).
    Net,
}

/// A full report.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Per-target reports.
    pub nets: Vec<NetReport>,
    /// Warnings.
    pub warnings: Vec<String>,
}

fn est(sum: f64, sum_sq: f64, n: u64) -> Estimate {
    Moments::from_sums(sum, sum_sq).estimate(n)
}

fn combine(a: Estimate, b: Estimate, how: Symmetrize) -> Estimate {
    match how {
        Symmetrize::None => a,
        Symmetrize::Average => Estimate {
            value: 0.5 * (a.value + b.value),
            std_err: 0.5 * a.std_err.hypot(b.std_err),
        },
        Symmetrize::MaxSide => {
            if a.std_err <= b.std_err {
                a
            } else {
                b
            }
        }
        Symmetrize::InverseVariance => {
            let (wa, wb) = (1.0 / (a.std_err * a.std_err), 1.0 / (b.std_err * b.std_err));
            if !(wa.is_finite() && wb.is_finite()) || wa + wb == 0.0 {
                return combine(a, b, Symmetrize::Average);
            }
            Estimate {
                value: (wa * a.value + wb * b.value) / (wa + wb),
                std_err: (1.0 / (wa + wb)).sqrt(),
            }
        }
    }
}

/// Builds a report from raw results.
///
/// With symmetrization enabled, each coupling between two targets is one
/// value shared by both rows (also when only one direction was sampled), and
/// the ground capacitance is the residual `total − Σ couplings`, so that every
/// row satisfies `total = ground + Σ couplings` exactly. The ground standard
/// error is then taken from the sampled ground aggregate (an approximation).
/// With `symmetrize = "none"`, rows use their own estimates and the sampled
/// ground aggregate.
#[must_use]
pub fn build_report(r: &ResultFile, cc: &CouplingConfig, ground_report: GroundReport) -> Report {
    let ground: HashSet<&str> = r.ground_nets.iter().map(String::as_str).collect();
    let targets: HashSet<&str> = r.targets.iter().map(|t| t.net.as_str()).collect();
    let is_ground_sink = |s: &str| ground.contains(s) || is_reserved_label(s);
    let mut totals: HashMap<&str, Estimate> = HashMap::new();
    // raw couplings (i, j) -> positive estimate; partners per target in first-seen order
    let mut raw: HashMap<(&str, &str), Estimate> = HashMap::new();
    let mut partners: HashMap<&str, Vec<&str>> = HashMap::new();
    for t in &r.targets {
        for s in &t.sinks {
            let e = est(s.sum, s.sum_sq, t.walks);
            if s.sink == t.net {
                totals.insert(&t.net, e);
            } else if !is_ground_sink(&s.sink) {
                raw.insert((&t.net, &s.sink), e.scaled(-1.0));
                partners.entry(&t.net).or_default().push(&s.sink);
            }
        }
    }
    let symmetric = cc.symmetrize != Symmetrize::None;
    if symmetric {
        // Fill in pairs sampled from one side only.
        let pairs: Vec<(&str, &str)> = raw.keys().copied().collect();
        for (i, j) in pairs {
            if targets.contains(j) && !raw.contains_key(&(j, i)) {
                partners.entry(j).or_default().push(i);
            }
        }
    }
    let total_of = |n: &str| totals.get(n).map_or(0.0, |e| e.value);
    let pair_value = |i: &str, j: &str| -> Estimate {
        match (raw.get(&(i, j)), raw.get(&(j, i))) {
            (Some(&a), Some(&b)) if symmetric => {
                // Order-independent combination so both rows get the same value.
                if i < j {
                    combine(a, b, cc.symmetrize)
                } else {
                    combine(b, a, cc.symmetrize)
                }
            }
            (Some(&a), _) => a,
            (None, Some(&b)) => b,
            (None, None) => Estimate {
                value: 0.0,
                std_err: 0.0,
            },
        }
    };

    let mut nets = Vec::new();
    let mut negative_ground: Vec<String> = Vec::new();
    for t in &r.targets {
        let total = totals.get(t.net.as_str()).copied().unwrap_or(Estimate {
            value: 0.0,
            std_err: f64::INFINITY,
        });
        let g0 = est(t.ground.sum, t.ground.sum_sq, t.walks).scaled(-1.0);
        let mut moved = 0.0;
        let mut moved_var = 0.0;
        let mut kept_sum = 0.0;
        let mut couplings = Vec::new();
        for &j in partners.get(t.net.as_str()).map_or(&[][..], Vec::as_slice) {
            let c = pair_value(&t.net, j);
            let (ti, tj) = (
                total.value,
                if targets.contains(j) {
                    total_of(j)
                } else {
                    total.value
                },
            );
            let mut tests = Vec::new();
            if cc.abs_threshold_ff > 0.0 {
                tests.push(c.value < cc.abs_threshold_ff);
            }
            if cc.rel_threshold > 0.0 {
                let r = cc.rel_threshold;
                tests.push(match cc.rel_ref {
                    RelRef::Either => c.value < r * ti || c.value < r * tj,
                    RelRef::Both => c.value < r * ti && c.value < r * tj,
                    RelRef::Min => c.value < r * ti.min(tj),
                    RelRef::Max => c.value < r * ti.max(tj),
                });
            }
            let by_threshold = !tests.is_empty()
                && match cc.threshold_op {
                    ThresholdOp::And => tests.iter().all(|&b| b),
                    ThresholdOp::Or => tests.iter().any(|&b| b),
                };
            let insignificant =
                cc.significance_sigma > 0.0 && c.value < cc.significance_sigma * c.std_err;
            let non_target = cc.to_non_target == NonTarget::Ground && !targets.contains(j);
            if cc.mode == CouplingMode::Ground || by_threshold || insignificant || non_target {
                moved += c.value;
                moved_var += c.std_err * c.std_err;
            } else {
                kept_sum += c.value;
                couplings.push(Coupling {
                    net: j.to_owned(),
                    c: c.into(),
                });
            }
        }
        let m = cc.ground_multiplier;
        let mut ground_value = if symmetric {
            total.value - kept_sum - moved + m * moved
        } else {
            g0.value + m * moved
        };
        if ground_value < 0.0 {
            // Symmetrized couplings can exceed a noisy total; never report a
            // negative (non-passive) ground capacitance.
            negative_ground.push(t.net.clone());
            ground_value = 0.0;
        }
        let kept_var: f64 = couplings.iter().map(|c| c.c.sigma * c.c.sigma).sum();
        let ground_sigma = if symmetric {
            (total.std_err * total.std_err + kept_var + (1.0 - m).powi(2) * moved_var).sqrt()
        } else {
            (g0.std_err * g0.std_err + m * m * moved_var).sqrt()
        };
        couplings.sort_by(|a, b| {
            b.c.value
                .total_cmp(&a.c.value)
                .then_with(|| a.net.cmp(&b.net))
        });
        let ground_by_source = if ground_report == GroundReport::BySource {
            let mut v: BTreeMap<&str, Estimate> = BTreeMap::new();
            for s in &t.sinks {
                if is_ground_sink(&s.sink) {
                    v.insert(&s.sink, est(s.sum, s.sum_sq, t.walks).scaled(-1.0));
                }
            }
            v.into_iter()
                .map(|(k, e)| Coupling {
                    net: k.to_owned(),
                    c: e.into(),
                })
                .collect()
        } else {
            Vec::new()
        };
        nets.push(NetReport {
            net: t.net.clone(),
            total: total.into(),
            ground: (ground_report != GroundReport::Drop).then_some(Value {
                value: ground_value,
                sigma: ground_sigma,
            }),
            ground_by_source,
            couplings,
            moved_to_ground: moved,
            walks: t.walks,
            converged: t.converged,
            aborted_walks: t.aborted_walks,
        });
    }
    let mut warnings = r.warnings.clone();
    for t in &r.targets {
        let boundary: f64 = t
            .sinks
            .iter()
            .filter(|s| {
                matches!(
                    crate::result::parse_fixed_sink(&s.sink),
                    Some(faradice_core::Sink::Boundary(_))
                )
            })
            .map(|s| -est(s.sum, s.sum_sq, t.walks).value)
            .sum();
        let total = total_of(&t.net);
        if total > 0.0 && boundary > BOUNDARY_WARN_RATIO * total {
            warnings.push(format!(
                "net `{}`: {:.1} % of the total capacitance goes to the domain boundary; \
                 increase domain.margin if the boundary is not meant to be physical",
                t.net,
                100.0 * boundary / total
            ));
        }
    }
    for n in &negative_ground {
        warnings.push(format!(
            "net `{n}`: symmetrized couplings exceed the total capacitance (statistical noise); \
             ground clamped to 0 — tighten accuracy or use coupling.symmetrize = \"none\""
        ));
    }
    for n in &nets {
        if !n.converged {
            warnings.push(format!(
                "net `{}` did not reach the requested accuracy",
                n.net
            ));
        }
        if n.aborted_walks > 0 {
            warnings.push(format!(
                "net `{}`: {} walks hit walk.max_steps (biased)",
                n.net, n.aborted_walks
            ));
        }
    }
    Report { nets, warnings }
}

impl Report {
    /// Long-format CSV: `net,kind,other,value_ff,sigma_ff`.
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut s = String::from("net,kind,other,value_ff,sigma_ff\n");
        for n in &self.nets {
            let _ = writeln!(
                s,
                "{},total,,{:.9e},{:.3e}",
                n.net, n.total.value, n.total.sigma
            );
            if let Some(g) = n.ground {
                let _ = writeln!(s, "{},ground,,{:.9e},{:.3e}", n.net, g.value, g.sigma);
            }
            for g in &n.ground_by_source {
                let _ = writeln!(
                    s,
                    "{},ground_source,{},{:.9e},{:.3e}",
                    n.net, g.net, g.c.value, g.c.sigma
                );
            }
            for c in &n.couplings {
                let _ = writeln!(
                    s,
                    "{},coupling,{},{:.9e},{:.3e}",
                    n.net, c.net, c.c.value, c.c.sigma
                );
            }
        }
        s
    }

    /// Capacitance-only SPEF (IEEE 1481) with one lumped node per net, using
    /// internal nodes `net:1`. See [`Report::to_spef_with`].
    #[must_use]
    pub fn to_spef(&self, design: &str) -> String {
        self.to_spef_with(design, SpefNodes::Internal)
    }

    /// Capacitance-only SPEF (IEEE 1481) with one lumped node per net.
    ///
    /// Each coupling is written in the sections of both nets (as readers such as
    /// OpenSTA attach a coupling capacitor only to the net being read), and each
    /// `*D_NET` total is the sum of the capacitors emitted in its section.
    ///
    /// **Limitation (M1):** no `*CONN` section is written, because GDS/box inputs
    /// carry no pin information. The file is suitable for inspection and
    /// capacitance back-annotation by net, but static timing analysis needs pin
    /// connectivity and distributed RC (not implemented yet).
    #[must_use]
    pub fn to_spef_with(&self, design: &str, nodes: SpefNodes) -> String {
        let node = |net: &str| match nodes {
            SpefNodes::Internal => format!("{net}:1"),
            SpefNodes::Net => net.to_owned(),
        };
        let mut s = String::new();
        let _ = writeln!(s, "*SPEF \"IEEE 1481-1998\"");
        let _ = writeln!(s, "*DESIGN \"{design}\"");
        let _ = writeln!(s, "*DATE \"\"");
        let _ = writeln!(s, "*VENDOR \"faradice\"");
        let _ = writeln!(s, "*PROGRAM \"faradice\"");
        let _ = writeln!(s, "*VERSION \"{}\"", env!("CARGO_PKG_VERSION"));
        let _ = writeln!(s, "*DESIGN_FLOW \"PIN_CAP NONE\"");
        let _ = writeln!(s, "*DIVIDER /\n*DELIMITER :\n*BUS_DELIMITER [ ]");
        let _ = writeln!(
            s,
            "*T_UNIT 1 NS\n*C_UNIT 1 FF\n*R_UNIT 1 OHM\n*L_UNIT 1 HENRY\n"
        );
        for n in &self.nets {
            let mut caps = String::new();
            let mut sum = 0.0;
            let mut k = 1;
            if let Some(g) = n.ground {
                let _ = writeln!(caps, "{k} {} {:.6}", node(&n.net), g.value);
                sum += g.value;
                k += 1;
            }
            for c in &n.couplings {
                let _ = writeln!(
                    caps,
                    "{k} {} {} {:.6}",
                    node(&n.net),
                    node(&c.net),
                    c.c.value
                );
                sum += c.c.value;
                k += 1;
            }
            let _ = writeln!(s, "*D_NET {} {sum:.6}\n*CAP\n{caps}*END\n", n.net);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{Completeness, NetRecord, SinkSums};

    fn rec(net: &str, sinks: &[(&str, f64)], ground: f64) -> NetRecord {
        // One "walk" per unit: store sums so that estimate(n=1) returns the value.
        NetRecord {
            net: net.into(),
            walks: 1,
            aborted_walks: 0,
            total_steps: 0,
            converged: true,
            gauss_offset: 0.0,
            sinks: sinks
                .iter()
                .map(|(s, v)| SinkSums {
                    sink: (*s).into(),
                    sum: *v,
                    sum_sq: v * v,
                })
                .collect(),
            ground: SinkSums {
                sink: "<GND>".into(),
                sum: ground,
                sum_sq: ground * ground,
            },
            cross: Vec::new(),
        }
    }

    fn file() -> ResultFile {
        ResultFile {
            schema_version: 2,
            faradice_version: String::new(),
            seed: 1,
            eps_stop: 0.0,
            elapsed_s: 0.0,
            nets_all: vec!["A".into(), "B".into(), "C".into()],
            ground_nets: vec![],
            targets: vec![
                rec(
                    "A",
                    &[("A", 10.0), ("B", -3.0), ("C", -0.2), ("<substrate>", -6.8)],
                    -6.8,
                ),
                rec("B", &[("B", 8.0), ("A", -3.0), ("<substrate>", -5.0)], -5.0),
            ],
            warnings: vec![],
            config_toml: String::new(),
            completeness: Completeness::Complete,
        }
    }

    #[test]
    fn thresholds_preserve_total() {
        let cc = CouplingConfig {
            rel_threshold: 0.05,
            ..CouplingConfig::default()
        };
        let r = build_report(&file(), &cc, GroundReport::Merged);
        let a = &r.nets[0];
        assert_eq!(a.couplings.len(), 1); // C (0.2 < 5% of 10) moved
        let g = a.ground.unwrap().value;
        let sum = g + a.couplings.iter().map(|c| c.c.value).sum::<f64>();
        assert!((sum - a.total.value).abs() < 1e-12);
    }

    #[test]
    fn symmetrization_keeps_rows_consistent() {
        // A sees B at 2, B sees A at 4 (equal weights): both rows report 3 and
        // totals stay equal to ground + couplings.
        let mut f = file();
        f.targets[0] = rec(
            "A",
            &[("A", 10.0), ("B", -2.0), ("<substrate>", -8.0)],
            -8.0,
        );
        f.targets[1] = rec("B", &[("B", 8.0), ("A", -4.0), ("<substrate>", -4.0)], -4.0);
        let cc = CouplingConfig {
            symmetrize: Symmetrize::Average,
            ..CouplingConfig::default()
        };
        let r = build_report(&f, &cc, GroundReport::Merged);
        for n in &r.nets {
            assert!((n.couplings[0].c.value - 3.0).abs() < 1e-12);
            let sum = n.ground.unwrap().value + n.couplings[0].c.value;
            assert!((sum - n.total.value).abs() < 1e-12);
        }
        // One-sided sampling: B never hit A, but both rows still report the pair.
        f.targets[1] = rec("B", &[("B", 8.0), ("<substrate>", -8.0)], -8.0);
        let r = build_report(&f, &cc, GroundReport::Merged);
        assert_eq!(r.nets[1].couplings.len(), 1);
        assert_eq!(r.nets[1].couplings[0].net, "A");
    }

    #[test]
    fn couple_to_ground_and_drop() {
        let cc = CouplingConfig {
            mode: CouplingMode::Ground,
            ..CouplingConfig::default()
        };
        let r = build_report(&file(), &cc, GroundReport::Merged);
        assert!(r.nets.iter().all(|n| n.couplings.is_empty()));
        assert!((r.nets[0].ground.unwrap().value - 10.0).abs() < 1e-12);
        let d = build_report(&file(), &CouplingConfig::default(), GroundReport::Drop);
        assert!(d.nets.iter().all(|n| n.ground.is_none()));
        // With ground dropped, the D_NET total is the sum of the emitted couplings.
        assert!(d.to_spef("t").contains("*D_NET A 3.200000"));
        assert!(d.to_spef("t").contains("A:1 B:1 3.000000"));
        assert!(d.to_spef("t").contains("B:1 A:1 3.000000"));
    }
}
