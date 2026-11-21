//! The capacitance matrix of a report: one row per target net, one column per
//! net (targets first, in row order, then the other nets by their largest
//! coupling) and a ground column. The diagonal holds the total capacitance,
//! the other cells the coupling capacitances, all positive (fF).

use std::collections::HashMap;

use faradice_io::report::{NetReport, Report};

/// Most net columns shown; the remaining nets are counted in
/// [`Columns::hidden`].
pub(crate) const MAX_COLUMNS: usize = 64;

/// Column order of a report, computed once per report (a row's cells are
/// then looked up on demand, so only the visible rows cost anything).
#[derive(Debug, Clone, Default)]
pub(crate) struct Columns {
    /// Net names of the net columns (the ground column comes last).
    pub(crate) nets: Vec<String>,
    /// Nets left out beyond [`MAX_COLUMNS`].
    pub(crate) hidden: usize,
    index: HashMap<String, usize>,
}

/// One cell of the matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Cell {
    /// Capacitance (fF).
    pub(crate) value: f64,
    /// Standard error (fF, 1σ).
    pub(crate) sigma: f64,
    /// `value / C_total` of the row.
    pub(crate) share: f64,
    /// `|value|` exceeds `confidence × σ`.
    pub(crate) significant: bool,
    /// The row's own total.
    pub(crate) diagonal: bool,
}

impl Columns {
    /// The columns of `report`, at most `max` net columns.
    pub(crate) fn new(report: &Report, max: usize) -> Self {
        let mut nets: Vec<String> = report.nets.iter().map(|n| n.net.clone()).collect();
        let mut seen: HashMap<&str, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        // Other nets, by their largest coupling to any target.
        let mut others: Vec<(&str, f64)> = Vec::new();
        for c in report.nets.iter().flat_map(|n| &n.couplings) {
            let v = c.c.value.abs();
            match seen.get(c.net.as_str()) {
                Some(&i) if i >= nets.len() => {
                    let o = &mut others[i - nets.len()].1;
                    *o = o.max(v);
                }
                Some(_) => {}
                None => {
                    seen.insert(&c.net, nets.len() + others.len());
                    others.push((&c.net, v));
                }
            }
        }
        others.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        nets.extend(others.into_iter().map(|(n, _)| n.to_owned()));
        let hidden = nets.len().saturating_sub(max);
        nets.truncate(max);
        let index = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        Self {
            nets,
            hidden,
            index,
        }
    }

    /// Number of columns including ground.
    pub(crate) fn len(&self) -> usize {
        self.nets.len() + 1
    }

    /// The cells of `row`: one per column, `None` where nothing was reported.
    pub(crate) fn cells(&self, row: &NetReport, confidence: f64) -> Vec<Option<Cell>> {
        let total = row.total.value.abs().max(f64::MIN_POSITIVE);
        let cell = |value: f64, sigma: f64, diagonal: bool| Cell {
            value,
            sigma,
            share: value / total,
            significant: value.abs() > confidence * sigma,
            diagonal,
        };
        let mut out = vec![None; self.len()];
        if let Some(&i) = self.index.get(&row.net) {
            out[i] = Some(cell(row.total.value, row.total.sigma, true));
        }
        for c in &row.couplings {
            if let Some(&i) = self.index.get(&c.net) {
                out[i] = Some(cell(c.c.value, c.c.sigma, false));
            }
        }
        if let Some(g) = row.ground {
            out[self.nets.len()] = Some(cell(g.value, g.sigma, false));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use faradice_io::report::{Coupling, Value};

    use super::*;

    fn v(value: f64, sigma: f64) -> Value {
        Value { value, sigma }
    }

    fn net(name: &str, total: f64, ground: Option<f64>, couplings: &[(&str, f64)]) -> NetReport {
        NetReport {
            net: name.into(),
            total: v(total, 0.01 * total),
            ground: ground.map(|g| v(g, 0.01)),
            ground_by_source: Vec::new(),
            couplings: couplings
                .iter()
                .map(|&(n, c)| Coupling {
                    net: n.into(),
                    c: v(c, 0.1),
                })
                .collect(),
            moved_to_ground: 0.0,
            walks: 1000,
            converged: true,
            aborted_walks: 0,
        }
    }

    fn report() -> Report {
        Report {
            nets: vec![
                net("a", 10.0, Some(5.0), &[("b", 3.0), ("x", 1.5), ("y", 0.2)]),
                net("b", 8.0, None, &[("a", 3.0), ("y", 4.0)]),
            ],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn targets_come_first_then_others_by_largest_coupling() {
        let c = Columns::new(&report(), MAX_COLUMNS);
        assert_eq!(c.nets, ["a", "b", "y", "x"]);
        assert_eq!((c.len(), c.hidden), (5, 0));
        let c = Columns::new(&report(), 3);
        assert_eq!(c.nets, ["a", "b", "y"]);
        assert_eq!(c.hidden, 1);
    }

    #[test]
    fn cells_hold_total_couplings_and_ground() {
        let r = report();
        let c = Columns::new(&r, MAX_COLUMNS);
        let row = c.cells(&r.nets[0], 2.0);
        let diag = row[0].unwrap();
        assert!(diag.diagonal && diag.share == 1.0 && diag.significant);
        assert_eq!(row[1].unwrap().value, 3.0);
        assert!((row[1].unwrap().share - 0.3).abs() < 1e-12);
        // 0.2 fF with σ = 0.1 fF is not significant at 2σ.
        assert!(!row[2].unwrap().significant);
        assert_eq!(row[3].unwrap().value, 1.5);
        assert_eq!(row[4].unwrap().value, 5.0);
        let row = c.cells(&r.nets[1], 2.0);
        assert!(row[3].is_none() && row[4].is_none());
    }
}
