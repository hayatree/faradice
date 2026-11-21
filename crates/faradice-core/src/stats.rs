//! Deterministic, mergeable accumulators for weighted Monte Carlo samples.
//!
//! A floating random walk estimates `C_ik` as the mean of per-walk weights
//! `w · 1[walk ends at k]` over all `N` walks launched from net `i`. Each sink
//! therefore only needs the running sums `Σw` and `Σw²`; the walk count `N`
//! is shared by all sinks of a target net and is passed in when estimating.
//!
//! Sums use Neumaier (improved Kahan) compensation so that 10⁸+ samples do not
//! lose precision. Results are bit-for-bit deterministic as long as samples and
//! merges are applied in the same order; the solver guarantees this by merging
//! batches in batch-id order.

/// A compensated floating-point sum (Neumaier's variant of Kahan summation).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NeumaierSum {
    sum: f64,
    compensation: f64,
}

impl NeumaierSum {
    /// An empty sum.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Adds one value.
    pub fn add(&mut self, value: f64) {
        let t = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        self.sum = t;
    }

    /// Adds another compensated sum.
    pub fn merge(&mut self, other: &Self) {
        self.add(other.sum);
        self.add(other.compensation);
    }

    /// The compensated total.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// First and second raw moments (`Σw`, `Σw²`) of the weights hitting one sink.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Moments {
    sum: NeumaierSum,
    sum_sq: NeumaierSum,
}

impl Moments {
    /// Empty moments.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sum: NeumaierSum::new(),
            sum_sq: NeumaierSum::new(),
        }
    }

    /// Reconstructs moments from stored sums (e.g. a result file). The
    /// compensation terms are folded into the sums.
    #[must_use]
    pub fn from_sums(sum: f64, sum_sq: f64) -> Self {
        let mut m = Self::new();
        m.sum.add(sum);
        m.sum_sq.add(sum_sq);
        m
    }

    /// Records one sample weight.
    pub fn add(&mut self, weight: f64) {
        self.sum.add(weight);
        self.sum_sq.add(weight * weight);
    }

    /// Merges moments accumulated elsewhere (e.g. another batch).
    pub fn merge(&mut self, other: &Self) {
        self.sum.merge(&other.sum);
        self.sum_sq.merge(&other.sum_sq);
    }

    /// `Σw`.
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.sum.value()
    }

    /// `Σw²`.
    #[must_use]
    pub fn sum_sq(&self) -> f64 {
        self.sum_sq.value()
    }

    /// Estimates the mean over `n_walks` samples, where walks that did not hit
    /// this sink count as zero-weight samples.
    ///
    /// The standard error uses the unbiased sample variance. With fewer than
    /// two walks the standard error is `+∞`.
    #[must_use]
    pub fn estimate(&self, n_walks: u64) -> Estimate {
        if n_walks == 0 {
            return Estimate {
                value: 0.0,
                std_err: f64::INFINITY,
            };
        }
        // Walk counts stay far below 2^53, so the conversion is exact in practice.
        #[allow(clippy::cast_precision_loss)]
        let n = n_walks as f64;
        let mean = self.sum() / n;
        if n_walks < 2 {
            return Estimate {
                value: mean,
                std_err: f64::INFINITY,
            };
        }
        let var = ((self.sum_sq() - n * mean * mean) / (n - 1.0)).max(0.0);
        Estimate {
            value: mean,
            std_err: (var / n).sqrt(),
        }
    }
}

/// A Monte Carlo estimate with its standard error (1σ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// The estimated mean.
    pub value: f64,
    /// The standard error of the mean (one standard deviation).
    pub std_err: f64,
}

impl Estimate {
    /// Relative standard error `σ / |value|` (`+∞` if the value is zero).
    #[must_use]
    pub fn rel_err(&self) -> f64 {
        if self.value == 0.0 {
            f64::INFINITY
        } else {
            self.std_err / self.value.abs()
        }
    }

    /// `true` if `sigma · std_err ≤ rel_tol · |value|`.
    #[must_use]
    pub fn meets(&self, rel_tol: f64, sigma: f64) -> bool {
        sigma * self.std_err <= rel_tol * self.value.abs()
    }

    /// Multiplies value and error by `factor` (e.g. to convert units).
    #[must_use]
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            value: self.value * factor,
            std_err: self.std_err * factor.abs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_recovers_lost_bits() {
        let mut s = NeumaierSum::new();
        for v in [1.0, 1e100, 1.0, -1e100] {
            s.add(v);
        }
        assert_eq!(s.value(), 2.0);
    }

    #[test]
    fn estimate_matches_closed_form() {
        // Samples 1, 2, 3, 4 hit the sink; 4 more walks hit elsewhere (weight 0).
        let mut m = Moments::new();
        for w in [1.0, 2.0, 3.0, 4.0] {
            m.add(w);
        }
        let e = m.estimate(8);
        let samples = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let mean = samples.iter().sum::<f64>() / 8.0;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 7.0;
        assert!((e.value - mean).abs() < 1e-15);
        assert!((e.std_err - (var / 8.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn merge_equals_sequential() {
        let weights: Vec<f64> = (0..1000).map(|i| f64::from(i).sin() * 3.0).collect();
        let mut all = Moments::new();
        weights.iter().for_each(|&w| all.add(w));

        let mut a = Moments::new();
        let mut b = Moments::new();
        weights[..400].iter().for_each(|&w| a.add(w));
        weights[400..].iter().for_each(|&w| b.add(w));
        a.merge(&b);

        let (x, y) = (all.estimate(1000), a.estimate(1000));
        assert!((x.value - y.value).abs() < 1e-14);
        assert!((x.std_err - y.std_err).abs() < 1e-14);
    }

    #[test]
    fn degenerate_counts() {
        let m = Moments::new();
        assert!(m.estimate(0).std_err.is_infinite());
        let mut one = Moments::new();
        one.add(2.0);
        let e = one.estimate(1);
        assert_eq!(e.value, 2.0);
        assert!(e.std_err.is_infinite());
        assert!(!e.meets(0.5, 1.0));
    }

    #[test]
    fn meets_and_rel_err() {
        let e = Estimate {
            value: -10.0,
            std_err: 0.1,
        };
        assert!((e.rel_err() - 0.01).abs() < 1e-15);
        assert!(e.meets(0.031, 3.0));
        assert!(!e.meets(0.029, 3.0));
        assert_eq!(
            e.scaled(2.0),
            Estimate {
                value: -20.0,
                std_err: 0.2
            }
        );
        assert!(
            Estimate {
                value: 0.0,
                std_err: 1.0
            }
            .rel_err()
            .is_infinite()
        );
    }
}
