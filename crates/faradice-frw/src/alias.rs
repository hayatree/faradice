//! Alias tables (Walker's method with Vose's stable construction).

use crate::rng::Rng;

/// O(1) sampler for a discrete distribution over `0..n`.
#[derive(Debug, Clone)]
pub struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl AliasTable {
    /// Builds a table from non-negative weights (they need not be normalized).
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty, longer than `u32::MAX`, contains a negative or
    /// non-finite value, or sums to zero.
    #[must_use]
    pub fn new(weights: &[f64]) -> Self {
        let n = weights.len();
        assert!(
            n > 0 && u32::try_from(n).is_ok(),
            "alias table size out of range"
        );
        assert!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "invalid weight"
        );
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "weights sum to zero");
        #[allow(clippy::cast_precision_loss)]
        let scale = n as f64 / total;
        let mut scaled: Vec<f64> = weights.iter().map(|w| w * scale).collect();
        let mut prob = vec![1.0; n];
        let mut alias: Vec<u32> = (0..n).map(|i| i as u32).collect();
        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..n).partition(|&i| scaled[i] < 1.0);
        while let (Some(s), Some(&l)) = (small.pop(), large.last()) {
            prob[s] = scaled[s];
            #[allow(clippy::cast_possible_truncation)]
            {
                alias[s] = l as u32;
            }
            scaled[l] = (scaled[l] + scaled[s]) - 1.0;
            if scaled[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        // Remaining entries have probability 1 up to rounding.
        Self { prob, alias }
    }

    /// Number of outcomes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.prob.len()
    }

    /// Always `false` (tables are never empty).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Draws an outcome.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let i = rng.below(self.prob.len() as u32) as usize;
        if rng.next_f64() < self.prob[i] {
            i
        } else {
            self.alias[i] as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequencies_match_weights() {
        let w = [1.0, 0.0, 3.0, 6.0];
        let t = AliasTable::new(&w);
        let mut rng = Rng::from_seed(7);
        let mut counts = [0u32; 4];
        let n = 400_000;
        for _ in 0..n {
            counts[t.sample(&mut rng)] += 1;
        }
        assert_eq!(counts[1], 0);
        for (c, wi) in counts.iter().zip(w) {
            let expected = wi / 10.0;
            assert!(
                (f64::from(*c) / f64::from(n) - expected).abs() < 0.004,
                "{counts:?}"
            );
        }
    }
}
