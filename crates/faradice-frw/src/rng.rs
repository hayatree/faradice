//! Reproducible random number streams.
//!
//! Each `(seed, net, batch)` triple maps to an independent xoshiro256++ stream
//! seeded through SplitMix64, so results do not depend on thread scheduling.
//! Both algorithms are public domain (Blackman & Vigna); they are implemented
//! here directly to keep the solver dependency-free and to allow an identical
//! implementation on the GPU.

/// SplitMix64 step, used for seeding and hashing stream keys.
#[must_use]
#[inline]
pub const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ generator.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Creates the stream for `(seed, net, batch)`.
    #[must_use]
    pub fn for_stream(seed: u64, net: u32, batch: u64) -> Self {
        let mut h = seed;
        let a = splitmix64(&mut h);
        let mut h = a ^ u64::from(net).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let b = splitmix64(&mut h);
        let mut h = b ^ batch.wrapping_mul(0xA076_1D64_78BD_642F);
        Self::from_seed(splitmix64(&mut h))
    }

    /// Creates a generator from a single 64-bit seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut st = seed;
        let s = [
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
        ];
        Self { s }
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)` with 53 random bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Exact: a 53-bit integer converts to f64 without rounding.
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 11) as f64;
        v * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `0..n` (Lemire's multiply-shift, bias < 2⁻³²·n; `n > 0`).
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let x = self.next_u64() >> 32;
        // The product of two u32-range values shifted by 32 fits in u32.
        #[allow(clippy::cast_possible_truncation)]
        let r = ((x * u64::from(n)) >> 32) as u32;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_reproducible_and_distinct() {
        let mut a = Rng::for_stream(1, 2, 3);
        let mut b = Rng::for_stream(1, 2, 3);
        let mut c = Rng::for_stream(1, 2, 4);
        let va: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let vb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let vc: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(va, vb);
        assert_ne!(va, vc);
    }

    #[test]
    fn uniform_moments() {
        let mut r = Rng::from_seed(42);
        let n = 200_000;
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let u = r.next_f64();
            assert!((0.0..1.0).contains(&u));
            s += u;
            s2 += u * u;
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        assert!((mean - 0.5).abs() < 0.005);
        assert!((var - 1.0 / 12.0).abs() < 0.002);
        let mut counts = [0u32; 6];
        for _ in 0..60_000 {
            counts[r.below(6) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| (9_500..10_500).contains(&c)));
    }
}
