//! Dense matrix products for the fast-diagonalization preconditioners of
//! [`crate::edge`] and [`crate::section`].

/// Output tile: `ROWS × COLS` sums stay in registers while the inner index
/// runs.
const ROWS: usize = 4;
const COLS: usize = 4;

/// `c[(o s + t) n + j] = Σ_r a[o k + r] b[r n + j]` for row-major `a`
/// (`m × k`) and `b` (`k × n`), with `[m, k, n] = dims` and `[s, t] = rows`:
/// row `o` of the product goes to row `o s + t` of `c` (each `n` long).
/// Other rows of `c` are left alone.
///
/// Every sum starts at `0.0` and adds its terms one by one in order of `r`,
/// as the plain triple loop would, so the result does not depend on the
/// tiling.
pub(crate) fn product(a: &[f64], b: &[f64], c: &mut [f64], dims: [usize; 3], rows: [usize; 2]) {
    let [m, k, n] = dims;
    let [stride, first] = rows;
    let mut o = 0;
    while o < m {
        let tr = ROWS.min(m - o);
        let mut j = 0;
        while j < n {
            let tc = COLS.min(n - j);
            if tr == ROWS && tc == COLS {
                let mut acc = [[0.0_f64; COLS]; ROWS];
                for r in 0..k {
                    let br: &[f64; COLS] = b[r * n + j..r * n + j + COLS]
                        .try_into()
                        .expect("tile width");
                    for (i, row) in acc.iter_mut().enumerate() {
                        let s = a[(o + i) * k + r];
                        // Plain multiply-add: `mul_add` is a library call
                        // without the FMA target feature and blocks
                        // vectorization.
                        for (t, v) in row.iter_mut().zip(br) {
                            *t += s * *v;
                        }
                    }
                }
                for (i, row) in acc.iter().enumerate() {
                    let at = ((o + i) * stride + first) * n + j;
                    c[at..at + COLS].copy_from_slice(row);
                }
            } else {
                for i in o..o + tr {
                    for jj in j..j + tc {
                        let mut t = 0.0;
                        for r in 0..k {
                            t += a[i * k + r] * b[r * n + jj];
                        }
                        c[(i * stride + first) * n + jj] = t;
                    }
                }
            }
            j += tc;
        }
        o += tr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_plain_loop_bit_for_bit() {
        let mut x = 0.3_f64;
        let mut next = || {
            x = (x * 3.7 + 0.11).fract();
            x - 0.5
        };
        for (m, k, n) in [(1, 1, 1), (4, 4, 4), (5, 7, 9), (9, 3, 13), (16, 16, 6)] {
            let a: Vec<f64> = (0..m * k).map(|_| next()).collect();
            let b: Vec<f64> = (0..k * n).map(|_| next()).collect();
            for (stride, first) in [(1, 0), (2, 0), (2, 1)] {
                let mut c = vec![f64::NAN; (m * stride + first) * n];
                product(&a, &b, &mut c, [m, k, n], [stride, first]);
                for i in 0..m {
                    for j in 0..n {
                        let mut t = 0.0;
                        for r in 0..k {
                            t += a[i * k + r] * b[r * n + j];
                        }
                        let got = c[(i * stride + first) * n + j];
                        assert_eq!(got.to_bits(), t.to_bits(), "{m}x{k}x{n} at ({i}, {j})");
                    }
                }
            }
        }
    }
}
