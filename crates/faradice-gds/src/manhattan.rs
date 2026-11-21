//! Manhattan polygon and path decomposition into rectangles.
//!
//! All coordinates are integers (half database units), so the decomposition
//! is exact. Rectangles are `[x0, y0, x1, y1]` with `x0 < x1`, `y0 < y1`.

use crate::flatten::ShapeKind;
use crate::{GdsError, Result};

/// An integer rectangle `[x0, y0, x1, y1]`.
pub type Rect = [i64; 4];

/// Decomposes a flattened shape into rectangles.
pub fn shape_rects(kind: &ShapeKind) -> Result<Vec<Rect>> {
    match kind {
        ShapeKind::Polygon(pts) => polygon_rects(pts),
        ShapeKind::Path {
            pathtype,
            half_width,
            bgn_ext,
            end_ext,
            pts,
        } => path_rects(*pathtype, *half_width, *bgn_ext, *end_ext, pts),
    }
}

/// Removes the closing point, consecutive duplicates and collinear vertices.
fn clean(pts: &[[i64; 2]]) -> Vec<[i64; 2]> {
    let mut v: Vec<[i64; 2]> = Vec::with_capacity(pts.len());
    for &p in pts {
        if v.last() != Some(&p) {
            v.push(p);
        }
    }
    if v.len() > 1 && v.first() == v.last() {
        v.pop();
    }
    // Drop collinear points (repeat until stable).
    loop {
        let n = v.len();
        if n < 3 {
            return v;
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b, c) = (v[(i + n - 1) % n], v[i], v[(i + 1) % n]);
            let collinear = (a[0] == b[0] && b[0] == c[0]) || (a[1] == b[1] && b[1] == c[1]);
            if !collinear {
                out.push(b);
            }
        }
        if out.len() == n {
            return out;
        }
        v = out;
    }
}

/// Decomposes a simple or self-touching Manhattan polygon (even-odd rule) into
/// rectangles by vertical slabs, merging consecutive slabs with equal intervals.
pub fn polygon_rects(pts: &[[i64; 2]]) -> Result<Vec<Rect>> {
    // Validate before cleaning so that non-Manhattan shapes (e.g. triangles,
    // which would clean down to three vertices) are rejected, not dropped.
    let n = pts.len();
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        if a[0] != b[0] && a[1] != b[1] {
            return Err(GdsError::Unsupported(format!(
                "non-Manhattan edge ({}, {}) → ({}, {}) (half-dbu)",
                a[0], a[1], b[0], b[1]
            )));
        }
    }
    let v = clean(pts);
    if v.len() < 4 {
        return Ok(Vec::new()); // degenerate (zero area)
    }
    let n = v.len();
    let mut horiz: Vec<(i64, i64, i64)> = Vec::new(); // (xmin, xmax, y)
    for i in 0..n {
        let (a, b) = (v[i], v[(i + 1) % n]);
        if a[1] == b[1] {
            horiz.push((a[0].min(b[0]), a[0].max(b[0]), a[1]));
        } else if a[0] != b[0] {
            return Err(GdsError::Unsupported(format!(
                "non-Manhattan edge ({}, {}) → ({}, {}) (half-dbu)",
                a[0], a[1], b[0], b[1]
            )));
        }
    }
    let mut xs: Vec<i64> = v.iter().map(|p| p[0]).collect();
    xs.sort_unstable();
    xs.dedup();
    let mut out: Vec<Rect> = Vec::new();
    let mut prev: Option<(i64, Vec<(i64, i64)>)> = None; // (x start, intervals)
    for w in xs.windows(2) {
        let (xa, xb) = (w[0], w[1]);
        let mut ys: Vec<i64> = horiz
            .iter()
            .filter(|h| h.0 <= xa && h.1 >= xb)
            .map(|h| h.2)
            .collect();
        ys.sort_unstable();
        // Even-odd pairing; equal coordinates cancel (touching edges).
        let mut intervals: Vec<(i64, i64)> = Vec::new();
        for pair in ys.chunks(2) {
            if let [y0, y1] = *pair {
                if y0 < y1 {
                    if let Some(last) = intervals.last_mut() {
                        if last.1 == y0 {
                            last.1 = y1;
                            continue;
                        }
                    }
                    intervals.push((y0, y1));
                }
            }
        }
        match prev.take() {
            Some((x0, iv)) if iv == intervals => prev = Some((x0, iv)),
            Some((x0, iv)) => {
                out.extend(iv.iter().map(|&(y0, y1)| [x0, y0, xa, y1]));
                prev = Some((xa, intervals));
            }
            None => prev = Some((xa, intervals)),
        }
    }
    if let (Some((x0, iv)), Some(&xl)) = (prev, xs.last()) {
        out.extend(iv.iter().map(|&(y0, y1)| [x0, y0, xl, y1]));
    }
    Ok(out)
}

/// Decomposes a Manhattan path into one rectangle per segment.
///
/// Interior joints are extended by the half width (square joins), which covers
/// the corners exactly. Path ends follow the path type: 0 flush, 2 half-width
/// extension, 4 custom extensions; type 1 (round) is approximated by type 2.
pub fn path_rects(
    pathtype: i16,
    hw: i64,
    bgn_ext: i64,
    end_ext: i64,
    pts: &[[i64; 2]],
) -> Result<Vec<Rect>> {
    let mut p: Vec<[i64; 2]> = Vec::with_capacity(pts.len());
    for &q in pts {
        if p.last() != Some(&q) {
            p.push(q);
        }
    }
    if p.len() < 2 || hw == 0 {
        return Ok(Vec::new());
    }
    let (e_begin, e_end) = match pathtype {
        0 => (0, 0),
        1 | 2 => (hw, hw),
        4 => (bgn_ext, end_ext),
        t => return Err(GdsError::Unsupported(format!("path type {t}"))),
    };
    let last = p.len() - 2;
    let mut out = Vec::with_capacity(p.len() - 1);
    for (i, w) in p.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        let ea = if i == 0 { e_begin } else { hw };
        let eb = if i == last { e_end } else { hw };
        if a[1] == b[1] {
            let (s, e) = if a[0] < b[0] {
                (a[0] - ea, b[0] + eb)
            } else {
                (b[0] - eb, a[0] + ea)
            };
            out.push([s, a[1] - hw, e, a[1] + hw]);
        } else if a[0] == b[0] {
            let (s, e) = if a[1] < b[1] {
                (a[1] - ea, b[1] + eb)
            } else {
                (b[1] - eb, a[1] + ea)
            };
            out.push([a[0] - hw, s, a[0] + hw, e]);
        } else {
            return Err(GdsError::Unsupported("non-Manhattan path segment".into()));
        }
    }
    Ok(out
        .into_iter()
        .filter(|r| r[0] < r[2] && r[1] < r[3])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(rs: &[Rect]) -> i64 {
        rs.iter().map(|r| (r[2] - r[0]) * (r[3] - r[1])).sum()
    }

    #[test]
    fn rectangle_and_l_shape() {
        let r = polygon_rects(&[[0, 0], [10, 0], [10, 4], [0, 4], [0, 0]]).unwrap();
        assert_eq!(r, vec![[0, 0, 10, 4]]);
        // L: 10x2 bottom + 2x8 left column
        let l = polygon_rects(&[[0, 0], [10, 0], [10, 2], [2, 2], [2, 10], [0, 10]]).unwrap();
        assert_eq!(area(&l), 10 * 2 + 2 * 8);
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn keyhole_polygon_with_hole() {
        // 10x10 square with a 4x4 hole connected by a zero-width cut at x=3.
        let pts = [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 3],
            [3, 3],
            [3, 7],
            [7, 7],
            [7, 3],
            [3, 3],
            [0, 3],
        ];
        let r = polygon_rects(&pts).unwrap();
        assert_eq!(area(&r), 100 - 16);
    }

    #[test]
    fn non_manhattan_is_rejected() {
        assert!(polygon_rects(&[[0, 0], [10, 0], [5, 5], [0, 5]]).is_err());
        assert!(polygon_rects(&[[0, 0], [10, 0], [0, 10], [0, 0]]).is_err()); // triangle
    }

    #[test]
    fn paths() {
        // L-shaped path, half width 1, flush ends.
        let r = path_rects(0, 1, 0, 0, &[[0, 0], [10, 0], [10, 10]]).unwrap();
        assert_eq!(r, vec![[0, -1, 11, 1], [9, -1, 11, 10]]);
        let r2 = path_rects(2, 1, 0, 0, &[[0, 0], [10, 0]]).unwrap();
        assert_eq!(r2, vec![[-1, -1, 11, 1]]);
        let r4 = path_rects(4, 1, 3, 5, &[[0, 0], [0, 10]]).unwrap();
        assert_eq!(r4, vec![[-1, -3, 1, 15]]);
    }
}
