//! Hierarchy flattening with exact integer arithmetic.
//!
//! Coordinates are stored in **half database units** (`2 × dbu`, `i64`) so
//! that half path widths stay exact. Supported transformations: mirroring
//! about x, rotations by multiples of 90°, magnification 1. Anything else is
//! rejected with [`GdsError::Unsupported`] rather than approximated.

use crate::stream::{Element, Library, Strans};
use crate::{GdsError, Result};

/// A flattened shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeKind {
    /// Polygon vertices (half-dbu), not necessarily closed.
    Polygon(Vec<[i64; 2]>),
    /// Path centre line (half-dbu) with half width and end extensions (half-dbu).
    Path {
        /// Path type (0, 1, 2, 4).
        pathtype: i16,
        /// Half width (half-dbu) = width in dbu.
        half_width: i64,
        /// Begin extension (half-dbu).
        bgn_ext: i64,
        /// End extension (half-dbu).
        end_ext: i64,
        /// Points.
        pts: Vec<[i64; 2]>,
    },
}

/// A flattened shape with its layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// GDS layer.
    pub layer: i16,
    /// GDS datatype.
    pub datatype: i16,
    /// Geometry.
    pub kind: ShapeKind,
}

/// A flattened text label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text {
    /// GDS layer.
    pub layer: i16,
    /// GDS text type.
    pub texttype: i16,
    /// Anchor (half-dbu).
    pub pos: [i64; 2],
    /// String.
    pub string: String,
    /// Hierarchy depth at which the label was defined (0 = top cell).
    pub depth: u32,
}

/// Flattened layout.
#[derive(Debug, Clone, Default)]
pub struct Flat {
    /// Shapes.
    pub shapes: Vec<Shape>,
    /// Labels.
    pub texts: Vec<Text>,
}

/// Flattening limits.
#[derive(Debug, Clone, Copy)]
pub struct FlattenOptions {
    /// Maximum number of shapes after flattening.
    pub max_shapes: usize,
}

impl Default for FlattenOptions {
    fn default() -> Self {
        Self {
            max_shapes: 50_000_000,
        }
    }
}

/// Affine map `p ↦ A·p + b` with `A` a signed permutation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Xf {
    a: [[i64; 2]; 2],
    b: [i64; 2],
}

impl Xf {
    const ID: Self = Self {
        a: [[1, 0], [0, 1]],
        b: [0, 0],
    };

    fn apply(&self, p: [i64; 2]) -> [i64; 2] {
        [
            self.a[0][0] * p[0] + self.a[0][1] * p[1] + self.b[0],
            self.a[1][0] * p[0] + self.a[1][1] * p[1] + self.b[1],
        ]
    }

    /// `self ∘ other` (apply `other` first).
    fn then_after(&self, other: &Self) -> Self {
        let a = |i: usize, j: usize| self.a[i][0] * other.a[0][j] + self.a[i][1] * other.a[1][j];
        let pb = self.apply(other.b);
        Self {
            a: [[a(0, 0), a(0, 1)], [a(1, 0), a(1, 1)]],
            b: pb,
        }
    }

    fn from_strans(s: &Strans, origin: [i64; 2]) -> Result<Self> {
        #[allow(clippy::float_cmp)]
        if s.mag != 1.0 {
            return Err(GdsError::Unsupported(format!(
                "magnification {} (only 1 is supported)",
                s.mag
            )));
        }
        if s.absolute {
            return Err(GdsError::Unsupported(
                "absolute magnification/angle flags".into(),
            ));
        }
        let q = s.angle / 90.0;
        if (q - q.round()).abs() > 1e-9 {
            return Err(GdsError::Unsupported(format!(
                "rotation by {}° (only multiples of 90°)",
                s.angle
            )));
        }
        #[allow(clippy::cast_possible_truncation)]
        let k = (q.round() as i64).rem_euclid(4);
        let rot = match k {
            0 => [[1, 0], [0, 1]],
            1 => [[0, -1], [1, 0]],
            2 => [[-1, 0], [0, -1]],
            _ => [[0, 1], [-1, 0]],
        };
        let refl = if s.reflect {
            [[1, 0], [0, -1]]
        } else {
            [[1, 0], [0, 1]]
        };
        let m = |i: usize, j: usize| rot[i][0] * refl[0][j] + rot[i][1] * refl[1][j];
        Ok(Self {
            a: [[m(0, 0), m(0, 1)], [m(1, 0), m(1, 1)]],
            b: origin,
        })
    }
}

fn half(p: [i32; 2]) -> [i64; 2] {
    [2 * i64::from(p[0]), 2 * i64::from(p[1])]
}

/// Flattens `top` (or the single top cell if `None`).
pub fn flatten(lib: &Library, top: Option<&str>, opts: FlattenOptions) -> Result<Flat> {
    let top = match top {
        Some(t) => t.to_owned(),
        None => {
            let tops = lib.top_cells();
            match tops.as_slice() {
                [t] => (*t).to_owned(),
                [] => {
                    return Err(GdsError::Hierarchy(
                        "no top cell (cyclic hierarchy?)".into(),
                    ));
                }
                many => {
                    return Err(GdsError::Hierarchy(format!(
                        "multiple top cells ({}); choose one with --top",
                        many.join(", ")
                    )));
                }
            }
        }
    };
    let idx = *lib
        .index
        .get(&top)
        .ok_or_else(|| GdsError::Hierarchy(format!("cell `{top}` not found")))?;
    let mut out = Flat::default();
    let mut stack = Vec::new();
    walk(lib, idx, Xf::ID, 0, &mut stack, &mut out, &opts)?;
    Ok(out)
}

fn walk(
    lib: &Library,
    idx: usize,
    xf: Xf,
    depth: u32,
    stack: &mut Vec<usize>,
    out: &mut Flat,
    opts: &FlattenOptions,
) -> Result<()> {
    if stack.contains(&idx) {
        return Err(GdsError::Hierarchy(format!(
            "reference cycle through `{}`",
            lib.structs[idx].0
        )));
    }
    stack.push(idx);
    for e in &lib.structs[idx].1 {
        match e {
            Element::Boundary {
                layer,
                datatype,
                xy,
            }
            | Element::Box {
                layer,
                boxtype: datatype,
                xy,
            } => {
                let pts = xy.iter().map(|&p| xf.apply(half(p))).collect();
                push(
                    out,
                    opts,
                    Shape {
                        layer: *layer,
                        datatype: *datatype,
                        kind: ShapeKind::Polygon(pts),
                    },
                )?;
            }
            Element::Path {
                layer,
                datatype,
                pathtype,
                width,
                bgn_ext,
                end_ext,
                xy,
            } => {
                let pts = xy.iter().map(|&p| xf.apply(half(p))).collect();
                let kind = ShapeKind::Path {
                    pathtype: *pathtype,
                    half_width: i64::from(*width).abs(),
                    bgn_ext: 2 * i64::from(*bgn_ext),
                    end_ext: 2 * i64::from(*end_ext),
                    pts,
                };
                push(
                    out,
                    opts,
                    Shape {
                        layer: *layer,
                        datatype: *datatype,
                        kind,
                    },
                )?;
            }
            Element::Text {
                layer,
                texttype,
                xy,
                string,
            } => {
                out.texts.push(Text {
                    layer: *layer,
                    texttype: *texttype,
                    pos: xf.apply(half(*xy)),
                    string: string.clone(),
                    depth,
                });
            }
            Element::Sref { sname, strans, xy } => {
                let child = child_index(lib, sname)?;
                let local = Xf::from_strans(strans, half(*xy))?;
                walk(
                    lib,
                    child,
                    xf.then_after(&local),
                    depth + 1,
                    stack,
                    out,
                    opts,
                )?;
            }
            Element::Aref {
                sname,
                strans,
                colrow,
                xy,
            } => {
                let child = child_index(lib, sname)?;
                let (cols, rows) = (i64::from(colrow[0]), i64::from(colrow[1]));
                if cols <= 0 || rows <= 0 {
                    return Err(GdsError::Format {
                        offset: 0,
                        msg: format!("AREF of `{sname}` with COLROW {colrow:?}"),
                    });
                }
                let (o, c, r) = (half(xy[0]), half(xy[1]), half(xy[2]));
                let step = |end: [i64; 2], n: i64| -> Result<[i64; 2]> {
                    let d = [end[0] - o[0], end[1] - o[1]];
                    if d[0] % n != 0 || d[1] % n != 0 {
                        return Err(GdsError::Unsupported(format!(
                            "AREF of `{sname}` with non-integer pitch"
                        )));
                    }
                    Ok([d[0] / n, d[1] / n])
                };
                let (dc, dr) = (step(c, cols)?, step(r, rows)?);
                for j in 0..rows {
                    for i in 0..cols {
                        let origin = [o[0] + i * dc[0] + j * dr[0], o[1] + i * dc[1] + j * dr[1]];
                        let local = Xf::from_strans(strans, origin)?;
                        walk(
                            lib,
                            child,
                            xf.then_after(&local),
                            depth + 1,
                            stack,
                            out,
                            opts,
                        )?;
                    }
                }
            }
        }
    }
    stack.pop();
    Ok(())
}

fn child_index(lib: &Library, name: &str) -> Result<usize> {
    lib.index
        .get(name)
        .copied()
        .ok_or_else(|| GdsError::Hierarchy(format!("reference to missing cell `{name}`")))
}

fn push(out: &mut Flat, opts: &FlattenOptions, s: Shape) -> Result<()> {
    if out.shapes.len() >= opts.max_shapes {
        return Err(GdsError::Unsupported(format!(
            "more than {} shapes after flattening",
            opts.max_shapes
        )));
    }
    out.shapes.push(s);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_compose() {
        let r90 = Xf::from_strans(
            &Strans {
                mag: 1.0,
                angle: 90.0,
                ..Strans::default()
            },
            [10, 0],
        )
        .unwrap();
        assert_eq!(r90.apply([2, 0]), [10, 2]);
        let refl = Xf::from_strans(
            &Strans {
                mag: 1.0,
                reflect: true,
                ..Strans::default()
            },
            [0, 0],
        )
        .unwrap();
        assert_eq!(refl.apply([3, 4]), [3, -4]);
        // reflect then rotate 90: (3,4) -> (3,-4) -> (4,3)
        let both = Xf::from_strans(
            &Strans {
                mag: 1.0,
                reflect: true,
                angle: 90.0,
                ..Strans::default()
            },
            [0, 0],
        )
        .unwrap();
        assert_eq!(both.apply([3, 4]), [4, 3]);
        let c = r90.then_after(&refl);
        assert_eq!(c.apply([3, 4]), r90.apply(refl.apply([3, 4])));
        assert!(
            Xf::from_strans(
                &Strans {
                    mag: 2.0,
                    ..Strans::default()
                },
                [0, 0]
            )
            .is_err()
        );
        assert!(
            Xf::from_strans(
                &Strans {
                    mag: 1.0,
                    angle: 45.0,
                    ..Strans::default()
                },
                [0, 0]
            )
            .is_err()
        );
    }
}
