//! GDSII stream reader.
//!
//! Supports the record types needed for physical geometry and labels. Unknown
//! records inside elements (properties, ELFLAGS, PLEX, …) are skipped.

use std::collections::HashMap;

use crate::{GdsError, Result};

/// Library units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Units {
    /// Database unit in user units.
    pub db_in_user: f64,
    /// Database unit in metres.
    pub db_in_meters: f64,
}

/// Transformation of a reference (applied as: reflect about x, magnify, rotate, translate).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Strans {
    /// Reflect about the x axis before rotation.
    pub reflect: bool,
    /// Magnification (only 1 is supported when flattening).
    pub mag: f64,
    /// Rotation angle in degrees (multiples of 90 supported when flattening).
    pub angle: f64,
    /// Absolute magnification / angle flags (unsupported when set).
    pub absolute: bool,
}

/// An element of a structure.
#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    /// Closed polygon (first point repeated at the end).
    Boundary {
        /// Layer.
        layer: i16,
        /// Datatype.
        datatype: i16,
        /// Vertices in database units.
        xy: Vec<[i32; 2]>,
    },
    /// Path.
    Path {
        /// Layer.
        layer: i16,
        /// Datatype.
        datatype: i16,
        /// Path type (0 flush, 1 round, 2 half-width extension, 4 custom).
        pathtype: i16,
        /// Width in database units (negative = absolute, treated as magnitude).
        width: i32,
        /// Begin extension (pathtype 4).
        bgn_ext: i32,
        /// End extension (pathtype 4).
        end_ext: i32,
        /// Centre line.
        xy: Vec<[i32; 2]>,
    },
    /// BOX element (treated like a boundary).
    Box {
        /// Layer.
        layer: i16,
        /// Box type.
        boxtype: i16,
        /// Vertices.
        xy: Vec<[i32; 2]>,
    },
    /// Text label.
    Text {
        /// Layer.
        layer: i16,
        /// Text type.
        texttype: i16,
        /// Anchor point.
        xy: [i32; 2],
        /// The string.
        string: String,
    },
    /// Structure reference.
    Sref {
        /// Referenced structure.
        sname: String,
        /// Transformation.
        strans: Strans,
        /// Origin.
        xy: [i32; 2],
    },
    /// Array reference.
    Aref {
        /// Referenced structure.
        sname: String,
        /// Transformation.
        strans: Strans,
        /// Columns, rows.
        colrow: [i16; 2],
        /// Origin, column end point, row end point.
        xy: [[i32; 2]; 3],
    },
}

/// A parsed library.
#[derive(Debug, Clone, Default)]
pub struct Library {
    /// Library name.
    pub name: String,
    /// Units.
    pub units: Option<Units>,
    /// Structures by name, in file order.
    pub structs: Vec<(String, Vec<Element>)>,
    /// Name → index into `structs`.
    pub index: HashMap<String, usize>,
}

impl Library {
    /// Top cells: structures that are never referenced.
    #[must_use]
    pub fn top_cells(&self) -> Vec<&str> {
        let mut referenced = std::collections::HashSet::new();
        for (_, els) in &self.structs {
            for e in els {
                if let Element::Sref { sname, .. } | Element::Aref { sname, .. } = e {
                    referenced.insert(sname.as_str());
                }
            }
        }
        self.structs
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| !referenced.contains(n))
            .collect()
    }
}

/// Converts a GDSII 8-byte real (excess-64, base-16) to `f64`.
#[must_use]
pub fn real8(b: [u8; 8]) -> f64 {
    let sign = if b[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from(b[0] & 0x7f) - 64;
    let mut mant: u64 = 0;
    for &x in &b[1..] {
        mant = (mant << 8) | u64::from(x);
    }
    #[allow(clippy::cast_precision_loss)]
    let m = mant as f64 / (1u64 << 56) as f64;
    sign * m * 16f64.powi(exp)
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

struct Record<'a> {
    rtype: u8,
    offset: usize,
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    fn next(&mut self) -> Result<Option<Record<'a>>> {
        if self.pos + 4 > self.data.len() {
            return Ok(None);
        }
        let len = usize::from(u16::from_be_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
        ]));
        if len == 0 {
            // Zero padding after ENDLIB.
            return Ok(None);
        }
        if len < 4 || self.pos + len > self.data.len() {
            return Err(GdsError::Format {
                offset: self.pos,
                msg: format!("bad record length {len}"),
            });
        }
        let rec = Record {
            rtype: self.data[self.pos + 2],
            offset: self.pos,
            data: &self.data[self.pos + 4..self.pos + len],
        };
        self.pos += len;
        Ok(Some(rec))
    }
}

impl Record<'_> {
    fn err(&self, msg: &str) -> GdsError {
        GdsError::Format {
            offset: self.offset,
            msg: format!("record 0x{:02x}: {msg}", self.rtype),
        }
    }
    fn i16s(&self) -> Vec<i16> {
        self.data
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]))
            .collect()
    }
    fn i16(&self) -> Result<i16> {
        self.i16s()
            .first()
            .copied()
            .ok_or_else(|| self.err("missing i16"))
    }
    fn i32s(&self) -> Vec<i32> {
        self.data
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn i32(&self) -> Result<i32> {
        self.i32s()
            .first()
            .copied()
            .ok_or_else(|| self.err("missing i32"))
    }
    fn reals(&self) -> Vec<f64> {
        self.data
            .chunks_exact(8)
            .map(|c| real8(c.try_into().unwrap_or([0; 8])))
            .collect()
    }
    fn string(&self) -> String {
        let end = self
            .data
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.data.len());
        String::from_utf8_lossy(&self.data[..end]).into_owned()
    }
    fn points(&self) -> Result<Vec<[i32; 2]>> {
        let v = self.i32s();
        if v.len() % 2 != 0 {
            return Err(self.err("odd XY count"));
        }
        Ok(v.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
    }
}

mod rt {
    pub(super) const BGNLIB: u8 = 0x01;
    pub(super) const LIBNAME: u8 = 0x02;
    pub(super) const UNITS: u8 = 0x03;
    pub(super) const ENDLIB: u8 = 0x04;
    pub(super) const BGNSTR: u8 = 0x05;
    pub(super) const STRNAME: u8 = 0x06;
    pub(super) const ENDSTR: u8 = 0x07;
    pub(super) const BOUNDARY: u8 = 0x08;
    pub(super) const PATH: u8 = 0x09;
    pub(super) const SREF: u8 = 0x0A;
    pub(super) const AREF: u8 = 0x0B;
    pub(super) const TEXT: u8 = 0x0C;
    pub(super) const LAYER: u8 = 0x0D;
    pub(super) const DATATYPE: u8 = 0x0E;
    pub(super) const WIDTH: u8 = 0x0F;
    pub(super) const XY: u8 = 0x10;
    pub(super) const ENDEL: u8 = 0x11;
    pub(super) const SNAME: u8 = 0x12;
    pub(super) const COLROW: u8 = 0x13;
    pub(super) const NODE: u8 = 0x15;
    pub(super) const TEXTTYPE: u8 = 0x16;
    pub(super) const STRING: u8 = 0x19;
    pub(super) const STRANS: u8 = 0x1A;
    pub(super) const MAG: u8 = 0x1B;
    pub(super) const ANGLE: u8 = 0x1C;
    pub(super) const PATHTYPE: u8 = 0x21;
    pub(super) const BOX: u8 = 0x2D;
    pub(super) const BOXTYPE: u8 = 0x2E;
    pub(super) const BGNEXTN: u8 = 0x30;
    pub(super) const ENDEXTN: u8 = 0x31;
}

#[derive(Default)]
struct ElemState {
    kind: u8,
    layer: i16,
    dtype: i16,
    pathtype: i16,
    width: i32,
    bgn: i32,
    end: i32,
    xy: Vec<[i32; 2]>,
    sname: String,
    strans: Strans,
    colrow: [i16; 2],
    string: String,
}

/// Parses a GDSII stream from bytes.
pub fn parse_library(data: &[u8]) -> Result<Library> {
    let mut r = Reader { data, pos: 0 };
    let mut lib = Library::default();
    let mut current: Option<(String, Vec<Element>)> = None;
    let mut el: Option<ElemState> = None;
    while let Some(rec) = r.next()? {
        match rec.rtype {
            rt::LIBNAME => lib.name = rec.string(),
            rt::UNITS => {
                let v = rec.reals();
                if v.len() != 2 {
                    return Err(rec.err("UNITS needs two reals"));
                }
                lib.units = Some(Units {
                    db_in_user: v[0],
                    db_in_meters: v[1],
                });
            }
            rt::BGNSTR => current = Some((String::new(), Vec::new())),
            rt::STRNAME => {
                if let Some(c) = current.as_mut() {
                    c.0 = rec.string();
                }
            }
            rt::ENDSTR => {
                let (name, els) = current
                    .take()
                    .ok_or_else(|| rec.err("ENDSTR without BGNSTR"))?;
                if lib.index.insert(name.clone(), lib.structs.len()).is_some() {
                    return Err(GdsError::Hierarchy(format!("duplicate structure `{name}`")));
                }
                lib.structs.push((name, els));
            }
            rt::BOUNDARY | rt::PATH | rt::SREF | rt::AREF | rt::TEXT | rt::BOX | rt::NODE => {
                el = Some(ElemState {
                    kind: rec.rtype,
                    strans: Strans {
                        mag: 1.0,
                        ..Strans::default()
                    },
                    ..ElemState::default()
                });
            }
            rt::ENDEL => {
                let e = el.take().ok_or_else(|| rec.err("ENDEL without element"))?;
                let c = current
                    .as_mut()
                    .ok_or_else(|| rec.err("element outside structure"))?;
                if let Some(element) = finish(e, &rec)? {
                    c.1.push(element);
                }
            }
            rt::ENDLIB => break,
            _ => {
                if let Some(e) = el.as_mut() {
                    match rec.rtype {
                        rt::LAYER => e.layer = rec.i16()?,
                        rt::DATATYPE | rt::TEXTTYPE | rt::BOXTYPE => e.dtype = rec.i16()?,
                        rt::PATHTYPE => e.pathtype = rec.i16()?,
                        rt::WIDTH => e.width = rec.i32()?,
                        rt::BGNEXTN => e.bgn = rec.i32()?,
                        rt::ENDEXTN => e.end = rec.i32()?,
                        rt::XY => e.xy = rec.points()?,
                        rt::SNAME => e.sname = rec.string(),
                        rt::STRING => e.string = rec.string(),
                        rt::STRANS => {
                            let f = u16::from_be_bytes([
                                rec.data.first().copied().unwrap_or(0),
                                rec.data.get(1).copied().unwrap_or(0),
                            ]);
                            e.strans.reflect = f & 0x8000 != 0;
                            e.strans.absolute = f & 0x0006 != 0;
                        }
                        rt::MAG => e.strans.mag = rec.reals().first().copied().unwrap_or(1.0),
                        rt::ANGLE => e.strans.angle = rec.reals().first().copied().unwrap_or(0.0),
                        rt::COLROW => {
                            let v = rec.i16s();
                            if v.len() != 2 {
                                return Err(rec.err("COLROW needs two values"));
                            }
                            e.colrow = [v[0], v[1]];
                        }
                        _ => {}
                    }
                }
            }
        }
        let _ = rt::BGNLIB;
    }
    if current.is_some() {
        return Err(GdsError::Format {
            offset: r.pos,
            msg: "unterminated structure".into(),
        });
    }
    Ok(lib)
}

fn finish(e: ElemState, rec: &Record<'_>) -> Result<Option<Element>> {
    let one = |xy: &[[i32; 2]]| xy.first().copied().ok_or_else(|| rec.err("missing XY"));
    Ok(Some(match e.kind {
        rt::BOUNDARY => Element::Boundary {
            layer: e.layer,
            datatype: e.dtype,
            xy: e.xy,
        },
        rt::BOX => Element::Box {
            layer: e.layer,
            boxtype: e.dtype,
            xy: e.xy,
        },
        rt::PATH => Element::Path {
            layer: e.layer,
            datatype: e.dtype,
            pathtype: e.pathtype,
            width: e.width,
            bgn_ext: e.bgn,
            end_ext: e.end,
            xy: e.xy,
        },
        rt::TEXT => Element::Text {
            layer: e.layer,
            texttype: e.dtype,
            xy: one(&e.xy)?,
            string: e.string,
        },
        rt::SREF => Element::Sref {
            sname: e.sname,
            strans: e.strans,
            xy: one(&e.xy)?,
        },
        rt::AREF => {
            if e.xy.len() != 3 {
                return Err(rec.err("AREF needs three XY points"));
            }
            Element::Aref {
                sname: e.sname,
                strans: e.strans,
                colrow: e.colrow,
                xy: [e.xy[0], e.xy[1], e.xy[2]],
            }
        }
        _ => return Ok(None), // NODE
    }))
}

/// Reads and parses a GDSII file.
pub fn read_library(path: &std::path::Path) -> Result<Library> {
    let data = std::fs::read(path).map_err(|source| GdsError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_library(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real8_values() {
        // 1e-3 and 1e-9 as written by common tools.
        assert!((real8([0x3E, 0x41, 0x89, 0x37, 0x4B, 0xC6, 0xA7, 0xF0]) - 1e-3).abs() < 1e-15);
        assert!((real8([0x39, 0x44, 0xB8, 0x2F, 0xA0, 0x9B, 0x5A, 0x54]) - 1e-9).abs() < 1e-21);
        assert_eq!(real8([0x41, 0x10, 0, 0, 0, 0, 0, 0]), 1.0);
        assert_eq!(real8([0xC1, 0x20, 0, 0, 0, 0, 0, 0]), -2.0);
        assert_eq!(real8([0; 8]), 0.0);
    }
}
