//! Minimal GDSII writer (used for tests and for generating example layouts).

use crate::stream::{Element, Library};

/// Encodes an `f64` as a GDSII 8-byte real.
#[must_use]
pub fn to_real8(v: f64) -> [u8; 8] {
    if v == 0.0 || !v.is_finite() {
        return [0; 8];
    }
    let sign = if v < 0.0 { 0x80u8 } else { 0 };
    let mut a = v.abs();
    let mut exp: i32 = 0;
    while a >= 1.0 {
        a /= 16.0;
        exp += 1;
    }
    while a < 1.0 / 16.0 {
        a *= 16.0;
        exp -= 1;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mant = (a * (1u64 << 56) as f64).round() as u64;
    let mut out = [0u8; 8];
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        out[0] = sign | ((exp + 64) as u8 & 0x7f);
    }
    for i in 0..7 {
        #[allow(clippy::cast_possible_truncation)]
        {
            out[1 + i] = (mant >> (8 * (6 - i))) as u8;
        }
    }
    out
}

struct W(Vec<u8>);

impl W {
    fn rec(&mut self, rtype: u8, dtype: u8, data: &[u8]) {
        let mut d = data.to_vec();
        if d.len() % 2 == 1 {
            d.push(0);
        }
        #[allow(clippy::cast_possible_truncation)]
        let len = (d.len() + 4) as u16;
        self.0.extend_from_slice(&len.to_be_bytes());
        self.0.push(rtype);
        self.0.push(dtype);
        self.0.extend_from_slice(&d);
    }
    fn i16s(&mut self, rtype: u8, v: &[i16]) {
        let d: Vec<u8> = v.iter().flat_map(|x| x.to_be_bytes()).collect();
        self.rec(rtype, 2, &d);
    }
    fn i32s(&mut self, rtype: u8, v: &[i32]) {
        let d: Vec<u8> = v.iter().flat_map(|x| x.to_be_bytes()).collect();
        self.rec(rtype, 3, &d);
    }
    fn reals(&mut self, rtype: u8, v: &[f64]) {
        let d: Vec<u8> = v.iter().flat_map(|x| to_real8(*x)).collect();
        self.rec(rtype, 5, &d);
    }
    fn str(&mut self, rtype: u8, s: &str) {
        self.rec(rtype, 6, s.as_bytes());
    }
    fn xy(&mut self, pts: &[[i32; 2]]) {
        let v: Vec<i32> = pts.iter().flat_map(|p| [p[0], p[1]]).collect();
        self.i32s(0x10, &v);
    }
    fn strans(&mut self, s: &crate::stream::Strans) {
        let flags: u16 = if s.reflect { 0x8000 } else { 0 };
        self.rec(0x1A, 1, &flags.to_be_bytes());
        #[allow(clippy::float_cmp)]
        if s.mag != 1.0 {
            self.reals(0x1B, &[s.mag]);
        }
        if s.angle != 0.0 {
            self.reals(0x1C, &[s.angle]);
        }
    }
}

/// Serializes a library to GDSII bytes.
#[must_use]
pub fn write_library(lib: &Library) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.i16s(0x00, &[600]);
    w.i16s(0x01, &[0; 12]);
    w.str(0x02, &lib.name);
    let u = lib.units.unwrap_or(crate::stream::Units {
        db_in_user: 1e-3,
        db_in_meters: 1e-9,
    });
    w.reals(0x03, &[u.db_in_user, u.db_in_meters]);
    for (name, els) in &lib.structs {
        w.i16s(0x05, &[0; 12]);
        w.str(0x06, name);
        for e in els {
            match e {
                Element::Boundary {
                    layer,
                    datatype,
                    xy,
                } => {
                    w.rec(0x08, 0, &[]);
                    w.i16s(0x0D, &[*layer]);
                    w.i16s(0x0E, &[*datatype]);
                    w.xy(xy);
                }
                Element::Box { layer, boxtype, xy } => {
                    w.rec(0x2D, 0, &[]);
                    w.i16s(0x0D, &[*layer]);
                    w.i16s(0x2E, &[*boxtype]);
                    w.xy(xy);
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
                    w.rec(0x09, 0, &[]);
                    w.i16s(0x0D, &[*layer]);
                    w.i16s(0x0E, &[*datatype]);
                    w.i16s(0x21, &[*pathtype]);
                    w.i32s(0x0F, &[*width]);
                    if *pathtype == 4 {
                        w.i32s(0x30, &[*bgn_ext]);
                        w.i32s(0x31, &[*end_ext]);
                    }
                    w.xy(xy);
                }
                Element::Text {
                    layer,
                    texttype,
                    xy,
                    string,
                } => {
                    w.rec(0x0C, 0, &[]);
                    w.i16s(0x0D, &[*layer]);
                    w.i16s(0x16, &[*texttype]);
                    w.xy(&[*xy]);
                    w.str(0x19, string);
                }
                Element::Sref { sname, strans, xy } => {
                    w.rec(0x0A, 0, &[]);
                    w.str(0x12, sname);
                    w.strans(strans);
                    w.xy(&[*xy]);
                }
                Element::Aref {
                    sname,
                    strans,
                    colrow,
                    xy,
                } => {
                    w.rec(0x0B, 0, &[]);
                    w.str(0x12, sname);
                    w.strans(strans);
                    w.i16s(0x13, colrow);
                    w.xy(xy);
                }
            }
            w.rec(0x11, 0, &[]);
        }
        w.rec(0x07, 0, &[]);
    }
    w.rec(0x04, 0, &[]);
    w.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::real8;

    #[test]
    fn real8_roundtrip() {
        for v in [1e-3, 1e-9, 1.0, -2.5, 90.0, 270.0, 0.0, 123_456.789] {
            let r = real8(to_real8(v));
            assert!((r - v).abs() <= 1e-15 * v.abs(), "{v} -> {r}");
        }
    }
}
