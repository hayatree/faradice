//! Axis and face enums for axis-aligned geometry.

use std::fmt;
use std::str::FromStr;

use crate::Error;

/// A Cartesian axis. `x` and `y` span the layout plane, `z` is height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    /// The x axis.
    X,
    /// The y axis.
    Y,
    /// The z (vertical) axis.
    Z,
}

impl Axis {
    /// All axes in order.
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Index into a `[f64; 3]` coordinate array.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// The lower or upper side along an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    /// The side with the smaller coordinate.
    Min,
    /// The side with the larger coordinate.
    Max,
}

impl Side {
    /// `-1.0` for [`Side::Min`], `+1.0` for [`Side::Max`] (outward normal sign).
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Min => -1.0,
            Self::Max => 1.0,
        }
    }
}

/// One of the six faces of an axis-aligned box or domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Face {
    /// The axis normal to the face.
    pub axis: Axis,
    /// Which side of the axis the face is on.
    pub side: Side,
}

impl Face {
    /// All six faces: xmin, xmax, ymin, ymax, zmin, zmax.
    pub const ALL: [Self; 6] = [
        Self::new(Axis::X, Side::Min),
        Self::new(Axis::X, Side::Max),
        Self::new(Axis::Y, Side::Min),
        Self::new(Axis::Y, Side::Max),
        Self::new(Axis::Z, Side::Min),
        Self::new(Axis::Z, Side::Max),
    ];

    /// Creates a face.
    #[must_use]
    pub const fn new(axis: Axis, side: Side) -> Self {
        Self { axis, side }
    }

    /// Index in `0..6`, consistent with [`Face::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.axis.index() * 2
            + match self.side {
                Side::Min => 0,
                Side::Max => 1,
            }
    }

    /// Canonical name, e.g. `"xmin"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match (self.axis, self.side) {
            (Axis::X, Side::Min) => "xmin",
            (Axis::X, Side::Max) => "xmax",
            (Axis::Y, Side::Min) => "ymin",
            (Axis::Y, Side::Max) => "ymax",
            (Axis::Z, Side::Min) => "zmin",
            (Axis::Z, Side::Max) => "zmax",
        }
    }
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Face {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|face| face.as_str() == lower)
            .ok_or_else(|| Error::UnknownFace(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_index_matches_all() {
        for (i, face) in Face::ALL.into_iter().enumerate() {
            assert_eq!(face.index(), i);
            assert_eq!(face.to_string().parse::<Face>().unwrap(), face);
        }
        assert!("top".parse::<Face>().is_err());
    }

    #[test]
    fn side_sign() {
        assert!(Side::Min.sign() < 0.0 && Side::Max.sign() > 0.0);
    }
}
