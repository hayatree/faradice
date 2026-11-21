//! Unit system.
//!
//! Internally faradice uses **micrometres (µm)** for lengths and
//! **femtofarads (fF)** for capacitances. With these units the vacuum
//! permittivity is [`EPS0_FF_PER_UM`].

use std::fmt;
use std::str::FromStr;

use crate::Error;

/// Vacuum permittivity ε0 in fF/µm (CODATA 2018: 8.8541878128e-12 F/m).
pub const EPS0_FF_PER_UM: f64 = 8.854_187_812_8e-3;

/// A length unit accepted in input files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LengthUnit {
    /// Nanometre.
    Nm,
    /// Micrometre (the internal unit).
    #[default]
    Um,
    /// Millimetre.
    Mm,
    /// Metre.
    M,
}

impl LengthUnit {
    /// Multiplicative factor converting a value in `self` to µm.
    #[must_use]
    pub const fn to_um_factor(self) -> f64 {
        match self {
            Self::Nm => 1e-3,
            Self::Um => 1.0,
            Self::Mm => 1e3,
            Self::M => 1e6,
        }
    }

    /// Converts `value` expressed in `self` to µm.
    #[must_use]
    pub fn to_um(self, value: f64) -> f64 {
        value * self.to_um_factor()
    }

    /// Canonical lowercase name (`"nm"`, `"um"`, `"mm"`, `"m"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nm => "nm",
            Self::Um => "um",
            Self::Mm => "mm",
            Self::M => "m",
        }
    }
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for LengthUnit {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nm" => Ok(Self::Nm),
            "um" | "µm" | "micron" => Ok(Self::Um),
            "mm" => Ok(Self::Mm),
            "m" => Ok(Self::M),
            _ => Err(Error::UnknownLengthUnit(s.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_convert() {
        assert_eq!("nm".parse::<LengthUnit>().unwrap(), LengthUnit::Nm);
        assert_eq!(" UM ".parse::<LengthUnit>().unwrap(), LengthUnit::Um);
        assert_eq!("µm".parse::<LengthUnit>().unwrap(), LengthUnit::Um);
        assert!("inch".parse::<LengthUnit>().is_err());
        assert!((LengthUnit::Nm.to_um(140.0) - 0.14).abs() < 1e-15);
        assert!((LengthUnit::M.to_um(1e-6) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn display_roundtrip() {
        for u in [
            LengthUnit::Nm,
            LengthUnit::Um,
            LengthUnit::Mm,
            LengthUnit::M,
        ] {
            assert_eq!(u.to_string().parse::<LengthUnit>().unwrap(), u);
        }
    }

    #[test]
    fn eps0_matches_si() {
        // 8.8541878128e-12 F/m = 8.8541878128e-12 * 1e15 fF / 1e6 µm
        let si = 8.854_187_812_8e-12_f64 * 1e15 / 1e6;
        assert!((EPS0_FF_PER_UM - si).abs() < 1e-18);
    }
}
