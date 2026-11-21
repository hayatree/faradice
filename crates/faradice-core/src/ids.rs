//! Strongly typed identifiers.

use std::fmt;

use crate::{Error, Result};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// Creates an identifier from a raw `u32`.
            #[must_use]
            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            /// Creates an identifier from a `usize` index, failing if it exceeds `u32::MAX`.
            pub fn from_index(index: usize) -> Result<Self> {
                u32::try_from(index).map(Self).map_err(|_| Error::IdOverflow(index))
            }

            /// The raw value.
            #[must_use]
            pub const fn raw(self) -> u32 {
                self.0
            }

            /// The value as a `usize` index.
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "#{}"), self.0)
            }
        }
    };
}

define_id!(
    /// Index of a net (an electrically connected set of conductor boxes).
    NetId,
    "net"
);
define_id!(
    /// Index of a single axis-aligned conductor box.
    BoxId,
    "box"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip() {
        let id = NetId::from_index(42).unwrap();
        assert_eq!(id.index(), 42);
        assert_eq!(id.raw(), 42);
        assert_eq!(id.to_string(), "net#42");
        assert_eq!(BoxId::new(7).to_string(), "box#7");
    }

    #[test]
    fn overflow_is_an_error() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            NetId::from_index(u32::MAX as usize + 1),
            Err(Error::IdOverflow(u32::MAX as usize + 1))
        );
    }
}
