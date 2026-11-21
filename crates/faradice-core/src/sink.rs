//! Walk termination targets.

use std::fmt;

use crate::{Face, NetId};

/// Label of the sum over ground-like sinks in result and sensitivity files.
pub const GROUND_SUM_LABEL: &str = "<GND>";

/// `true` for labels of the reserved form `<…>` (the ground sum and the
/// fixed ground sinks). Net names may not take this form, so sink labels
/// never collide with net names.
#[must_use]
pub fn is_reserved_label(s: &str) -> bool {
    s.len() >= 2 && s.starts_with('<') && s.ends_with('>')
}

/// Where a floating random walk terminated.
///
/// Its label (`Display`) is `<substrate>`, `<boundary:FACE>` or
/// `<infinity>` for the fixed ground sinks (see [`is_reserved_label`]).
///
/// Every walk launched from a target net ends at exactly one sink; the
/// estimator accumulates the walk weight per sink. Ground-like sinks are kept
/// separate so that reports can either merge them or show them by source
/// (see `ground.report` in the M1 design).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Sink {
    /// A conductor belonging to a net.
    Net(NetId),
    /// The substrate ground plane.
    GroundPlane,
    /// A grounded (Dirichlet) face of the simulation domain.
    Boundary(Face),
    /// Infinity, reached through an `open` boundary.
    Infinity,
}

impl Sink {
    /// `true` for sinks that are always at ground potential regardless of net
    /// configuration (ground plane, grounded boundary, infinity).
    #[must_use]
    pub const fn is_fixed_ground(self) -> bool {
        !matches!(self, Self::Net(_))
    }

    /// The net, if this sink is a conductor.
    #[must_use]
    pub const fn net(self) -> Option<NetId> {
        match self {
            Self::Net(id) => Some(id),
            _ => None,
        }
    }
}

impl fmt::Display for Sink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Net(id) => write!(f, "{id}"),
            Self::GroundPlane => f.write_str("<substrate>"),
            Self::Boundary(face) => write!(f, "<boundary:{face}>"),
            Self::Infinity => f.write_str("<infinity>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Axis, Side};

    #[test]
    fn display_and_classification() {
        let n = Sink::Net(NetId::new(3));
        assert_eq!(n.to_string(), "net#3");
        assert!(!n.is_fixed_ground());
        assert_eq!(n.net(), Some(NetId::new(3)));

        let b = Sink::Boundary(Face::new(Axis::Z, Side::Max));
        assert_eq!(b.to_string(), "<boundary:zmax>");
        assert!(b.is_fixed_ground());
        assert_eq!(Sink::GroundPlane.to_string(), "<substrate>");
        assert_eq!(Sink::Infinity.to_string(), "<infinity>");
        for l in ["<GND>", "<substrate>", "<boundary:zmax>", "<x>"] {
            assert!(is_reserved_label(l), "{l}");
        }
        for l in ["GND", "net<3>", "<a", "a>", "<", "bus<0>x"] {
            assert!(!is_reserved_label(l), "{l}");
        }
    }
}
