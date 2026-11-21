//! Stratified dielectric stacks (design M2a, `docs/design/M2_multi_dielectric.md`).
//!
//! The input is a list of layers from the bottom up, each with the height of
//! its top interface, plus the permittivity above the last layer. The solver
//! works on the **effective stack** (§3.5 of the design): interfaces clipped to
//! the domain, mirrored across reflecting z faces, with adjacent layers of
//! equal permittivity merged, so that a stack whose layers all share one ε is
//! exactly the homogeneous problem.

use crate::boundary::BoundaryModel;

/// One dielectric layer: the medium below `z_top` (and above the previous
/// layer's `z_top`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DielectricLayer {
    /// Height of the layer's top interface (µm).
    pub z_top: f64,
    /// Relative permittivity.
    pub eps_r: f64,
}

/// Checks a layer list: finite heights, strictly increasing, and finite
/// permittivities `>= 1`.
///
/// # Errors
///
/// A message describing the first violation.
pub fn validate_layers(layers: &[DielectricLayer]) -> Result<(), String> {
    let ok_eps = |e: f64| e.is_finite() && e >= 1.0;
    for (i, l) in layers.iter().enumerate() {
        if !l.z_top.is_finite() {
            return Err(format!("dielectric layer {i}: z_top must be finite"));
        }
        if !ok_eps(l.eps_r) {
            return Err(format!(
                "dielectric layer {i}: eps_r must be finite and >= 1"
            ));
        }
        if i > 0 && l.z_top <= layers[i - 1].z_top {
            return Err(format!(
                "dielectric layer {i}: z_top must be above the previous layer's z_top"
            ));
        }
    }
    Ok(())
}

/// The effective piecewise-constant permittivity ε(z) seen by the walk.
#[derive(Debug, Clone, PartialEq)]
pub struct Stack {
    /// Interface heights, strictly increasing (including images across
    /// reflecting z faces).
    interfaces: Vec<f64>,
    /// Relative permittivity of each layer: `eps[i]` lies below
    /// `interfaces[i]`, the last entry above the last interface.
    eps: Vec<f64>,
    /// Largest cube half-side whose vertical extent the images cover: the
    /// domain height if both z faces reflect (a grounded face bounds the
    /// clearance by the domain height anyway), else unbounded.
    max_half: f64,
}

impl Stack {
    /// A homogeneous medium.
    #[must_use]
    pub fn homogeneous(eps_r: f64) -> Self {
        Self {
            interfaces: Vec::new(),
            eps: vec![eps_r],
            max_half: f64::INFINITY,
        }
    }

    /// Builds the effective stack of `layers` (bottom-up) under `eps_top` for
    /// the domain `domain` (validated input assumed, see [`validate_layers`]).
    #[must_use]
    pub fn effective(eps_top: f64, layers: &[DielectricLayer], domain: &BoundaryModel) -> Self {
        let raw = |z: f64| {
            layers
                .iter()
                .find(|l| z < l.z_top)
                .map_or(eps_top, |l| l.eps_r)
        };
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        let (mut rlo, mut rhi) = (false, false);
        if let Some(b) = domain.bounds() {
            lo = b.min()[2];
            hi = b.max()[2];
            (rlo, rhi) = domain.reflect()[2];
        }
        if let Some(z0) = domain.substrate_z() {
            // The substrate replaces the bottom face only if it bounds the domain.
            if z0 >= lo {
                lo = z0;
                rlo = false;
            }
        }
        // Medium seen at any height: fold into [lo, hi] across reflecting faces.
        let fold = |mut z: f64| {
            for _ in 0..8 {
                if rlo && z < lo {
                    z = 2.0 * lo - z;
                } else if rhi && z > hi {
                    z = 2.0 * hi - z;
                } else {
                    break;
                }
            }
            z
        };
        let inside: Vec<f64> = layers
            .iter()
            .map(|l| l.z_top)
            .filter(|&z| lo < z && z < hi)
            .collect();
        let mut cand = inside.clone();
        // Images up to two reflections cover every interface a cube can reach:
        // cubes never extend more than the domain height beyond a reflecting face.
        if rlo {
            cand.extend(inside.iter().map(|&z| 2.0 * lo - z));
        }
        if rhi {
            cand.extend(inside.iter().map(|&z| 2.0 * hi - z));
        }
        if rlo && rhi {
            let h = hi - lo;
            cand.extend(inside.iter().map(|&z| z + 2.0 * h));
            cand.extend(inside.iter().map(|&z| z - 2.0 * h));
        }
        cand.sort_by(f64::total_cmp);
        cand.dedup();
        let mid = |a: f64, b: f64| match (a.is_finite(), b.is_finite()) {
            (true, true) => 0.5 * (a + b),
            (true, false) => a + 1.0,
            (false, true) => b - 1.0,
            (false, false) => 0.0,
        };
        // Labels of the segments inside the domain, sampled strictly inside them.
        let bounds: Vec<f64> = std::iter::once(lo)
            .chain(inside.iter().copied())
            .chain(std::iter::once(hi))
            .collect();
        let labels: Vec<f64> = bounds.windows(2).map(|w| raw(mid(w[0], w[1]))).collect();
        // Medium at any height: fold across reflecting faces, then the segment
        // containing it (heights outside a non-reflecting face keep the label of
        // the adjacent interior segment; walks never get there).
        let eps_between = |a: f64, b: f64| {
            let z = fold(mid(a, b));
            labels[inside.partition_point(|&zi| zi < z)]
        };
        let mut interfaces = Vec::new();
        let mut eps = Vec::new();
        let mut prev = f64::NEG_INFINITY;
        for &z in &cand {
            let e = eps_between(prev, z);
            if eps.last() != Some(&e) {
                if !eps.is_empty() {
                    interfaces.push(prev);
                }
                eps.push(e);
            }
            prev = z;
        }
        let e = eps_between(prev, f64::INFINITY);
        if eps.last() != Some(&e) {
            if !eps.is_empty() {
                interfaces.push(prev);
            }
            eps.push(e);
        }
        let max_half = if rlo && rhi { hi - lo } else { f64::INFINITY };
        Self {
            interfaces,
            eps,
            max_half,
        }
    }

    /// Largest cube half-side, centred inside the domain, for which the
    /// interfaces listed here (with their images) are all those the cube
    /// contains.
    #[must_use]
    pub const fn max_half(&self) -> f64 {
        self.max_half
    }

    /// `true` if there is no interface (a single permittivity).
    #[must_use]
    pub fn is_homogeneous(&self) -> bool {
        self.interfaces.is_empty()
    }

    /// Interface heights, strictly increasing.
    #[must_use]
    pub fn interfaces(&self) -> &[f64] {
        &self.interfaces
    }

    /// Relative permittivities, one more than [`Self::interfaces`].
    #[must_use]
    pub fn eps(&self) -> &[f64] {
        &self.eps
    }

    /// Relative permittivity at height `z` (a point exactly on an interface
    /// belongs to the layer above).
    #[must_use]
    pub fn eps_at(&self, z: f64) -> f64 {
        self.eps[self.interfaces.partition_point(|&zi| zi <= z)]
    }

    /// Relative permittivity just below height `z` (the layer under an
    /// interface at `z`).
    #[must_use]
    pub fn eps_below(&self, z: f64) -> f64 {
        self.eps[self.interfaces.partition_point(|&zi| zi < z)]
    }

    /// The nearest interface to height `z`: its index, the distance `d₁` to it
    /// and the distance `d₂` to the second-nearest interface
    /// (`f64::INFINITY` if absent). `None` for a homogeneous stack.
    #[must_use]
    pub fn nearest(&self, z: f64) -> Option<(usize, f64, f64)> {
        let i = self.interfaces.partition_point(|&zi| zi < z);
        let mut best: Option<(usize, f64)> = None;
        let mut second = f64::INFINITY;
        for j in [i.wrapping_sub(2), i.wrapping_sub(1), i, i + 1] {
            if let Some(&zi) = self.interfaces.get(j) {
                let d = (z - zi).abs();
                match best {
                    Some((_, b)) if d >= b => second = second.min(d),
                    Some((_, b)) => {
                        second = second.min(b);
                        best = Some((j, d));
                    }
                    None => best = Some((j, d)),
                }
            }
        }
        best.map(|(j, d)| (j, d, second))
    }

    /// Distances from height `z` to the nearest and the second-nearest
    /// interface (`f64::INFINITY` if absent).
    #[must_use]
    pub fn nearest_two(&self, z: f64) -> (f64, f64) {
        let i = self.interfaces.partition_point(|&zi| zi < z);
        let mut d = [f64::INFINITY; 4];
        for (k, j) in [i.wrapping_sub(2), i.wrapping_sub(1), i, i + 1]
            .into_iter()
            .enumerate()
        {
            if let Some(&zi) = self.interfaces.get(j) {
                d[k] = (z - zi).abs();
            }
        }
        d.sort_by(f64::total_cmp);
        (d[0], d[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{Boundary, FaceKind};
    use faradice_core::{Axis, Face, Side};
    use faradice_geom::Aabb;

    fn layer(z_top: f64, eps_r: f64) -> DielectricLayer {
        DielectricLayer { z_top, eps_r }
    }

    fn boxed(zmin: f64, zmax: f64, reflect_lo: bool, reflect_hi: bool) -> BoundaryModel {
        let mut faces = [FaceKind::Ground; 6];
        if reflect_lo {
            faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Reflect;
        }
        if reflect_hi {
            faces[Face::new(Axis::Z, Side::Max).index()] = FaceKind::Reflect;
        }
        let b = Aabb::new([0.0, 0.0, zmin], [1.0, 1.0, zmax]).unwrap();
        BoundaryModel::new(&Boundary::Box { bounds: b, faces }, None, b)
    }

    #[test]
    fn cube_size_is_capped_only_between_two_reflecting_faces() {
        let layers = [layer(1.0, 3.0)];
        for (lo, hi, h) in [
            (true, true, 4.0),
            (true, false, f64::INFINITY),
            (false, true, f64::INFINITY),
            (false, false, f64::INFINITY),
        ] {
            let s = Stack::effective(1.0, &layers, &boxed(0.0, 4.0, lo, hi));
            assert_eq!(
                s.max_half().total_cmp(&h),
                std::cmp::Ordering::Equal,
                "{lo} {hi}"
            );
        }
    }

    #[test]
    fn validation() {
        assert!(validate_layers(&[layer(1.0, 3.9), layer(2.0, 4.5)]).is_ok());
        assert!(validate_layers(&[]).is_ok());
        assert!(validate_layers(&[layer(2.0, 3.9), layer(2.0, 4.5)]).is_err());
        assert!(validate_layers(&[layer(f64::NAN, 3.9)]).is_err());
        assert!(validate_layers(&[layer(1.0, 0.9)]).is_err());
    }

    #[test]
    fn clips_and_merges() {
        let d = boxed(0.0, 10.0, false, false);
        let s = Stack::effective(
            1.0,
            &[
                layer(-1.0, 7.0),
                layer(2.0, 3.9),
                layer(3.0, 3.9),
                layer(5.0, 4.5),
            ],
            &d,
        );
        assert_eq!(s.interfaces(), &[3.0, 5.0]);
        assert_eq!(s.eps(), &[3.9, 4.5, 1.0]);
        assert_eq!(s.eps_at(4.0), 4.5);
        assert_eq!(s.eps_at(5.0), 1.0);
        assert_eq!(s.nearest_two(4.2), (0.7999999999999998, 1.2000000000000002));
        assert_eq!(
            s.nearest(4.2),
            Some((1, 0.7999999999999998, 1.2000000000000002))
        );
        assert_eq!(s.nearest(3.0), Some((0, 0.0, 2.0)));
        assert_eq!(s.nearest(9.0), Some((1, 4.0, 6.0)));
        // Equal permittivity everywhere: homogeneous.
        let h = Stack::effective(3.9, &[layer(2.0, 3.9), layer(5.0, 3.9)], &d);
        assert!(h.is_homogeneous());
        assert_eq!(h, Stack::homogeneous(3.9));
    }

    #[test]
    fn layer_top_on_the_domain_top_keeps_the_interior_label() {
        let d = boxed(0.0, 1.0, false, false);
        let s = Stack::effective(4.0, &[layer(0.5, 4.0), layer(1.0, 5.0)], &d);
        assert_eq!(s.interfaces(), &[0.5]);
        assert_eq!(s.eps(), &[4.0, 5.0]);
        // Translated domain entirely inside one layer: homogeneous with its ε.
        let d = boxed(10.0, 11.0, false, false);
        let s = Stack::effective(1.0, &[layer(5.0, 3.0), layer(20.0, 7.0)], &d);
        assert_eq!(s, Stack::homogeneous(7.0));
        // Reflecting top face on a layer top: the interior label is mirrored.
        let d = boxed(0.0, 1.0, false, true);
        let s = Stack::effective(4.0, &[layer(0.5, 4.0), layer(1.0, 5.0)], &d);
        assert_eq!(s.interfaces(), &[0.5, 1.5]);
        assert_eq!(s.eps(), &[4.0, 5.0, 4.0]);
    }

    #[test]
    fn substrate_below_the_box_keeps_a_reflecting_bottom() {
        let mut faces = [FaceKind::Ground; 6];
        faces[Face::new(Axis::Z, Side::Min).index()] = FaceKind::Reflect;
        let b = Aabb::new([0.0, 0.0, 1.0], [1.0, 1.0, 10.0]).unwrap();
        let d = BoundaryModel::new(&Boundary::Box { bounds: b, faces }, Some(0.0), b);
        let s = Stack::effective(1.0, &[layer(2.0, 4.0)], &d);
        assert_eq!(s.interfaces(), &[0.0, 2.0]);
        assert_eq!(s.eps(), &[1.0, 4.0, 1.0]);
    }

    #[test]
    fn substrate_drops_lower_interfaces() {
        let b = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 10.0]).unwrap();
        let d = BoundaryModel::new(&Boundary::Open, Some(1.0), b);
        let s = Stack::effective(1.0, &[layer(1.0, 3.9), layer(4.0, 4.5)], &d);
        assert_eq!(s.interfaces(), &[4.0]);
        assert_eq!(s.eps(), &[4.5, 1.0]);
    }

    #[test]
    fn reflecting_faces_mirror_the_medium() {
        // Interface at 2 in [0, 10], reflecting top: image at 18.
        let d = boxed(0.0, 10.0, false, true);
        let s = Stack::effective(1.0, &[layer(2.0, 4.0)], &d);
        assert_eq!(s.interfaces(), &[2.0, 18.0]);
        assert_eq!(s.eps(), &[4.0, 1.0, 4.0]);
        // An interface on the reflecting face disappears.
        let s = Stack::effective(1.0, &[layer(10.0, 4.0)], &d);
        assert!(s.is_homogeneous());
        assert_eq!(s.eps(), &[4.0]);
        // Both faces reflecting: single and double images.
        let d = boxed(0.0, 10.0, true, true);
        let s = Stack::effective(1.0, &[layer(2.0, 4.0)], &d);
        assert_eq!(s.interfaces(), &[-18.0, -2.0, 2.0, 18.0, 22.0]);
        assert_eq!(s.eps(), &[4.0, 1.0, 4.0, 1.0, 4.0, 1.0]);
    }
}
