//! Conductor geometry: boxes grouped into named nets.

use std::collections::HashMap;

use faradice_core::{BoxId, NetId};

use crate::{Aabb, GeomError, GridOptions, Result, UniformGrid};

/// Incrementally collects nets and boxes, then validates them into a [`Geometry`].
#[derive(Debug, Default, Clone)]
pub struct GeometryBuilder {
    net_names: Vec<String>,
    name_to_id: HashMap<String, NetId>,
    boxes: Vec<Aabb>,
    box_net: Vec<NetId>,
}

impl GeometryBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the id of net `name`, creating the net if it does not exist yet.
    pub fn net(&mut self, name: &str) -> Result<NetId> {
        if let Some(&id) = self.name_to_id.get(name) {
            return Ok(id);
        }
        if faradice_core::is_reserved_label(name) {
            return Err(GeomError::ReservedNetName(name.to_owned()));
        }
        let id = NetId::from_index(self.net_names.len())?;
        self.net_names.push(name.to_owned());
        self.name_to_id.insert(name.to_owned(), id);
        Ok(id)
    }

    /// Adds a box to `net`.
    ///
    /// # Panics
    ///
    /// Panics if `net` was not created by this builder.
    pub fn add_box(&mut self, net: NetId, aabb: Aabb) -> Result<BoxId> {
        assert!(net.index() < self.net_names.len(), "unknown {net}");
        let id = BoxId::from_index(self.boxes.len())?;
        self.boxes.push(aabb);
        self.box_net.push(net);
        Ok(id)
    }

    /// Validates and freezes the geometry.
    ///
    /// Boxes of the same net may overlap or touch (vias, abutting shapes).
    /// Boxes of different nets must be separated by a positive gap; otherwise
    /// [`GeomError::NetsShorted`] is returned.
    pub fn build(self) -> Result<Geometry> {
        if self.boxes.is_empty() {
            return Err(GeomError::Empty);
        }
        let mut net_boxes: Vec<Vec<BoxId>> = vec![Vec::new(); self.net_names.len()];
        for (i, net) in self.box_net.iter().enumerate() {
            net_boxes[net.index()].push(BoxId::from_index(i)?);
        }
        if let Some(i) = net_boxes.iter().position(Vec::is_empty) {
            return Err(GeomError::EmptyNet(self.net_names[i].clone()));
        }
        check_shorts(&self.boxes, &self.box_net, &self.net_names)?;
        let bounds = self
            .boxes
            .iter()
            .skip(1)
            .fold(self.boxes[0], |acc, b| acc.union(b));
        Ok(Geometry {
            net_names: self.net_names,
            name_to_id: self.name_to_id,
            boxes: self.boxes,
            box_net: self.box_net,
            net_boxes,
            bounds,
        })
    }
}

/// Finds touching/overlapping boxes of different nets.
///
/// Broad phase: a uniform grid. Two closed boxes that intersect share a point
/// `p`, and both are registered in the cell containing `p`, so every
/// intersecting pair shares at least one cell. The cost is the sum of squared
/// cell occupancies rather than O(n²) for long parallel wires.
fn check_shorts(boxes: &[Aabb], box_net: &[NetId], names: &[String]) -> Result<()> {
    // Occupancy only: no distance transform, and a cell budget proportional to the
    // box count so sparse layouts do not allocate huge empty grids.
    let opts = GridOptions {
        max_cells: boxes.len().saturating_mul(8).max(64),
        clearance_transform: false,
        ..GridOptions::default()
    };
    let grid = UniformGrid::from_boxes(boxes, opts);
    let mut first: Option<(usize, usize)> = None;
    for items in grid.cell_items() {
        for (k, &i) in items.iter().enumerate() {
            for &j in &items[k + 1..] {
                let (i, j) = (i as usize, j as usize);
                if box_net[i] != box_net[j] && boxes[i].intersects_closed(&boxes[j]) {
                    let pair = (i.min(j), i.max(j));
                    if first.is_none_or(|f| pair < f) {
                        first = Some(pair);
                    }
                }
            }
        }
    }
    if let Some((a, b)) = first {
        return Err(GeomError::NetsShorted {
            a: BoxId::from_index(a)?,
            net_a: names[box_net[a].index()].clone(),
            b: BoxId::from_index(b)?,
            net_b: names[box_net[b].index()].clone(),
        });
    }
    Ok(())
}

/// Validated conductor geometry.
#[derive(Debug, Clone)]
pub struct Geometry {
    net_names: Vec<String>,
    name_to_id: HashMap<String, NetId>,
    boxes: Vec<Aabb>,
    box_net: Vec<NetId>,
    net_boxes: Vec<Vec<BoxId>>,
    bounds: Aabb,
}

impl Geometry {
    /// All boxes, indexed by [`BoxId`].
    #[must_use]
    pub fn boxes(&self) -> &[Aabb] {
        &self.boxes
    }

    /// The box with id `id`.
    #[must_use]
    pub fn box_(&self, id: BoxId) -> &Aabb {
        &self.boxes[id.index()]
    }

    /// The net that box `id` belongs to.
    #[must_use]
    pub fn box_net(&self, id: BoxId) -> NetId {
        self.box_net[id.index()]
    }

    /// Number of nets.
    #[must_use]
    pub fn num_nets(&self) -> usize {
        self.net_names.len()
    }

    /// All net ids in order.
    pub fn nets(&self) -> impl ExactSizeIterator<Item = NetId> + '_ {
        // Ids were created with `from_index`, so every index fits into u32.
        #[allow(clippy::cast_possible_truncation)]
        (0..self.net_names.len()).map(|i| NetId::new(i as u32))
    }

    /// Name of net `id`.
    #[must_use]
    pub fn net_name(&self, id: NetId) -> &str {
        &self.net_names[id.index()]
    }

    /// Looks up a net by name.
    #[must_use]
    pub fn net_id(&self, name: &str) -> Option<NetId> {
        self.name_to_id.get(name).copied()
    }

    /// Boxes belonging to net `id`.
    #[must_use]
    pub fn net_boxes(&self, id: NetId) -> &[BoxId] {
        &self.net_boxes[id.index()]
    }

    /// Bounding box of all conductors.
    #[must_use]
    pub const fn bounds(&self) -> Aabb {
        self.bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(c: [f64; 6]) -> Aabb {
        Aabb::from_array(c).unwrap()
    }

    #[test]
    fn build_and_query() {
        let mut g = GeometryBuilder::new();
        let a = g.net("A").unwrap();
        let bnet = g.net("B").unwrap();
        assert_eq!(g.net("A").unwrap(), a);
        g.add_box(a, b([0.0, 0.0, 0.0, 10.0, 1.0, 1.0])).unwrap();
        g.add_box(a, b([9.0, 0.0, 0.0, 10.0, 5.0, 1.0])).unwrap(); // overlaps same net: ok
        g.add_box(bnet, b([0.0, 2.0, 0.0, 8.0, 3.0, 1.0])).unwrap();
        let geom = g.build().unwrap();
        assert_eq!(geom.num_nets(), 2);
        assert_eq!(geom.net_boxes(a).len(), 2);
        assert_eq!(geom.net_id("B"), Some(bnet));
        assert_eq!(geom.net_name(bnet), "B");
        assert_eq!(geom.bounds().max(), [10.0, 5.0, 1.0]);
        assert_eq!(geom.nets().count(), 2);
    }

    #[test]
    fn reserved_net_names_are_refused() {
        let mut g = GeometryBuilder::new();
        assert!(matches!(g.net("<GND>"), Err(GeomError::ReservedNetName(_))));
        assert!(matches!(
            g.net("<substrate>"),
            Err(GeomError::ReservedNetName(_))
        ));
        for ok in ["GND", "GND:substrate", "bus<3>", "<a"] {
            g.net(ok).unwrap();
        }
    }

    #[test]
    fn touching_different_nets_is_a_short() {
        let mut g = GeometryBuilder::new();
        let a = g.net("A").unwrap();
        let bnet = g.net("B").unwrap();
        g.add_box(a, b([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
        g.add_box(bnet, b([1.0, 0.0, 0.0, 2.0, 1.0, 1.0])).unwrap();
        assert!(matches!(g.build(), Err(GeomError::NetsShorted { .. })));
    }

    #[test]
    fn many_parallel_wires_build_quickly() {
        // 5k separated wires sharing the same x interval: an O(n²) sweep needs ~1.25e7 tests.
        let mut g = GeometryBuilder::new();
        for i in 0..5_000 {
            let net = g.net(&format!("n{i}")).unwrap();
            let y = f64::from(i) * 0.28;
            g.add_box(net, b([0.0, y, 0.0, 100.0, y + 0.14, 0.36]))
                .unwrap();
        }
        assert_eq!(g.build().unwrap().num_nets(), 5_000);
    }

    #[test]
    fn sparse_geometry_builds_quickly() {
        let mut g = GeometryBuilder::new();
        let a = g.net("A").unwrap();
        let c = g.net("B").unwrap();
        g.add_box(a, b([0.0, 0.0, 0.0, 0.14, 0.14, 0.36])).unwrap();
        g.add_box(c, b([1000.0, 1000.0, 10.0, 1000.14, 1000.14, 10.36]))
            .unwrap();
        let t = std::time::Instant::now();
        g.build().unwrap();
        assert!(t.elapsed().as_secs_f64() < 0.5);
    }

    #[test]
    fn empty_inputs_are_rejected() {
        assert!(matches!(
            GeometryBuilder::new().build(),
            Err(GeomError::Empty)
        ));
        let mut g = GeometryBuilder::new();
        let a = g.net("A").unwrap();
        g.net("unused").unwrap();
        g.add_box(a, b([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
        assert_eq!(g.build().unwrap_err(), GeomError::EmptyNet("unused".into()));
    }
}
