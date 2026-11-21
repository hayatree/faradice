//! The non-stratified part of the medium: material boxes and their interface
//! faces (design M2c §3).
//!
//! [`BoxMedium::build`] turns [`DielectricRegions`] into disjoint material
//! boxes: conformal shells (grown conductor boxes minus those boxes) in rule
//! order, then explicit boxes on top. ε(r) is the box containing r, else the
//! stratified stack. Reflecting faces of the domain mirror the whole medium:
//! lookups fold the point into the domain first.
//!
//! The **interface faces** are the parts of box faces across which ε jumps.
//! They are found per plane: the box faces on the plane are overlaid (and, if
//! vertical, split at the stack heights), and each piece is kept if the
//! materials on its two sides differ. Faces lying on a reflecting face vanish
//! (the box continues across it).
//!
//! The indices hold the faces and their single reflections across each
//! reflecting domain face. For cubes of half-side up to [`BoxMedium::reach`]
//! centred in the domain this is complete for what the walk asks: the L∞
//! distance to the nearest face of each normal, and the plane coordinates of
//! the faces normal to an axis. Composite images (reflected across faces of two
//! different axes) are not indexed: each lies in the plane of a single image
//! and is never nearer than it, because a reflection across a face that the
//! centre lies inside of only moves a face away from the centre.

use std::collections::BTreeMap;

use faradice_core::{Axis, BoxId};
use faradice_geom::{
    Aabb, Geometry, GridOptions, Paint, PaintItem, Painted, Rect, RectIndex, SpatialIndex,
    UniformGrid, paint,
};

use crate::boundary::BoundaryModel;
use crate::regions::DielectricRegions;
use crate::stack::Stack;

/// Material boxes with their interface faces (see the module docs).
#[derive(Debug, Clone)]
pub struct BoxMedium {
    /// Disjoint material boxes (clipped to the domain) and their ε.
    boxes: Vec<(Aabb, f64)>,
    volume: UniformGrid,
    /// Interface faces (with single reflections) per normal axis.
    faces: [RectIndex; 3],
    /// Per indexed face: ε below and above its plane (along its normal).
    sides: [Vec<[f64; 2]>; 3],
    /// Interface faces inside the domain, without reflections (reports, tests).
    own_faces: usize,
    /// Per axis: the first `own[axis]` indexed faces are the faces inside the
    /// domain (the rest are reflections).
    own: [usize; 3],
    /// Volume where materials of different ε were painted over each other.
    overlap_volume: f64,
    /// Largest cube half-side for which single reflections suffice.
    reach: f64,
    domain: BoundaryModel,
    stack: Stack,
}

impl BoxMedium {
    /// Builds the medium. Returns `None` if `regions` adds nothing inside the
    /// domain.
    ///
    /// `regions` must be valid for `geom` ([`DielectricRegions::validate`]).
    #[must_use]
    pub fn build(
        regions: &DielectricRegions,
        geom: &Geometry,
        stack: &Stack,
        domain: &BoundaryModel,
    ) -> Option<Self> {
        let mut materials = Materials::default();
        let items = region_items(regions, &mut materials, |b| vec![geom.boxes()[b.index()]]);
        let painted = paint(&items);
        Self::assemble(&painted, &materials.0, stack, domain)
    }

    /// The medium `M₁` of design D3 §4.4c: the medium of the geometry in
    /// which conductor box `edited` is replaced by `with` (all inside it, and
    /// tagged like it), **inside** the nominal box, and the nominal medium
    /// outside it. Returns `None` if the result adds nothing to the stack.
    ///
    /// `regions` must be valid for `geom` ([`DielectricRegions::validate`]).
    #[must_use]
    pub fn edited_inside(
        regions: &DielectricRegions,
        geom: &Geometry,
        edited: BoxId,
        with: &[Aabb],
        stack: &Stack,
        domain: &BoundaryModel,
    ) -> Option<Self> {
        let mut materials = Materials::default();
        let nominal = paint(&region_items(regions, &mut materials, |b| {
            vec![geom.boxes()[b.index()]]
        }));
        let after = paint(&region_items(regions, &mut materials, |b| {
            if b == edited {
                with.to_vec()
            } else {
                vec![geom.boxes()[b.index()]]
            }
        }));
        let inside = geom.boxes()[edited.index()];
        let mut items: Vec<PaintItem> = nominal
            .boxes
            .iter()
            .map(|&(b, l)| PaintItem {
                region: b,
                paint: Paint::Set(l),
            })
            .collect();
        items.push(PaintItem {
            region: inside,
            paint: Paint::Clear,
        });
        items.extend(after.boxes.iter().filter_map(|&(b, l)| {
            Some(PaintItem {
                region: intersect(&b, &inside)?,
                paint: Paint::Set(l),
            })
        }));
        let mut painted = paint(&items);
        painted.conflict_volume = nominal.conflict_volume;
        Self::assemble(&painted, &materials.0, stack, domain)
    }

    /// Clips painted material boxes to the domain and builds their faces.
    fn assemble(
        painted: &Painted,
        materials: &[f64],
        stack: &Stack,
        domain: &BoundaryModel,
    ) -> Option<Self> {
        let clip = clip_region(domain);
        let boxes: Vec<(Aabb, f64)> = painted
            .boxes
            .iter()
            .filter_map(|&(b, l)| {
                let b = match &clip {
                    Some(c) => intersect(&b, c)?,
                    None => b,
                };
                Some((b, materials[l as usize]))
            })
            .collect();
        if boxes.is_empty() {
            return None;
        }
        let regions: Vec<Aabb> = boxes.iter().map(|&(b, _)| b).collect();
        // Coarser than the conductor default, as for the face indices
        // (`faradice_geom::RECT_CELL_FACTOR`): material boxes are thin.
        let volume = UniformGrid::from_boxes(
            &regions,
            GridOptions {
                cell_factor: faradice_geom::RECT_CELL_FACTOR,
                ..GridOptions::default()
            },
        );
        let mut medium = Self {
            boxes,
            volume,
            faces: [
                RectIndex::new(Vec::new()),
                RectIndex::new(Vec::new()),
                RectIndex::new(Vec::new()),
            ],
            sides: [Vec::new(), Vec::new(), Vec::new()],
            own_faces: 0,
            own: [0; 3],
            overlap_volume: painted.conflict_volume,
            reach: reach(domain),
            domain: domain.clone(),
            stack: stack.clone(),
        };
        let mut own: Vec<(Rect, [f64; 2])> = Vec::new();
        for axis in Axis::ALL {
            own.extend(medium.interface_faces(axis));
        }
        medium.own_faces = own.len();
        let mut all: [Vec<(Rect, [f64; 2])>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for f in &own {
            all[f.0.normal().index()].push(*f);
        }
        medium.own = [all[0].len(), all[1].len(), all[2].len()];
        if let Some(bounds) = domain.bounds() {
            for (b, &(rmin, rmax)) in domain.reflect().iter().enumerate() {
                for (on, plane) in [(rmin, bounds.min()[b]), (rmax, bounds.max()[b])] {
                    if on {
                        for (f, e) in &own {
                            // A reflection across the face's own axis swaps its sides.
                            let e = if f.normal().index() == b {
                                [e[1], e[0]]
                            } else {
                                *e
                            };
                            all[f.normal().index()].push((reflected(f, b, plane), e));
                        }
                    }
                }
            }
        }
        medium.sides = all.each_ref().map(|v| v.iter().map(|&(_, e)| e).collect());
        medium.faces = all.map(|v| RectIndex::new(v.into_iter().map(|(f, _)| f).collect()));
        Some(medium)
    }

    /// The disjoint material boxes and their ε (clipped to the domain).
    #[must_use]
    pub fn boxes(&self) -> &[(Aabb, f64)] {
        &self.boxes
    }

    /// Interface faces normal to `axis`, including their single reflections
    /// across reflecting domain faces (complete for distances and plane
    /// coordinates within [`reach`](Self::reach), see the module docs).
    #[must_use]
    pub fn faces(&self, axis: Axis) -> &RectIndex {
        &self.faces[axis.index()]
    }

    /// `true` if the closed rectangle (or box) `[lo, hi]` meets an interface
    /// face inside the domain (closed: touching counts).
    #[must_use]
    pub fn touches(&self, lo: [f64; 3], hi: [f64; 3]) -> bool {
        let c: [f64; 3] = std::array::from_fn(|i| 0.5 * (lo[i] + hi[i]));
        let r = (0..3).map(|i| 0.5 * (hi[i] - lo[i])).fold(0.0, f64::max);
        let meets = |f: &Rect| (0..3).all(|i| f.min()[i] <= hi[i] && lo[i] <= f.max()[i]);
        Axis::ALL.iter().any(|&axis| {
            let index = &self.faces[axis.index()];
            let own = self.own[axis.index()];
            // Faces within the L∞ ball of the rectangle's half-extent, one at a
            // time (most rectangles have none).
            let mut seen: Vec<u32> = Vec::new();
            loop {
                let Some(n) =
                    index.nearest_linf_where(c, r, |id| (id as usize) < own && !seen.contains(&id))
                else {
                    return false;
                };
                let id = n.box_id.index();
                if meets(&index.rects()[id]) {
                    return true;
                }
                #[allow(clippy::cast_possible_truncation)]
                seen.push(id as u32);
            }
        })
    }

    /// ε below and above (along `axis`) of the indexed face `id` of
    /// [`faces`](Self::faces).
    #[must_use]
    pub fn face_sides(&self, axis: Axis, id: u32) -> [f64; 2] {
        self.sides[axis.index()][id as usize]
    }

    /// Number of interface faces inside the domain (without reflections).
    #[must_use]
    pub const fn num_faces(&self) -> usize {
        self.own_faces
    }

    /// Volume where materials of different ε overlapped in the input (the
    /// later one won).
    #[must_use]
    pub const fn overlap_volume(&self) -> f64 {
        self.overlap_volume
    }

    /// Largest cube half-side for which the indexed faces (with single
    /// reflections) are complete: the smallest domain width across a pair of
    /// axes with a reflecting face (`∞` without reflecting faces).
    #[must_use]
    pub const fn reach(&self) -> f64 {
        self.reach
    }

    /// Relative permittivity at `p` (folded into the domain first).
    #[must_use]
    pub fn eps_at(&self, mut p: [f64; 3]) -> f64 {
        self.domain.fold(&mut p);
        self.box_eps(p).unwrap_or_else(|| self.stack.eps_at(p[2]))
    }

    /// L∞ distance from `p` to the nearest material box (`0` inside one),
    /// or `None` beyond `max_r`.
    #[must_use]
    pub fn volume_distance(&self, p: [f64; 3], max_r: f64) -> Option<f64> {
        self.volume.nearest_linf(p, max_r).map(|n| n.distance)
    }

    /// The material boxes within L∞ distance `radius` of `centre`, for
    /// repeated exact [`Self::volume_distance`] queries near it.
    pub fn volume_local(&self, centre: [f64; 3], radius: f64, out: &mut faradice_geom::LocalBoxes) {
        self.volume.local(centre, radius, out);
    }

    /// ε of a material box containing `p` (closed, after folding), if any.
    pub(crate) fn box_eps_closed(&self, mut p: [f64; 3]) -> Option<f64> {
        self.domain.fold(&mut p);
        self.box_eps(p)
    }

    /// Largest L∞ distance from `p` to the boundary of a material box piece
    /// containing it (`0` if none does): a cube of that half-side at `p` lies
    /// in one material, whatever the stack does there.
    pub(crate) fn inner_clearance(&self, p: [f64; 3]) -> f64 {
        self.volume
            .items_at(p)
            .iter()
            .map(|&i| self.boxes[i as usize].0)
            .filter(|b| b.contains(p))
            .map(|b| {
                (0..3)
                    .map(|i| (p[i] - b.min()[i]).min(b.max()[i] - p[i]))
                    .fold(f64::INFINITY, f64::min)
            })
            .fold(0.0, f64::max)
    }

    fn box_eps(&self, p: [f64; 3]) -> Option<f64> {
        self.volume
            .items_at(p)
            .iter()
            .map(|&i| self.boxes[i as usize])
            .find(|(b, _)| b.contains(p))
            .map(|(_, e)| e)
    }

    /// Interface faces normal to `axis` inside the domain (see the module docs).
    ///
    /// Classification is symbolic: on a plane, the material on each side of a
    /// piece is the box whose face (on that side) covers the piece, else the
    /// stack. A box crossing the plane cannot cover a piece of another box's
    /// face, because boxes are disjoint, so the boxes with a face on the plane
    /// are the only candidates.
    fn interface_faces(&self, axis: Axis) -> Vec<(Rect, [f64; 2])> {
        let a = axis.index();
        let (u, v) = match axis {
            Axis::X => (1, 2),
            Axis::Y => (0, 2),
            Axis::Z => (0, 1),
        };
        // Faces by plane: (u0, u1, v0, v1, ε, the box lies above the plane).
        let mut planes: BTreeMap<u64, Vec<PlaneRect>> = BTreeMap::new();
        for &(b, eps) in &self.boxes {
            let r = |above| PlaneRect {
                u: [b.min()[u], b.max()[u]],
                v: [b.min()[v], b.max()[v]],
                eps,
                above,
            };
            planes
                .entry(order_key(b.min()[a]))
                .or_default()
                .push(r(true));
            planes
                .entry(order_key(b.max()[a]))
                .or_default()
                .push(r(false));
        }
        let (rmin, rmax) = self.domain.reflect()[a];
        let bounds = self.domain.bounds();
        let mut out = Vec::new();
        for (key, rects) in planes {
            let c = from_order_key(key);
            // A reflecting face mirrors the inner side onto the outer side.
            let mirror = bounds.is_some_and(|b| {
                (rmin && c.to_bits() == b.min()[a].to_bits())
                    || (rmax && c.to_bits() == b.max()[a].to_bits())
            });
            let stack_side = |above: bool, v0: f64| -> f64 {
                if axis == Axis::Z {
                    if above {
                        self.stack.eps_at(c)
                    } else {
                        self.stack.eps_below(c)
                    }
                } else {
                    // v is z, and no stack height lies inside the cell.
                    self.stack.eps_at(v0)
                }
            };
            let mut pairs: Vec<[f64; 2]> = Vec::new();
            let cells = self.plane_cells(&rects, v == 2, |v0, below, above| {
                let lo = below.unwrap_or_else(|| stack_side(false, v0));
                let hi = if mirror {
                    lo
                } else {
                    above.unwrap_or_else(|| stack_side(true, v0))
                };
                (lo.to_bits() != hi.to_bits()).then(|| {
                    let pair = [lo, hi];
                    let k = pairs
                        .iter()
                        .position(|q| q.map(f64::to_bits) == pair.map(f64::to_bits))
                        .unwrap_or_else(|| {
                            pairs.push(pair);
                            pairs.len() - 1
                        });
                    u32::try_from(k).expect("few material pairs")
                })
            });
            // Merge the cells with the same side materials into rectangles.
            for (b, k) in paint(&cells).boxes {
                let (mut lo, mut hi) = ([c; 3], [c; 3]);
                lo[u] = b.min()[0];
                hi[u] = b.max()[0];
                lo[v] = b.min()[1];
                hi[v] = b.max()[1];
                out.push((
                    Rect::new(lo, hi).expect("merged faces are rectangles"),
                    pairs[k as usize],
                ));
            }
        }
        out
    }

    /// Cells of a plane's arrangement where `jump(v0, ε below, ε above)` returns a
    /// label (the pair of side materials),
    /// as unit-thickness boxes in (u, v) for merging. A sweep over u keeps the
    /// rectangles crossing each u slab; within a slab, the v intervals of the
    /// two sides (each disjoint) are overlaid, split at the stack heights if
    /// `v_is_z`.
    fn plane_cells(
        &self,
        rects: &[PlaneRect],
        v_is_z: bool,
        mut jump: impl FnMut(f64, Option<f64>, Option<f64>) -> Option<u32>,
    ) -> Vec<PaintItem> {
        let mut us: Vec<f64> = rects.iter().flat_map(|r| r.u).collect();
        us.sort_by(f64::total_cmp);
        us.dedup();
        let mut order: Vec<usize> = (0..rects.len()).collect();
        order.sort_by(|&i, &j| rects[i].u[0].total_cmp(&rects[j].u[0]));
        let mut next = 0;
        let mut active: Vec<usize> = Vec::new();
        let mut cells = Vec::new();
        for w in us.windows(2) {
            let (u0, u1) = (w[0], w[1]);
            while next < order.len() && rects[order[next]].u[0] <= u0 {
                active.push(order[next]);
                next += 1;
            }
            active.retain(|&i| rects[i].u[1] > u0);
            if active.is_empty() {
                continue;
            }
            let mut vs: Vec<f64> = active.iter().flat_map(|&i| rects[i].v).collect();
            let (vmin, vmax) = vs
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                    (a.min(x), b.max(x))
                });
            if v_is_z {
                vs.extend(
                    self.stack
                        .interfaces()
                        .iter()
                        .copied()
                        .filter(|&z| vmin < z && z < vmax),
                );
            }
            vs.sort_by(f64::total_cmp);
            vs.dedup();
            // Each side's faces are disjoint: sort them by v and walk along.
            let side = |above: bool| {
                let mut ivs: Vec<([f64; 2], f64)> = active
                    .iter()
                    .map(|&i| &rects[i])
                    .filter(|r| r.above == above)
                    .map(|r| (r.v, r.eps))
                    .collect();
                ivs.sort_by(|x, y| x.0[0].total_cmp(&y.0[0]));
                ivs
            };
            let (lower, upper) = (side(false), side(true));
            let (mut il, mut iu) = (0, 0);
            let cover = |ivs: &[([f64; 2], f64)], k: &mut usize, v0: f64, v1: f64| {
                while *k < ivs.len() && ivs[*k].0[1] <= v0 {
                    *k += 1;
                }
                ivs.get(*k)
                    .filter(|iv| iv.0[0] <= v0 && v1 <= iv.0[1])
                    .map(|iv| iv.1)
            };
            for wv in vs.windows(2) {
                let (v0, v1) = (wv[0], wv[1]);
                let below = cover(&lower, &mut il, v0, v1);
                let above = cover(&upper, &mut iu, v0, v1);
                let label = if below.is_some() || above.is_some() {
                    jump(v0, below, above)
                } else {
                    None
                };
                if let Some(label) = label {
                    cells.push(PaintItem {
                        region: Aabb::new([u0, v0, 0.0], [u1, v1, 1.0])
                            .expect("cells have positive extent"),
                        paint: Paint::Set(label),
                    });
                }
            }
        }
        cells
    }
}

/// A box face on a plane, in the plane's (u, v) coordinates.
#[derive(Debug, Clone, Copy)]
struct PlaneRect {
    u: [f64; 2],
    v: [f64; 2],
    eps: f64,
    /// The box lies on the upper side of the plane.
    above: bool,
}

/// `b` grown laterally by `side` and upwards by `top`.
/// Material labels by ε (bitwise), in order of first use.
#[derive(Default)]
struct Materials(Vec<f64>);

impl Materials {
    fn label(&mut self, eps: f64) -> u32 {
        let i = self
            .0
            .iter()
            .position(|e| e.to_bits() == eps.to_bits())
            .unwrap_or_else(|| {
                self.0.push(eps);
                self.0.len() - 1
            });
        u32::try_from(i).expect("few materials")
    }
}

/// Paint items of `regions` (M2c §6.2): per rule its shells (grown boxes
/// minus the rule's own boxes) in rule order, then the explicit boxes.
/// `own(b)` gives the conductor boxes that a tagged box id stands for.
fn region_items(
    regions: &DielectricRegions,
    materials: &mut Materials,
    own: impl Fn(BoxId) -> Vec<Aabb>,
) -> Vec<PaintItem> {
    let mut items = Vec::new();
    for r in regions.conformal.iter().filter(|r| !r.boxes.is_empty()) {
        let l = materials.label(r.eps_r);
        let own: Vec<Aabb> = r.boxes.iter().flat_map(|&b| own(b)).collect();
        let mut shell_items: Vec<PaintItem> = own
            .iter()
            .map(|b| PaintItem {
                region: grown(b, r.side, r.top),
                paint: Paint::Set(l),
            })
            .collect();
        shell_items.extend(own.iter().map(|&b| PaintItem {
            region: b,
            paint: Paint::Clear,
        }));
        items.extend(
            paint(&shell_items)
                .boxes
                .into_iter()
                .map(|(b, l)| PaintItem {
                    region: b,
                    paint: Paint::Set(l),
                }),
        );
    }
    for b in &regions.boxes {
        let l = materials.label(b.eps_r);
        items.push(PaintItem {
            region: b.region,
            paint: Paint::Set(l),
        });
    }
    items
}

pub(crate) fn grown(b: &Aabb, side: f64, top: f64) -> Aabb {
    let (lo, hi) = (b.min(), b.max());
    Aabb::new(
        [lo[0] - side, lo[1] - side, lo[2]],
        [hi[0] + side, hi[1] + side, hi[2] + top],
    )
    .expect("growing keeps a box valid")
}

/// The region that holds the medium: the domain box above the substrate.
fn clip_region(domain: &BoundaryModel) -> Option<Aabb> {
    let b = domain.bounds()?;
    let mut lo = b.min();
    if let Some(z0) = domain.substrate_z() {
        lo[2] = lo[2].max(z0);
    }
    Aabb::new(lo, b.max()).ok()
}

fn intersect(a: &Aabb, b: &Aabb) -> Option<Aabb> {
    let lo: [f64; 3] = std::array::from_fn(|i| a.min()[i].max(b.min()[i]));
    let hi: [f64; 3] = std::array::from_fn(|i| a.max()[i].min(b.max()[i]));
    Aabb::new(lo, hi).ok()
}

/// Smallest domain width along an axis with a reflecting face.
fn reach(domain: &BoundaryModel) -> f64 {
    let Some(b) = domain.bounds() else {
        return f64::INFINITY;
    };
    (0..3)
        .filter(|&a| {
            let (lo, hi) = domain.reflect()[a];
            lo || hi
        })
        .map(|a| b.max()[a] - b.min()[a])
        .fold(f64::INFINITY, f64::min)
}

/// Mirror image of `f` across the plane `axis b = plane`.
fn reflected(f: &Rect, b: usize, plane: f64) -> Rect {
    let (mut lo, mut hi) = (f.min(), f.max());
    lo[b] = 2.0 * plane - f.max()[b];
    hi[b] = 2.0 * plane - f.min()[b];
    Rect::new(lo, hi).expect("reflection keeps a rectangle")
}

/// Order-preserving key of a finite `f64` (for grouping faces by plane).
fn order_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

fn from_order_key(k: u64) -> f64 {
    f64::from_bits(if k >> 63 == 1 { k & !(1 << 63) } else { !k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::{Boundary, FaceKind};
    use crate::regions::{ConformalRule, DielectricBox};
    use crate::stack::DielectricLayer;
    use faradice_core::{BoxId, Face, Side};
    use faradice_geom::GeometryBuilder;

    fn domain(reflect_x: bool) -> BoundaryModel {
        let bounds = Aabb::new([-4.0, -4.0, 0.0], [4.0, 4.0, 4.0]).unwrap();
        let mut faces = [FaceKind::Ground; 6];
        if reflect_x {
            faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Reflect;
        }
        BoundaryModel::new(&Boundary::Box { bounds, faces }, Some(0.0), bounds)
    }

    fn wire() -> Geometry {
        let mut g = GeometryBuilder::new();
        let n = g.net("a").unwrap();
        g.add_box(
            n,
            Aabb::from_array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.5]).unwrap(),
        )
        .unwrap();
        g.build().unwrap()
    }

    fn stack(d: &BoundaryModel) -> Stack {
        // Interface at the wire bottom: 4.05 below, 4.5 above.
        Stack::effective(
            4.5,
            &[DielectricLayer {
                z_top: 1.0,
                eps_r: 4.05,
            }],
            d,
        )
    }

    fn shell(side: f64, top: f64) -> DielectricRegions {
        DielectricRegions {
            boxes: Vec::new(),
            conformal: vec![ConformalRule {
                layer: "met1".into(),
                boxes: vec![BoxId::new(0)],
                side,
                top,
                eps_r: 3.5,
            }],
        }
    }

    fn faces_of(m: &BoxMedium, axis: Axis) -> Vec<Rect> {
        m.interface_faces(axis)
            .into_iter()
            .map(|(f, _)| f)
            .collect()
    }

    #[test]
    fn order_keys_round_trip_and_sort() {
        let xs = [-3.5, -0.0, 0.0, 1e-300, 2.25, 7.0];
        for w in xs.windows(2) {
            assert!(order_key(w[0]) <= order_key(w[1]));
        }
        for x in xs {
            assert_eq!(from_order_key(order_key(x)).to_bits(), x.to_bits());
        }
    }

    #[test]
    fn sidewall_shell_faces() {
        let d = domain(false);
        let m = BoxMedium::build(&shell(0.25, 0.0), &wire(), &stack(&d), &d).unwrap();
        assert_eq!(m.eps_at([1.1, 0.0, 1.2]), 3.5);
        assert_eq!(m.eps_at([1.3, 0.0, 1.2]), 4.5);
        assert_eq!(m.eps_at([1.1, 0.0, 0.9]), 4.05);
        // Shell volume: (2.5² − 2²) × 0.5.
        let vol: f64 = m.boxes().iter().map(|(b, _)| b.volume()).sum();
        assert!((vol - 1.125).abs() < 1e-12, "{vol}");
        // x faces: outer faces at ±1.25 and inner faces at ±1 (shell against the
        // stack inside the conductor), each spanning z ∈ [1, 1.5].
        let mut xs: Vec<f64> = faces_of(&m, Axis::X).iter().map(Rect::coord).collect();
        xs.sort_by(f64::total_cmp);
        xs.dedup();
        assert_eq!(xs, vec![-1.25, -1.0, 1.0, 1.25]);
        for f in faces_of(&m, Axis::X) {
            assert_eq!((f.min()[2], f.max()[2]), (1.0, 1.5));
        }
        // z faces: the shell bottom (on the stack interface, 3.5 above vs 4.05
        // below) and the shell top (3.5 below vs 4.5 above), ring-shaped.
        let z = faces_of(&m, Axis::Z);
        let area = |c: f64| -> f64 {
            z.iter()
                .filter(|f| f.coord().to_bits() == c.to_bits())
                .map(Rect::area)
                .sum()
        };
        assert!((area(1.0) - 2.25).abs() < 1e-12);
        assert!((area(1.5) - 2.25).abs() < 1e-12);
        assert_eq!(
            m.num_faces(),
            faces_of(&m, Axis::X).len() + faces_of(&m, Axis::Y).len() + z.len()
        );
    }

    #[test]
    fn boxes_matching_the_stack_have_no_faces() {
        // A box of the stack's ε inside one layer, and one spanning the stack
        // interface with the lower layer's ε: only the upper part of the
        // latter's side faces, and its top face, are interfaces.
        let d = domain(false);
        let regions = DielectricRegions {
            boxes: vec![
                DielectricBox {
                    region: Aabb::from_array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]).unwrap(),
                    eps_r: 4.5,
                },
                DielectricBox {
                    region: Aabb::from_array([-3.0, -3.0, 0.5, -2.0, -2.0, 1.5]).unwrap(),
                    eps_r: 4.05,
                },
            ],
            conformal: Vec::new(),
        };
        let m = BoxMedium::build(&regions, &wire(), &stack(&d), &d).unwrap();
        for f in faces_of(&m, Axis::X).iter().chain(&faces_of(&m, Axis::Y)) {
            assert_eq!((f.min()[2], f.max()[2]), (1.0, 1.5), "{f:?}");
            assert!(f.max()[0] <= -2.0 && f.max()[1] <= -2.0);
        }
        let z = faces_of(&m, Axis::Z);
        assert_eq!(z.len(), 1);
        assert_eq!(z[0].coord(), 1.5);
    }

    #[test]
    fn reflecting_faces_continue_boxes() {
        // A box against the reflecting x max face has no face there, and the
        // images of its other x face are indexed.
        let d = domain(true);
        let regions = DielectricRegions {
            boxes: vec![DielectricBox {
                region: Aabb::from_array([3.0, -1.0, 2.0, 5.0, 1.0, 3.0]).unwrap(),
                eps_r: 7.0,
            }],
            conformal: Vec::new(),
        };
        let m = BoxMedium::build(&regions, &wire(), &stack(&d), &d).unwrap();
        // Clipped at the domain face x = 4.
        assert_eq!(m.boxes()[0].0.max()[0], 4.0);
        let own: Vec<f64> = faces_of(&m, Axis::X).iter().map(Rect::coord).collect();
        assert_eq!(own, vec![3.0]);
        // The image of x = 3 across x = 4 is x = 5, at distance 0.5 from x = 4.5.
        let n = m
            .faces(Axis::X)
            .nearest_linf([4.5, 0.0, 2.5], 10.0)
            .unwrap();
        assert!((n.distance - 0.5).abs() < 1e-12);
        assert_eq!(m.eps_at([4.5, 0.0, 2.5]), 7.0);
        assert_eq!(m.reach(), 8.0);
    }

    /// The Lemma of design M2c §3.2 on random box sets in a stack: inside a
    /// cube free of faces normal to the two other axes (stack planes count as
    /// z-faces), ε depends on the a-coordinate only.
    #[test]
    fn lemma_on_random_boxes() {
        let mut s = 0x1234_5678_9abc_def1_u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            #[allow(clippy::cast_precision_loss)]
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            u
        };
        let d = domain(true);
        let st = Stack::effective(
            1.0,
            &[
                DielectricLayer {
                    z_top: 1.0,
                    eps_r: 2.0,
                },
                DielectricLayer {
                    z_top: 2.5,
                    eps_r: 3.0,
                },
            ],
            &d,
        );
        let lattice = |u: f64| (u * 16.0).floor() * 0.5 - 4.0;
        // Samples whose cube contains an interface (the check is not vacuous).
        let mut nontrivial = 0;
        for _ in 0..20 {
            // Disjoint boxes on a lattice (overlaps are painted over).
            let mut boxes = Vec::new();
            for _ in 0..8 {
                let lo = [lattice(next()), lattice(next()), lattice(next()).abs()];
                let hi = [
                    lo[0] + 0.5 + (next() * 4.0).floor() * 0.5,
                    lo[1] + 0.5 + (next() * 4.0).floor() * 0.5,
                    lo[2] + 0.5 + (next() * 4.0).floor() * 0.5,
                ];
                let eps = [1.0, 2.0, 3.0, 5.0][(next() * 4.0) as usize];
                let region = Aabb::new(lo, hi).unwrap();
                if boxes
                    .iter()
                    .all(|b: &DielectricBox| !overlaps(&b.region, &region))
                {
                    boxes.push(DielectricBox { region, eps_r: eps });
                }
            }
            let regions = DielectricRegions {
                boxes,
                conformal: Vec::new(),
            };
            let Some(m) = BoxMedium::build(&regions, &wire(), &st, &d) else {
                continue;
            };
            for _ in 0..200 {
                let p = [next() * 8.0 - 4.0, next() * 8.0 - 4.0, next() * 4.0];
                for axis in Axis::ALL {
                    let a = axis.index();
                    let mut r = m.reach();
                    for b in Axis::ALL.into_iter().filter(|&b| b != axis) {
                        let df = m
                            .faces(b)
                            .nearest_linf(p, f64::INFINITY)
                            .map_or(f64::INFINITY, |n| n.distance);
                        r = r.min(df);
                        if b == Axis::Z {
                            r = r.min(st.nearest_two(p[2]).0);
                        }
                    }
                    if !(r.is_finite() && r > 1e-9) {
                        continue;
                    }
                    let r = r * (1.0 - 1e-9);
                    for _ in 0..20 {
                        let mut q: [f64; 3] =
                            std::array::from_fn(|i| p[i] + r * (2.0 * next() - 1.0));
                        let along = m.eps_at(q);
                        if along.to_bits() != m.eps_at(p).to_bits() {
                            nontrivial += 1;
                        }
                        // Same a-coordinate, transverse coordinates of p.
                        let t = q[a];
                        q = p;
                        q[a] = t;
                        assert_eq!(
                            along.to_bits(),
                            m.eps_at(q).to_bits(),
                            "axis {axis:?} p {p:?} r {r}"
                        );
                    }
                }
            }
        }
        assert!(nontrivial > 1000, "{nontrivial}");
    }

    fn overlaps(a: &Aabb, b: &Aabb) -> bool {
        (0..3).all(|i| a.min()[i] < b.max()[i] && b.min()[i] < a.max()[i])
    }

    fn unit_domain(reflect_x_max: bool) -> BoundaryModel {
        let bounds = Aabb::new([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]).unwrap();
        let mut faces = [FaceKind::Ground; 6];
        if reflect_x_max {
            faces[Face::new(Axis::X, Side::Max).index()] = FaceKind::Reflect;
        }
        BoundaryModel::new(&Boundary::Box { bounds, faces }, None, bounds)
    }

    fn x_planes(m: &BoxMedium) -> Vec<f64> {
        let mut xs: Vec<f64> = faces_of(m, Axis::X).iter().map(Rect::coord).collect();
        xs.sort_by(f64::total_cmp);
        xs.dedup();
        xs
    }

    #[test]
    fn faces_near_a_reflecting_face_are_kept() {
        // A probe-based classification folded the outer probe back into the box.
        let d = unit_domain(true);
        let regions = DielectricRegions {
            boxes: vec![DielectricBox {
                region: Aabb::from_array([0.2, -0.5, 0.2, 0.99, 0.5, 0.8]).unwrap(),
                eps_r: 2.0,
            }],
            conformal: Vec::new(),
        };
        let m = BoxMedium::build(&regions, &wire(), &Stack::homogeneous(1.0), &d).unwrap();
        assert_eq!(x_planes(&m), vec![0.2, 0.99]);
    }

    #[test]
    fn one_ulp_gaps_keep_their_faces() {
        let d = unit_domain(false);
        let up = f64::from_bits(0.5_f64.to_bits() + 1);
        let regions = DielectricRegions {
            boxes: vec![
                DielectricBox {
                    region: Aabb::from_array([0.0, -0.5, 0.2, 0.5, 0.5, 0.8]).unwrap(),
                    eps_r: 2.0,
                },
                DielectricBox {
                    region: Aabb::from_array([up, -0.5, 0.2, 0.75, 0.5, 0.8]).unwrap(),
                    eps_r: 3.0,
                },
            ],
            conformal: Vec::new(),
        };
        let m = BoxMedium::build(&regions, &wire(), &Stack::homogeneous(1.0), &d).unwrap();
        assert_eq!(x_planes(&m), vec![0.0, 0.5, up, 0.75]);
    }

    #[test]
    fn many_coplanar_faces_are_fast() {
        // 4000 disjoint boxes sharing their y and z planes (a bus of shells).
        let d = BoundaryModel::new(
            &Boundary::grounded_box(Aabb::new([-1.0, -1.0, 0.0], [9000.0, 2.0, 2.0]).unwrap()),
            None,
            Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
        );
        let boxes = (0..4000)
            .map(|i| DielectricBox {
                region: Aabb::from_array([
                    2.0 * f64::from(i),
                    0.0,
                    0.5,
                    2.0 * f64::from(i) + 1.0,
                    1.0,
                    1.5,
                ])
                .unwrap(),
                eps_r: 3.5,
            })
            .collect();
        let regions = DielectricRegions {
            boxes,
            conformal: Vec::new(),
        };
        let t = std::time::Instant::now();
        let m = BoxMedium::build(&regions, &wire(), &Stack::homogeneous(4.5), &d).unwrap();
        assert_eq!(m.num_faces(), 6 * 4000);
        assert!(t.elapsed().as_secs_f64() < 20.0, "{:?}", t.elapsed());
    }

    #[test]
    fn nothing_inside_the_domain_is_none() {
        let d = domain(false);
        let regions = DielectricRegions {
            boxes: vec![DielectricBox {
                region: Aabb::from_array([10.0, 10.0, 1.0, 11.0, 11.0, 2.0]).unwrap(),
                eps_r: 7.0,
            }],
            conformal: Vec::new(),
        };
        assert!(BoxMedium::build(&regions, &wire(), &stack(&d), &d).is_none());
    }

    /// D3 §4.4c: `M₁` takes the edited medium inside the nominal box only.
    #[test]
    fn edited_inside_fills_the_vacated_slab() {
        let d = domain(false);
        let (g, st) = (wire(), stack(&d));
        let regions = shell(0.1, 0.0);
        let nominal = BoxMedium::build(&regions, &g, &st, &d).unwrap();
        // y+ face moved in by 0.05: the slab becomes shell, the old shell
        // outside the box stays.
        let y_in = Aabb::from_array([-1.0, -1.0, 1.0, 1.0, 0.95, 1.5]).unwrap();
        let m = BoxMedium::edited_inside(&regions, &g, BoxId::new(0), &[y_in], &st, &d).unwrap();
        assert_eq!(m.eps_at([0.0, 0.97, 1.2]), 3.5);
        assert_eq!(nominal.eps_at([0.0, 0.97, 1.2]), 4.5);
        for p in [
            [0.0, 1.05, 1.2],
            [0.0, 1.12, 1.2],
            [1.05, 0.0, 1.2],
            [0.0, 0.0, 1.6],
        ] {
            assert_eq!(m.eps_at(p), nominal.eps_at(p), "{p:?}");
        }
        // z+ face moved down with top = 0: the edited shell no longer reaches
        // the old top, so the slab takes the stack above the wire.
        let z_in = Aabb::from_array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.45]).unwrap();
        let m = BoxMedium::edited_inside(&regions, &g, BoxId::new(0), &[z_in], &st, &d).unwrap();
        assert_eq!(m.eps_at([0.0, 0.0, 1.47]), 4.5);
        // The ring of the side shell keeps its nominal height outside the box.
        assert_eq!(m.eps_at([0.0, 1.05, 1.47]), 3.5);
        // With top > 0 the slab is shell.
        let regions = shell(0.1, 0.1);
        let m = BoxMedium::edited_inside(&regions, &g, BoxId::new(0), &[z_in], &st, &d).unwrap();
        assert_eq!(m.eps_at([0.0, 0.0, 1.47]), 3.5);
    }
}
