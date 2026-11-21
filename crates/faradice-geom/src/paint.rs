//! Labelled Boolean operations on axis-aligned boxes by painting.
//!
//! Items are painted in order: each one sets its label on, or clears, the
//! region it covers, and later items win. The result is a set of disjoint
//! labelled boxes (design M2c §6.2: conformal shells are "grow, then clear
//! the conductors", and shells and explicit dielectric boxes are combined
//! with a precedence).
//!
//! The sweep splits space into z slabs at every item height, each slab into
//! x slabs at every x coordinate of the items crossing it, and paints y
//! intervals within each x slab. Neighbouring x slabs with identical y
//! intervals, and neighbouring z slabs with identical rectangles, are merged.
//! Active items are maintained incrementally, so the cost is about the sum,
//! over all (z slab, x slab) pairs, of the number of items covering the pair
//! (times a log). This is proportional to the output for disjoint or
//! side-by-side items (wires and their shells), but deeply nested items can
//! cost more than their output.
//! Output boxes reuse the input coordinates exactly (no arithmetic), and the
//! output order depends only on the input.

use std::collections::{BTreeSet, HashMap};

use crate::Aabb;

/// What an item does to the region it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Paint {
    /// Sets this label.
    Set(u32),
    /// Removes any label.
    Clear,
}

/// One painting step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaintItem {
    /// The covered region.
    pub region: Aabb,
    /// What happens there.
    pub paint: Paint,
}

/// Result of [`paint`].
#[derive(Debug, Clone, PartialEq)]
pub struct Painted {
    /// Disjoint labelled boxes covering exactly the labelled region.
    pub boxes: Vec<(Aabb, u32)>,
    /// Volume covered by `Set` items of two or more different labels,
    /// whatever the final label (overlaps between materials).
    pub conflict_volume: f64,
}

/// Paints `items` in order (later items win) and returns disjoint labelled boxes.
#[must_use]
pub fn paint(items: &[PaintItem]) -> Painted {
    let mut zs: Vec<f64> = items
        .iter()
        .flat_map(|it| [it.region.min()[2], it.region.max()[2]])
        .collect();
    sort_dedup(&mut zs);

    let mut out: Vec<(Aabb, u32)> = Vec::new();
    let mut conflict = 0.0;
    // Boxes ending at the current slab's bottom, keyed by their (x, y) rectangle and label.
    let mut open: HashMap<RectKey, usize> = HashMap::new();
    // Items by increasing bottom; `active` holds those crossing the current slab.
    let mut by_z: Vec<usize> = (0..items.len()).collect();
    by_z.sort_by(|&a, &b| items[a].region.min()[2].total_cmp(&items[b].region.min()[2]));
    let mut next = 0;
    let mut active: Vec<usize> = Vec::new();
    for w in zs.windows(2) {
        let (z0, z1) = (w[0], w[1]);
        while next < by_z.len() && items[by_z[next]].region.min()[2] <= z0 {
            active.push(by_z[next]);
            next += 1;
        }
        // z0 and z1 are consecutive heights: crossing [z0, z1] means ending above z0.
        active.retain(|&i| items[i].region.max()[2] > z0);
        let (rects, area) = paint_slab(items, &active);
        conflict += area * (z1 - z0);
        let mut next_open = HashMap::with_capacity(rects.len());
        for (x0, x1, y0, y1, label) in rects {
            let key = RectKey::new(x0, x1, y0, y1, label);
            let idx = if let Some(&i) = open.get(&key) {
                let b = out[i].0;
                out[i].0 = Aabb::closed(b.min(), [b.max()[0], b.max()[1], z1]);
                i
            } else {
                out.push((Aabb::closed([x0, y0, z0], [x1, y1, z1]), label));
                out.len() - 1
            };
            next_open.insert(key, idx);
        }
        open = next_open;
    }
    Painted {
        boxes: out,
        conflict_volume: conflict,
    }
}

type SlabRect = (f64, f64, f64, f64, u32);

/// Labelled y intervals of one x slab.
type Line = Vec<(f64, f64, u32)>;

/// Paints one z slab given the items crossing it (ascending order). Returns the
/// labelled rectangles and the conflict area.
fn paint_slab(items: &[PaintItem], active: &[usize]) -> (Vec<SlabRect>, f64) {
    let mut xs: Vec<f64> = active
        .iter()
        .flat_map(|&i| [items[i].region.min()[0], items[i].region.max()[0]])
        .collect();
    sort_dedup(&mut xs);
    let mut by_x = active.to_vec();
    by_x.sort_by(|&a, &b| items[a].region.min()[0].total_cmp(&items[b].region.min()[0]));
    let mut next = 0;
    let mut covering: Vec<usize> = Vec::new();
    let mut rects = Vec::new();
    let mut conflict = 0.0;
    // Current run of x slabs with identical y intervals.
    let mut run: Option<(f64, Line)> = None;
    let mut prev_x1 = f64::NAN;
    for w in xs.windows(2) {
        let (x0, x1) = (w[0], w[1]);
        while next < by_x.len() && items[by_x[next]].region.min()[0] <= x0 {
            covering.push(by_x[next]);
            next += 1;
        }
        covering.retain(|&i| items[i].region.max()[0] > x0);
        let (ivals, len) = paint_line(items, &covering);
        conflict += len * (x1 - x0);
        match &mut run {
            // x slabs are consecutive, so equal lines extend the run.
            Some((_, prev)) if *prev == ivals => {}
            _ => {
                if let Some((start, prev)) = run.take() {
                    flush(&mut rects, start, prev_x1, &prev);
                }
                run = Some((x0, ivals));
            }
        }
        prev_x1 = x1;
    }
    if let Some((start, prev)) = run {
        flush(&mut rects, start, prev_x1, &prev);
    }
    (rects, conflict)
}

fn flush(rects: &mut Vec<SlabRect>, x0: f64, x1: f64, ivals: &[(f64, f64, u32)]) {
    rects.extend(ivals.iter().map(|&(y0, y1, l)| (x0, x1, y0, y1, l)));
}

/// Paints the y line of one x slab. Returns the merged labelled intervals and
/// the conflict length.
fn paint_line(items: &[PaintItem], covering: &[usize]) -> (Line, f64) {
    // Events: (y, start?, item). Ends sort before starts at equal y.
    let mut ev: Vec<(f64, bool, usize)> = covering
        .iter()
        .flat_map(|&i| {
            let r = &items[i].region;
            [(r.min()[1], true, i), (r.max()[1], false, i)]
        })
        .collect();
    ev.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut active: BTreeSet<usize> = BTreeSet::new();
    let mut labels: HashMap<u32, usize> = HashMap::new();
    let mut out: Vec<(f64, f64, u32)> = Vec::new();
    let mut conflict = 0.0;
    for (k, &(y, start, i)) in ev.iter().enumerate() {
        let set_label = match items[i].paint {
            Paint::Set(l) => Some(l),
            Paint::Clear => None,
        };
        if start {
            active.insert(i);
            if let Some(l) = set_label {
                *labels.entry(l).or_insert(0) += 1;
            }
        } else {
            active.remove(&i);
            if let Some(l) = set_label {
                let c = labels.get_mut(&l).expect("label counted at start");
                *c -= 1;
                if *c == 0 {
                    labels.remove(&l);
                }
            }
        }
        let Some(&(y1, _, _)) = ev.get(k + 1) else {
            break;
        };
        if y1 <= y {
            continue;
        }
        if labels.len() >= 2 {
            conflict += y1 - y;
        }
        if let Some(&top) = active.last() {
            if let Paint::Set(l) = items[top].paint {
                match out.last_mut() {
                    // Exact: coordinates are copied, never computed.
                    Some(last) if last.1.to_bits() == y.to_bits() && last.2 == l => last.1 = y1,
                    _ => out.push((y, y1, l)),
                }
            }
        }
    }
    (out, conflict)
}

fn sort_dedup(v: &mut Vec<f64>) {
    v.sort_by(f64::total_cmp);
    v.dedup();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RectKey([u64; 4], u32);

impl RectKey {
    fn new(x0: f64, x1: f64, y0: f64, y1: f64, label: u32) -> Self {
        Self([x0, x1, y0, y1].map(f64::to_bits), label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(c: [f64; 6]) -> Aabb {
        Aabb::from_array(c).unwrap()
    }

    fn set(c: [f64; 6], l: u32) -> PaintItem {
        PaintItem {
            region: bx(c),
            paint: Paint::Set(l),
        }
    }

    fn clear(c: [f64; 6]) -> PaintItem {
        PaintItem {
            region: bx(c),
            paint: Paint::Clear,
        }
    }

    /// Label at `p` by painting the items one after the other.
    fn brute(items: &[PaintItem], p: [f64; 3]) -> Option<u32> {
        let mut l = None;
        for it in items {
            if it.region.contains(p) {
                l = match it.paint {
                    Paint::Set(v) => Some(v),
                    Paint::Clear => None,
                };
            }
        }
        l
    }

    fn label_at(out: &Painted, p: [f64; 3]) -> Option<u32> {
        let hits: Vec<u32> = out
            .boxes
            .iter()
            .filter(|(b, _)| b.contains(p))
            .map(|&(_, l)| l)
            .collect();
        assert!(hits.len() <= 1, "boxes overlap at {p:?}");
        hits.first().copied()
    }

    #[test]
    fn shell_around_an_l_shaped_conductor() {
        // Two conductor boxes forming an L, grown by 0.5 laterally, then cleared.
        let a = [0.0, 0.0, 0.0, 4.0, 1.0, 1.0];
        let b = [0.0, 1.0, 0.0, 1.0, 4.0, 1.0];
        let grow = |c: [f64; 6]| [c[0] - 0.5, c[1] - 0.5, c[2], c[3] + 0.5, c[4] + 0.5, c[5]];
        let items = [set(grow(a), 7), set(grow(b), 7), clear(a), clear(b)];
        let out = paint(&items);
        let vol: f64 = out.boxes.iter().map(|(b, _)| b.volume()).sum();
        // Grown L: 5×2 + 2×3 = 16; conductor: 4 + 3 = 7.
        assert!((vol - 9.0).abs() < 1e-12, "{vol}");
        assert_eq!(out.conflict_volume, 0.0);
        assert!(out.boxes.iter().all(|&(_, l)| l == 7));
        assert_eq!(label_at(&out, [2.0, 0.5, 0.5]), None);
        assert_eq!(label_at(&out, [2.0, 1.2, 0.5]), Some(7));
        assert_eq!(label_at(&out, [1.2, 1.2, 0.5]), Some(7)); // inner corner
    }

    #[test]
    fn later_items_win_and_conflicts_are_measured() {
        let items = [
            set([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], 1),
            set([1.0, 1.0, 1.0, 3.0, 3.0, 3.0], 2),
        ];
        let out = paint(&items);
        assert_eq!(label_at(&out, [1.5, 1.5, 1.5]), Some(2));
        assert_eq!(label_at(&out, [0.5, 0.5, 0.5]), Some(1));
        assert!((out.conflict_volume - 1.0).abs() < 1e-12);
        let vol: f64 = out.boxes.iter().map(|(b, _)| b.volume()).sum();
        assert!((vol - 15.0).abs() < 1e-12);
    }

    #[test]
    fn identical_slabs_merge() {
        // A long bar split into two touching boxes with the same label merges in x and z.
        let items = [
            set([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3),
            set([1.0, 0.0, 0.0, 2.0, 1.0, 1.0], 3),
            set([0.0, 0.0, 1.0, 2.0, 1.0, 2.0], 3),
        ];
        let out = paint(&items);
        assert_eq!(out.boxes.len(), 1, "{:?}", out.boxes);
        assert_eq!(out.boxes[0].0, bx([0.0, 0.0, 0.0, 2.0, 1.0, 2.0]));
    }

    /// Random items on a lattice against the brute-force painter at every
    /// lattice-cell centre; total volume against the cell count.
    #[test]
    fn matches_brute_force() {
        let mut s = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % n
        };
        for _ in 0..40 {
            let mut items = Vec::new();
            for _ in 0..12 {
                #[allow(clippy::cast_precision_loss)]
                let mut c = [0.0; 6];
                for a in 0..3 {
                    let lo = next(7);
                    let hi = lo + 1 + next(7 - lo);
                    #[allow(clippy::cast_precision_loss)]
                    {
                        c[a] = lo as f64;
                        c[a + 3] = hi as f64;
                    }
                }
                let paint = if next(4) == 0 {
                    Paint::Clear
                } else {
                    #[allow(clippy::cast_possible_truncation)]
                    Paint::Set(next(3) as u32)
                };
                items.push(PaintItem {
                    region: bx(c),
                    paint,
                });
            }
            let out = paint(&items);
            let mut labelled = 0.0;
            let mut conflict = 0.0;
            for x in 0..8 {
                for y in 0..8 {
                    for z in 0..8 {
                        let p = [f64::from(x) + 0.5, f64::from(y) + 0.5, f64::from(z) + 0.5];
                        let want = brute(&items, p);
                        assert_eq!(label_at(&out, p), want, "{p:?}");
                        if want.is_some() {
                            labelled += 1.0;
                        }
                        let mut ls: Vec<u32> = items
                            .iter()
                            .filter(|it| it.region.contains(p))
                            .filter_map(|it| match it.paint {
                                Paint::Set(l) => Some(l),
                                Paint::Clear => None,
                            })
                            .collect();
                        ls.sort_unstable();
                        ls.dedup();
                        if ls.len() >= 2 {
                            conflict += 1.0;
                        }
                    }
                }
            }
            let vol: f64 = out.boxes.iter().map(|(b, _)| b.volume()).sum();
            assert!((vol - labelled).abs() < 1e-9, "{vol} vs {labelled}");
            assert!((out.conflict_volume - conflict).abs() < 1e-9);
        }
    }
}
