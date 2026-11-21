//! The random walk itself.

use faradice_core::{BoxId, NetId, Sink};
use faradice_geom::{Aabb, Geometry, SpatialIndex, UniformGrid};

use crate::axis::BoxSteps;
use crate::boundary::BoundaryModel;
use crate::green::GreenTables;
use crate::medium::Medium;
use crate::rng::Rng;

/// How a walk ended.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WalkEnd {
    /// The sink, or `None` if the walk hit `max_steps`.
    pub(crate) sink: Option<Sink>,
    /// `±1`; flips when the walk crosses the substrate plane in the odd image
    /// extension used by the open boundary.
    pub(crate) sign: f64,
    pub(crate) steps: u32,
    /// Where the walk stopped (the last point visited).
    #[cfg_attr(not(feature = "sensitivity"), allow(dead_code))]
    pub(crate) pos: [f64; 3],
}

pub(crate) struct Walker<'a> {
    pub(crate) geom: &'a Geometry,
    pub(crate) index: &'a UniformGrid,
    pub(crate) green: &'a GreenTables,
    pub(crate) domain: &'a BoundaryModel,
    /// Stratified dielectric (design M2a); `None` if homogeneous.
    pub(crate) medium: Option<&'a Medium>,
    /// Dielectric boxes (design M2c); `None` if there are none.
    pub(crate) boxes: Option<&'a BoxSteps>,
    pub(crate) eps_stop: f64,
    pub(crate) far_step: f64,
    pub(crate) max_steps: u32,
    /// A local edit of the geometry (sensitivity continuations): see
    /// [`Overlay`]. `None` for the nominal geometry.
    pub(crate) overlay: Option<&'a Overlay>,
    /// Conductors added on top of the geometry (replay of additions, design
    /// D3 §8). `None` for the nominal geometry.
    pub(crate) added: Option<&'a [(Aabb, NetId)]>,
    /// Window first hops (design M2f); off for recorded and replayed walks.
    pub(crate) windows: bool,
}

/// The geometry with box `base` replaced by `boxes` (all inside `base`, of
/// the same net), without rebuilding the index: the nominal index stays a
/// valid lower bound, because the replacement only removes conductor.
///
/// When the nominal nearest box is `base` (at distance `d_b`), the nearest
/// of the edited geometry is among `boxes`, the boxes `local` within `reach`
/// of `base`, and the boundary. Any other box `c` is at least
/// `reach − d_b` away (L∞ triangle inequality), and every conductor at
/// least `d_b`, so `max(d_b, min(·, reach − d_b))` is a valid step. It is
/// exact whenever it is below the stopping distance, provided
/// `reach ≥ 2 ε_stop`.
#[derive(Debug, Clone)]
pub(crate) struct Overlay {
    pub(crate) base: BoxId,
    pub(crate) net: NetId,
    pub(crate) boxes: Vec<Aabb>,
    pub(crate) local: Vec<(Aabb, NetId)>,
    pub(crate) reach: f64,
}

impl Overlay {
    /// Builds the overlay of `base` → `boxes` in `geom`, listing the boxes
    /// within `reach` of `base`.
    #[cfg_attr(not(feature = "sensitivity"), allow(dead_code))]
    pub(crate) fn new(geom: &Geometry, base: BoxId, boxes: Vec<Aabb>, reach: f64) -> Self {
        let b = *geom.box_(base);
        let local = geom
            .boxes()
            .iter()
            .enumerate()
            .filter(|&(i, c)| {
                #[allow(clippy::cast_possible_truncation)]
                let other = BoxId::new(i as u32) != base;
                other && box_gap(&b, c) < reach
            })
            .map(|(i, c)| {
                #[allow(clippy::cast_possible_truncation)]
                (*c, geom.box_net(BoxId::new(i as u32)))
            })
            .collect();
        Self {
            base,
            net: geom.box_net(base),
            boxes,
            local,
            reach,
        }
    }

    #[inline]
    fn clearance(&self, x: [f64; 3], d_b: f64, d_bnd: f64, bnd_sink: Sink) -> (f64, Sink) {
        let mut best = (d_bnd, bnd_sink);
        for b in &self.boxes {
            let d = b.linf_distance(x);
            if d < best.0 {
                best = (d, Sink::Net(self.net));
            }
        }
        for (c, n) in &self.local {
            let d = c.linf_distance(x);
            if d < best.0 {
                best = (d, Sink::Net(*n));
            }
        }
        let far = self.reach - d_b;
        if far < best.0 {
            // Unlisted boxes may be nearer than `best`: a conservative step,
            // at least `reach / 2 ≥ ε_stop`.
            return (d_b.max(far), bnd_sink);
        }
        best
    }
}

/// L∞ gap between two boxes (0 if they touch or overlap).
#[cfg_attr(not(feature = "sensitivity"), allow(dead_code))]
fn box_gap(a: &Aabb, b: &Aabb) -> f64 {
    (0..3)
        .map(|i| {
            (b.min()[i] - a.max()[i])
                .max(a.min()[i] - b.max()[i])
                .max(0.0)
        })
        .fold(0.0, f64::max)
}

impl Walker<'_> {
    /// Largest conductor- and boundary-free cube half-side at `x`, and the sink
    /// that would be hit if that distance is below the stopping threshold.
    #[inline]
    pub(crate) fn clearance(&self, x: [f64; 3]) -> (f64, Sink) {
        let r = self.base_clearance(x);
        match self.added {
            None => r,
            // Strictly nearer only: where an added box does not cut the
            // nominal cube, the step (and so the walk) is unchanged.
            Some(added) => added.iter().fold(r, |best, (b, n)| {
                let d = b.linf_distance(x);
                if d < best.0 { (d, Sink::Net(*n)) } else { best }
            }),
        }
    }

    /// [`Self::clearance`] capped by the distance to the nearest dielectric
    /// box: the largest cube at `x` that sees the stack only.
    #[inline]
    pub(crate) fn stack_clearance(&self, x: [f64; 3]) -> (f64, Sink) {
        let (h, sink) = self.clearance(x);
        match self.boxes {
            Some(b) => (h.min(b.box_distance(x, h)), sink),
            None => (h, sink),
        }
    }

    #[inline]
    fn base_clearance(&self, x: [f64; 3]) -> (f64, Sink) {
        let (d_bnd, bnd_sink) = self.domain.dirichlet_distance(x);
        let lb = self.index.clearance_lower_bound(x);
        if lb >= d_bnd {
            return (d_bnd, bnd_sink);
        }
        // The shortcut never stops a walk: with an explicit stopping distance
        // above `far_step`, a bound below `eps_stop` needs the exact query
        // for the right sink.
        if lb >= self.far_step && lb >= self.eps_stop {
            // Far from all conductors: the bound is a valid (slightly conservative) step.
            return (lb, bnd_sink);
        }
        match self.index.nearest_linf(x, d_bnd) {
            Some(n) if n.distance < d_bnd => match self.overlay {
                Some(o) if o.base == n.box_id => o.clearance(x, n.distance, d_bnd, bnd_sink),
                _ => (n.distance, Sink::Net(self.geom.box_net(n.box_id))),
            },
            _ => (d_bnd, bnd_sink),
        }
    }

    /// Walks from `x` until a sink is reached.
    #[inline]
    pub(crate) fn run(&self, x: [f64; 3], rng: &mut Rng) -> WalkEnd {
        self.run_traced(x, rng, &mut |_, _| {})
    }

    /// [`Self::run`], reporting the centre and half-side of every clearance
    /// cube (including the final one below `ε_stop`, and the carried-over
    /// lower bound where it replaces the clearance query). Every step stays
    /// within the reported cube.
    pub(crate) fn run_traced<T: FnMut([f64; 3], f64)>(
        &self,
        mut x: [f64; 3],
        rng: &mut Rng,
        trace: &mut T,
    ) -> WalkEnd {
        let mut sign = 1.0;
        // Lower bound on the clearance at `x` (design M2a §8).
        let mut bound = 0.0_f64;
        for steps in 0..self.max_steps {
            if let Some((c, radius)) = self.domain.sphere {
                let d = [x[0] - c[0], x[1] - c[1], x[2] - c[2]];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                // Only jump when the walk is well outside the sphere: just outside it the
                // return probability radius/r is ~1 and the walk would bounce on the sphere.
                if r > 2.0 * radius {
                    match exterior_sphere_step(x, c, radius, r, rng) {
                        None => {
                            return WalkEnd {
                                sink: Some(Sink::Infinity),
                                sign,
                                steps,
                                pos: x,
                            };
                        }
                        Some(mut y) => {
                            if let Some(z0) = self.domain.substrate_z {
                                if y[2] < z0 {
                                    y[2] = 2.0 * z0 - y[2];
                                    sign = -sign;
                                }
                            }
                            x = y;
                            bound = 0.0;
                            continue;
                        }
                    }
                }
            }
            // In a stratified medium, a step off an interface whose cube is
            // limited by the second-nearest interface needs no exact
            // clearance: a lower bound carried over from the previous step is
            // a valid (conductor-free) choice.
            let skip = self.boxes.is_none()
                && self
                    .medium
                    .is_some_and(|m| m.bound_suffices(x[2], bound, self.eps_stop));
            let (h, sink) = if skip {
                (bound, Sink::Infinity)
            } else {
                self.clearance(x)
            };
            trace(x, h);
            if h < self.eps_stop {
                return WalkEnd {
                    sink: Some(sink),
                    sign,
                    steps,
                    pos: x,
                };
            }
            // `reach`: the max-norm distance of the new point (on the surface
            // of a domain centred on `x`).
            let reach = match (self.boxes, self.medium) {
                (Some(b), m) => match b.step(m, self.green, &mut x, h, rng) {
                    Some(r) => r,
                    // No cube makes progress (a dielectric edge): abort.
                    None => {
                        return WalkEnd {
                            sink: None,
                            sign,
                            steps,
                            pos: x,
                        };
                    }
                },
                (None, None) => {
                    let u = self.green.sample_transition(rng);
                    x = [x[0] + h * u[0], x[1] + h * u[1], x[2] + h * u[2]];
                    h
                }
                (None, Some(m)) => m.step(self.green, &mut x, h, rng),
            };
            let unfolded = x;
            self.domain.fold(&mut x);
            // Clearance is 1-Lipschitz in the max norm; a folded point starts
            // afresh (exact comparison: `fold` leaves unreflected points untouched).
            #[allow(clippy::float_cmp)]
            let unchanged = x == unfolded;
            bound = if unchanged { h - reach } else { 0.0 };
        }
        WalkEnd {
            sink: None,
            sign,
            steps: self.max_steps,
            pos: x,
        }
    }
}

/// One step of the exterior harmonic measure of a sphere (centre `c`, radius
/// `radius`) from a point at distance `r > radius`: with probability
/// `1 − radius/r` the walk escapes to infinity (`None`); otherwise it lands on
/// the sphere with density `∝ 1/s³` in the distance `s` to the landing point.
fn exterior_sphere_step(
    x: [f64; 3],
    c: [f64; 3],
    radius: f64,
    r: f64,
    rng: &mut Rng,
) -> Option<[f64; 3]> {
    if rng.next_f64() >= radius / r {
        return None;
    }
    let (a, b) = (1.0 / (r - radius), 1.0 / (r + radius));
    let s = 1.0 / (a - rng.next_f64() * (a - b));
    let cos_t = ((r * r + radius * radius - s * s) / (2.0 * r * radius)).clamp(-1.0, 1.0);
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * rng.next_f64();
    let e = [(x[0] - c[0]) / r, (x[1] - c[1]) / r, (x[2] - c[2]) / r];
    // Orthonormal basis (t1, t2) perpendicular to e.
    let helper = if e[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1 = normalize(cross(e, helper));
    let t2 = cross(e, t1);
    let (cp, sp) = (phi.cos(), phi.sin());
    Some(std::array::from_fn(|k| {
        c[k] + radius * (cos_t * e[k] + sin_t * (cp * t1[k] + sp * t2[k]))
    }))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The overlay step against brute force on the edited box list: never
    /// larger than the true clearance, and exact (with the right net) below
    /// the stopping distance.
    #[test]
    fn overlay_matches_brute_force() {
        use faradice_geom::{GeometryBuilder, GridOptions};
        let mut rng = Rng::from_seed(5);
        let mut gb = GeometryBuilder::new();
        let nets: Vec<NetId> = (0..3).map(|i| gb.net(&format!("n{i}")).unwrap()).collect();
        // A row of thin wires with small gaps, three nets interleaved.
        for i in 0..40 {
            let x0 = f64::from(i) * 0.3;
            let b = Aabb::from_array([x0, 0.0, 0.0, x0 + 0.2, 4.0, 0.3]).unwrap();
            gb.add_box(nets[i as usize % 3], b).unwrap();
        }
        let geom = gb.build().unwrap();
        let index = UniformGrid::new(&geom, GridOptions::default());
        let eps = 1e-3;
        let reach = (2.0 * index.cell_size()).max(4.0 * eps);
        let base = BoxId::new(17);
        let b = *geom.box_(base);
        // Shrink the y+ face by 0.5 and keep a filler on part of the strip.
        let moved = Aabb::from_array([b.min()[0], 0.0, 0.0, b.max()[0], 3.5, 0.3]).unwrap();
        let filler = Aabb::from_array([b.min()[0], 3.5, 0.0, b.max()[0], 3.7, 0.3]).unwrap();
        let o = Overlay::new(&geom, base, vec![moved, filler], reach);
        let mut edited: Vec<(Aabb, NetId)> = geom
            .boxes()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 17)
            .map(|(i, c)| (*c, geom.box_net(BoxId::new(u32::try_from(i).unwrap()))))
            .collect();
        edited.push((moved, o.net));
        edited.push((filler, o.net));
        let (mut checked, mut exact_hits) = (0, 0);
        for _ in 0..200_000 {
            // Points around the edited box, some within ε_stop of surfaces.
            let mut x = [
                b.min()[0] - 0.5 + 1.2 * rng.next_f64(),
                3.0 + 1.5 * rng.next_f64(),
                -0.2 + 0.7 * rng.next_f64(),
            ];
            if rng.next_f64() < 0.3 {
                // Snap one coordinate close to a nearby surface.
                x[1] = 3.5 + (rng.next_f64() - 0.5) * 2.0 * eps;
            }
            let Some(n) = index.nearest_linf(x, f64::INFINITY) else {
                continue;
            };
            if n.box_id != base || n.distance <= 0.0 {
                continue;
            }
            let (h, sink) = o.clearance(x, n.distance, f64::INFINITY, Sink::Infinity);
            let (t, tn) = edited
                .iter()
                .map(|(c, net)| (c.linf_distance(x), *net))
                .fold(
                    (f64::INFINITY, nets[0]),
                    |a, v| if v.0 < a.0 { v } else { a },
                );
            assert!(h <= t + 1e-12, "step {h} > clearance {t} at {x:?}");
            if h < eps {
                exact_hits += 1;
                assert!((h - t).abs() <= 1e-12, "inexact {h} vs {t}");
                assert_eq!(sink, Sink::Net(tn));
            }
            checked += 1;
        }
        assert!(
            checked > 10_000 && exact_hits > 100,
            "{checked} {exact_hits}"
        );
    }

    #[test]
    fn exterior_step_return_probability() {
        let mut rng = Rng::from_seed(11);
        let (c, radius, x) = ([0.0; 3], 1.0, [3.0, 0.0, 0.0]);
        let n = 100_000;
        let mut back = 0;
        let mut mean_x = 0.0;
        for _ in 0..n {
            if let Some(y) = exterior_sphere_step(x, c, radius, 3.0, &mut rng) {
                back += 1;
                let rr = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
                assert!((rr - 1.0).abs() < 1e-12);
                mean_x += y[0];
            }
        }
        let p = f64::from(back) / f64::from(n);
        assert!((p - 1.0 / 3.0).abs() < 0.005, "p={p}");
        // Harmonic measure is biased towards the walker: E[y_x | return] > 0.
        assert!(mean_x / f64::from(back) > 0.2);
    }
}
