//! Uniform-grid spatial index with exact L∞ nearest-box queries.
//!
//! Boxes are registered in every grid cell they overlap (CSR layout). A query
//! visits cells in Chebyshev rings `k = 0, 1, 2, …` around the query cell.
//! After ring `k` has been visited, any box not yet seen lies entirely in rings
//! `≥ k + 1` and is therefore strictly farther than `k · h` (cell size `h`) in
//! the L∞ metric, which gives an exact termination criterion.
//!
//! In addition, a chessboard distance transform over the cells stores, for
//! every cell, the Chebyshev distance `m` (in cells) to the nearest occupied
//! cell. Every box is then at least `(m - 1) · h` away from any point of that
//! cell, which lets queries skip empty rings and gives an O(1)
//! [`clearance_lower_bound`](SpatialIndex::clearance_lower_bound).

use faradice_core::BoxId;

use crate::{Aabb, Geometry, Nearest, SpatialIndex};

/// Options controlling grid construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridOptions {
    /// Cell edge length in µm. `None` chooses it automatically as
    /// [`cell_factor`](Self::cell_factor) × the median of the boxes' smallest extents.
    pub cell_size: Option<f64>,
    /// Multiple of the median smallest box extent used for the automatic cell size.
    ///
    /// Benchmarks on routed-bus geometry show a broad optimum around 2–6× the
    /// wire width; the default is 3.
    pub cell_factor: f64,
    /// Upper bound on the total number of cells (the cell size grows to respect it).
    pub max_cells: usize,
    /// Upper bound on (cell, box) registrations, i.e. index memory in `u32`s
    /// (the cell size grows to respect it).
    pub max_registrations: usize,
    /// Compute the chessboard distance transform used by clearance bounds and
    /// ring skipping. Disable for occupancy-only uses (e.g. broad phases).
    pub clearance_transform: bool,
    /// Precompute, for every cell within two cells of a conductor, the boxes that
    /// can be nearest to any point of the cell. Exact queries there scan only
    /// this short list instead of searching rings. Requires `clearance_transform`.
    /// Lists longer than 64 boxes, and lists beyond a total of
    /// [`max_registrations`](Self::max_registrations) entries, are not cached.
    pub near_cache: bool,
}

impl Default for GridOptions {
    fn default() -> Self {
        Self {
            cell_size: None,
            cell_factor: 3.0,
            max_cells: 1 << 24,
            max_registrations: 1 << 25,
            clearance_transform: true,
            near_cache: true,
        }
    }
}

/// Borrowed view of a [`UniformGrid`]'s arrays (see [`UniformGrid::raw`]).
///
/// Cells are linearized as `(z * dims[1] + y) * dims[0] + x`; cell `[i, j, k]`
/// spans `origin + [i, j, k] * cell` to `origin + [i + 1, j + 1, k + 1] * cell`.
#[derive(Debug, Clone, Copy)]
pub struct GridData<'a> {
    /// Indexed boxes (box ids index this slice).
    pub boxes: &'a [Aabb],
    /// Lower corner of cell `[0, 0, 0]`.
    pub origin: [f64; 3],
    /// Cell edge length.
    pub cell: f64,
    /// Cells per axis.
    pub dims: [usize; 3],
    /// CSR offsets into `items` (length `cells + 1`).
    pub cell_start: &'a [u32],
    /// Box ids registered per cell.
    pub items: &'a [u32],
    /// Chessboard distance (cells) to the nearest occupied cell; empty if the
    /// transform was disabled.
    pub clearance: &'a [u32],
    /// CSR offsets of the near-cell candidate lists (empty if disabled).
    pub near_start: &'a [u32],
    /// Near-cell candidate box ids. A cell with a non-empty list may be queried
    /// exactly by scanning only that list.
    pub near_items: &'a [u32],
    /// Bounding box of all boxes.
    pub bounds: Aabb,
}

/// A uniform grid over the bounding box of all conductors.
#[derive(Debug, Clone)]
pub struct UniformGrid {
    boxes: Vec<Aabb>,
    origin: [f64; 3],
    cell: f64,
    inv_cell: f64,
    dims: [usize; 3],
    /// CSR offsets: items of cell `c` are `items[cell_start[c]..cell_start[c + 1]]`.
    cell_start: Vec<u32>,
    items: Vec<u32>,
    /// Chessboard distance (in cells) from each cell to the nearest occupied cell.
    clearance: Vec<u32>,
    /// Near-cell candidate lists (CSR); empty when disabled. An empty list for a
    /// cell means "not cached" (cached cells always have at least one candidate).
    near_start: Vec<u32>,
    near_items: Vec<u32>,
    bounds: Aabb,
}

impl UniformGrid {
    /// Builds a grid over the boxes of `geom`.
    #[must_use]
    pub fn new(geom: &Geometry, opts: GridOptions) -> Self {
        Self::from_boxes(geom.boxes(), opts)
    }

    /// Builds a grid over arbitrary boxes (box ids are slice indices).
    ///
    /// # Panics
    ///
    /// Panics if `boxes` is empty or has more than `u32::MAX` entries.
    #[must_use]
    pub fn from_boxes(boxes: &[Aabb], opts: GridOptions) -> Self {
        assert!(!boxes.is_empty(), "UniformGrid requires at least one box");
        assert!(u32::try_from(boxes.len()).is_ok(), "too many boxes");
        let bounds = boxes.iter().skip(1).fold(boxes[0], |acc, b| acc.union(b));
        let origin = bounds.min();
        let extent = bounds.extent();
        let max_cells = opts.max_cells.max(1);

        let mut cell = opts.cell_size.unwrap_or_else(|| {
            let mut mins: Vec<f64> = boxes.iter().map(Aabb::min_extent).collect();
            let mid = mins.len() / 2;
            let (_, median, _) = mins.select_nth_unstable_by(mid, f64::total_cmp);
            *median * opts.cell_factor
        });
        if !(cell.is_finite() && cell > 0.0) {
            cell = extent[0].max(extent[1]).max(extent[2]);
        }
        let dims_for = |h: f64| -> [usize; 3] {
            // Extents are finite and h > 0; the loop below keeps the cell count bounded.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            extent.map(|e| ((e / h).ceil() as usize).max(1))
        };
        let registrations_for = |h: f64| -> f64 {
            #[allow(clippy::cast_precision_loss)]
            boxes
                .iter()
                .map(|b| {
                    b.extent()
                        .iter()
                        .map(|e| (e / h).floor() + 2.0)
                        .product::<f64>()
                })
                .sum()
        };
        #[allow(clippy::cast_precision_loss)]
        let too_big = |h: f64| {
            let dims = dims_for(h);
            dims.iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .is_none_or(|n| n > max_cells)
                || registrations_for(h) > opts.max_registrations as f64
        };
        // Stop growing once the whole domain fits in one cell: the registration
        // estimate has a floor of 8 per box, so a tiny budget could never be met.
        while too_big(cell) && dims_for(cell).iter().any(|&d| d > 1) {
            cell *= 1.25;
        }
        let dims = dims_for(cell);
        let inv_cell = 1.0 / cell;
        let n_cells = dims[0] * dims[1] * dims[2];

        let mut grid = Self {
            boxes: boxes.to_vec(),
            origin,
            cell,
            inv_cell,
            dims,
            cell_start: Vec::new(),
            items: Vec::new(),
            clearance: Vec::new(),
            near_start: Vec::new(),
            near_items: Vec::new(),
            bounds,
        };

        // Two-pass CSR construction.
        let mut counts = vec![0u32; n_cells + 1];
        for b in boxes {
            let (lo, hi) = grid.cell_range(b);
            grid.for_each_cell(lo, hi, |c| counts[c + 1] += 1);
        }
        for c in 0..n_cells {
            counts[c + 1] += counts[c];
        }
        let mut fill = counts.clone();
        let mut items = vec![0u32; counts[n_cells] as usize];
        for (i, b) in boxes.iter().enumerate() {
            let (lo, hi) = grid.cell_range(b);
            #[allow(clippy::cast_possible_truncation)]
            grid.for_each_cell(lo, hi, |c| {
                items[fill[c] as usize] = i as u32;
                fill[c] += 1;
            });
        }
        grid.cell_start = counts;
        grid.items = items;
        grid.clearance = if opts.clearance_transform {
            grid.chessboard_transform()
        } else {
            // Every cell conservatively reports zero clearance.
            vec![0; n_cells]
        };
        if opts.clearance_transform && opts.near_cache {
            // The cache may use as much memory as the registrations themselves.
            grid.build_near_cache(opts.max_registrations);
        }
        grid
    }

    /// Clearance (in cells) at or below which a cell gets a candidate list.
    const NEAR_CELLS: u32 = 2;

    /// Longest candidate list cached for one cell; longer lists fall back to
    /// the ring search.
    const NEAR_MAX_PER_CELL: usize = 64;

    /// Builds the near-cell candidate lists (see [`GridOptions::near_cache`]).
    ///
    /// For a cell `C`, let `U = min_B max_{p∈C} d(p, B)`. The nearest box of any
    /// point of `C` is within `U`, so only boxes with `min_{p∈C} d(p, B) ≤ U` can
    /// be nearest. Boxes are gathered ring by ring until the next ring is
    /// provably farther than `U`.
    fn build_near_cache(&mut self, budget: usize) {
        let [nx, ny, nz] = self.dims;
        let n_cells = nx * ny * nz;
        let mut start = Vec::with_capacity(n_cells + 1);
        let mut items: Vec<u32> = Vec::new();
        let mut stamp = vec![u32::MAX; self.boxes.len()];
        let mut cand: Vec<(u32, f64)> = Vec::new();
        start.push(0u32);
        let pad = 1e-9 * self.cell;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let lin = self.linear(x, y, z);
                    if self.clearance[lin] <= Self::NEAR_CELLS {
                        #[allow(clippy::cast_precision_loss)]
                        let lo = [
                            self.origin[0] + x as f64 * self.cell - pad,
                            self.origin[1] + y as f64 * self.cell - pad,
                            self.origin[2] + z as f64 * self.cell - pad,
                        ];
                        let hi = [
                            lo[0] + self.cell + 2.0 * pad,
                            lo[1] + self.cell + 2.0 * pad,
                            lo[2] + self.cell + 2.0 * pad,
                        ];
                        #[allow(clippy::cast_possible_wrap)]
                        let c = [x as i64, y as i64, z as i64];
                        #[allow(clippy::cast_possible_wrap)]
                        let n = [nx as i64, ny as i64, nz as i64];
                        cand.clear();
                        let mut upper = f64::INFINITY;
                        let mut k = 0i64;
                        #[allow(clippy::cast_possible_truncation)]
                        let stamp_id = lin as u32;
                        loop {
                            self.for_each_ring_cell(c, n, k, |cell| {
                                let (s0, e0) = (
                                    self.cell_start[cell] as usize,
                                    self.cell_start[cell + 1] as usize,
                                );
                                for &i in &self.items[s0..e0] {
                                    if stamp[i as usize] == stamp_id {
                                        continue;
                                    }
                                    stamp[i as usize] = stamp_id;
                                    let b = &self.boxes[i as usize];
                                    let (dmin, dmax) = cell_box_distances(lo, hi, b);
                                    upper = upper.min(dmax);
                                    cand.push((i, dmin));
                                }
                            });
                            #[allow(clippy::cast_precision_loss)]
                            let next_ring_min = k as f64 * self.cell;
                            let max_k =
                                (0..3).map(|a| c[a].max(n[a] - 1 - c[a])).max().unwrap_or(0);
                            if next_ring_min > upper || k >= max_k {
                                break;
                            }
                            k += 1;
                        }
                        let limit = upper * (1.0 + 1e-12) + pad;
                        let before = items.len();
                        items.extend(cand.iter().filter(|(_, d)| *d <= limit).map(|(i, _)| *i));
                        // Cells with long lists (dense duplicates, via arrays) or beyond
                        // the memory budget stay uncached: an empty list means "use the
                        // ring search", which is exact either way.
                        let len = items.len() - before;
                        if len > Self::NEAR_MAX_PER_CELL || items.len() > budget {
                            items.truncate(before);
                        } else {
                            items[before..].sort_unstable();
                        }
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    start.push(items.len() as u32);
                }
            }
        }
        self.near_start = start;
        self.near_items = items;
    }

    /// Calls `f` for every in-grid cell of Chebyshev ring `k` around `c`.
    fn for_each_ring_cell(&self, c: [i64; 3], n: [i64; 3], k: i64, mut f: impl FnMut(usize)) {
        let lo = |a: usize| (c[a] - k).max(0);
        let hi = |a: usize| (c[a] + k).min(n[a] - 1);
        for z in lo(2)..=hi(2) {
            let z_on = (z - c[2]).abs() == k;
            for y in lo(1)..=hi(1) {
                let y_on = (y - c[1]).abs() == k;
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let row = |x: i64| self.linear(x as usize, y as usize, z as usize);
                if z_on || y_on || k == 0 {
                    for x in lo(0)..=hi(0) {
                        f(row(x));
                    }
                } else {
                    for x in [c[0] - k, c[0] + k] {
                        if (0..n[0]).contains(&x) {
                            f(row(x));
                        }
                    }
                }
            }
        }
    }

    /// Exact chessboard (L∞) distance transform over cells, in cell units.
    ///
    /// Two-pass chamfer with the full 3×3×3 neighbourhood, which is exact for
    /// the chessboard metric.
    fn chessboard_transform(&self) -> Vec<u32> {
        let [nx, ny, nz] = self.dims;
        let mut d: Vec<u32> = (0..nx * ny * nz)
            .map(|c| {
                if self.cell_start[c] == self.cell_start[c + 1] {
                    u32::MAX
                } else {
                    0
                }
            })
            .collect();
        let relax = |d: &mut Vec<u32>, x: usize, y: usize, z: usize, forward: bool| {
            let c = self.linear(x, y, z);
            let mut best = d[c];
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        // Forward pass looks at already-visited (lexicographically smaller) neighbours.
                        let key = (dz, dy, dx);
                        let earlier = key < (0, 0, 0);
                        if key == (0, 0, 0) || earlier != forward {
                            continue;
                        }
                        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                        let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                        #[allow(clippy::cast_possible_wrap)]
                        let inside = (0..nx as i64).contains(&xx)
                            && (0..ny as i64).contains(&yy)
                            && (0..nz as i64).contains(&zz);
                        if inside {
                            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                            let n = d[self.linear(xx as usize, yy as usize, zz as usize)];
                            best = best.min(n.saturating_add(1));
                        }
                    }
                }
            }
            d[c] = best;
        };
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    relax(&mut d, x, y, z, true);
                }
            }
        }
        for z in (0..nz).rev() {
            for y in (0..ny).rev() {
                for x in (0..nx).rev() {
                    relax(&mut d, x, y, z, false);
                }
            }
        }
        d
    }

    /// Linear index of cell `c`, or `None` outside the grid.
    #[inline]
    fn cell_index(&self, c: [i64; 3]) -> Option<usize> {
        let inside = (0..3).all(|a| {
            #[allow(clippy::cast_possible_wrap)]
            let n = self.dims[a] as i64;
            (0..n).contains(&c[a])
        });
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        inside.then(|| self.linear(c[0] as usize, c[1] as usize, c[2] as usize))
    }

    /// Clearance (in cells) of the cell containing `c`, or `None` if `c` is outside the grid.
    #[inline]
    fn cell_clearance(&self, c: [i64; 3]) -> Option<u32> {
        let inside = (0..3).all(|a| {
            #[allow(clippy::cast_possible_wrap)]
            let n = self.dims[a] as i64;
            (0..n).contains(&c[a])
        });
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        inside.then(|| self.clearance[self.linear(c[0] as usize, c[1] as usize, c[2] as usize)])
    }

    /// Cell edge length in µm.
    #[must_use]
    pub const fn cell_size(&self) -> f64 {
        self.cell
    }

    /// Number of cells along each axis.
    #[must_use]
    pub const fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Box ids registered in the cell containing `p` (empty outside the grid).
    ///
    /// Every box that contains `p` (closed) is in this list.
    #[must_use]
    pub fn items_at(&self, p: [f64; 3]) -> &[u32] {
        let mut c = self.cell_of(p);
        for a in 0..3 {
            #[allow(clippy::cast_possible_wrap)]
            let n = self.dims[a] as i64;
            // Points on the closed upper bound belong to the last cell, matching how
            // box registrations are clamped.
            if c[a] >= n && p[a] <= self.bounds.max()[a] {
                c[a] = n - 1;
            }
            if !(0..n).contains(&c[a]) {
                return &[];
            }
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let l = self.linear(c[0] as usize, c[1] as usize, c[2] as usize);
        &self.items[self.cell_start[l] as usize..self.cell_start[l + 1] as usize]
    }

    /// Box ids registered in each cell holding at least two boxes (broad-phase pair tests).
    pub fn cell_items(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.cell_start
            .windows(2)
            .map(|w| &self.items[w[0] as usize..w[1] as usize])
            .filter(|c| c.len() > 1)
    }

    /// Read-only view of the internal arrays, for uploading the index to other
    /// devices (e.g. a GPU). Query semantics are those of
    /// [`SpatialIndex::nearest_linf`] and [`SpatialIndex::clearance_lower_bound`].
    #[must_use]
    pub fn raw(&self) -> GridData<'_> {
        GridData {
            boxes: &self.boxes,
            origin: self.origin,
            cell: self.cell,
            dims: self.dims,
            cell_start: &self.cell_start,
            items: &self.items,
            clearance: &self.clearance,
            near_start: &self.near_start,
            near_items: &self.near_items,
            bounds: self.bounds,
        }
    }

    /// Total number of (cell, box) registrations, a measure of memory use.
    #[must_use]
    pub fn num_registrations(&self) -> usize {
        self.items.len()
    }

    /// Unclamped integer cell coordinate of `p`.
    #[inline]
    fn cell_of(&self, p: [f64; 3]) -> [i64; 3] {
        // Saturating float-to-int conversion is fine: far-away points just get large indices.
        #[allow(clippy::cast_possible_truncation)]
        std::array::from_fn(|a| ((p[a] - self.origin[a]) * self.inv_cell).floor() as i64)
    }

    fn cell_range(&self, b: &Aabb) -> ([usize; 3], [usize; 3]) {
        let lo = self.cell_of(b.min());
        let hi = self.cell_of(b.max());
        let clamp = |v: i64, a: usize| -> usize {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_possible_wrap
            )]
            (v.clamp(0, self.dims[a] as i64 - 1) as usize)
        };
        (
            std::array::from_fn(|a| clamp(lo[a], a)),
            std::array::from_fn(|a| clamp(hi[a], a)),
        )
    }

    fn for_each_cell(&self, lo: [usize; 3], hi: [usize; 3], mut f: impl FnMut(usize)) {
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    f(self.linear(x, y, z));
                }
            }
        }
    }

    #[inline]
    const fn linear(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    /// Conservative L∞ distance from `p` to the outside of the `(2k+1)³` cell
    /// block centred on cell `c`: every box not registered in that block is at
    /// least this far away. This is at least `k·h` and up to `(k+1)·h`, so a
    /// point deep inside its cell can stop after ring 0.
    #[inline]
    /// Magnitude of the quantities rounded in `cell_of`, box registration and
    /// the block bounds at ring `k`: guards scale with it.
    fn round_scale(&self, p: [f64; 3], k: i64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let mut scale = self.cell * k as f64;
        for a in 0..3 {
            scale = scale
                .max(p[a].abs())
                .max(self.origin[a].abs())
                .max((p[a] - self.origin[a]).abs());
        }
        scale
    }

    fn block_reach(&self, p: [f64; 3], c: [i64; 3], k: i64) -> f64 {
        let mut reach = f64::INFINITY;
        for a in 0..3 {
            #[allow(clippy::cast_precision_loss)]
            let lo = self.origin[a] + (c[a] - k) as f64 * self.cell;
            #[allow(clippy::cast_precision_loss)]
            let hi = self.origin[a] + (c[a] + k + 1) as f64 * self.cell;
            reach = reach.min(p[a] - lo).min(hi - p[a]);
        }
        let scale = self.round_scale(p, k + 1);
        // Guard against rounding in `cell_of` and in box registration.
        #[allow(clippy::cast_precision_loss)]
        let floor = (k as f64 * self.cell - 1e-12 * scale).max(0.0);
        (reach - 1e-12 * scale).max(floor.min(reach.max(0.0)))
    }

    #[inline]
    fn visit_cell(
        &self,
        p: [f64; 3],
        max_r: f64,
        c: usize,
        keep: &impl Fn(u32) -> bool,
        best: &mut Option<Nearest>,
    ) {
        let (s, e) = (self.cell_start[c] as usize, self.cell_start[c + 1] as usize);
        for &i in &self.items[s..e] {
            if keep(i) {
                self.consider(p, max_r, i, best);
            }
        }
    }

    #[inline]
    fn consider(&self, p: [f64; 3], max_r: f64, i: u32, best: &mut Option<Nearest>) {
        let distance = self.boxes[i as usize].linf_distance(p);
        if distance > max_r {
            return;
        }
        let cand = Nearest {
            distance,
            box_id: BoxId::new(i),
        };
        if best.is_none_or(|b| cand.better_than(&b)) {
            *best = Some(cand);
        }
    }
}

/// The boxes of a [`UniformGrid`] within an L∞ radius `radius` of a centre,
/// for repeated exact queries near it ([`UniformGrid::local`]).
///
/// A query at `p` with radius `max_r` is exact when
/// `‖p − centre‖∞ + max_r ≤ radius`: every box within `max_r` of `p` is then
/// within `radius` of the centre (L∞ triangle inequality), and is listed.
#[derive(Debug, Clone, Default)]
pub struct LocalBoxes {
    centre: [f64; 3],
    radius: f64,
    ids: Vec<u32>,
    boxes: Vec<Aabb>,
    /// Per box of the grid, the call of [`UniformGrid::local`] that last
    /// saw it (`epoch`).
    seen: Vec<u32>,
    epoch: u32,
}

impl LocalBoxes {
    /// `true` if a query at `p` with radius `max_r` is exact on this list.
    /// A relative margin of 1e-12 of the coordinates covers rounding in the
    /// triangle inequality.
    #[inline]
    #[must_use]
    pub fn covers(&self, p: [f64; 3], max_r: f64) -> bool {
        let mut off = 0.0_f64;
        let mut scale = self.radius;
        for a in 0..3 {
            off = off.max((p[a] - self.centre[a]).abs());
            scale = scale.max(p[a].abs()).max(self.centre[a].abs());
        }
        off + max_r <= self.radius - 1e-12 * scale
    }

    /// Like [`UniformGrid::nearest_linf_where`]; exact if
    /// [`Self::covers`]`(p, max_r)`. Ties are broken by the smaller box id.
    #[inline]
    pub fn nearest_linf_where(
        &self,
        p: [f64; 3],
        max_r: f64,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Nearest> {
        let mut best: Option<Nearest> = None;
        for (b, &i) in self.boxes.iter().zip(&self.ids) {
            let distance = b.linf_distance(p);
            if distance > max_r || !keep(i) {
                continue;
            }
            let cand = Nearest {
                distance,
                box_id: BoxId::new(i),
            };
            if best.is_none_or(|b| cand.better_than(&b)) {
                best = Some(cand);
            }
        }
        best
    }

    /// The nearest box within `max_r` of `p`, and the nearest one whose
    /// `key` differs from the first's (ties by the smaller id), in one pass;
    /// exact if [`Self::covers`]`(p, max_r)`.
    #[inline]
    pub fn nearest_two(
        &self,
        p: [f64; 3],
        max_r: f64,
        key: impl Fn(u32) -> u64,
    ) -> (Option<Nearest>, Option<Nearest>) {
        let mut first: Option<(Nearest, u64)> = None;
        let mut second: Option<Nearest> = None;
        for (b, &i) in self.boxes.iter().zip(&self.ids) {
            let distance = b.linf_distance(p);
            if distance > max_r {
                continue;
            }
            let cand = Nearest {
                distance,
                box_id: BoxId::new(i),
            };
            let k = key(i);
            match first {
                Some((f, fk)) if !cand.better_than(&f) => {
                    if k != fk && second.is_none_or(|s| cand.better_than(&s)) {
                        second = Some(cand);
                    }
                }
                Some((f, fk)) => {
                    // The old first is better than every other box seen, so
                    // it is the best with a key other than `k`.
                    if fk != k {
                        second = Some(f);
                    }
                    first = Some((cand, k));
                }
                None => first = Some((cand, k)),
            }
        }
        (first.map(|f| f.0), second)
    }

    /// [`Self::nearest_two`] for a list of faces normal to the axis `normal`
    /// (boxes of zero extent along it): the same answers.
    #[inline]
    pub fn nearest_two_normal(
        &self,
        p: [f64; 3],
        max_r: f64,
        normal: usize,
        key: impl Fn(u32) -> u64,
    ) -> (Option<Nearest>, Option<Nearest>) {
        // One loop per axis, so the plane coordinate is a fixed field.
        match normal {
            0 => self.two_normal::<0>(p, max_r, key),
            1 => self.two_normal::<1>(p, max_r, key),
            _ => self.two_normal::<2>(p, max_r, key),
        }
    }

    #[inline]
    fn two_normal<const N: usize>(
        &self,
        p: [f64; 3],
        max_r: f64,
        key: impl Fn(u32) -> u64,
    ) -> (Option<Nearest>, Option<Nearest>) {
        let mut first: Option<(Nearest, u64)> = None;
        let mut second: Option<Nearest> = None;
        for (b, &i) in self.boxes.iter().zip(&self.ids) {
            // A face normal to `N` is at least its plane's distance away
            // (exactly that component of `linf_distance`): most listed faces
            // fail on it, before the full distance.
            if (p[N] - b.min()[N]).abs() > max_r {
                continue;
            }
            let distance = b.linf_distance(p);
            if distance > max_r {
                continue;
            }
            let cand = Nearest {
                distance,
                box_id: BoxId::new(i),
            };
            let k = key(i);
            match first {
                Some((f, fk)) if !cand.better_than(&f) => {
                    if k != fk && second.is_none_or(|s| cand.better_than(&s)) {
                        second = Some(cand);
                    }
                }
                Some((f, fk)) => {
                    // The old first is better than every other box seen, so
                    // it is the best with a key other than `k`.
                    if fk != k {
                        second = Some(f);
                    }
                    first = Some((cand, k));
                }
                None => first = Some((cand, k)),
            }
        }
        (first.map(|f| f.0), second)
    }

    /// The number of listed boxes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// `true` if no box is listed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl UniformGrid {
    /// The boxes within L∞ distance `radius` of `centre` (see [`LocalBoxes`]),
    /// written into `out` (its buffers are reused).
    pub fn local(&self, centre: [f64; 3], radius: f64, out: &mut LocalBoxes) {
        out.centre = centre;
        out.radius = radius;
        out.ids.clear();
        out.boxes.clear();
        // The cells of the query cube, one cell wider on each side against
        // rounding in the cell assignment of boxes.
        let lo = self.cell_of(std::array::from_fn(|a| centre[a] - radius));
        let hi = self.cell_of(std::array::from_fn(|a| centre[a] + radius));
        #[allow(clippy::cast_possible_wrap)]
        let n: [i64; 3] = self.dims.map(|d| d as i64);
        if (0..3).any(|a| hi[a] < -1 || lo[a] > n[a]) {
            return;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let clamp = |v: i64, a: usize| v.clamp(0, n[a] - 1) as usize;
        let lo = std::array::from_fn(|a| clamp(lo[a] - 1, a));
        let hi = std::array::from_fn(|a| clamp(hi[a] + 1, a));
        // A box spanning many cells is listed in each; the stamps of the
        // boxes seen in this call skip the repeats.
        if out.seen.len() != self.boxes.len() {
            out.seen.clear();
            out.seen.resize(self.boxes.len(), 0);
            out.epoch = 0;
        }
        out.epoch = out.epoch.wrapping_add(1);
        if out.epoch == 0 {
            out.seen.fill(0);
            out.epoch = 1;
        }
        let (ids, seen, epoch) = (&mut out.ids, &mut out.seen, out.epoch);
        self.for_each_cell(lo, hi, |c| {
            let (s, e) = (self.cell_start[c] as usize, self.cell_start[c + 1] as usize);
            for &i in &self.items[s..e] {
                let k = i as usize;
                if seen[k] != epoch {
                    seen[k] = epoch;
                    if self.boxes[k].linf_distance(centre) <= radius {
                        ids.push(i);
                    }
                }
            }
        });
        ids.sort_unstable();
        out.boxes
            .extend(ids.iter().map(|&i| self.boxes[i as usize]));
    }
}

impl SpatialIndex for UniformGrid {
    fn nearest_linf(&self, p: [f64; 3], max_r: f64) -> Option<Nearest> {
        let c = self.cell_of(p);
        if !self.near_start.is_empty() {
            if let Some(lin) = self.cell_index(c) {
                let (s, e) = (
                    self.near_start[lin] as usize,
                    self.near_start[lin + 1] as usize,
                );
                if s < e {
                    let mut best = None;
                    for &i in &self.near_items[s..e] {
                        self.consider(p, max_r, i, &mut best);
                    }
                    return best;
                }
            }
        }
        self.ring_search(p, max_r, &|_| true)
    }

    fn clearance_lower_bound(&self, p: [f64; 3]) -> f64 {
        let c = self.cell_of(p);
        match self.cell_clearance(c) {
            #[allow(clippy::cast_precision_loss)]
            Some(m) => f64::from(m.saturating_sub(1)) * self.cell,
            // Outside the grid: all boxes lie inside the bounds.
            None => self.bounds.linf_distance(p),
        }
    }
}

impl UniformGrid {
    /// Like [`SpatialIndex::nearest_linf`], restricted to the boxes for which
    /// `keep(box_id)` is `true`. Ties are broken by the smaller box id.
    ///
    /// This always uses the ring search (the near-cell lists are built for the
    /// unfiltered query), so it is slower than the unfiltered query.
    pub fn nearest_linf_where(
        &self,
        p: [f64; 3],
        max_r: f64,
        keep: impl Fn(u32) -> bool,
    ) -> Option<Nearest> {
        self.ring_search(p, max_r, &keep)
    }

    /// Exact ring search over the grid cells, considering only boxes kept by `keep`.
    fn ring_search(&self, p: [f64; 3], max_r: f64, keep: &impl Fn(u32) -> bool) -> Option<Nearest> {
        let c = self.cell_of(p);
        #[allow(clippy::cast_possible_wrap)]
        let n: [i64; 3] = self.dims.map(|d| d as i64);
        // Rings closer than `k_start` contain no in-grid cells; rings beyond `k_end` neither.
        let k_start = (0..3)
            .map(|a| (-c[a]).max(c[a] - (n[a] - 1)).max(0))
            .max()
            .unwrap_or(0);
        let k_end = (0..3)
            .map(|a| c[a].max(n[a] - 1 - c[a]).max(0))
            .max()
            .unwrap_or(0);
        // Rings closer than the cell's clearance contain no boxes either.
        let k_start = match self.cell_clearance(c) {
            Some(m) => k_start.max(i64::from(m)),
            None => k_start,
        };
        let mut best: Option<Nearest> = None;

        let mut k = k_start;
        loop {
            // Unvisited boxes are strictly farther than (k - 1) * h before visiting
            // ring k, up to rounding in `cell_of` and in box registration (the
            // guard of `block_reach`).
            #[allow(clippy::cast_precision_loss)]
            let lower = ((k - 1).max(0) as f64).mul_add(self.cell, -1e-12 * self.round_scale(p, k));
            // Also before the first ring: a cell clearance beyond `max_r`
            // leaves nothing to find.
            if lower >= max_r {
                break;
            }
            self.visit_ring(p, max_r, c, n, k, keep, &mut best);
            let reach = self.block_reach(p, c, k);
            if best.is_some_and(|b| b.distance <= reach) || k >= k_end || reach >= max_r {
                break;
            }
            k += 1;
        }
        best
    }

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_arguments
    )]
    fn visit_ring(
        &self,
        p: [f64; 3],
        max_r: f64,
        c: [i64; 3],
        n: [i64; 3],
        k: i64,
        keep: &impl Fn(u32) -> bool,
        best: &mut Option<Nearest>,
    ) {
        let lo = |a: usize| (c[a] - k).max(0);
        let hi = |a: usize| (c[a] + k).min(n[a] - 1);
        let full_row = |y: i64, z: i64, best: &mut Option<Nearest>| {
            for x in lo(0)..=hi(0) {
                self.visit_cell(
                    p,
                    max_r,
                    self.linear(x as usize, y as usize, z as usize),
                    keep,
                    best,
                );
            }
        };
        // The ring's x faces (cells x = c ± k); none if both lie outside the grid.
        let mut faces = [0i64; 2];
        let mut nf = 0;
        for x in [c[0] - k, c[0] + k] {
            if (0..n[0]).contains(&x) && (nf == 0 || faces[0] != x) {
                faces[nf] = x;
                nf += 1;
            }
        }
        let x_faces = &faces[..nf];
        for z in lo(2)..=hi(2) {
            if (z - c[2]).abs() == k {
                for y in lo(1)..=hi(1) {
                    full_row(y, z, best);
                }
                continue;
            }
            if x_faces.is_empty() {
                // Only the two y faces of this layer can be in the grid: skip the
                // interior rows instead of scanning them (flat grids).
                for y in [c[1] - k, c[1] + k] {
                    if (0..n[1]).contains(&y) {
                        full_row(y, z, best);
                    }
                }
                continue;
            }
            for y in lo(1)..=hi(1) {
                if (y - c[1]).abs() == k {
                    full_row(y, z, best);
                } else {
                    for &x in x_faces {
                        self.visit_cell(
                            p,
                            max_r,
                            self.linear(x as usize, y as usize, z as usize),
                            keep,
                            best,
                        );
                    }
                }
            }
        }
    }
}

/// `(min, max)` over points of the closed box `[lo, hi]` of the L∞ distance to `b`.
fn cell_box_distances(lo: [f64; 3], hi: [f64; 3], b: &Aabb) -> (f64, f64) {
    let (bl, bh) = (b.min(), b.max());
    let mut dmin = 0.0_f64;
    let mut dmax = 0.0_f64;
    for a in 0..3 {
        dmin = dmin.max(bl[a] - hi[a]).max(lo[a] - bh[a]);
        // The per-axis distance max(bl - p, p - bh, 0) is convex in p: maximal at an end.
        let at = |p: f64| (bl[a] - p).max(p - bh[a]).max(0.0);
        dmax = dmax.max(at(lo[a]).max(at(hi[a])));
    }
    (dmin.max(0.0), dmax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BruteForce;

    fn boxes() -> Vec<Aabb> {
        vec![
            Aabb::from_array([0.0, 0.0, 0.0, 10.0, 0.1, 0.1]).unwrap(),
            Aabb::from_array([0.0, 1.0, 0.0, 10.0, 1.1, 0.1]).unwrap(),
            Aabb::from_array([5.0, 5.0, 2.0, 5.5, 5.5, 2.5]).unwrap(),
        ]
    }

    #[test]
    fn matches_brute_force_on_simple_points() {
        let bx = boxes();
        let grid = UniformGrid::from_boxes(
            &bx,
            GridOptions {
                cell_size: Some(0.3),
                ..Default::default()
            },
        );
        let brute = BruteForce::from_boxes(&bx);
        for p in [
            [0.5, 0.5, 0.05],
            [5.2, 5.2, 1.0],
            [-20.0, 3.0, 9.0],
            [10.0, 1.05, 0.05],
            [5.25, 5.25, 2.25],
        ] {
            for r in [0.01, 0.5, 3.0, f64::INFINITY] {
                assert_eq!(
                    grid.nearest_linf(p, r),
                    brute.nearest_linf(p, r),
                    "p={p:?} r={r}"
                );
            }
        }
    }

    #[test]
    fn auto_cell_size_respects_cap() {
        let bx = boxes();
        let grid = UniformGrid::from_boxes(
            &bx,
            GridOptions {
                max_cells: 8,
                ..Default::default()
            },
        );
        assert!(grid.dims().iter().product::<usize>() <= 8);
        assert!(grid.num_registrations() >= bx.len());
        assert!(grid.cell_size() > 0.0);
    }
}

#[cfg(test)]
mod transform_tests {
    use super::*;

    /// The two-pass chamfer must equal a brute-force chessboard transform.
    #[test]
    fn chessboard_transform_is_exact() {
        let boxes: Vec<Aabb> = [
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            [7.2, 3.1, 1.0, 7.6, 9.0, 1.4],
            [3.0, 8.5, 4.0, 3.3, 8.9, 6.0],
        ]
        .into_iter()
        .map(|c| Aabb::from_array(c).unwrap())
        .collect();
        let grid = UniformGrid::from_boxes(
            &boxes,
            GridOptions {
                cell_size: Some(0.7),
                ..Default::default()
            },
        );
        let [nx, ny, nz] = grid.dims();
        let occupied: Vec<[usize; 3]> = (0..nz)
            .flat_map(|z| (0..ny).flat_map(move |y| (0..nx).map(move |x| [x, y, z])))
            .filter(|&[x, y, z]| {
                let c = grid.linear(x, y, z);
                grid.cell_start[c] != grid.cell_start[c + 1]
            })
            .collect();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let brute = occupied
                        .iter()
                        .map(|o| x.abs_diff(o[0]).max(y.abs_diff(o[1])).max(z.abs_diff(o[2])))
                        .min()
                        .unwrap();
                    assert_eq!(
                        grid.clearance[grid.linear(x, y, z)] as usize,
                        brute,
                        "cell {x},{y},{z}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod items_at_tests {
    use super::*;

    #[test]
    fn upper_bound_points_see_touching_boxes() {
        let bx = vec![Aabb::from_array([0.0, 0.0, 0.0, 6.0, 2.0, 2.0]).unwrap(); 2];
        let g = UniformGrid::from_boxes(
            &bx,
            GridOptions {
                cell_size: Some(6.0),
                ..GridOptions::default()
            },
        );
        assert_eq!(g.items_at([6.0, 1.0, 1.0]).len(), 2);
        assert!(g.items_at([6.0 + 1e-9, 1.0, 1.0]).is_empty());
    }

    #[test]
    fn dense_duplicates_do_not_blow_up_the_cache() {
        let mut bx = vec![Aabb::from_array([5.5, 5.5, 5.5, 5.6, 5.6, 5.6]).unwrap(); 2000];
        bx.push(Aabb::from_array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]).unwrap());
        bx.push(Aabb::from_array([10.0, 10.0, 10.0, 10.1, 10.1, 10.1]).unwrap());
        let g = UniformGrid::from_boxes(
            &bx,
            GridOptions {
                cell_size: Some(1.0),
                ..GridOptions::default()
            },
        );
        // Without the per-cell cap, each duplicate would be listed in ~125 cells.
        assert!(
            g.near_items.len() < 125 * 2000 / 10,
            "{}",
            g.near_items.len()
        );
        for p in [
            [5.0, 5.0, 5.0],
            [6.2, 5.55, 4.9],
            [1.0, 0.5, 0.2],
            [9.5, 9.9, 10.5],
        ] {
            let exact = bx
                .iter()
                .map(|b| b.linf_distance(p))
                .fold(f64::INFINITY, f64::min);
            let got = g.nearest_linf(p, f64::INFINITY).map(|n| n.distance);
            assert_eq!(got, Some(exact), "{p:?}");
        }
    }
}
