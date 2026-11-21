//! Connectivity tracing and net naming.
//!
//! 1. Every mapped shape is decomposed into rectangles and lifted to a 3D box
//!    using its layer's z span (vias span from the top of their bottom layer
//!    to the bottom of their top layer).
//! 2. Physical components are built with union-find: two boxes connect iff
//!    they share a contact of **positive area** (overlapping volumes or
//!    touching faces with overlapping extents). Edge- or corner-only contacts
//!    do not connect. Vias therefore connect exactly the layers they touch.
//! 3. Text labels attach to the component whose box on the label's conductor
//!    layer contains the label point. Conflicting names on one component and
//!    repeated names on disconnected components are reported, never silently
//!    merged (unless `join_same_name` is set).

use std::collections::{BTreeMap, HashMap};

use faradice_geom::{Aabb, Geometry, GeometryBuilder, GridOptions, UniformGrid};

use crate::Result;
use crate::flatten::Flat;
use crate::manhattan::shape_rects;
use crate::stream::Units;
use crate::tech::Tech;

/// Tracing options.
#[derive(Debug, Clone, Copy, Default)]
pub struct TraceOptions {
    /// Only use labels defined at hierarchy depth ≤ this (None = all depths).
    pub max_label_depth: Option<u32>,
    /// Merge disconnected components that carry the same label.
    pub join_same_name: bool,
}

/// A diagnostic produced while tracing.
#[derive(Debug, Clone, PartialEq)]
pub enum Diagnostic {
    /// Several different labels on one physical component (a short, or a label error).
    ConflictingLabels {
        /// Chosen name.
        chosen: String,
        /// All names found.
        names: Vec<String>,
    },
    /// One name on several disconnected components.
    DisconnectedNet {
        /// Name.
        name: String,
        /// Number of components.
        parts: usize,
    },
    /// A label that touches no shape on its layer.
    DanglingLabel {
        /// Label text.
        name: String,
        /// Layer name.
        layer: String,
        /// Position (µm).
        at: [f64; 2],
    },
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConflictingLabels { chosen, names } => {
                write!(
                    f,
                    "labels {names:?} are on one connected component (named `{chosen}`): short or label error"
                )
            }
            Self::DisconnectedNet { name, parts } => {
                write!(
                    f,
                    "net `{name}` consists of {parts} disconnected components (named {name}, {name}:2, …)"
                )
            }
            Self::DanglingLabel { name, layer, at } => {
                write!(
                    f,
                    "label `{name}` on {layer} at ({:.4}, {:.4}) µm touches no shape",
                    at[0], at[1]
                )
            }
        }
    }
}

/// One physically connected component.
#[derive(Debug, Clone)]
pub struct Component {
    /// Net name (label, label with `:k` suffix, or `$k` if unlabeled).
    pub name: String,
    /// Indices into [`Traced::boxes`].
    pub boxes: Vec<usize>,
    /// Labels attached to this component.
    pub labels: Vec<String>,
}

/// Tracing result.
#[derive(Debug, Clone)]
pub struct Traced {
    /// All conductor boxes (µm).
    pub boxes: Vec<Aabb>,
    /// Span index (conductor or via, see [`Tech::span_name`]) of each box.
    pub box_span: Vec<usize>,
    /// Components, ordered by their smallest box index.
    pub components: Vec<Component>,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

struct Dsu(Vec<usize>);

impl Dsu {
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.0[hi] = lo;
        }
    }
}

/// `true` if two closed boxes touch with positive contact area.
#[must_use]
pub fn positive_contact(a: &Aabb, b: &Aabb) -> bool {
    let mut zero = 0;
    for t in 0..3 {
        let o = a.max()[t].min(b.max()[t]) - a.min()[t].max(b.min()[t]);
        if o < 0.0 {
            return false;
        }
        if o == 0.0 {
            zero += 1;
        }
    }
    zero <= 1
}

/// Traces connectivity of a flattened layout.
pub fn trace(flat: &Flat, units: Units, tech: &Tech, opts: TraceOptions) -> Result<Traced> {
    // half-dbu → µm
    let k = units.db_in_meters * 1e6 / 2.0;
    #[allow(clippy::cast_precision_loss)]
    let um = |v: i64| v as f64 * k;
    let mut boxes = Vec::new();
    let mut box_span = Vec::new();
    for s in &flat.shapes {
        let Some(span) = tech.span(s.layer, s.datatype) else {
            continue;
        };
        for r in shape_rects(&s.kind)? {
            boxes.push(Aabb::new(
                [um(r[0]), um(r[1]), span.z0],
                [um(r[2]), um(r[3]), span.z1],
            )?);
            box_span.push(span.index);
        }
    }
    let n = boxes.len();
    let mut dsu = Dsu((0..n).collect());
    if n > 1 {
        let grid = UniformGrid::from_boxes(
            &boxes,
            GridOptions {
                max_cells: n.saturating_mul(8).max(64),
                clearance_transform: false,
                ..GridOptions::default()
            },
        );
        for items in grid.cell_items() {
            for (a, &i) in items.iter().enumerate() {
                for &j in &items[a + 1..] {
                    if positive_contact(&boxes[i as usize], &boxes[j as usize]) {
                        dsu.union(i as usize, j as usize);
                    }
                }
            }
        }
    }
    // Components ordered by smallest box index (roots are minimal indices).
    let mut comp_of_root: HashMap<usize, usize> = HashMap::new();
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let r = dsu.find(i);
        let c = *comp_of_root.entry(r).or_insert_with(|| {
            comps.push(Vec::new());
            comps.len() - 1
        });
        comps[c].push(i);
    }
    let box_comp: Vec<usize> = (0..n).map(|i| comp_of_root[&dsu.find(i)]).collect();

    // Labels.
    let mut diagnostics = Vec::new();
    let mut comp_labels: Vec<Vec<String>> = vec![Vec::new(); comps.len()];
    let label_grid = (n > 0).then(|| {
        UniformGrid::from_boxes(
            &boxes,
            GridOptions {
                max_cells: n.saturating_mul(8).max(64),
                clearance_transform: false,
                ..GridOptions::default()
            },
        )
    });
    for t in &flat.texts {
        if opts.max_label_depth.is_some_and(|d| t.depth > d) {
            continue;
        }
        let Some(layer) = tech.label_layer(t.layer, t.texttype) else {
            continue;
        };
        let l = &tech.layers[layer];
        let p = [um(t.pos[0]), um(t.pos[1]), l.zmin + 0.5 * l.thickness];
        let hit = label_grid.as_ref().and_then(|g| {
            g.items_at(p)
                .iter()
                .map(|&i| i as usize)
                .find(|&i| box_span[i] == layer && boxes[i].contains(p))
        });
        match hit {
            Some(i) => {
                let c = box_comp[i];
                if !comp_labels[c].contains(&t.string) {
                    comp_labels[c].push(t.string.clone());
                }
            }
            None => diagnostics.push(Diagnostic::DanglingLabel {
                name: t.string.clone(),
                layer: l.name.clone(),
                at: [p[0], p[1]],
            }),
        }
    }
    // Primary name per component.
    let mut primary: Vec<Option<String>> = Vec::with_capacity(comps.len());
    for labels in &mut comp_labels {
        labels.sort();
        if labels.len() > 1 {
            diagnostics.push(Diagnostic::ConflictingLabels {
                chosen: labels[0].clone(),
                names: labels.clone(),
            });
        }
        primary.push(labels.first().cloned());
    }
    // Same name on several components.
    let mut by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (c, p) in primary.iter().enumerate() {
        if let Some(p) = p {
            by_name.entry(p.clone()).or_default().push(c);
        }
    }
    let mut names: Vec<String> = (0..comps.len()).map(|c| format!("${c}")).collect();
    let mut merged_into: Vec<usize> = (0..comps.len()).collect();
    for (name, cs) in &by_name {
        if cs.len() > 1 && !opts.join_same_name {
            diagnostics.push(Diagnostic::DisconnectedNet {
                name: name.clone(),
                parts: cs.len(),
            });
        }
        for (k, &c) in cs.iter().enumerate() {
            if opts.join_same_name {
                names[c] = name.clone();
                merged_into[c] = cs[0];
            } else {
                names[c] = if k == 0 {
                    name.clone()
                } else {
                    format!("{name}:{}", k + 1)
                };
            }
        }
    }
    let mut components: Vec<Component> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    for c in 0..comps.len() {
        let target = merged_into[c];
        let s = *slot.entry(target).or_insert_with(|| {
            components.push(Component {
                name: names[target].clone(),
                boxes: Vec::new(),
                labels: Vec::new(),
            });
            components.len() - 1
        });
        components[s].boxes.extend_from_slice(&comps[c]);
        for l in &comp_labels[c] {
            if !components[s].labels.contains(l) {
                components[s].labels.push(l.clone());
            }
        }
    }
    Ok(Traced {
        boxes,
        box_span,
        components,
        diagnostics,
    })
}

impl Traced {
    /// Finds a component by name.
    #[must_use]
    pub fn component(&self, name: &str) -> Option<&Component> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Builds solver geometry, keeping components accepted by `keep`.
    pub fn to_geometry(&self, keep: impl Fn(&str) -> bool) -> Result<Geometry> {
        Ok(self.to_geometry_with_spans(keep)?.0)
    }

    /// Like [`to_geometry`](Self::to_geometry), also returning the span index
    /// (see [`Tech::span_name`]) of every geometry box, indexed by box id.
    pub fn to_geometry_with_spans(
        &self,
        keep: impl Fn(&str) -> bool,
    ) -> Result<(Geometry, Vec<usize>)> {
        let mut g = GeometryBuilder::new();
        let mut spans = Vec::new();
        for c in self.components.iter().filter(|c| keep(&c.name)) {
            let net = g.net(&c.name)?;
            for &b in &c.boxes {
                g.add_box(net, self.boxes[b])?;
                spans.push(self.box_span[b]);
            }
        }
        Ok((g.build()?, spans))
    }
}
