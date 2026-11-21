# faradice-gds

GDSII stream input for faradice:

- a dependency-light GDSII reader (BOUNDARY, PATH, BOX, TEXT, SREF, AREF),
- hierarchy flattening with exact integer coordinates (90° rotations, mirroring, unit magnification),
- Manhattan polygon and path decomposition into rectangles,
- a technology file mapping GDS layers to conductor layers, vias and label layers,
- connectivity tracing: physical components first (positive-area contact in 3D), then net names
  from text labels, with diagnostics for conflicting and repeated labels.

Licensed under either of MIT or Apache-2.0 at your option.
