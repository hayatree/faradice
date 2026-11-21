# faradice-geom

Conductor geometry for faradice:

- `Aabb`: validated axis-aligned box with L∞ (Chebyshev) distance queries
- `Geometry` / `GeometryBuilder`: boxes grouped into named nets, with validation
  (no overlapping or touching boxes of different nets)
- `SpatialIndex` and `UniformGrid`: exact nearest-box queries in the L∞ metric, which give
  the largest conductor-free transition cube around a point

Licensed under either of MIT or Apache-2.0 at your option.
