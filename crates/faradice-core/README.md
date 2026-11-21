# faradice-core

Core, dependency-light types shared by all faradice crates:

- `units`: length units, vacuum permittivity in fF/µm
- `ids`: `NetId`, `BoxId` newtypes
- `geom`: `Axis`, `Face`
- `sink`: where a random walk terminates (`Sink`)
- `stats`: deterministic, mergeable weighted-sample accumulators
- `error`: the shared `Error` type

Licensed under either of MIT or Apache-2.0 at your option.
