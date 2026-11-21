# faradice-sens

Capacitance sensitivities and what-if analysis for
[faradice](https://github.com/hayatree/faradice), behind `faradice sens`, `faradice whatif`
and `faradice map`. Results carry propagated statistical errors (σ); see the user
reference for the exceptions.

- Parameter files (`*.sens.toml`): faces selected by box or by a point on them, weights,
  translations and process sigmas.
- `dC/dp` by finite differences with path continuation, or by shape derivatives on faces
  without geometric edges.
- Exact deletions and additions of conductors (additions by replaying the recorded walks).
- Insertion maps: where a small grounded or floating object would change a coupling most.
- Exact edits of an extracted Maxwell matrix: ground, float, merge or swap conductors
  without re-solving.

The methods are described in [`docs/design/D3_sensitivity.md`](../../docs/design/D3_sensitivity.md);
the file format is in the [user reference](../../docs/reference.md).
