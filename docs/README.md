# faradice documentation

## Using faradice

- [User reference](reference.md): the CLI, the input files (structure, configuration,
  technology, sensitivity parameters) and the outputs.
- [Data provenance](data-provenance.md): sources and licenses of the technology files and
  examples.
- The [top-level README](../README.md) has a quick start, and [`gui/README.md`](../gui/README.md)
  describes the desktop GUI.

## How it works: design documents

The design documents record the method behind each part of the solver, the published
papers it builds on, and the alternatives that were considered. They are written for
readers who want to understand or change the solver; you do not need them to use it. Each
starts with a status line.

The codes group the work: **M** is the solver and its dielectric models, **D** the tools
around it, **G** the GPU engine.

| Document | Topic |
|---|---|
| [M1](design/M1_frw_solver.md) | The FRW solver in a uniform dielectric: transition cubes, Gaussian surfaces, estimators, stopping, ground handling. The original design; parts of its interface proposals were not implemented (see its status line) |
| [M2](design/M2_multi_dielectric.md) (also called M2a) | Stratified dielectric stacks: exact transition cubes crossing one planar interface |
| [M2b](design/M2b_interface_anchored.md) | Walks anchored at interfaces, for thin layers |
| [M2c](design/M2c_conformal_dielectrics.md) | Dielectric boxes and conformal shells around conductors |
| [M2d](design/M2d_edge_cubes.md) | Transition cubes centred on dielectric edges |
| [M2e](design/M2e_layered_first_hops.md) | First hops between facing interfaces |
| [M2f](design/M2f_window_first_hops.md) | First hops for thin 2D cross-sections |
| [M2g](design/M2g_shell_walk_cost.md) | Cost of walks near conformal shells (ongoing) |
| [M2h](design/M2h_two_sided_landing.md) | Two-sided landing near dielectric edges (measured, not adopted) |
| [D1](design/D1_gui.md) | The desktop GUI |
| [D2](design/D2_agent_api_mcp.md) | An engine API and an MCP server for AI agents (proposal) |
| [D3](design/D3_sensitivity.md) | Capacitance sensitivities and what-if analysis |
| [G1](design/G1_gpu.md) | The experimental GPU walk engine |

## Validation reports

| Report | Covers | Case names |
|---|---|---|
| [M1](validation/M1.md) | Uniform dielectric: exact solutions, bias studies, comparison with Palace and FasterCap, scaling | V1–V8 |
| [M2](validation/M2.md) | Stratified stacks, dielectric boxes and conformal shells: exact solutions, comparison with Palace, parameter and cost studies | L1–L6, C1–C10 |
| [D3](validation/D3.md) | Sensitivities, deletions, additions, maps and what-if edits | S1–S14 |

The structures of the cases compared with other solvers are in
[`bench/compare/cases/`](../bench/compare/cases/); most others are built in the tests of
`crates/faradice-frw/tests/` and `crates/faradice-sens/tests/`.

## Related work

- [Capacitance extraction methods](research/01_capacitance_extraction_survey.md): a survey
  of field solvers, pattern-based extraction, FRW and machine learning approaches.
- [Open-source extractors](research/02_oss_landscape_novelty.md): existing open-source
  tools and how faradice relates to them.
