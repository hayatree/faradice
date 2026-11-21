# faradice D2 Design: Engine API and MCP Server for AI Agents

- Date: 2026-11-15
- Status: Partially implemented (open questions in §12). Implemented: the shared run engine
  `faradice-engine` (projects, runs, cancellable background jobs with events; used by the CLI and
  the GUI) and `--json` output of `solve`, `report`, `sens`, `whatif` and `gds trace`. Proposed
  (not implemented): the versioned JSON-RPC engine API, the MCP server, the operation table,
  artifact store, published JSON Schemas and Python bindings
- Target: originally engine API v1 and the MCP server with v0.3 (M2), GDS tools with M3; the
  solver milestones M2 and M3 are implemented, the API and MCP phases of §11 are unscheduled
- Related: [M1 FRW solver](M1_frw_solver.md), [D1 GUI](D1_gui.md)

---

## 0. Summary

faradice exposes **one operation set** through five surfaces: the Rust library, a versioned
JSON-RPC engine API, the CLI (`--json`), Python bindings, and an MCP server. The MCP server is
not a separate product; it is a thin adapter over the engine API, like the GUI (D1 §5).

The design is optimized for **agent experience (AX)**: an AI agent with no prior knowledge of
faradice should be able to discover what it can do, validate a request, learn what it will
cost, run it without blocking, read a bounded answer, and understand why a number is what it
is. FRW suits agents well: cost is controllable (per-net, per-accuracy), results carry σ, and
runs are deterministic given a seed.

---

## 1. AX Principles

| # | Principle | Concrete rule in faradice |
|---|---|---|
| P1 | Stable, typed, self-describing | Every operation has a name, a version, and **JSON Schema** for input and output (`docs/schema/api/v1/*.json`), served at runtime |
| P2 | Small composable tools | One verb per tool (inspect, validate, estimate, start, status, results, explain, compare, export). No "do everything" tool |
| P3 | Long jobs are jobs | `start_extraction` returns a `job_id` immediately. Status by poll or stream. Results fetched separately. Jobs survive client disconnect |
| P4 | Idempotent and resumable | `start_extraction` takes an optional `idempotency_key`; the same key and identical request returns the existing job. A finished result is reused (`cache_hit: true`) only when its **cache identity** matches (§5.2): content hashes of every input, not paths |
| P5 | Validate before run | `validate_config` and `dry_run: true` resolve defaults and check inputs without solving. `estimate_cost` predicts walks and time |
| P6 | Structured errors | Machine-readable `code`, human `message`, `details`, and `hints[]` with concrete remediation (a field to change, a value to try, a tool to call) |
| P7 | Bounded outputs | Summary first; details on request. Pagination (`cursor`, `limit`), top-k couplings. **Every** response, including resource contents and text renderings, has a hard serialized-size cap (§5.3) with `truncated: true` and a continuation |
| P8 | Deterministic | Seeded; bit-identical across thread counts (M1 §5.6). The seed is always echoed |
| P9 | Provenance everywhere | Every result names its `run_id`, input hashes, resolved config hash, tool version, Green-table hash |
| P10 | Safe defaults | Read-only by default; writes only under a designated output root; resource limits on by default |
| P11 | No hidden state | No "current project" or "current net" implicit in a session. Every call names its inputs by path or ID |
| P13 | Cancellable everywhere | Deadlines and cancellation propagate into preprocessing and the walk loop (§5.1), not only to the API boundary |
| P12 | Units explicit | Every quantity is `{ "value": 1.23, "sigma": 0.01, "unit": "fF" }` or a field suffixed with its unit (`margin_um`) |

Anti-patterns we explicitly avoid: free-text-only outputs, tools that block for minutes,
megabyte JSON matrices returned by default, errors that say "invalid input", options whose
meaning depends on another call made earlier.

---

## 2. Layered Surfaces

```
  Rust library (faradice-frw, faradice-io)    -- pure functions: solve(), postprocess()
        │
  faradice-engine (Rust)                      -- operations, jobs, artifacts, schemas  ← the API
        │
        ├── JSON-RPC 2.0 over stdio / Unix socket / HTTP (+ SSE for events)   "engine API v1"
        ├── CLI: faradice <verb> --json        (same request/response types, NDJSON events)
        ├── Python: faradice.Engine (PyO3)     (same operations, returns dataclasses / dicts)
        ├── MCP: faradice mcp                   (tools/resources/prompts mapped 1:1 to ops)
        └── GUI: EngineClient                   (D1)
```

### 2.1 Operation catalog (shared by all surfaces)

| Operation | Kind | CLI | MCP tool |
|---|---|---|---|
| `layout.inspect` | read | `faradice inspect FILE --json` | `inspect_layout` |
| `layout.list_nets` | read, paginated | `faradice nets FILE --json` | `list_nets` |
| `layout.trace_net` (M3) | job (short) | `faradice gds trace --net N --json` | `trace_net` |
| `config.validate` | read | `faradice solve ... --dry-run --json` | `validate_config` |
| `config.schema` | read | `faradice schema config` | resource `faradice://schema/...` |
| `run.estimate` | read | `faradice estimate ... --json` | `estimate_cost` |
| `run.start` | job | `faradice solve ... --json --detach` | `start_extraction` |
| `job.status` / `job.events` | read / stream | `faradice job status ID` / `faradice job watch ID` | `get_job_status` |
| `job.cancel` | write | `faradice job cancel ID` | `cancel_job` |
| `result.get` | read, paginated | `faradice result ID --view summary --json` | `get_results` |
| `result.explain` | read | `faradice explain ID --coupling A:B --json` | `explain_result` |
| `result.compare` | read | `faradice compare ID1 ID2 --json` | `compare_runs` |
| `result.report` | write | `faradice report ID -c post.toml` | `postprocess` |
| `result.export` | write | `faradice export ID --format spef` | `export_spef` |
| `run.list` | read, paginated | `faradice runs --json` | `list_runs` |

Rules:
- Operation names are `noun.verb`; MCP tool names are `verb_noun` snake_case (agents and MCP
  clients handle these best). The mapping is a static table, tested.
- **CLI `--json`** prints exactly the engine response object on stdout; logs go to stderr;
  `--json` with a job prints NDJSON events then the final object. Exit codes: 0 ok, 2 usage,
  3 validation error, 4 job failed, 5 not converged (results still written), 130 cancelled.
- Python: `Engine().start_extraction(...)` returns a `Job` with `.events()` iterator and
  `.result()`; raw dicts are available for schema-level access.

### 2.2 Engine API v1 transport

- JSON-RPC 2.0. Methods are the operation names. `params` and `result` validate against the
  published schemas. Event streams use JSON-RPC notifications (`job.event`) on stdio/socket and
  SSE on HTTP (`GET /v1/jobs/{id}/events`, `Last-Event-ID` for resume).
- HTTP also exposes REST-style aliases (`POST /v1/runs`, `GET /v1/jobs/{id}`), generated from
  the same operation table, for curl users and CI.
- Every response includes `api_version: "1.x"`; requests may pin `"api_version": "1"`.

---

## 3. Common Types

```jsonc
// Quantity with uncertainty (all capacitances)
{ "value": 0.212, "sigma": 0.009, "unit": "fF" }

// Error (JSON-RPC error.data, MCP structuredContent when isError = true)
{
  "code": "input.overlap_different_nets",       // stable, dotted, documented
  "message": "Boxes of nets 'A' and 'B' overlap at [9.86,0.42,1.0]-[10.0,0.56,1.36] µm.",
  "details": { "nets": ["A", "B"], "box_ids": [1, 2], "file": "bus.faradice.toml", "line": 18 },
  "hints": [
    { "action": "fix_input", "text": "Remove the overlap or assign both boxes to the same net." },
    { "action": "call_tool", "tool": "inspect_layout", "args": { "path": "bus.faradice.toml", "region": [9.8,0.4,0.9,10.1,0.6,1.4] } }
  ],
  "retryable": false,
  "docs": "faradice://docs/errors#input.overlap_different_nets"
}

// Provenance (every result and artifact)
{ "run_id": "r_01JD3K7Q", "faradice_version": "0.3.0", "git": "0123abc",
  "input_sha256": { "bus.faradice.toml": "1c2e…" }, "config_sha256": "88ab…",
  "green_table_sha256": "5f0d…", "backend": {"kind": "cpu"}, "seed": 1, "threads": 16,
  "deterministic": true, "completeness": "complete" }   // or "cancelled" | "time_limited"

// Run input (ONE schema, shared by validate_config, estimate_cost and start_extraction)
{ "structure": "path", "tech": "path or PDK name (GDS only)",
  "config": "path or inline object", "overrides": { /* partial config, same keys as config */ },
  "nets": ["globs"], "preset": "fast|default|signoff|reference", "seed": 7 }
```

**Run input precedence** (lowest to highest): built-in defaults → `preset` → `config` →
`overrides` → the shortcut fields `nets` (= `overrides.extract.nets`), `preset`, `seed`
(= `overrides.run.seed`). `seed` has **no default in the schema**: when omitted, `config.run.seed`
is used, and only if that is absent the built-in default (1). The resolved value and its source
are always echoed in `resolved_config.source`. `validate_config`, `estimate_cost` and
`start_extraction` accept exactly this object (plus operation-specific fields such as
`max_seconds` or `pilot`), so a request that validates is the request that runs.

Error code families (initial set):

| Family | Examples | Retryable |
|---|---|---|
| `input.*` | `input.parse`, `input.overlap_different_nets`, `input.no_target_nets`, `input.unknown_net` | no |
| `config.*` | `config.unknown_key`, `config.invalid_value`, `config.conflict` (mutually exclusive options) | no |
| `tech.*` (M3) | `tech.unknown_layer`, `tech.missing_stack`, `tech.pdk_not_found` | no |
| `limits.*` | `limits.walk_budget_exceeded`, `limits.memory`, `limits.output_too_large` | yes, with changed args |
| `security.*` | `security.path_outside_roots`, `security.write_denied` | no |
| `job.*` | `job.not_found`, `job.cancelled`, `job.not_finished` | `job.not_finished` yes |
| `internal.*` | `internal.bug` (includes a report bundle path) | maybe |

`config.unknown_key` always includes the nearest valid key (`did_you_mean`). Warnings use the
same shape with `severity: "warning"` (e.g. `result.boundary_ratio_high`, `result.not_converged`,
`config.ignore_nets_changes_physics`) and are attached to results, never only logged.

---

## 4. MCP Server

### 4.1 Process and transport

- `faradice mcp` speaks MCP over **stdio** (default; what desktop agents and IDEs launch).
- `faradice mcp --http 127.0.0.1:7421` serves **Streamable HTTP** for shared or remote use,
  bearer-token auth required for any non-loopback address. The server **validates the `Origin`
  header on every request**, as the Streamable HTTP transport requires, to defeat DNS rebinding
  against loopback listeners: requests with an `Origin` not on the allowlist (default: none, i.e.
  only requests without `Origin` from non-browser clients, plus `--allow-origin URL`) are
  rejected with HTTP 403 before any JSON-RPC processing. Loopback binding and the token are
  additional layers, not replacements.
- Implementation: the official Rust SDK (`rmcp`) if its API is stable at implementation time,
  otherwise a minimal hand-written JSON-RPC layer; either way the tool table is generated from
  the engine operation catalog so schemas never drift.
- **Pinned protocol revisions.** The server declares an explicit list of supported MCP
  revisions and implements each behind a **compatibility adapter** (`mcp::rev_2025_06_18`,
  `mcp::rev_2026_07_28`, …) that maps one internal tool/resource model onto the wire format. The
  initial target is the current revision, **2026-07-28**, with 2025-06-18 as the compatibility
  baseline (`outputSchema` / `structuredContent`, progress, cancellation, resource links).
- Changes in 2026-07-28 that the adapter must handle, **to verify against the published spec at
  implementation time**: `initialize` replaced by discovery via `server/discover`; per-request
  protocol/capability metadata; results carrying a `resultType`; subscriptions via
  `subscriptions/listen`; the Tasks mechanism as an **optional extension** negotiated per client.
  Where Tasks is negotiated, jobs are also exposed as tasks; the explicit
  `start_extraction` / `get_job_status` tools remain and work in every client.
- The adapter layer is the only code that knows protocol revisions; golden session tests run per
  supported revision (§10).

### 4.2 Tool list

Each tool returns `structuredContent` conforming to its `outputSchema`, plus a short text
rendering (≤ 30 lines) for clients that only show text. Annotations: `readOnlyHint`,
`idempotentHint`, `destructiveHint: false` on all tools (none delete user data).

**inspect_layout** (read-only) — what is in this file.
```jsonc
// input
{ "type": "object", "required": ["path"],
  "properties": { "path": {"type": "string"}, "tech": {"type": "string", "description": "tech file or PDK name, GDS only"},
                  "region": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6} } }
// output (abbrev.)
{ "format": "faradice-toml", "units": "um", "net_count": 42, "box_count": 3180,
  "bbox_um": [-2,-2,0,120,80,6.1], "layers": ["met1","met2","met3"],
  "ground_candidates": ["VSS"], "power_candidates": ["VPWR"], "issues": [], "provenance": {…} }
```

**list_nets** (read-only, paginated) — `{path, filter?: glob, role?: enum, cursor?, limit?: 1..500 (default 100)}`
→ `{ items: [{name, boxes, bbox_um, layers, suggested_role}], next_cursor, total }`.

**trace_net** (M3; short job, returns inline if < 10 s) — `{gds, tech, net, label_layer?, top_cell?}`
→ `{ net, polygons, layers, connected_labels, opens: [...], shorts: [{other_net, location_um}], structure_path }`.
A short to another label is an `input.label_short` warning with both labels and coordinates.

**validate_config** (read-only) — the shared run input (§3)
→ `{ valid, resolved_config: {...}, target_nets: {count, sample[≤20]}, errors: [], warnings: [], cli: "faradice solve ..." }`.
`resolved_config` contains every option with defaults filled in and a `source` map
(`default` / `preset:signoff` / `config` / `override`), so the agent can see why a value is set.

**estimate_cost** (read-only) — the shared run input (§3) plus `threads?`, `pilot?: bool`
(default true), `pilot_walks?` (default 2×10⁴ per net) and `max_seconds?` (default 10 s). The
pilot uses small batches (`batch_walks = 256`) and a reduced `walk.max_steps`, and its deadline
is also checked **inside** the walk loop (every 1024 steps per worker); unfinished batches are
discarded and queued work is dropped, so the cap holds to within one check interval. The estimate
is then returned with `method: "pilot_partial"`. The pilot runs as an internal job with the same
cancellation token as extractions (§5.1), but unlike a detached extraction it is **bound to its
request**: `notifications/cancelled` for the request, or (Streamable HTTP) the client closing the
request's response stream, cancels it.
```jsonc
{ "walks_estimate": 2.1e6, "walks_range": [1.2e6, 4.0e6], "seconds_estimate": 38, "seconds_range": [20, 75],
  "per_net": [{"net": "clk", "walks": 6.0e5, "limiting": "coupling clk→vin_p (1.7% of total)"}],
  "method": "pilot", "pilot_walks": 20000,
  "advice": ["Coupling tracking at rel=1% dominates cost; rel=5% would take ≈ 9 s."] }
```
Method: a short pilot run (seeded) measures per-net variance and throughput, then applies
N ∝ σ²/(ε·C)² per tracked quantity (M1 §4.4). `method: "heuristic"` (geometry-only, instant)
when `pilot: false`.

**start_extraction** — returns immediately.
```jsonc
// input: the shared run input (§3; structure, tech, config, overrides, nets, preset, seed) plus
{ "type": "object", "required": ["structure"],
  "properties": {
    "max_seconds": {"type": "number", "description": "wall-clock budget; checked between batches (§5.1); job stops with partial results"},
    "max_walks_total": {"type": "number"},
    "idempotency_key": {"type": "string"},
    "dry_run": {"type": "boolean", "default": false} } }
// output
{ "job_id": "j_01JD3K7Q", "run_id": "r_01JD3K7Q", "state": "queued", "cache_hit": false,
  "estimate": {"seconds_estimate": 38}, "poll_after_ms": 2000,
  "resources": ["faradice://runs/r_01JD3K7Q/manifest"] }
```

**get_job_status** (read-only) — `{job_id, wait_ms?: 0..30000}`. Long-poll up to `wait_ms` for a
state change, so an agent needs few calls.
```jsonc
{ "job_id": "j_01JD3K7Q", "state": "running", "progress": 0.62, "elapsed_s": 21, "eta_s": 14,
  "nets": {"total": 4, "converged": 2, "running": 2, "not_converged": 0},
  "worst": [{"net": "clk_b", "rel_err": 0.010, "target": 0.005}],
  "warnings": [{"code": "result.boundary_ratio_high", "net": "clk_b", "ratio": 0.018}] }
```
States: `queued | running | succeeded | not_converged | failed | cancelled`.
`not_converged` is a success with caveats: results exist and say which nets missed targets.

**cancel_job** — `{job_id, mode?: "now" | "after_batch"}` → `{state, partial_results_available}`.
`now` sets the job's cancellation token; it takes effect at the next check point (§5.1), typically
within one batch duration, and the tool returns `state: "cancelling"` until the job reaches
`cancelled`. A cancel racing with completion returns the final state (`succeeded` wins if the last
batch already merged); cancelling a finished job is a no-op that returns its state.

**get_results** (read-only, bounded) — the main read tool.
```jsonc
// input
{ "run_id": "...", "view": {"enum": ["summary", "nets", "top_couplings", "matrix", "ground_breakdown", "warnings"]},
  "nets": ["clk*"], "top_k": 10, "min_rel": 0.01, "significant_only": true,
  "postprocess": {"$ref": "faradice://schema/config/v1#/$defs/coupling"},  // optional, recomputed from raw stats
  "cursor": "…", "limit": 100 }
// output, view = top_couplings (abbrev.)
{ "run_id": "r_01JD3K7Q", "state": "succeeded",
  "items": [{ "net": "clk", "total": {"value": 12.431, "sigma": 0.006, "unit": "fF"},
              "couplings": [
                {"to": "VSS(ground)", "c": {"value": 3.110, "sigma": 0.007, "unit": "fF"}, "frac": 0.250, "tracked": true},
                {"to": "clk_b", "c": {"value": 1.902, "sigma": 0.005, "unit": "fF"}, "frac": 0.153, "tracked": true},
                {"to": "vin_p", "c": {"value": 0.212, "sigma": 0.003, "unit": "fF"}, "frac": 0.017, "tracked": true}],
              "omitted": {"count": 31, "sum": {"value": 0.084, "sigma": 0.004, "unit": "fF"}} }],
  "next_cursor": null, "truncated": false, "provenance": {…} }
```
`summary` (default) is always small: counts, convergence, worst nets (≤ 10), warnings (≤ 20, with
`omitted_warnings` count), top 5 couplings overall. `nets` and `top_couplings` paginate by net
(`limit` ≤ 100) and cap `top_k` at 50. `matrix` is refused above 50×50 with
`limits.output_too_large` and a hint to use the chunked `matrix.csv` resource (§4.3) or a net
filter. Note `sigma` here is 1σ; the
configured `confidence_sigma` is reported separately so agents never confuse the two.

**explain_result** (read-only) — why a number is what it is. Requires recorded **evidence**
(versioned schema `evidence/v1`, below). For runs without it (older runs, or runs where recording
was disabled) the tool returns `{"evidence": "unavailable", "reason": "...", "hints": [{"action":
"call_tool", "tool": "start_extraction", "args": {"overrides": {"debug": {"evidence": true}}}}]}`
rather than guessing.
- Input: `{run_id, net, coupling_to?, question?: "why_large" | "why_noisy" | "boundary" | "ground"}`.
- Output: structured findings, each with evidence:
```jsonc
{ "findings": [
   {"kind": "proximity", "text": "62% ± 3% of the clk→vin_p coupling is collected by vin_p met2 segment x∈[40,58] µm, 0.14 µm from clk.",
    "evidence": {"contribution_fraction": {"value": 0.62, "sigma": 0.03}, "hit_fraction": 0.48,
                 "region_um": [40,10.2,1.0,58,10.5,1.36], "min_gap_um": 0.14}},
   {"kind": "boundary", "severity": "warning",
    "text": "<boundary:xmax> collects 1.8% of clk_b total (limit 1%). Domain margin 2.0 µm is small vs 6.1 µm stack height.",
    "remediation": {"set": {"domain.margin": 8.0}, "expected_effect": "boundary ratio < 0.5%, cost +15%"}},
   {"kind": "significance", "text": "clk_b→vin_n = 0.018 ± 0.007 fF is 2.6σ; not tracked; treat as upper bound."}] }
```
Explanations are computed from recorded statistics, not generated text; the text field is a
template rendering of the evidence. Evidence `v1` stores, per target net and per sink box (or
box face), the **sum and sum of squares of the signed estimator weights** (antithetic pairs
combined per sample, as in the main estimator). That gives each region's contribution a proper
value and σ, but **not** the σ of a ratio or of a sum over regions: one sample can contribute to
several regions and sinks (both walks of an antithetic pair), so those need cross-moments. `v1`
therefore also records, per sample, the joint moments needed for the reported derived quantities
only — the region-vs-own-sink cross moment for `contribution_fraction`, and the per-sample sum
over omitted couplings for `omitted.sum` (accumulated like the solver's ground aggregate). Any
other derived uncertainty is reported as `"sigma": null` with `"sigma_unavailable":
"needs_joint_statistics"` rather than computed from per-region moments as if independent. Raw hit counts are stored too but only reported as `hit_fraction`: because weights
vary in magnitude and sign, hit density is not proportional to capacitance and must never be
presented as such. Evidence is bounded (at most the boxes of sink nets, aggregated to ≤ 10⁴
regions per net) and recorded under a config key added with a config `schema_version` bump
(the current parser rejects unknown `debug.*` keys).

**compare_runs** (read-only) — `{run_a, run_b, nets?, top_k?: 20, z_threshold?: 3}` →
`{config_diff: [{key, a, b}], input_diff: [...], independence: "independent_seeds" | "shared_seed_unknown_covariance",
changes: [{net, to, a, b, delta_rel, z, verdict: "significant" | "noise" | "indeterminate"}]}`.
`z = Δ / sqrt(σa² + σb² − 2·Cov(a,b))`. Runs with different seeds are independent (Cov = 0).
Runs sharing a seed share random streams and are correlated with unknown covariance: the tool
then reports `independence: "shared_seed_unknown_covariance"`, computes `z` with Cov = 0 only as
a labeled approximation, and marks borderline verdicts `indeterminate`. The
`compare_layout_revisions` prompt therefore runs A and B with **different seeds**.

**postprocess** — apply `coupling.*` / `ground.report` to an existing run without recomputation;
returns a new `report_id`. **export_spef** — `{run_id | report_id, path?, divider?, name_map?}` →
`{path, bytes, nets, sha256, resource}`. Writes only under the output root (§6).

**list_runs** (read-only, paginated) — `{project?, since?, net?, cursor?, limit?}` → runs with
state, preset, target nets, created time, input hash.

### 4.3 Resources

| URI | Content |
|---|---|
| `faradice://runs/{run_id}/manifest` | Provenance and resolved config (JSON) |
| `faradice://runs/{run_id}/report.json` | Postprocessed report |
| `faradice://runs/{run_id}/matrix.csv?part={n}` | Full coupling matrix with σ columns, in chunks |
| `faradice://runs/{run_id}/result.json?part={n}` | Raw statistics in chunks (agents should prefer tools) |
| `faradice://runs/{run_id}/spef/{name}` | Exported SPEF files |
| `faradice://schema/api/v1/{operation}` | Input/output JSON Schemas |
| `faradice://schema/config/v1` | Run configuration schema (M1 §4 options with descriptions, defaults, units) |
| `faradice://docs/options/{key}` | One option's doc: meaning, layer A/B/C, cost/physics impact |
| `faradice://docs/errors` | Error code catalog with remediation |
| `faradice://pdk/{name}/tech` (M3) | Tech/stack summary for installed open PDKs |

Run resources are listed via `resources/list` (paginated, newest first) and tools return
resource links instead of inlining big content. **Resource contents are bounded too**:
pagination of the listing does not bound a single `resources/read`, so every large artifact is
served in parts of at most `--max-response-kb` (default 64 KB, split at row/record boundaries).
Part 0 starts with a small header `{total_parts, total_bytes, sha256, next: "…?part=1"}`; each
part names its successor. Clients that need the whole file in one piece get a path under the
runs directory (stdio, same host) or an HTTP download URL (Streamable HTTP), never an unbounded
message. Clients that subscribe (via the revision-specific mechanism, e.g. `resources/subscribe` or
`subscriptions/listen`, handled by the adapter) are notified when a run's report changes.

### 4.4 Prompts

| Prompt | Arguments | What it scripts |
|---|---|---|
| `extract_critical_nets_signoff` | `structure`, `tech`, `nets` | inspect → validate (signoff) → estimate → confirm budget with user → start → poll → check warnings → explain any boundary/convergence issue → export SPEF |
| `debug_unexpected_coupling` | `run_id`, `net_a`, `net_b` | get_results(top_couplings) → explain_result(why_large) → suggest geometry/config changes |
| `compare_layout_revisions` | `structure_a`, `structure_b`, `nets` | Same config, **independent seeds** on both, compare_runs, report only significant changes |
| `choose_accuracy_preset` | `structure`, `nets`, `time_budget_s` | estimate_cost for each preset, recommend the tightest that fits |

Prompts are guidance; every step is an ordinary tool call the agent can also do on its own.

---

## 5. Long-Running Jobs in MCP

- `start_extraction` never blocks beyond validation (target < 1 s, plus up to the pilot time if
  `estimate: true` is requested).
- **Progress notifications are tied to requests.** `start_extraction` returns immediately, so
  its `progressToken` cannot report the detached job and is not used for it. Progress of a job is
  observed by `get_job_status` (long-poll) or, where negotiated, the Tasks extension / resource
  update subscriptions. `notifications/progress` is only sent during requests that themselves
  last (an `estimate_cost` pilot, a `get_job_status` long-poll, an inline `trace_net`), and its
  `progress` value is **monotonically non-decreasing** and stops at completion of that request:
  we report completed walks against a fixed per-request denominator, never a fraction of a
  re-estimated total that could go backwards.
- `get_job_status(wait_ms)` long-polls so an agent loop of "wait 30 s, check" costs few tokens.
- MCP `notifications/cancelled` for the `start_extraction` request does **not** cancel the job
  (the request already completed); `cancel_job` does. This is documented in the tool description.
  Request-bound work (pilots in `estimate_cost`, long `get_job_status(wait_ms)` polls) is
  different: on Streamable HTTP, closing the request's response stream is the cancellation signal
  in the pinned revision, and the server maps it to that request's cancellation token. Detached
  extraction jobs are never tied to a response stream.
- Jobs are owned by the engine, not the MCP session: they continue if the client disconnects,
  and a new session can find them with `list_runs`. Results are kept until `retention`
  (default 14 days or 20 GB under the runs root) and are content-addressed by cache identity
  (§5.2).
- Partial results: `get_results` on a running job returns the current estimates with
  `state: "running"` and per-value σ, so an agent can stop early when it has enough.

### 5.1 Cancellation and deadlines

- Every job carries a **cancellation token** and an optional **deadline** (`max_seconds`, server
  limit). Both are passed into the library: preprocessing (parsing, GDS flattening and tracing,
  index construction) checks them between phases and every bounded chunk of work, and the solver
  checks them **between batches** of each net (the existing merge point of M1 §5.6), so a stop
  never leaves a half-merged batch. Hard deadlines (pilots, `max_seconds`) are additionally
  checked inside the walk loop every 1024 steps; a batch interrupted that way is discarded, not
  merged, and the parallel wave in flight is abandoned. The library API gains `solve_cancellable(..., &Cancel)`;
  `solve()` stays as the uncancellable convenience.
- Latency: one batch duration per net (default `batch_walks = 4096`; milliseconds for typical
  nets). A walk that runs very long is bounded by `walk.max_steps`, so the worst case is
  documented as "one batch of max-length walks". The server may lower `batch_walks` for jobs with
  tight deadlines.
- On stop, the merged statistics so far are **persisted** with `completeness: "cancelled"` or
  `"time_limited"` and per-net walk counts; they are never reported as converged, and they are
  never used as a cache hit for a complete run (§5.2).

### 5.2 Cache identity

A finished run is reused only if its **cache identity** equals that of the new request:
sha256 of the bytes of every input (structure/GDS, tech file, included files; never paths or
mtimes), sha256 of the fully resolved solver configuration (after defaults, presets, overrides
and seed), the faradice version, Green-table hash, and backend settings that affect results
(CPU vs GPU, device class). Only `completeness: "complete"` runs are cache candidates. A file
changed in place at the same path therefore never hits a stale result.

### 5.3 Response size bounds

The cap (`--max-response-kb`, default 64 KB) applies to the **complete serialized response**:
structured content, text rendering and embedded resources together. Each list field has an
explicit maximum (polygons in `trace_net`: ≤ 1000, beyond which they are returned only as a
resource; warnings ≤ 20; per-net estimates ≤ 100 per page). When a response would exceed the
cap, it is cut at an item boundary and returns `truncated: true` with a `next_cursor` or
resource link that yields the rest.

---

## 6. Security and Resource Limits

| Concern | Default | Configuration |
|---|---|---|
| Filesystem read | Only under **server-owned read roots** (`--read-root`, default: the working directory). MCP client roots can only **narrow** access: the effective set is the intersection of client roots with server roots. A client nominating `/` or any path outside the server roots gains nothing | `--read-root PATH` (repeatable) |
| Filesystem write | **Read-only** mode by default for MCP: no export, no runs dir writes outside the state dir | `--allow-write PATH`; runs always go to `$XDG_STATE_HOME/faradice/runs` or `--runs-dir` |
| Path handling | Canonicalized, symlinks resolved, then checked against roots; `security.path_outside_roots` otherwise | — |
| CPU | `threads = min(cores, 8)` for MCP-started jobs; one concurrent job | `--max-threads`, `--max-concurrent-jobs` |
| Walk / time budget | 10⁹ walks and 30 min per job unless the request asks for more within server limits | `--max-walks`, `--max-seconds` |
| Memory | Refuse inputs whose estimated index size exceeds a cap (`limits.memory`) | `--max-memory-gb` |
| Output size | Tool responses ≤ 64 KB structured content; larger data via resources | `--max-response-kb` |
| Network | stdio: none. HTTP: loopback only unless `--auth-token` set; `Origin` validated on every request (§4.1); no outbound connections ever | `--allow-origin URL` |
| Code execution | None. Configs are data (TOML/JSON); no scripts, no plugins loaded from inputs | — |

Prompt-injection hygiene: net names, labels and file contents are returned as data fields,
never interpolated into tool descriptions or prompts; text renderings quote them.

---

## 7. Versioning and Compatibility

- Engine API and MCP tool schemas are versioned together as **API v1**. Within v1: only
  additive changes (new optional inputs, new output fields, new tools, new error codes).
  Agents must ignore unknown output fields (stated in every schema via `additionalProperties`).
- Breaking changes create `v2` operations served alongside v1 for at least two minor releases.
  MCP tool names do not carry versions; a v2 server may expose `start_extraction` with v2
  semantics only after a deprecation period in which v1 remains the default.
- `server_info` (the MCP discovery result of the negotiated revision, e.g. `initialize` in
  2025-06-18 or `server/discover` in 2026-07-28, and `faradice version --json`) lists API version,
  supported MCP revisions,
  supported config `schema_version`s, PDKs installed, GPU backends available, and limits.
- Config `schema_version` (M1 §1.3) is independent of API version; the API accepts any
  supported config version and returns the resolved config in the latest.
- Tool descriptions are part of the contract and are snapshot-tested; changing one is a
  reviewed change, because agents behave differently on different descriptions.

---

## 8. Writing for Agents: Descriptions and Messages

Tool descriptions follow one template: what it does (one line), when to use it, when not to,
cost, and the next tool usually called. Example:

> `start_extraction`: Start a capacitance extraction job and return immediately with a job_id.
> Use after `validate_config` succeeds. Takes seconds to hours depending on nets and preset;
> call `estimate_cost` first if unsure. Poll with `get_job_status(wait_ms=30000)`; read results
> with `get_results(view="summary")`. Re-sending the same request with the same
> `idempotency_key` returns the same job.

Error messages: state the fact with numbers, the consequence, and the fix:

> `config.ignore_ground_changes_physics` (warning): `ground.ignore = "physics"` removes the
> substrate plane and ground nets from the solved geometry and turns grounded domain faces
> into reflecting ones, so every capacitance changes. The open boundary still keeps infinity at
> 0 V, so `ground.report = "by_source"` will report only the infinity part. If you wanted the
> same physics without ground in the report, use `ground.ignore = "report"`.

(This matches the current `faradice-io` semantics: `physics` with `by_source` is accepted, and
the open-domain behavior is reported as a warning. Turning it into an error would be a
versioned incompatibility and is not planned.)

Warnings in results are written so an agent can relay them verbatim to a human, and each one
carries a `remediation` object the agent can apply mechanically.

---

## 9. Example Agent Transcripts

### 9.1 Sign-off extraction of critical nets

```
user: Extract clk and clk_b in designs/pll/vco.gds for sign-off with sky130A, under 5 minutes.

agent → inspect_layout {"path":"designs/pll/vco.gds","tech":"sky130A"}
      ← {net_count: 214, box_count: 51882, layers: [li1, met1..met4], ground_candidates: ["VGND"], ...}
agent → validate_config {"structure":"designs/pll/vco.gds","tech":"sky130A",
                         "overrides":{"extract":{"nets":["clk","clk_b"]},"accuracy":{"preset":"signoff"}}}
      ← {valid: true, target_nets: {count: 2}, warnings: [], cli: "faradice solve designs/pll/vco.gds --tech sky130A --nets clk,clk_b --preset signoff --seed 1"}
agent → estimate_cost {... same ...}
      ← {seconds_estimate: 140, seconds_range: [90, 260], per_net: [{net:"clk_b", limiting:"coupling clk_b→vctrl (2.1%)"}]}
agent → start_extraction {... same ..., "max_seconds": 300, "idempotency_key": "vco-clk-signoff-1"}
      ← {job_id: "j_7Q", run_id: "r_7Q", state: "queued", poll_after_ms: 2000}
agent → get_job_status {"job_id":"j_7Q","wait_ms":30000}   (×4)
      ← {state: "succeeded", warnings: [{code:"result.boundary_ratio_high", net:"clk_b", ratio:0.013}]}
agent → explain_result {"run_id":"r_7Q","net":"clk_b","question":"boundary"}
      ← {findings:[{kind:"boundary", remediation:{set:{"domain.margin":10.0}, expected_effect:"cost +12%"}}]}
agent → start_extraction {..., "overrides":{"domain":{"margin":10.0}}, "idempotency_key":"vco-clk-signoff-2"}
      ...
agent → get_results {"run_id":"r_8A","view":"top_couplings","top_k":5}
agent → export_spef {"run_id":"r_8A","path":"out/vco_clk.spef"}
      ← error security.write_denied: "Server is read-only; out/ is not an allowed write root.
         Hint: restart with --allow-write out/ or read faradice://runs/r_8A/spef/default"
agent: Totals: clk 12.43 ± 0.02 fF, clk_b 11.90 ± 0.02 fF (3σ, sign-off targets met). Largest
       couplings: ... The first run's domain was too small for clk_b (1.3% boundary); I reran with
       margin 10 µm. I could not write out/vco_clk.spef (server is read-only); the SPEF is
       available as resource faradice://runs/r_8A/spef/default.
```

### 9.2 Recovering from a structured error

```
agent → start_extraction {"structure":"bus.faradice.toml","nets":["data[0-7"]}
      ← isError: {code:"config.invalid_value", message:"Glob 'data[0-7' is unterminated.",
                  details:{field:"nets[0]"}, hints:[{action:"set", field:"nets[0]", value:"data[0-7]"}]}
agent → start_extraction {"structure":"bus.faradice.toml","nets":["data[0-7]"]}
      ← {job_id: "j_9C", ...}
```

### 9.3 Is the shield worth it?

```
agent → compare_runs {"run_a":"r_A","run_b":"r_B","nets":["clk"],"z_threshold":3}
      ← {config_diff: [], input_diff: [{file:"vco.gds", a:"1c2e…", b:"88ab…"}],
         changes: [{net:"clk", to:"vin_p", a:0.212, b:0.041, delta_rel:-0.807, z:-15.8, verdict:"significant"},
                   {net:"clk", to:"VGND", delta_rel:+0.094, z:+10.1, verdict:"significant"}]}
agent: The shield cuts clk→vin_p by 81% (z = 16, far beyond noise) at the cost of +9% clk ground cap.
```

---

## 10. Implementation Notes

- Request/response types live in `faradice-engine::api::v1` as Rust structs with `serde` and
  `schemars`; JSON Schemas are generated, committed under `docs/schema/api/v1/`, and CI fails
  if they drift from the code.
- The MCP tool table, CLI subcommands and Python methods are generated from (or tested against)
  one `OPERATIONS` table, so a new operation appears on every surface or none.
- Golden tests: recorded MCP sessions (JSON lines) replayed against the server for each release
  and each supported protocol revision;
  description snapshots; error-code coverage test (every code has docs and at least one hint).
- An **agent eval suite** (`evals/agent/`): scripted tasks ("extract X within budget",
  "find why coupling is large") run against a model through the MCP server, scoring success,
  tool-call count and tokens. Used to compare description and output-shape changes, not as CI
  gating.

---

## 11. Roadmap

| Phase | With | Content |
|---|---|---|
| A | M1 | `--json` on `solve`, `report`, `check`; structured error codes in `faradice-core`/`faradice-io`; JSON Schemas for config and result |
| B | M2 | `faradice-engine` with operation table, jobs, artifact store, provenance; `estimate`; JSON-RPC over stdio/Unix socket |
| C | M2 | `faradice mcp` (stdio): inspect, list_nets, validate, estimate, start, status, cancel, get_results, list_runs; resources for runs and schemas; read-only default |
| D | M3 | GDS tools (`trace_net`, tech resources), `explain_result`, `compare_runs`, `postprocess`, `export_spef`; prompts; Python bindings over the engine |
| E | M4 | Streamable HTTP + auth, `faradice serve`, KLayout bridge on the same API; agent eval suite |
| F | M5 | GPU backend selection in requests and `server_info`; sweep jobs (parameter studies) as a first-class operation |

---

## 12. Open Questions

1. **MCP Tasks extension**: it is optional and negotiated; should `start_extraction` return a
   task where negotiated? Proposal: yes, and keep the explicit status tool through API v1.
2. **Estimate accuracy**: is a 2×10⁴-walk pilot good enough for heavy-tailed nets (vias, thin
   gaps)? Need data from M1 validation cases; report calibration error in `estimate_cost`.
3. **Explain evidence storage**: weighted per-box accumulators (`evidence/v1`) cost memory per
   target net. Always on (bounded, ≤ 10⁴ regions) or opt-in? Leaning always on once the config
   key exists.
4. **Read-only default vs usability**: should `export_spef` to the runs directory be allowed in
   read-only mode (it writes only server-owned state)? Proposal: yes; arbitrary paths need
   `--allow-write`.
5. **Multi-user HTTP server**: per-user quotas and isolation are out of scope for v1; is a
   single shared token sufficient for lab servers?
6. **Python surface**: expose the engine (jobs, artifacts) or only the pure library
   (`solve()` returning results)? Proposal: both, with `faradice.solve()` as sugar over the engine.
7. **Units in outputs**: always fF, or let clients request aF for tiny couplings? Proposal:
   always fF in structured data; text rendering picks a readable prefix.
