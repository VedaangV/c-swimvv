# Virginia Islanding-Based Mitigation: Implementation Plan

Data-driven islanding for the Virginia+PJM GIC pipeline, extending the existing
C-SWIM Virginia adaptation (see `VIRGINIA_ADAPTATION.md`). Supersedes the
`dc_blocker/README.md` placeholder as the mitigation strategy actually being
built; DC blockers remain a valid future extension but are not the near-term
deliverable.

Goal: given the substations, GIC risk, and economic exposure already computed
by the existing pipeline, (1) build a Substation Criticality Index (SCI) that
identifies which parts of the Virginia grid matter most, (2) use a seeded
region-growing algorithm to identify a specific, minimal-cut island around the
highest-SCI region, (3) re-run the existing GIC/fragility/economic pipeline on
the islanded topology to quantify avoided loss, and (4) repeat for a second,
structurally independent island as a light validation of the method's
generality.

Target: 10 working days (~2 weeks). See "Descope order" at the end for what
to cut if behind schedule.

## Phase 1 — Substation network graph (Day 1-2)

**New file: `scripts/build_virginia_network_graph.py`**

Build a `networkx.Graph` from `run_virginia_admittance.py`'s own output --
`substation_info.csv`'s `buses` column (bus indices per substation, already
remapped through `bus_ids_map` at write time) and `transmission_lines.csv`'s
`from_bus`/`to_bus` (same remapped bus-index space) -- rather than the raw
`va_pjm_lines.geojson`/`va_pjm_substations.geojson`, so the graph's topology
matches what's actually electrically solved. Doesn't need to touch the raw
`substation_buses.pkl`/`bus_ids_map.pkl`/`sub_look_up.pkl` pickles directly.

- Nodes: substations (`sub_id`), attributed with lat/lon.
- Edges: transmission lines between substations. Two subtleties handled
  explicitly: same-substation internal lines are dropped (not real
  inter-substation edges), and parallel circuits between the same two
  substations are aggregated onto one edge with a `line_ids`/
  `n_parallel_lines` count rather than collapsed to a single line, so
  Phase 3's min-cut cost reflects the real number of lines to open.
- Node attributes: betweenness centrality, degree, articulation-point flag
  (`networkx.articulation_points`), connected-component id.
- Output: `data/virginia/network/va_network_graph.pkl` +
  `va_topology_metrics.csv` (sub_id, betweenness, degree,
  is_articulation_point, component_id, in_main_component).

**Result (as run against real VA+PJM data)**: 2624 substations, 2967
substation-pair edges (4586 physical lines), 49 connected components -- a
2476-substation main component plus 148 substations across 48 small
components (mostly isolated pairs). Of those 148, 11 are inside Virginia;
the rest are elsewhere in PJM's 13-state footprint. This is a real, expected
consequence of modeling EHV-only (>=200kV) lines -- the small components are
almost certainly generator substations with a single private tie-line to one
interconnection point, not a bug in the join logic (0 unresolved bus
endpoints, 0 anomalous internal-line skips beyond the expected same-site
connections). The 11 VA orphan substations are excluded from SCI ranking and
islanding candidacy by design (`in_main_component=False`) -- they can't be
meaningfully centrality-ranked or islanded against a network they aren't
connected to in this dataset. Documented here as a known limitation, not
silently dropped: `['816921387', '217112412', '82899127', '1234109369',
'97721028', '638385315', '82899118', '1234109368', '149931761',
'1464257782', '46831932']`.

**Phase 1 complete.**

## Phase 2 — Substation Criticality Index (Day 3)

**New file: `scripts/compute_virginia_criticality.py`**

Join three (optionally four) signals per substation, each min-max normalized
to [0, 1]:

```
SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i) + w_t * norm(betweenness_i) [+ w_c * norm(critical_load_i)]
```

- `GIC_risk_i`: 100-year mean GIC from `va_gic_summary.csv`'s
  `100-year-hazard A/ph_mean` column (same column `plot_virginia_gic_risk.py`
  already ranks on, for consistency with existing figures).
- `econ_exposure_i`: sum of `GDP_*` columns + `POP20` from the areal
  interpolation output (`run_virginia_areal_interpolation.py`'s
  `aggregate_gdf`).
- `betweenness_i`: from Phase 1's `va_topology_metrics.csv`.
- `critical_load_i` (Phase 3b, optional): binary/weighted flag from external
  critical-facility data.
- Default weights: equal (1/3 each, or 1/4 with critical_load included).
  Treat weight choice as an explicit modeling assumption — worth a brief
  sensitivity check (e.g., report SCI rankings under 2-3 weight sets) in the
  same spirit as the NZ paper's four-method robustness comparison, not a
  single hardcoded number presented as ground truth.

Output: `data/virginia/econ/va_substation_criticality.csv`
(SS_ID, SCI, and each component score, for transparency).

**New file: `viz/plot_virginia_criticality.py`** — ranked map figure, styled
consistently with `plot_virginia_gic_risk.py`.

This phase alone is a complete, defensible deliverable even if nothing past
it finishes in time.

## Phase 2b — Critical-load data (Day 4, droppable)

Source Loudoun County / NoVA data center locations (public GIS) and HIFLD
critical-facility layers (hospitals, water systems), spatial-join to nearest
substation by `SS_ID`, fold into `critical_load_i` in Phase 2's formula.

**Constraint**: I can't browse and download arbitrary external GIS sites
directly from this session. You'll likely need to pull the file(s) locally
and hand them to me (or find a directly-fetchable public API/CSV I can pull
via WebFetch). If this stalls, it's fully droppable — SCI still works on
three terms.

## Phase 3 — Seeded region-growing islanding algorithm (Day 5-6)

**New file: `scripts/select_virginia_island.py`**

Implements the algorithm agreed on:

1. **Seed**: `argmax(SCI)` over eligible substations (exclude any substation
   flagged as outside the main connected component in Phase 1, and — for a
   second island — exclude substations already inside Island A or its cut
   boundary).
2. **Initialize**: island = `{seed}`; boundary = every line connecting the
   seed to a neighbor.
3. **Score**: `value = sum(SCI over island members)`, `cost = len(boundary
   lines)`, `ratio = value / cost`.
4. **Grow**: for every substation adjacent to the current boundary, compute
   the resulting `ratio` if added (accounting for boundary lines that become
   internal vs. new external lines the neighbor introduces). Add whichever
   candidate produces the best resulting ratio.
5. **Stop** the moment no candidate improves the ratio.
6. **Safety valve**: hard cap on island size (e.g., 15 substations) so a
   long thin bridge or near-tied scores can't produce a runaway island;
   trigger a warning if the cap is hit rather than the ratio naturally
   turning over, since that's a sign the graph/weights need a look.

Output: `data/virginia/network/va_island_a.json` — island member SS_IDs,
boundary line IDs (the actual cut set), and the growth trace (for the
methods section: which substation was added at each step and why).

This is a pure graph-algorithm step with no dependency on the electrical
solve — testable and debuggable in isolation before touching the admittance
pipeline.

## Phase 4 — Topology-edit support in the admittance/GIC pipeline (Day 6-8, highest risk)

**Modify**: `scripts/run_virginia_admittance.py`, `scripts/run_virginia_gic.py`
(or add thin wrapper scripts `scripts/run_virginia_islanded_admittance.py` /
`scripts/run_virginia_islanded_gic.py`, following this project's established
pattern of not editing shared/national logic in place — see how the econ
scripts monkeypatch module-level path constants rather than forking the
underlying functions).

- Accept a "removed lines" list (Phase 3's boundary cut set) and exclude
  those lines when building the bus-branch structures, before the Y-matrix
  gets constructed.
- Re-solve GIC with the modified topology via the existing Monte Carlo
  machinery — same `N_MONTE_CARLO` sample count for direct comparability
  with baseline results.
- **This is the block most likely to run long.** Every topology-touching step
  in this project so far has hit a real bug on first contact with actual VA
  data (duplicate columns, Windows multiprocessing, schema mismatches).
  Concrete risks specific to this step: the islanded sub-network might be too
  small for the earthing-impedance calculation's existing assumptions, or a
  substation inside the island might lose all its connections if a boundary
  line was its only link out (need to verify islands are internally
  connected before calling the solve, not just cut-minimal).
- If this is still broken by end of Day 8, that's the trigger to fall back to
  reporting Island A/B as *topological candidates only* (which lines would
  need to open, and why, based on SCI) without a re-solved electrical
  validation — weaker, but honest and completable. Frame explicitly as
  "candidate identification" with electrical validation as future work.

## Phase 5 — Economic re-evaluation (Day 8-9)

Push the islanded-topology GIC output through the existing fragility Monte
Carlo (`run_virginia_fragility.py`, parameterized to read the islanded GIC
results and write to a separate `va_island_a_econ/` subdirectory rather than
overwriting baseline output) and `run_virginia_econ_analysis.py`.

Produces avoided-GDP-loss numbers directly comparable to the existing
$145.5M/day (100yr) baseline figure. Compute a rough benefit-cost ratio using
an assumed switching-cost estimate (labor-based, same style as the NZ paper's
$250k figure) — **explicitly flag this as an assumption**, not a real utility
quote, in whatever gets written up.

## Phase 6 — Second island, comparison (Day 9, droppable)

Rerun Phases 3-5 with the seed restricted to substations outside Island A
and its boundary. Purpose: light validation that the method generalizes
rather than being tuned to one hand-picked case — if Island B's avoided-loss
scales with its SCI relative to Island A's, that's evidence the index is
doing real work, not just describing the one region you already knew
mattered. If nearly everything with high SCI turns out inside/adjacent to
Island A (plausible given NoVA's concentration), that's still a legitimate,
reportable finding — not a failure, just report it honestly rather than
forcing a second island that isn't really independent.

**First thing to cut if behind schedule.**

## Phase 7 — Figures and writeup (Day 10)

- Criticality map (Phase 2's `plot_virginia_criticality.py`).
- Islanded-topology GIC map, mirroring `plot_virginia_gic_risk.py`'s style,
  showing the cut boundary.
- Before/after economic comparison bars, mirroring
  `plot_virginia_econ_impact.py`'s style.
- Update `VIRGINIA_ADAPTATION.md` and the abstract's mitigation section —
  replace the placeholder DC-blocker claim with what was actually built.

## File manifest (new files this plan creates)

```
scripts/build_virginia_network_graph.py     Phase 1 — graph + centrality
scripts/compute_virginia_criticality.py     Phase 2 — SCI
viz/plot_virginia_criticality.py            Phase 2 — ranked map
scripts/select_virginia_island.py           Phase 3 — seeded region growing
scripts/run_virginia_islanded_admittance.py Phase 4 — topology-edited admittance
scripts/run_virginia_islanded_gic.py        Phase 4 — topology-edited GIC solve
scripts/run_virginia_islanded_econ_analysis.py  Phase 5 — econ re-evaluation
viz/plot_virginia_islanding_comparison.py   Phase 7 — before/after figures
```

## Descope order (in order of what to cut first if running behind)

1. Island B / comparison (Phase 6).
2. Critical-load data enrichment (Phase 2b) — SCI still works on three terms.
3. Electrical re-solve of the island (Phase 4/5) — fall back to reporting
   the topological candidate only, future work for validation.

Not attempted in this plan at all: MIP optimization with generation-balance
constraints (needs EIA-860 generation data joined in first, its own
multi-day task) — explicit future work in the paper, not a 2-week item.
