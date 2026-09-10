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

- `GIC_risk_i`: 100-year mean **ground** GIC from `va_ground_gic_summary.csv`'s
  `GIC_100_mean` column. (Originally planned as winding GIC from
  `va_gic_summary.csv`'s `100-year-hazard A/ph_mean` column — switched during
  implementation; see "Result" below.)
- `econ_exposure_i`: GDP + population, normalized separately then averaged
  (not establishment counts — see the script's module docstring for why).
  Sourced from the areal interpolation output
  (`run_virginia_areal_interpolation.py`'s `aggregate_gdf`).
- `betweenness_i`: from Phase 1's `va_topology_metrics.csv`.
- `critical_load_i` (Phase 2b, optional): binary/weighted flag from external
  critical-facility data.
- Default weights: equal (1/3 each, or 1/4 with critical_load included).
  Treat weight choice as an explicit modeling assumption — worth a brief
  sensitivity check (e.g., report SCI rankings under 2-3 weight sets) in the
  same spirit as the NZ paper's four-method robustness comparison, not a
  single hardcoded number presented as ground truth. Not yet done — still
  open for Phase 7.

Output: `data/virginia/econ/va_substation_criticality.csv`
(sub_id, SCI, and each component score, for transparency).

**New file: `viz/plot_virginia_criticality.py`** — ranked map figure, styled
consistently with `plot_virginia_gic_risk.py`. **Not yet built** — deferred
to Phase 7 alongside the other figures.

**Result (as run against real VA+PJM data)**: 413 Virginia substations
identified (2624 total − 2211 non-Virginia, matching Phase 1's known
in-state count); 402 eligible after excluding the 11 main-component orphans
documented in Phase 1. No missing-data warnings — every eligible substation
resolved both a GIC risk value and an economic exposure value, confirming
the Virginia-scoping fix lines up `va_topology_metrics.csv` (unfiltered) with
`va_ground_gic_summary.csv`/`va_substation_exposure.gpkg` (Virginia-only).
Top-ranked substation (`708152212`, gic_risk=844.71 A) hand-verified against
the SCI formula: `(1.0 + 0.037398 + 0.472921) / 3 = 0.503440`, exact match to
the script's output.

The GIC signal was switched from winding GIC to ground GIC partway through
implementation, for two compounding reasons surfaced while validating this
phase's output: (1) both are signed quantities, and averaging signed values
without taking `abs()` first can rank a substation with large but
direction-varying current as *safer* than one with small but consistent
current — a real bug, fixed by taking `abs()` before averaging in
`extract_virginia_results.py` for both signals; (2) even after that fix,
ground GIC is the more defensible choice on physical-consistency grounds —
it's the same quantity the already-validated national fragility Monte Carlo
uses for the headline $/day figures, and it's a single scalar per substation
per realization rather than something assembled by combining several
transformers' separately-signed windings. `va_top_substations.csv` and
`viz/plot_virginia_gic_risk.py`'s risk map were updated to match, so the SCI,
the top-15 table, and the risk map all now rank on the same signal.

**Phase 2 complete.**

## Phase 2b — Critical-load data (Day 4, droppable)

**New file: `scripts/fetch_virginia_critical_loads.py`**

Source hospitals, wastewater treatment plants, and data-center locations,
point-in-polygon join each against the Voronoi substation service areas
`run_virginia_areal_interpolation.py` already computes (the same "nearest
substation" geometry GDP/population get allocated through — no separate
radius/distance rule needed), and fold a total facility count into
`critical_load_i` in Phase 2's formula as a fourth, equally-weighted term.

**Constraint, as predicted**: I can't browse/verify arbitrary external GIS
sites from this sandbox — confirmed directly (hard egress denials against
`hifld-geoplatform.opendata.arcgis.com` and `services.arcgis.com`, and
`maps.nccs.nasa.gov` didn't even resolve). Every data source below had to
be run and verified by the user locally, not by me.

**Source-finding, three rounds**:
1. Wastewater treatment plants: an EPA-hosted ArcGIS FeatureServer, found
   via search, worked on the first real run — but its first real run also
   surfaced a genuine pagination bug (the fetch script stopped on any page
   shorter than requested, when it should only stop on a genuinely *empty*
   page — ArcGIS services commonly cap a page below what's asked for
   regardless). Silently truncated this category at exactly 1000 records
   (server's own cap) on the first run; fixed, and the real re-run recovered
   386 records, up from 288.
2. Hospitals: HIFLD Open's Hospitals dataset (the search-found FeatureServer
   URL) returned 0 records on a real run — turned out the whole dataset is
   gone from HIFLD's public catalog (DHS restricted several HIFLD Open
   layers, hospitals included, to agency-only access), confirmed by hand
   search. Switched to OpenStreetMap via Overpass (`amenity=hospital`),
   reusing `build_virginia_network.py`'s already-proven Overpass
   infrastructure (same servers, User-Agent, retry pattern) — worked
   immediately, 146 Virginia hospitals.
3. Data centers: no Loudoun County GIS layer was findable at all (neither
   an API URL nor a manually-browsable dataset page — only an ArcGIS Hub
   "Experience" map with no exposed underlying service), confirmed by the
   user's own search. Also switched to Overpass (`telecom=data_center`) —
   385 Virginia matches. Coverage here is the least certain of the three
   (some operators avoid being mapped, and the tag can also catch smaller
   telecom equipment sites, not just large-scale facilities) — worth a
   name spot-check before the methods write-up cites this figure as a
   facility count rather than an OSM-tag count.

**Result (as run against real VA+PJM data)**: 917 facilities fetched (386
wastewater treatment plants, 385 data centers, 146 hospitals), 888 matched
into a substation's Voronoi service area (29 fell outside every cell — a
border edge effect, same known limitation as the areal-interpolation step),
landing on 236 distinct substations. SCI re-computed with four equally-
weighted terms; formula hand-verified exactly on two rows (top two:
0.457031 and 0.377580).

**Correction — the ranking above was wrong, and here's how it was caught.**
The first real run's #1 substation, `707088636`, was assumed here to be
"almost certainly part of the Ashburn/Loudoun data-center cluster" on the
strength of its 35 critical facilities and high population (185,417). That
assumption was never checked against its actual coordinates before being
written down — and it was wrong. Plotting the SCI map
(`viz/plot_virginia_criticality.py`) surfaced it visually: `707088636` sat
alone in far southwestern Virginia, at (36.86, -81.42) — Damascus, right on
the Tennessee border — as by far the largest, darkest point on the map,
nowhere near any real concentration of population or infrastructure.
Pulling its full row and comparing its Voronoi service-area size against
the rest confirmed the mechanism: a **6,747 km² cell (99.5th percentile;
median across 413 VA substations is 64 km²)**, over 100x typical, that had
absorbed population and facility counts from a huge swath of sparsely-
populated southwestern Virginia purely because of its size — not because
`707088636` is actually near real exposure. This is a known, pre-existing
limitation of the areal-interpolation step (`run_virginia_areal_
interpolation.py`'s own docstring already flags it: border substations
near non-PJM neighbors get poorly-bounded, oversized Voronoi cells since
out-of-state generator points aren't included) — SCI's ranking is simply
the first thing to have ever ranked individual substations by these values,
which is what made a pre-existing, previously-invisible data issue
suddenly load-bearing for Phase 3's seed selection.

**Fix**: `econ_exposure_i` and `critical_load_i` now use *density* (per
km² of Voronoi cell), not raw totals, before normalizing — implemented in
`compute_virginia_criticality.py`, verified with a regression fixture that
reproduces the exact bug (two substations with identical raw totals but
very different cell sizes now correctly re-rank by density instead of
tying). GIC risk and betweenness are untouched (neither is Voronoi-derived,
so neither was ever affected).

**Result (density-corrected, real data)**: `707088636` no longer appears in
the top 10 at all. New #1 is `708152212` (SCI=0.368896, gic_risk_norm=1.0,
hand-verified exact match) — the same substation that led the very first
three-term SCI run, before critical_load_i or the density bug were ever in
the picture. Its high GIC risk (844.7 A) and moderate betweenness now
correctly dominate, with a low economic-density term (this kind of high-
GIC bulk-transmission node isn't necessarily inside a population center).
Formula hand-verified exactly on two more rows (0.368896 and 0.319636).
The corrected top 10 has no single outlier value — SCI descends smoothly
from 0.369 to 0.224, consistent with a sane, non-degenerate ranking.

**Second finding from the same real run, unrelated to the density bug**:
the criticality chart's bar labels showed what looked like two duplicate
entries ("115000 substation (lat, lon)" appearing twice). Not a data bug —
two genuinely different substations share a voltage class and both lack an
OSM name tag, so `_load_substation_labels()`'s fallback produced
visually-near-identical labels differing only in small-print coordinates.
Fixed in both `plot_virginia_criticality.py` and `plot_virginia_gic_risk.py`
(which duplicate this helper) by appending the substation's sub_id to any
fallback label, guaranteeing visual uniqueness.

**Phase 2b complete.**

## Phase 3 — Seeded region-growing islanding algorithm (Day 5-6)

**New file: `scripts/select_virginia_island.py`**

Implements the algorithm agreed on:

1. **Seed**: `argmax(SCI)` over eligible substations (exclude any substation
   flagged as outside the main connected component in Phase 1, and — for a
   second island — exclude substations already inside Island A or its cut
   boundary).
2. **Initialize**: island = `{seed}`; boundary = every line connecting the
   seed to a neighbor.
3. **Score**: `value = sum(SCI over island members)`, `cost = voltage-weighted
   sum over boundary lines` (kV/100 per physical parallel line, not a flat
   per-line count — cutting a 500kV line is a bigger deal than cutting a
   138kV one; an explicit modeling assumption, worth a sensitivity check
   later), `ratio = value / cost`.
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

**Literature grounding**: the algorithm's direct ancestor is Clauset (2005),
"Finding local community structure in networks" — seed-node greedy
expansion maximizing a local ratio metric, stopping when no neighbor
improves it, specifically so it needs no global network knowledge. The
departure here: Clauset's ratio is raw unweighted edge counts; this project
replaces it with a domain-weighted ratio (SCI-weighted value over real
physical line-cut cost) — adapting a general local-community-detection
template to a risk-weighted, engineering-cost-aware setting, not inventing
greedy region-growing from scratch. Explicitly **not** mainstream power-
systems "controlled islanding" (spectral-clustering/slow-coherency methods
optimizing generator dynamic-stability coherency for real-time response to
cascading contingencies across the whole network) — a different problem
with different constraints, not a lighter version of that literature.
GIC-specific precedent for topology reconfiguration as a mitigation
strategy: "Optimal Transmission Line Switching under Geomagnetic
Disturbances" (arXiv:1701.01469). Islanding specifically, combined with
GIC, doesn't have a direct precedent paper — an honest gap, not a hidden
one; the contribution is a defensible combination of separately-established
ideas applied to an under-explored specific use case. Full reasoning in
`scripts/select_virginia_island.py`'s module docstring.

**Clarification that came out of reviewing the first real run — protective,
not resilience, islanding.** Working through the actual New Zealand
precedent in detail (Oughton et al. 2017's economic-impact methodology, and
the closely related arXiv:2507.12495 assessment of NZ mitigation
strategies) confirmed what this project's islanding means: a **temporary,
planned disconnection during a storm's peak-risk window**, to break the
conductive path GIC needs to flow through a transformer, then reconnect
once the storm passes. It is *not* about the island continuing to operate
independently on local generation — the NZ paper doesn't claim continuity
of service either; it treats the island's own short outage as a real but
small cost, dwarfed by the alternative (a damaged large power transformer
can take 12-24+ months to replace, versus a few hours of planned
disconnection). This matters for Phase 5's design — see below — and
confirms Phase 3/4 as already planned don't need generator/generation-
balance data to be methodologically sound.

The NZ paper's headline results are also **whole-region**, not island-only:
GDP loss is computed for the whole country under each mitigation scenario
(including islanding) and compared to the no-mitigation baseline. Phase 4/5
were already planned to match this — re-solve GIC for the *entire* VA+PJM
network with Island A's boundary cut (not a standalone island-only
re-solve), then run the same whole-Virginia economic pipeline on that
output, producing a whole-VA $/day figure directly comparable to the
$145.5M/day baseline. No change needed there, just confirmed as the right
target.

**New file (optional, bonus): `scripts/check_island_generation.py`** — not
part of the core methodology (islanding doesn't require the island to stay
powered, per above), but a cheap factual check worth having: does Island A
happen to already contain real generation capacity? Matches EIA-860 plant
locations (manually downloaded — see the script's docstring) to Island A's
members within `SNAP_DISTANCE_M` and reports total matched nameplate
capacity. If Island A turns out to have real capacity, that's a genuinely
stronger result worth reporting; if not, that's expected and doesn't change
anything else.

**Result (as run against real VA+PJM data)**: seed `708152212` (the same
substation that topped the very first three-term SCI run and the density-
corrected run — highest raw GIC risk, 844.7 A). Grew to 3 members
(`708152212`, `1495710287`, `96421965`) before stopping on ratio non-
improvement, well under the 15-substation cap. 2 boundary lines cut,
voltage-weighted cost 10.00, final ratio 0.061 (up from the seed-only
baseline of 0.037 — the ratio improved at every step, as it must by
construction). Growth trace, hand-verifiable step by step:
```
step 0: added 708152212 -- seed (argmax SCI)         (size=1, value=0.3689, cost=10.00)
step 1: added 1495710287 -- ratio 0.0369 -> 0.0386    (size=2, value=0.5635, cost=14.60)
step 2: added 96421965  -- ratio 0.0386 -> 0.0608     (size=3, value=0.6084, cost=10.00)
```

**`check_island_generation.py` result (EIA-860 2025 Early Release, run against
real data)**: 0 MW matched — none of Island A's 3 members have EIA-860
generation capacity within `SNAP_DISTANCE_M`. Consistent with, not
surprising given, the seed's own profile from Phase 2b: `708152212` is a
high-GIC bulk-transmission node, not a population- or generation-colocated
site (low economic-density term in its SCI breakdown). Confirms the
"protective, not resilience" framing above is the operative one for this
particular island — there's no local generation to fall back on, so the
temporary planned-outage accounting in Phase 5 (not an assumption of
continued local service) is the right model, not a simplification of it.

**Phase 3 complete.**

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

**Implemented**: `scripts/run_virginia_islanded_gic.py`, a thin wrapper
around `scripts/run_virginia_gic.py`'s now-parameterized `main()`
(`exclude_line_ids`/`gic_dir`/`gnd_gic_dir`, defaults reproducing the exact
original baseline). Line removal happens once on `df_lines` before the
Monte Carlo loop; both the admittance matrix and geoelectric injection
currents consume the same filtered table, so nothing else needs editing.
Two safety checks run before the solve: Island A's internal connectivity
(hard-fails if violated — should be structurally impossible per how Phase
3 grows islands) and whether cutting the boundary fragments the rest of
the network further (logged, not blocking — the existing Monte Carlo
loop's non-zero-row filtering already handles any resulting isolated node
numerically). Verified against synthetic graph/`df_lines` fixtures — the
real HDF5-backed Monte Carlo loop itself needs real VA+PJM data not
available in this sandbox, so **not yet run against real data** — that's
the next step, on the user's machine.

## Phase 5 — Economic re-evaluation (Day 8-9)

Push the islanded-topology GIC output through the existing fragility Monte
Carlo (`run_virginia_fragility.py`, parameterized to read the islanded GIC
results and write to a separate `va_island_a_econ/` subdirectory rather than
overwriting baseline output) and `run_virginia_econ_analysis.py`.

Produces avoided-GDP-loss numbers directly comparable to the existing
$145.5M/day (100yr) baseline figure — computed for the **whole VA+PJM
network** with Island A's boundary cut, not an island-only re-solve, so the
result is a whole-Virginia $/day figure under "Island A protected," matching
the NZ paper's own whole-region reporting style (see Phase 3's writeup).

**Benefit-cost accounting, refined after clarifying what islanding actually
means here (see Phase 3)**: since islanding is a temporary, planned
disconnection (not a resilience/local-generation scenario), the net benefit
must account for the cost side honestly, not just the avoided-damage side:

- **Benefit**: avoided transformer-damage/failure-probability cost, from
  comparing baseline vs. islanded fragility/econ output (the existing plan).
- **Cost, two terms**:
  1. Switching-operation labor cost (the original plan's estimate, same
     style as the NZ paper's $250k figure — explicitly flagged as an
     assumption, not a real utility quote).
  2. **New**: the economic cost of Island A's own population/GDP going
     without grid power for the assumed disconnection duration (a storm's
     peak-risk window — hours, an explicit flagged assumption, not a real
     utility procedure). Uses the population/GDP-density data already
     computed for SCI in Phase 2 — no new data source needed, just a new
     use of data already on hand.

Net benefit-cost ratio = avoided-damage benefit / (switching cost + island's
own temporary-outage cost) — reported in the same style as the NZ paper's
700:1-class figures, but now honestly netting out the island's own cost
rather than only counting the benefit side.

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

**Reframed from "does the method generalize" to "does islanding help at all here."** Island A's real result (Phase 4/5, above) is net-negative for whole-Virginia $/day in every scenario checked, with the redistribution traced to a small, consistent set of substations (led by `96396693`) 25-106 km from Island A -- too far for a local boundary tweak to route around. Island B is now the direct next data point for the open question from that finding: is this a property of cutting near `708152212` specifically, or of islanding anything in this part of the network. Same purpose as originally planned (a second, independent island, not overlapping or adjacent to Island A), just a sharper reason to run it.

**Implemented**: `scripts/select_virginia_island.py` now takes `island_id`/`exclude_island_ids` (`--island b --exclude a`), reusing `select_island()`'s existing `excluded_nodes` parameter -- built for exactly this. Every downstream Phase 4/5 script (`run_virginia_islanded_gic.py`, `run_virginia_islanded_fragility.py`, `run_virginia_islanded_econ_analysis.py`, `compare_island_gic.py`, `compare_virginia_gic.py`) generalized to take `--island b` the same way, verified against fixtures to reproduce Island A's exact original paths/behavior when called with no arguments.

**Run against real data.** Island A is net-negative for whole-Virginia $/day in all 30 scenarios checked. Island B is mostly negative but genuinely positive for the real Gannon storm scenario. Combining A+B (`scripts/combine_islands.py` + `--island ab`) confirmed island effects are ~additive (AB's diff vs. baseline matches A's diff + B's diff to within ~0.5 percentage points across all 30 scenarios) -- stacking a net-negative island onto a net-positive one erased the win rather than averaging out. `scripts/search_candidate_islands.py` then searched three more candidates (C, D, E); all three had negligible effect in either direction. Net: 5 candidates selected by the SCI-driven greedy algorithm, none consistently positive Virginia-wide.

**Why, diagnosed two ways.** (1) SCI's betweenness term is shortest-path betweenness (Freeman 1977; Brandes 2001) -- counts a node as critical if it lies on many shortest paths, which is the wrong physical model for GIC's actual behavior (current splits and redistributes across all parallel paths in proportion to conductance, not just the shortest one). `scripts/check_redistribution_proxy.py` built a current-flow/random-walk betweenness proxy (Newman 2005; Brandes & Fleischer 2005, computed via the graph Laplacian) as a more physically faithful alternative -- it shows real but incomplete predictive signal on Islands A and B (misses the single worst redistribution hotspot in both cases), so betweenness choice alone doesn't explain the consistently negative results. (2) Reading the actual NZ precedent paper (arXiv:2507.12495) in full surfaced four structural differences from our setup: their strongest scenario (Scenario 5) is anchored on Manapouri, a real hydro generation site, wired to directly supply the protected load (Tiwai Point smelter) -- ours have zero local generation and rely purely on disconnection; their damage model is threshold-based (transformer offline above ~500A GIC, driving outage duration into an IO-model GDP loss) and insensitive to sub-threshold redistribution, unlike our continuous fragility-curve model which prices every redistributed amp; their islanding is one piece of a much larger expert-engineered national switching plan, not a standalone algorithmic cut; and their benefit-cost denominator is a small labor-only cost, not a redistribution cost netted out the way ours is. (1) revises the earlier claim in this document's Phase 3 section that Phase 3/4 "don't need generator/generation-balance data to be methodologically sound" -- that's still true for protective-only islanding, but the evidence now says protective-only islanding without a generation anchor doesn't consistently help Virginia-wide.

**Generation-anchored selection.** `select_virginia_island.py --require-local-generation` restricted SEED choice (not growth) to substations with real nearby EIA-860 generation capacity (via `check_island_generation.py`'s `compute_generation_by_substation()`, matched within `SNAP_DISTANCE_M`), mirroring Manapouri's anchor role in Scenario 5 rather than requiring every member to have its own generation.

**Real result: overlap exists, but only moderately.** `scripts/check_generation_sci_overlap.py` (run against real data) found 50/402 eligible substations have any matched EIA-860 generation nearby; the best of those by SCI (`82442537`, SCI 0.1891, 20 MW) sits at the 95.5th percentile of all 402 -- technically "top 5%," but still ~4-5x below Island A/B's seed SCI (~0.85-0.9), since SCI is heavily right-skewed with a handful of substations far above everyone else. Virginia doesn't have a substation that's simultaneously top-tier-critical *and* generation-backed; the best available tradeoff is moderate-risk with modest local generation. A first real run (`--island f --min-generation-mw 50`) missed this best candidate entirely -- the 50 MW floor excluded `82442537` (20 MW) in favor of a lower-SCI, higher-capacity node (`81595538`, SCI 0.1257, 175 MW) -- so a rerun with a lower floor is needed to actually test the best-available tradeoff.

Also added a population-based demand proxy (`estimate_demand_mw()` in the same script) to sanity-check generation against local demand before running the heavy pipeline -- the pipeline has no real per-substation load (MW) data anywhere (`econ_exposure_i` is population/GDP density, not electrical demand), so this converts each substation's Voronoi-cell population into a rough average/peak MW estimate via a single national per-capita consumption constant. Explicitly an order-of-magnitude proxy, not a real load estimate (ignores customer mix, timing, losses) -- documented as such in the script's module docstring, used only to catch an obvious generation-vs-demand mismatch before committing to a candidate.

**Resolved: Island F's original seed is validated, no rerun needed.** Running the demand proxy against real data settled the question -- `82442537` (SCI 0.1891, the seemingly-better candidate the 50 MW floor had excluded) has `generation_covers_peak_demand_ratio` = 0.58: its 20 MW of nearby generation doesn't even cover its own ~34.5 MW estimated peak demand, so anchoring there would just reproduce the "island loses power" problem rather than solving it. `81595538` (SCI 0.1257, Island F's actual seed) has a coverage ratio of 11.2 (175 MW vs. ~15.7 MW estimated demand) -- comfortably self-sufficient by this proxy, and the best-SCI substation among all 39 generation-backed substations that clear coverage >= 1. The original `--min-generation-mw 50` floor, while arbitrary, happened to land on the right candidate. Next: run Island F (already selected, `va_island_f.json` on disk) through Phase 4/5 as-is.

Also worth keeping in mind if Island F's econ result doesn't pan out: `81818612` (SCI 0.0824, coverage ratio ~491x -- 2103 MW of generation against only ~4.3 MW estimated local demand) is a real bulk generation source vastly exceeding its own substation's demand, arguably a stronger Manapouri-style anchor than a barely-self-sufficient one, just at lower SCI. A candidate island G if F doesn't work out.

**Island F's real econ result: 6/10 scenarios net-positive, including the real Gannon storm (+$1.15M/day at mean confidence)**, with a coherent severity trend -- losses grow with severity at the low end (50yr -$5.42M/day, 75yr -$3.76M/day) but flip to a growing benefit at the high end (150yr +$0.86M/day climbing to 250yr +$5.42M/day). This is the first candidate (of A/B/AB/C/D/E/F) where a majority of scenarios are positive and the pattern has a physically sensible shape (protection value grows with hazard severity, as it should).

**Important scope correction, prompted directly by the user questioning whether this result assumed Island F was "offline, even when it can be powered."** Checked: it does not, but not in the direction that sounds reassuring -- `econ/scripts/econ_analysis.py`'s $/day figures come entirely from fragility-curve transformer-failure probability, identical mechanism for baseline and every island. There is no term anywhere in the code for "the island's own population/GDP loses service during the planned disconnection window," despite this being explicitly designed in this Phase 5 section above (the "island's own temporary-outage cost" bullet) -- it was documented intent, never implemented. So generation was doing nothing at all in the $/day numbers being compared; its only actual effect anywhere in this project, before this fix, was steering `select_island()`'s seed choice.

**Fixed**: `scripts/compute_island_benefit_cost.py` implements the missing cost side of Phase 5's original design -- switching-labor cost (scaled by boundary_line_count, an explicitly flagged $25k/line assumption) plus the island's own-outage cost (its members' summed annual GDP x an assumed disconnection-hours fraction of a year, also explicitly flagged), converts the $/day econ diff into a one-time benefit via a repair-duration assumption (reported as low/mid/high 180/365/730-day variants rather than one number, since this is the calculation's biggest uncertainty), and -- this is where generation actually enters -- discounts the own-outage cost by how much of the island's own demand-proxy its matched EIA-860 generation covers (full coverage zeroes it; zero generation, as in Islands A-E, leaves it unchanged).

**Run against real data: the cost side turned out to be second-order.** Island A: $0.0594M total one-time cost (switching + fully-undiscounted own-outage, zero generation) against benefit magnitudes of ~$1800-2300M once integrated over the mid_365d repair-duration variant -- 0/10 scenarios positive, same as the raw diff, cost accounting changes nothing. Island F: whole-island generation (190 MW, summed over all 11 members) covers only 38% of whole-island estimated peak demand (503.8 MW) -- a much weaker coverage ratio than the seed substation's own 11.2x, revealing that generation-anchoring only genuinely covers the anchor itself, not the other 10 members grown around it -- but the resulting $0.324M total cost is still 3-4 orders of magnitude smaller than the benefit swings, so the 6/10-positive verdict is unchanged by adding costs. In both cases, whether an island helps or hurts is still determined almost entirely by the sign of the raw fragility-driven $/day diff, i.e. by which substations got selected and how current redistributes when they're cut -- which is what motivated the strategy revision below.

## Phase 6b — Strategy revision: redistribution-sensitivity-guided selection

**The SCI-driven greedy method is retired.** Six real candidate islands (A, B, combined AB, C, D, E, F) were selected by seeding on `argmax(SCI)` and growing by a local SCI-weighted value/cost ratio. Five were net-negative or negligible; Island F's apparent exception traced back to `require_local_generation` steering the same flawed algorithm into a different, luckier region of the graph, not to the algorithm itself working (see above). All SCI-selection code, and the two visualizations built around it, moved to `scripts/deprecated_sci_islanding/` (with a README explaining the full reasoning) -- kept working, not deleted, but no longer the recommended path.

**Diagnosis, confirmed by literature review.** Every real run showed the same pattern: cutting a boundary concentrated redistributed current onto a small, consistent set of substations elsewhere (`96396693` for Island A, `96284749` for Island F, etc.) large enough to flip the whole-Virginia result. This is a known, named failure mode in the GIC mitigation literature, not a Virginia-specific fluke -- Nagarajan et al.'s GIC-blocker-placement work states directly that "installing a blocking device at one location can exacerbate GIC levels at others across the power grid," and that "simplistic strategies like greedy methods are likely inadequate" for this problem class as a result (arXiv:2310.09409; review at arXiv:2402.07302). The field's actual methods -- exact optimization over the GIC circuit (arXiv:1701.01469), sensitivity-based placement, robust optimization under uncertain storm severity (arXiv:2006.02461, arXiv:2104.00204) -- share one property this project's SCI-greedy method lacked: every candidate is scored by its measured effect on the *whole* network, never by a local index at the cut site alone.

**Replacement: `scripts/select_island_by_redistribution_sensitivity.py`.** Same seeded-region-growing shape (so it's a drop-in producer of the same `va_island_{id}.json` schema every Phase 4-6 script already consumes), but every candidate -- seed and growth step alike -- is scored by actually cutting it, recomputing current-flow betweenness (Newman 2005; Brandes & Fleischer 2005 -- the same electrically-faithful betweenness `check_redistribution_proxy.py` used to diagnose SCI's shortest-path-betweenness flaw) on the rest of the network, and comparing to a one-time baseline. `net_score = benefit - redistribution_cost - switching_cost_weight * switching_cost`, where `redistribution_cost` is the *measured* mean risk-weighted current-flow-betweenness change elsewhere in the network, not an assumption. Risk weights are now two-term only (GIC risk + economic exposure) -- no static betweenness term baked in, since redistribution is measured dynamically instead. Extracted three new shared modules (`virginia_risk_data.py`, `island_graph_utils.py`, `virginia_demand_proxy.py`) so this and every future selection strategy can reuse the real per-substation data loaders and graph utilities without depending on the deprecated code.

Verified against synthetic fixtures (redistribution-cost sign correctness, self-consistent net_score-driven seed selection vs. naive argmax(risk_weight), growth termination, excluded_nodes, two-term-only risk weights, end-to-end `main()` wiring) -- **not yet run against real Virginia data**, which is genuinely the next step. Explicitly a greedy heuristic fix, not the field's full optimization formulation (MIQCP/DRO) -- honest about that in its own module docstring's LIMITATIONS section, and if it still doesn't produce a consistently positive result, that formal-optimization rung (or PowerModelsGMD.jl, an existing open-source tool for exactly this) is the next honest step, not another heuristic tweak.

**Real run #1 found a real bug: redistribution_cost was numerically negligible at real network scale.** `--island g --exclude a b c d e f` and an unconstrained rerun (`--island test_growth`, no exclusions -- same seed chosen either way, ruling out "ran out of room" as the explanation) both selected a single-substation island (`73989207`) with growth stopping immediately. `scripts/show_seed_candidate_trace.py` (new, prints the full seed-candidate comparison the console log never showed) revealed why: `redistribution_cost` for every one of the 10 evaluated candidates was ~1e-5 to 1e-8, versus `benefit` (~0.24-0.54) and `switching_cost` (~2.3-14.6) -- three to five orders of magnitude too small to affect `net_score` at all. `net_score` matched `benefit - 0.1*switching_cost` to 5+ decimal places for every candidate; the expensive current-flow-betweenness computation was being done but silently ignored.

Root cause: `evaluate_cut()` used the MEAN (divided by rest_node_count, ~2475 at real scale) instead of the SUM, out of a since-disproven concern that summing over thousands of substations would make redistribution_cost *dominate* benefit instead. Real data showed the opposite problem -- `current_flow_betweenness_centrality`'s normalized per-node values are already tiny at real network scale, so dividing by ~2475 crushed the signal to numerical irrelevance, degenerating net_score into "pick highest benefit, penalize switching cost" -- functionally reintroducing the exact local-proxy blindness this algorithm exists to fix, just with extra computation spent to get there. **Fixed**: switched to SUM, added a `redistribution_cost_weight` multiplier (default 1.0) for future tuning, and added a regression test (`test_redistribution_cost_is_sum_not_mean_scaled`) that directly checks the function's output against an independently-computed sum within one fixed graph -- an earlier version of that test tried to catch this by varying graph size instead and got confounded by `current_flow_betweenness_centrality`'s own normalization also shrinking with graph size (a separate, legitimate effect, not a bug).

**Re-confirmed against real data -- redistribution_cost is now real and meaningful** (candidates showed genuine positive and negative values, ~0.001-0.009 magnitude, vs. the old ~1e-8), but the SAME seed (`73989207`) still won every time, including an unconstrained rerun with zero exclusions (ruling out "ran out of room"). `scripts/show_seed_candidate_trace.py` (new -- the console log only ever printed the winning candidate, never the full comparison) revealed why: `73989207` has `boundary_line_count=1` -- a genuine degree-1 dead-end substation. Real GIC risk, real economic exposure, minimal switching cost, and (correctly, not a bug) ~zero redistribution effect, since nothing routes through a true leaf -- the actual best-scoring candidate by this criterion, just structurally unable to grow past itself. Added `--min-lines` (with `node_line_count()`, physical line count summed across parallel circuits, not graph degree) to restrict SEED choice only to substations with at least N physical lines, without touching growth mechanics.

**`--min-lines 2` immediately exposed a second, separate calibration problem: SWITCHING_COST_WEIGHT (0.1) was too large relative to benefit.** A single 230kV line's weighted contribution (0.23) alone exceeded every real candidate's benefit (observed range 0.15-0.34) -- every multi-line candidate scored net_score < 0 regardless of its actual redistribution profile, since switching_cost structurally dominated. Growth still worked mechanically (seed 96396692 grew to 2 members) since the stopping rule only requires improvement, not a positive score, but the ranking couldn't meaningfully differentiate candidates by quality. Recalibrated to 0.015 using the same real observed magnitudes -- checked against the actual trace that motivated this: all 10 real seed candidates that were uniformly negative at 0.1 became positive and meaningfully differentiated at 0.015. Like the mean/sum fix, this is a real-data-grounded correction, not a guess -- worth naming explicitly if this goes in the paper's methods section: the first real run against Virginia data caught two separate scale-calibration issues in the new algorithm within one afternoon, both fixed the same way (checking real output against real observed magnitudes, not synthetic fixtures alone).

Rerun with the corrected weight (`--island g --exclude a b c d e f --min-lines 2`) selected seed `96396692`, grown to 2 members (`+638514006`), `net_score = 0.16`. Pushed through Phase 4/5's real fragility/econ pipeline (`scripts/compute_island_benefit_cost.py --island g`): **9/10 scenarios net-positive** -- the best real result of any candidate so far (A-E: 0/10; F: 6/10). The one negative scenario is the real Gannon storm: `diff_per_day_musd = -1.089293`, own-outage cost $0.0129M (0% generation coverage), switching cost $0.05M, `net_musd` (mid_365d) = -397.66. See Phase 6c below -- this Gannon-specific loss, despite 9/10 return-period scenarios being positive, is the concrete case that motivated the directional-blindness investigation.

## Phase 6c — Directional blindness, and a cheap real-storm validation step

**Finding: `redistribution_cost` is direction-blind, and so are most of this project's "real-data" scenarios.** `redistribution_cost` (Phase 6b) is computed from `current_flow_betweenness_centrality`, which models current flowing between every pair of nodes in proportion to conductance -- an isotropic quantity with no storm direction in it at all, effectively an average over every possible direction rather than a measurement of what any one real storm does. This matters more than it might sound like it should, because it turns out most of this project's own "real" scenarios aren't independently directional either: rereading `scripts/est_gic.py`'s `load_and_process_gic_data` shows the nine statistical return-period scenarios (50-250yr, the backbone of Section 4's econ Monte Carlo and `compute_island_benefit_cost.py`'s 10-scenario set) get their *magnitude* from a lognormal return-period fit but have their per-line *sign* deliberately forced to match Gannon's own observed polarity (`sgn_full`) rather than drawn independently. Only three scenarios in the whole pipeline carry genuinely independent real directional information: Gannon (2024), Halloween (2003), and St. Patrick's Day (2015) -- each with its own measured per-line E-field, magnitude and polarity, from real magnetotelluric data.

This explains a finding from earlier real Island G output that was previously just flagged as an anomaly: Island G's own members' GIC INCREASED under the real Gannon forcing (+56.23A) but DECREASED under the 100-year return-period forcing (-59.01A) -- opposite signs, despite the 100yr scenario sharing Gannon's own polarity -- and the dominant redistribution-hotspot substation was completely different (`96396693` for 100yr vs. `96284749` for Gannon). A candidate validated only against `net_score` or against the return-period scenarios has not actually been checked against a real storm's direction, because none of those signals are independently directional in the first place.

No fix via better forecasting is available here -- checked directly (WebSearch): CME departure gives ~2-3 days of coarse warning with no reliable field-orientation information that early; L1 monitoring (ACE/DSCOVR) gives real measured IMF/Bz data but only ~15-60 minutes before impact; and precise, substation-resolved geoelectric-field *direction* forecasting beyond that lead time is stated directly in the literature as an open problem (regional GIC forecasting has "largely relied on forecasting dB/dt as a proxy," with "little effort devoted to forecasting geoelectric fields" directly -- Kelly et al., swsc-journal.org; the LSTM/"LiveWire" approaches, e.g. Hu et al. 2024 JGR:ML, top out around ~1 hour lead time). So the right response isn't "predict the direction and select for it" -- it's "check whether a candidate's benefit holds up across the range of real directions actually observed," which is what this phase implements.

**Implemented: `scripts/validate_island_directionality.py`.** A cheap real-physics filter between the topology-only search (Phase 6b, unchanged) and a full commitment to Phase 4/5's expensive Monte Carlo. For each finalist candidate: one single-iteration real admittance/GIC solve (`run_virginia_gic.py`'s `main()`, `start_idx=0, end_idx=1` -- ~1/`N_MONTE_CARLO` the cost of a full run) with the candidate's boundary actually cut, compared against an equivalent single-iteration baseline using the *same* seed draw (`seed_base + 0` for both, since neither run advances past iteration 0) -- isolating the topology change as the only source of difference, not random noise. Reuses `VA_GND_GIC_DIR/ground_gic0.csv` as the baseline if the full baseline Monte Carlo has already produced it (iteration 0 of a completed run is identical to a fresh single-iteration run), avoiding a redundant solve. Writes candidate output to new dedicated `gic_directional_check_{id}`/`gnd_gic_directional_check_{id}` directories (`configs.virginia_config.get_directional_check_dirs`) rather than `get_island_dirs(id)`'s Phase-4 paths, so a cheap one-iteration check can never collide with or be mistaken for a real, complete Phase 4 run for the same island id.

For every one of the twelve available storm forcings, computes the real-physics analogs of `evaluate_cut`'s two key terms: `own_effect` (risk-weighted change in `|GIC|` summed over the candidate's own members) and `redistribution_cost_real` (the same, summed over the rest of the network) -- `|GIC|`, not signed GIC, since magnitude is what drives transformer fragility, not direction. A candidate is **directionally consistent** if `redistribution_cost_real`'s sign agrees across all three genuinely independent real storms (gannon, halloween, st_patricks); **directionally fragile** if it flips sign between them -- the exact failure mode Island G's Gannon-vs-100yr divergence demonstrated. The nine return-period scenarios are still computed and reported (for continuity with Section 4/econ results) but are explicitly not counted toward the directional-consistency verdict, since they share Gannon's polarity rather than contributing independent directional evidence.

This does not replace Phase 4/5 and produces no $/day figure by itself -- it's a filter to avoid spending an overnight Monte Carlo run on a candidate a cheap single-iteration check already shows is direction-fragile.

**Run against real data on Island G.** Verdict: `directionally_consistent_across_real_storms = True`. But the per-storm numbers matter more than the pass/fail flag: gannon (own_effect +15.84, redistribution_cost_real -4.67), halloween (+9.50, -2.95), and st_patricks (+2.99, -0.97) all agree with each other -- own members get worse, rest of network gets better -- while all nine return-period scenarios (50-250yr) show the opposite pattern (own members better, rest of network worse), scaling smoothly with severity. Real vs. synthetic is a clean, unanimous split, not noise. Since Island G's 9/10-positive econ result (above) is 9 return-period scenarios plus 1 real one (Gannon, the one that lost), this raises the real possibility that the headline "9/10 positive" is dominated by scenarios systematically unlike any real storm. Halloween and St. Patrick's Day have real GIC data (`GIC_halloween`/`GIC_st_patricks` columns already computed by `est_gic.py`) but are not currently wired into `run_virginia_fragility.py`/`econ/scripts/econ_analysis.py` -- getting real `net_musd` figures for those two, not just the raw-GIC comparison this script gives, is the concrete next step before Island G's result can be called validated rather than merely mechanically consistent.

## Phase 6d — Generation removed; the search itself scored by real storms

**Generation-anchoring removed from active code, not just deprecated.** `scripts/check_island_generation.py` and `scripts/virginia_demand_proxy.py` moved to `scripts/deprecated_sci_islanding/` (imports fixed in their remaining consumers there); `scripts/compute_island_benefit_cost.py`'s own-outage cost is unconditional again -- no generation-coverage discount. Direct comparison settled it: Island G, selected with zero generation requirement, reached 9/10 net-positive, beating generation-anchored Island F's 6/10, without constraining seed choice to wherever generation happens to exist. See `scripts/deprecated_sci_islanding/README.md`'s "Generation-anchoring is retired too" for the full reasoning.

**The search itself now scores by real storms, replacing current_flow_betweenness_centrality entirely.** Phase 6c's real-storm check was a post-hoc filter applied *after* `select_island_by_sensitivity()` already picked a winner using the isotropic, direction-blind betweenness proxy -- which is exactly how Island G could pass selection and still lose on the one real, independent-direction storm (Gannon) in its econ set. `evaluate_cut()` (`scripts/select_island_by_redistribution_sensitivity.py`) now scores every candidate by one real single-iteration admittance/GIC solve (reusing Phase 6c's machinery, now factored into a shared module `scripts/real_storm_gic.py`) against gannon/halloween/st_patricks -- the three genuinely independent real historical storms -- and takes the **worst case across them, not the average**, directly targeting "a method that is positive for all cases" rather than positive on average.

**Implementation, for reuse across many candidate solves in one process.** `scripts/run_virginia_gic.py`'s `main()` was split into `load_gic_context()` (the expensive, exclude_line_ids-independent setup -- network topology, transformer-composition samples, storm E-field data, ~10-20s observed) and `run_monte_carlo(ctx, ...)` (the actual per-iteration solve, now taking a pre-loaded context). `main()` itself is unchanged -- every existing caller (`run_virginia_islanded_gic.py`, `validate_island_directionality.py`) keeps working identically -- but a search loop can now call `load_gic_context()` once and reuse it across dozens of candidate evaluations instead of repaying that setup cost every time. `scripts/real_storm_gic.py` holds the shared real-storm scoring logic (`solve_single_iteration`, `ensure_baseline_single_iteration`, `real_storm_redistribution`, `worst_case_redistribution_cost`) used by both `validate_island_directionality.py` (refactored onto this shared module, now also reusing one loaded context across all `--candidates` instead of reloading per candidate) and the search itself. A new `get_search_scratch_dirs()` config helper gives the search a reusable scratch location for per-candidate solves -- deliberately overwritten every candidate (`solve_single_iteration` deletes the previous output before solving the next), since a search evaluates dozens of candidates that are never saved as a real island.

**Cost tradeoff, stated plainly.** One admittance solve already computes all twelve storm columns at once, so scoring against three real storms costs the same as scoring against one -- but that one solve still costs ~20-40s on real data (dominated by admittance-matrix construction and per-period GIC calculation, not by the one-time context load), versus the old betweenness proxy's near-instant graph computation. A full search run's total real-solve count is roughly `n_seed_candidates + sum(boundary fan-out across growth steps)` -- expect a search run to take meaningfully longer than before; lower `--n-seed-candidates`/`--max-size` if that's impractical.

**`SWITCHING_COST_WEIGHT`/`REDISTRIBUTION_COST_WEIGHT` need a third real-data calibration pass.** Both were tuned for current_flow_betweenness's normalized scale (~0.001-0.009 typical redistribution_cost magnitude); `real_storm_gic`'s redistribution_cost lives in raw risk-weighted `|GIC|` amps instead (Phase 6c's real Island G check showed per-storm magnitudes in the 1-30 range) -- a completely different scale. Same pattern as the two calibration fixes already made to this file (SUM-vs-MEAN, then the 0.1-vs-0.015 switching-cost-weight fix): expect to check real output magnitudes and recalibrate, not treat the current defaults as still correct.

**Run against real data (no `--exclude` -- the point now is the single best real candidate, not a structurally-independent second island, so nothing was excluded): Island H.** Seed `107946151`, grown to 3 members (`107946150`, `108016246`), 4 boundary lines cut, `net_score=1.9486` (`benefit=1.2581`, worst-case `redistribution_cost=-0.8285`). No weight recalibration was actually needed this run -- `benefit` and the redistribution term's contribution came out comparable orders of magnitude (~1.26 vs ~0.83) on real data, unlike both prior scale-mismatch bugs; worth continued watching on future runs, but not an emergency this time.

All three real storms agreed in sign (`redistribution_cost_real`: gannon -3.4978, halloween -2.9808, st_patricks -0.8285 -- all favorable) with small, positive `own_effect` (+0.9223, +0.2074, +0.2135) -- `directionally_consistent_across_real_storms = True`, and critically, unlike Island G, the small own-member cost is dominated by the larger network-wide benefit under every real storm, not the reverse.

**Pushed through Phase 4/5: 10/10 scenarios net-positive, including the real Gannon storm** (`net_musd` mid_365d = +18.49, benefit-cost ratio ~118) -- the first candidate to clear Gannon. All nine return-period scenarios also positive, scaling monotonically with severity (50yr +$58.4M/ratio 370 through 250yr +$455.8M/ratio 2,884), same shape as Island F. This is a real predictive-validity confirmation for the cheap real-storm check: the raw-GIC own-cost/network-benefit ratio predicted a net-positive Gannon result before the expensive Monte Carlo ran, and the full pipeline confirmed it -- the same kind of validation as the `96396693` hotspot match earlier in this project, now for the corrective method itself.

**Resolved -- and the real answer is more nuanced than hoped.** Halloween and St. Patrick's Day are now wired in (Phase 6d's `SCENARIOS` monkeypatch), giving real dollar figures for all three real storms: `gannon-year` +$18.49M net (positive), `halloween-year` **-$4.51M net (negative)**, `st_patricks-year` **$0.00M** (exactly zero diff, net -$0.16M once costs are included). Full 12-scenario tally: **10/12 net-positive** (all nine return periods plus Gannon) -- still the best result of any candidate -- but the real-storm-only tally is **1/3 positive, not 3/3**.

This is a real, demonstrated limitation of the raw-GIC directional check above, not just a theoretical one: that check found `redistribution_cost_real` favorable (negative) for all three real storms and reported `directionally_consistent_across_real_storms = True`, yet Halloween's actual $/day sign came out negative. Sign agreement in aggregated `|GIC|` magnitude is not sufficient evidence of $/day-level agreement, because the fragility curve is nonlinear and the econ diff depends on which specific substations see the GIC change (and their population/GDP exposure), not just a network-wide sum. The cheap check correctly predicted Gannon's sign -- it has real predictive value -- but it is not a substitute for the full pipeline, and this is the first concrete case in the project demonstrating that gap rather than just asserting it.

St. Patrick's Day's exact $0.00M plausibly reflects a genuine floor effect (a materially weaker real storm than Gannon, plausibly not crossing the fragility failure threshold anywhere in Virginia under either topology) rather than a data artifact -- worth confirming directly against `va_ghosh_results.csv`'s raw `total_impact` for that scenario (baseline vs. Island H) before treating it as settled, rather than assuming.

**Honest status for the paper**: Island H is real-storm-positive for the most severe historical event available (Gannon) and the entire return-period family, negative for a weaker real storm (Halloween), neutral for another (St. Patrick's). No candidate this project has produced clears all three real storms. That is a stronger, more defensible result to report than a clean sweep would have been -- it shows the real-storm validation methodology is actually discriminating between outcomes, not confirming whatever the search selects.

**Implemented: `run_virginia_fragility.py` now processes Halloween and St. Patrick's Day too.** Turned out to be a one-line gap, not a missing mechanism: `econ/preprocess/p_gic_files.py`'s `process_gic_file()` already has a rename map for `GIC_halloween`/`GIC_st_patricks` (was there all along), and every downstream script (`econ_analysis.py`, `compare_island_econ.py`, `compute_island_benefit_cost.py`) discovers scenarios dynamically from the data rather than a hardcoded list -- the only thing gating them out was `SCENARIOS` (`configs.settings.REGULAR_SCENARIOS`, ten entries, national/shared code left untouched per this project's convention), which `run_virginia_fragility.py` now monkeypatches to add the two extra columns, the same technique it already used for `DATA_DIR`/`DATA_LOC`/`OUTPUT_FILES`.

**Cannot be resumed from existing checkpoints.** `process_all_files()`'s resume logic is a pure batch-file count with no awareness of which SCENARIOS list produced them -- resuming after this change would silently combine old batches (missing Halloween/St. Patrick's) with new ones (that have them) into one corrupted output. The script now hard-fails with a clear error if it finds existing checkpoint batches under `econ_dir`, rather than resuming into that silently-wrong state. **This means both the baseline (`VA_ECON_DIR`) and Island H's econ_dir need their `gic_processing_batches`/`gnd_gic_processing_batches` directories cleared and the full fragility Monte Carlo rerun from scratch** -- a real, expensive undertaking ("at least as long as the GIC Monte Carlo itself, likely considerably longer," per the script's own warning, now across 12 scenarios instead of 10) -- not something to kick off without expecting it to take a while. Not yet run against real data.

## Phase 6e — Closing the proxy-objective gap: fragility-weighted scoring

**Confirmed first: Island H/I's limitation is the scoring function, not search breadth.** Rerunning the search with a much larger budget (`--n-seed-candidates 10 --max-size 15`, no `--exclude`) reproduced Island H exactly -- same seed, same 3 members, same net_score, same natural stopping point at 3 members (not a size-cap artifact; growth stopped on `no_improving_candidate` well under the cap in both runs). A wider search finding the identical answer rules out "the small search budget missed something better" -- the ceiling is `redistribution_cost`'s definition itself.

**The gap, precisely.** `real_storm_redistribution`'s `redistribution_cost` is a raw risk-weighted `|GIC|` magnitude change. The actual $/day objective goes through a nonlinear, threshold-sensitive fragility curve (`econ/preprocess/p_gic_files.py`'s `Pf = Phi((ln Cpen + ln|GIC| - ln theta)/beta)`) and is weighted by each substation's specific GDP exposure, not an aggregated network-wide sum. Island H/I's Halloween result is the demonstrated case: raw-`|GIC|` sign agreed across all three real storms, but the real $/day sign for Halloween came out negative anyway.

**Implemented: `scripts/real_storm_gic.py` gained a fragility-weighted alternative.** `load_fragility_exposure()` loads `va_substation_exposure.gpkg` (the same POP20/GDP_*/EST_* table the real fragility Monte Carlo uses). `fragility_weighted_per_substation()` calls `econ/preprocess/p_gic_files.py`'s own `_simulate_scenario()`/`one_draw_vec()` directly -- not a reimplementation, the exact real fragility function -- at a small fixed draw count (`CHEAP_FRAGILITY_DRAWS=300` vs. the real pipeline's 20,000+ with early-stop convergence checking), returning each substation's failure-probability-weighted GDP exposure. `fragility_weighted_redistribution()` computes the before/after delta of that quantity, split into `own_effect` (members) and `redistribution_cost_real` (everyone else) -- same shape as the raw-`|GIC|` version, same `worst_case_redistribution_cost()` (unchanged, reused as-is). Two fragility-model calls per real storm per candidate, each cheap relative to the admittance solve that's already the dominant cost.

Restricted to Virginia by construction (the exposure table only covers Virginia's ~413 substations) -- a closer match to the real whole-Virginia econ objective than the raw-`|GIC|` version's PJM-wide scope.

`select_island_by_redistribution_sensitivity.py`'s `evaluate_cut()`/`select_island_by_sensitivity()`/`main()` now thread an `exposure` argument through (loaded once per search run, like `gic_ctx`) and call `fragility_weighted_redistribution()` instead of `real_storm_redistribution()`. `scripts/validate_island_directionality.py` (the post-hoc, all-12-period report) is unchanged -- it still uses the raw-`|GIC|` comparison, which remains a valid, different-purpose diagnostic (continuity with the return-period scenario set), not the search's in-loop objective anymore.

**Weights need a fourth real-data calibration pass.** `SWITCHING_COST_WEIGHT=0.015`/`REDISTRIBUTION_COST_WEIGHT=1.0` were tuned for raw-`|GIC|`-scale magnitudes (1-30 per storm). Fragility-weighted `redistribution_cost` is a probability (typically << 1) times a GDP exposure value -- almost certainly a much smaller number. Same pattern as every prior scale change in this file: check real output magnitudes, recalibrate, don't assume the current defaults still hold.

**Status: implemented, not yet run against real data.** Next real candidate should be saved as Island J (H and I are identical and already fully validated/documented). Expect a real timing check first (`--n-seed-candidates 5 --max-size 5` before committing to a full run, same caution as before) -- the fragility calls are cheap relative to the admittance solve, but real magnitudes and any GPU/CPU availability differences haven't been observed yet.

## Phase 6f — Fixed: a real precision bug was silently zeroing small effects

**Island J's exact-zero gannon/st_patricks results (Phase 6e) were a bug, not a genuine finding.** Traced methodically against real data: confirmed real, substantial per-substation differences at the raw ground-GIC level (realization 5, several substations differing by multiple amps); confirmed real, varying per-realization fragility output in batch 0; confirmed the fully-combined 3000-realization dataset (reconstructed fresh via the same `xr.concat` `load_gic_results()` uses) still shows that same real variation (realization 19's `GDP_AGR_affected` = 0.17338990542618557, matching the raw fragility batch exactly, not baseline's value). Every stage up through the combined dataset is correct.

**Root cause: `econ/scripts/econ_analysis.py`'s `get_confidence_intervals()` ends with `return df.round(2)`**, rounding the "mean" column -- the actual value `run_ghosh_value_added()` reads as the IO model's shock input -- to 2 decimal places *before* it reaches the model. For Islands A/B/F/G/H, the true difference between baseline's and the island's mean GDP-affected always exceeded 0.01, so this was invisible. Island J is a single-substation cut -- a much smaller lever -- and for the real Gannon storm specifically, baseline's and Island J's true means rounded to the identical 2-decimal figure. Since the Ghosh propagation is a deterministic linear transform, identical rounded input produced bit-for-bit identical `total_impact` (confirmed: `-66.45523912599468` exactly, both sides) despite genuinely different underlying data.

**Fixed**: `scripts/run_virginia_econ_analysis.py` no longer calls the shared `get_confidence_intervals()` -- it has its own `get_confidence_intervals_unrounded()`, an exact fork of that function's non-alpha-beta branch (the only branch this project ever exercises) with the final `.round(2)` removed. Reimplemented rather than editing the shared file in place, same convention `run_virginia_gic.py` already follows for `est_gic.py`'s loop body. `run_virginia_islanded_econ_analysis.py` picks up the fix automatically (it calls this file's `main()`).

**What this means for results reported so far.** Islands A, B, F, G, H's headline verdicts (net-negative / 6/10 / 9/10 / 10/10) are unlikely to flip -- their effects were large enough to survive 2-decimal rounding -- but their exact dollar figures were computed against rounded shock inputs and should be treated as approximate until rerun. This is cheap to fix: `run_virginia_econ_analysis.py`/`run_virginia_islanded_econ_analysis.py` only redo the IO-model step (fast, not the fragility Monte Carlo), so rerunning econ analysis for the baseline and every island (A-H, J) with the fix is a same-session task, not an overnight one. Island J needs its full `compare_island_econ.py`/`compute_island_benefit_cost.py` sequence rerun after that to get its real (no longer bugged) gannon/st_patricks figures.

## Phase 6g — Bounded lookahead: one-step-stopping couldn't see non-monotonic growth

**Real-data motivation.** Once Phase 6f's precision fix was applied and the full econ re-run was done for every island (baseline + A, B, AB, C, D, E, F, G, H, J), Island G's real result stood out sharply: 9/10 scenarios net-positive, net_musd sum **+115.86** — an order of magnitude larger than Island H (+6.11) or Island J (+1.12), this file's own real-storm/fragility-weighted search's selections. Island G was built under the now-fully-deprecated generation-anchored/current-flow-betweenness method (Phase 6/6d's predecessor) and is no longer reachable by this file's search at all. That a strictly better real candidate exists outside the search's reach is a live demonstration of a real limitation, not a hypothetical one.

**The specific blind spot.** `select_island_by_sensitivity()`'s growth loop only ever adds one member per step and stops the moment the single best candidate would make `net_score` worse than the current island's. That one-step rule cannot distinguish "this branch is a dead end" from "this branch is temporarily worse but leads somewhere better" — e.g. `{1}` scoring higher than `{1,2}`, but `{1,2,3}` scoring higher than either. Since growth never backtracks or removes a member, the old rule stops permanently at `{1}` and never finds `{1,2,3}`.

**Fix: `_try_bounded_lookahead()`.** When the best growth candidate at a step would make `net_score` worse (the old stopping trigger), the search no longer gives up immediately. It tentatively commits to that candidate and keeps greedily growing (still single-best-candidate-per-step, not branching) for up to `--lookahead-steps` more steps (default `LOOKAHEAD_STEPS = 2`), checking after each step whether `net_score` has climbed back past the score the island had *before* the tentative step. The moment it does, the whole tentative path up to that point is committed for real and normal growth resumes. If the lookahead budget runs out without recovering, the tentative extension is discarded entirely and the search stops exactly where the old rule would have — a strict relaxation, never worse than before, sometimes better. `--lookahead-steps 0` reproduces the old strict behavior exactly (kept as an explicit escape hatch, not the default).

**Not exhaustive.** Only the single greedy-best candidate is explored at each lookahead step, not every branch — a full branch-and-bound would cost combinatorially more real admittance/GIC solves per rejected step. This catches the reported `(1,2,3) > (1) > (1,2)` failure mode specifically, not every possible non-monotonic path a true joint optimization (the MIQCP/DRO methods already named in this file's LIMITATIONS section) would find.

**Status: run against real data — Island K.** `select_island_by_redistribution_sensitivity.py --island k` (default `--lookahead-steps 2`, no `--exclude`) used the lookahead twice: once at 4 members (dip to net_score 2.0494 at step 4, recovered to 2.5014 at step 5, both committed) and once at 9 members (dip to 3.5586 at step 9, recovered to 4.2551 at step 10). Growth continued to 12 members and a final net_score of 5.3762 — versus the 4-member/2.3618 island the old strict one-step rule would have stopped at. The final stop (12 members, best rejected candidate 4.3976 vs. current 5.3762) held even with lookahead, i.e. a real stop, not a premature one.

Full pipeline result (`run_virginia_islanded_gic.py` → `run_virginia_islanded_fragility.py` → `run_virginia_islanded_econ_analysis.py` → `compare_island_econ.py` → `compute_island_benefit_cost.py`, all 12 scenarios including Halloween/St. Patrick's from the start): **11/12 scenarios net-positive** at mean confidence — only `st_patricks-year` misses, and barely (diff `-0.000217`, net `-0.46` musd). Per real-storm comparison against H and J:

| Storm | H | J | K |
|---|---|---|---|
| gannon | +0.036036 | +0.002409 | +0.027673 |
| halloween | -0.031292 | -0.031279 | **+0.006126** |
| st_patricks | -0.000564 | -0.000136 | -0.000217 |

K is the first candidate to come back positive on Halloween — the real storm that originally motivated fragility-weighted scoring (Island H/I's Halloween failure, Phase 6e). This is a genuine, concrete case where bounded lookahead found a materially different, more real-storm-robust candidate that the old one-step-stopping rule structurally could not reach (growth never backtracks, so once it stops at 4 members it can never see the 12-member island). K's total `net_musd` sum across all 12 scenarios (mid_365d variant) is ≈$1914M, somewhat below H's ≈$2227M -- a real dollar-size-vs-direction-robustness tradeoff, not a strict dominance either way, worth reporting as such.

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
