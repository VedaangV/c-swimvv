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
