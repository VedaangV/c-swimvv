# Deprecated: SCI-driven greedy islanding

Everything in this folder implements the **first** islanding-selection
strategy tried in `ISLANDING_MITIGATION_PLAN.md` (Phases 2-3, and the
Phase 6 extensions built on top of it): compute a per-substation
"Substation Criticality Index" (SCI — a weighted blend of GIC risk,
economic exposure, shortest-path betweenness, and optionally critical-load
density), seed a region on the highest-SCI substation, and greedily grow
it outward by a local SCI-weighted value/cost ratio.

**This strategy is retired.** It selected six candidate islands (A, B,
C, D, E, F) against real Virginia data; five were net-negative or
negligible for whole-Virginia $/day loss in every scenario tested, and
the one exception (Island F) turned out not to be attributable to the
method itself (see below). Literature review turned up a direct,
peer-reviewed explanation for why greedy local selection doesn't work
for this problem class — see **Why it was retired** below — and the
active replacement is a sensitivity-based approach that scores each
candidate cut by its actual whole-network effect instead of a local
proxy: `scripts/select_island_by_redistribution_sensitivity.py`.

## Why it was retired

Two compounding problems, diagnosed over the course of this project:

1. **The betweenness term used the wrong physics.** SCI's betweenness
   component was shortest-path betweenness (Freeman 1977) — "how much
   traffic takes the single cheapest route through this node." GIC
   doesn't route that way: it's Ohm's-law current splitting across every
   parallel path simultaneously, in proportion to conductance. A
   current-flow-betweenness proxy (`check_redistribution_proxy.py`,
   below) showed real but incomplete predictive power once tested.

2. **The bigger problem: no term in SCI, or in the greedy growth ratio,
   ever modeled where the current goes once a boundary is cut.** The
   algorithm only asked "is this a good local trade" (SCI-weighted value
   inside the island vs. voltage-weighted cost of the cut), never "what
   does this cut do to the rest of the network." Every real run showed
   the same pattern: a small number of substations elsewhere absorbed a
   large share of the redistributed current, consistently enough to flip
   the whole-Virginia result negative even when the cut looked
   reasonable by its own local metric.

   This turned out to be a **known, named problem** in the GIC
   mitigation literature, not a Virginia-specific fluke. Nagarajan et
   al.'s review of the GIC blocker placement problem states it directly:
   "installing a blocking device at one location can exacerbate GIC
   levels at others across the power grid," which is why "simplistic
   strategies like greedy methods are likely inadequate" for this class
   of problem (arXiv:2310.09409, arXiv:2402.07302). The field's actual
   approaches — exact optimization over the GIC circuit
   (arXiv:1701.01469), sensitivity-based placement, and robust
   optimization under uncertain storm severity (arXiv:2006.02461,
   arXiv:2104.00204) — all share the property this project's SCI-greedy
   method lacked: they score a candidate intervention by its real effect
   on the whole network, not a local index at the cut site.

3. **Island F's apparent counterexample doesn't rescue the method.**
   Island F (seeded via `require_local_generation`, anchored on a
   substation with real nearby EIA-860 generation) was the only
   candidate with a majority-positive result. Investigating why revealed
   it isn't evidence SCI-greedy growth works when generation-anchored —
   the GIC/fragility/econ pipeline never modeled supply continuity
   anywhere; `require_local_generation` only changed which 11
   substations got selected as seed and grown, landing the algorithm in
   a different, better-performing region of the graph than Island A's
   SCI-only seed. That's a real result, but it's evidence generation
   happened to point the SAME flawed greedy algorithm somewhere luckier
   — not evidence the algorithm's blindness to redistribution stopped
   mattering. See `ISLANDING_MITIGATION_PLAN.md`'s Phase 6 section for
   the full trace of this reasoning.

## Generation-anchoring is retired too (moved here later, not part of the original SCI retirement above)

`check_island_generation.py` and `virginia_demand_proxy.py` were still
active (used by `scripts/compute_island_benefit_cost.py` to discount an
island's own-outage cost by matched EIA-860 generation coverage) after
the SCI method itself was retired — moved here once that stopped being
true too. The direct comparison settled it: Island G, selected by the
*current* method (`select_island_by_redistribution_sensitivity.py`) with
no generation requirement at all, reached 9/10 scenarios net-positive —
better than Island F's 6/10, without constraining seed choice to the
substations where generation happens to exist. Combined with the earlier
finding that best-redistribution-sensitivity and generation-backed
substations barely overlap in Virginia (Section 5.4,
`VIRGINIA_ADAPTATION.md`), requiring or crediting generation looks like a
net cost to candidate quality now, not a benefit — so
`compute_island_benefit_cost.py`'s own-outage cost is unconditional
again (no coverage discount), matching how every island except F was
already being scored.

## What's in this folder

| File | Was |
|---|---|
| `compute_virginia_criticality.py` | Computes SCI (Phase 2). |
| `select_virginia_island.py` | Seeded greedy region-growing on SCI (Phase 3), including the `require_local_generation` seed-anchoring extension. |
| `check_redistribution_proxy.py` | Current-flow-betweenness proxy, built to patch SCI's shortest-path betweenness flaw specifically. |
| `check_generation_sci_overlap.py` | Diagnostic: does any substation clear both a real SCI threshold and a real generation threshold. |
| `search_candidate_islands.py` | Orchestrates select-and-evaluate for multiple SCI-selected candidates in sequence (used for C, D, E). |
| `plot_virginia_criticality.py` (moved from `viz/`) | SCI ranking map. |
| `plot_virginia_island.py` (moved from `viz/`) | Sanity-check figure for one SCI-selected island's members/boundary/growth trace. |
| `check_island_generation.py` (moved from `scripts/`) | EIA-860 generation-capacity matching -- fed `require_local_generation` seed-anchoring above, and later `compute_island_benefit_cost.py`'s own-outage discount, until that was also retired (see below). |
| `virginia_demand_proxy.py` (moved from `scripts/`) | Population-based demand proxy -- existed only to compute generation-coverage ratios for the discount above. |

## What's still active and shared

These modules hold genuinely reusable pieces that were embedded inside
the files above but have nothing to do with SCI specifically — they
were extracted so active code (including the new sensitivity-based
selector) doesn't depend on deprecated code:

- `scripts/virginia_risk_data.py` — raw per-substation data loaders (GIC
  risk, economic exposure, critical-facility counts, topology metrics).
- `scripts/island_graph_utils.py` — graph structure utilities (load the
  network graph, compute a boundary cut, cost it, attach real
  conductance for current-flow analysis).

Everything downstream of *selecting* an island (Phase 4's topology-edited
GIC re-solve, Phase 5's fragility/econ re-evaluation, the GIC/econ
comparison and benefit-cost scripts) is selection-method-agnostic — it
just reads a `va_island_{id}.json` with `members`/`boundary_line_ids`,
regardless of how those were chosen — so none of it moved here.

## If you need to run something in this folder

It still works (verified against fixtures after the extraction above),
but treat any new result from it as measuring the old strategy, not as
a candidate for the project's current recommendation.
