"""
Replacement for scripts/deprecated_sci_islanding/select_virginia_island.py
(see that folder's README.md for why the SCI-driven greedy method was
retired). Same seeded-region-growing SHAPE -- pick a seed, grow it by
adding whichever adjacent substation most improves a score, stop when
nothing improves -- so this stays a drop-in producer of the same
va_island_{id}.json schema (members, boundary_line_ids, seed,
growth_trace) that every downstream Phase 4-6 script already consumes.

PHASE 6d: SCORED BY REAL STORMS, NOT current_flow_betweenness_centrality
--------------------------------------------------------------------------
Two scoring engines have now been tried, both for the same underlying
reason: local greedy criteria can't see that a cut's benefit at one
substation gets paid for -- or overpaid for -- by increased current
importance somewhere else in the network (six real island selections,
A-F, and a literature review both converged on this; see Nagarajan et
al., "Heuristic Algorithms for Placing Geomagnetically Induced Current
Blocking Devices" (arXiv:2310.09409): "installing a blocking device at
one location can exacerbate GIC levels at others across the power grid").

1. First fix (retired this revision): current_flow_betweenness_
   centrality (Newman 2005; Brandes & Fleischer 2005) on the rest of the
   network, before vs. after the cut. Cheap, and a real improvement over
   the old SCI method's shortest-path betweenness -- but current-flow
   betweenness has no storm in it. It models current flowing between
   every pair of nodes in proportion to conductance, isotropically --
   an average over every possible direction, not a measurement of what
   any ONE real storm's geoelectric field would actually do. This is
   exactly why Island G (selected by this proxy) came back 9/10
   scenarios net-positive in the real econ pipeline, with the one loss
   being the real Gannon storm -- and why a real-storm check
   (scripts/validate_island_directionality.py, Phase 6c) found Island
   G's own members' GIC flips sign between the real Gannon forcing and
   the (non-independently-directional) 100-year return-period scenario.
2. Second fix (retired this revision too): `evaluate_cut()` scored every
   candidate by an actual single-iteration real admittance/GIC solve
   (scripts/real_storm_gic.py, reusing the exact machinery Phase 6c
   already proved out) against the three genuinely independent real
   historical storms this pipeline has data for -- gannon, halloween,
   st_patricks -- taking the WORST case across them, not the average.
   This is what selected Island H/I. Real result once pushed through the
   full econ pipeline with Halloween/St. Patrick's finally wired in
   (Phase 6d's SCENARIOS fix): 10/12 scenarios net-positive, but only
   1/3 real storms (Gannon) -- Halloween came back net-NEGATIVE despite
   agreeing in sign with Gannon and St. Patrick's at the raw-|GIC| level.
   Widening the search (more seed candidates, larger max_size) reproduced
   the exact same candidate, ruling out "search wasn't broad enough" --
   the ceiling was the scoring function itself.
3. Current fix: raw |GIC| magnitude is not the quantity that actually
   determines $/day economic impact -- the fragility model
   (econ/preprocess/p_gic_files.py's lognormal Pf = Phi((ln Cpen +
   ln|GIC| - ln theta)/beta) curve) is nonlinear and threshold-sensitive,
   so two substations with the same |GIC| CHANGE can have very different
   effects on failure probability depending on where they sit relative
   to theta, and the whole-Virginia econ diff depends on which SPECIFIC
   substations see that change, weighted by THEIR GDP exposure -- not on
   an aggregated network-wide |GIC| sum. `evaluate_cut()` now scores
   redistribution_cost via scripts.real_storm_gic.fragility_weighted_
   redistribution(), which reuses econ/preprocess/p_gic_files.py's OWN
   fragility function directly (not a reimplementation) at a small draw
   count (CHEAP_FRAGILITY_DRAWS=300 vs. the real pipeline's 20,000+), to
   get each substation's expected-GDP-affected before/after the cut. This
   closes the specific gap Island H/I's Halloween result demonstrated:
   raw-|GIC| sign agreement is not sufficient evidence of $/day-level
   agreement.

SCORING
-------
For a candidate node set `members` (a single seed, or the current
island plus one growth candidate):
1. boundary = induced cut of members (scripts/island_graph_utils.py);
   boundary_line_ids = the physical lines that cut represents.
2. One single-iteration real admittance/GIC solve
   (scripts.real_storm_gic.solve_single_iteration) with those lines cut,
   compared against a one-time real single-iteration BASELINE (no cut) --
   same seed_base+0 Monte Carlo draw for both, so any difference is
   attributable to the topology change, not random noise.
3. redistribution_cost = MAX over gannon/halloween/st_patricks of the
   fragility-weighted expected-GDP-affected CHANGE summed over every
   substation NOT in `members`
   (scripts.real_storm_gic.fragility_weighted_redistribution +
   worst_case_redistribution_cost). Positive means the cut made the rest
   of Virginia's expected fragility-priced exposure worse, under the
   worst of the three real directions; negative means every one of the
   three real storms improved it. Restricted to substations with real
   economic exposure data (Virginia only, ~413 substations) -- a closer
   match to the actual whole-Virginia econ objective than the prior
   PJM-wide raw-|GIC| version.
4. benefit -- SUPERSEDED by Phase 6i below. Used to be the static sum of
   risk_weight over `members`; is now -worst_case_total_effect, the real,
   worst-case-across-real-storms whole-Virginia effect. See PHASE 6i.
5. switching_cost = voltage-weighted boundary cost (unchanged --
   see scripts/island_graph_utils.py's edge_cost()).

   SWITCHING_COST_WEIGHT (0.015) was calibrated for the raw-|GIC|
   version's scale (per-storm magnitudes in the 1-30 range); benefit and
   redistribution_cost's scales have both changed since (fragility-
   weighted, Phase 6e; then real-worst-case-total, Phase 6i) without a
   fresh switching_cost_weight recalibration against the new numbers --
   expect this to need another pass once real magnitudes are observed,
   same "check real output magnitudes, recalibrate" step this file has
   needed every time its scoring function's underlying units changed.
6. net_score -- SUPERSEDED by Phase 6i below. Used to be benefit -
   redistribution_cost_weight * redistribution_cost - switching_cost_
   weight * switching_cost, mixing a static topology proxy (benefit) with
   a real worst-cased quantity (redistribution_cost). See PHASE 6i for
   why that was a real gap and what replaced it.

RISK WEIGHTS
------------
Two-term only: 0.5*normalized GIC risk + 0.5*normalized economic
exposure density (scripts/virginia_risk_data.py's loaders). No
generation term anywhere -- see scripts/deprecated_sci_islanding/
README.md's "Generation-anchoring is retired too" for why (Island G,
selected with zero generation requirement, beat generation-anchored
Island F's 6/10 with 9/10).

PHASE 6i (see below) removed risk_weight from net_score/benefit entirely
-- it now serves ONLY as a candidate-discovery heuristic (which
substations rank in the top-`n_seed_candidates` seed pool), same
supporting role current-flow betweenness plays for Phase 6h. It no
longer has any vote in which candidate the search actually selects.

SEED CANDIDATE POOL
--------------------
n_seed_candidates restricts seed evaluation to the top-N substations by
raw risk_weight (highest GIC risk + economic exposure) rather than every
eligible substation. This bound matters much more now than it did under
the cheap betweenness proxy: each seed candidate now costs one real
admittance/GIC solve (~20-40s observed on real data, see scripts/
real_storm_gic.py's module docstring), not a fast graph computation.
Growth-step candidates are bounded only by the existing boundary-
adjacency constraint (a boundary's fan-out, typically single digits to
tens) -- also now each a real solve. A full search run's total real-solve
count is roughly (n_seed_candidates + sum of boundary fan-out across
growth steps) -- lower --n-seed-candidates and/or --max-size if a run is
impractically slow.

min_lines filters the SEED pool (not growth) to substations with at
least that many physical lines incident to them (summed line_ids across
all their graph edges) -- added after a real run kept selecting a
genuine degree-1 dead-end substation as seed every time: real GIC risk,
real economic exposure, minimal switching cost, and (correctly, not a
bug) essentially zero redistribution cost, since nothing routes through
a true leaf. min_lines >= 2 forces seed selection toward substations
that could plausibly anchor a multi-substation region, without touching
growth itself.

PHASE 6g: BOUNDED LOOKAHEAD -- ONE-STEP GREEDY CAN'T SEE NON-MONOTONIC PATHS
-----------------------------------------------------------------------------
Real-data motivation: once Phase 6f's precision-bug fix was applied, Island
G -- built under the now-fully-deprecated generation-anchored/current-flow-
betweenness method, no longer reachable by THIS file's own search -- came
back with a whole-Virginia net_musd sum an order of magnitude larger than
Island H or J, this file's own real-storm/fragility-weighted selections.
That is a real instance of the failure mode this section fixes: the old
stopping rule (`if best_candidate_eval["net_score"] <= current_eval["net_
score"]: stop`) only ever looks one step ahead. It cannot tell the
difference between "this branch is a dead end" and "this branch is
temporarily worse but leads somewhere better" -- e.g. members {1} scoring
higher than {1,2}, but {1,2,3} scoring higher than either. Since growth
only ever adds one member and never removes one, the old rule would stop
permanently at {1} and never discover {1,2,3}.

Fix: `_try_bounded_lookahead()`. When the single best growth candidate at
a step would make net_score worse (the old stopping condition), the search
no longer gives up immediately -- it tentatively commits to that candidate
anyway and keeps greedily growing (still picking the single best candidate
at each step, not branching) for up to `lookahead_steps` MORE steps
(`--lookahead-steps`, default LOOKAHEAD_STEPS below), watching whether
net_score ever climbs back past the score it had BEFORE the tentative step
(not just past the immediately-rejected candidate's score). The moment it
does, the whole tentative path up to that recovery point is committed for
real and normal growth resumes from there. If the lookahead budget runs out
without recovering, the entire tentative extension is discarded and the
search stops exactly where the old rule would have -- so this is a strict
relaxation, never worse than the old behavior, only sometimes better.

Cost: each lookahead attempt is up to `lookahead_steps` more real
admittance/GIC solves per boundary candidate at each extra step, but this
path only triggers at points where growth would otherwise have stopped
(typically once or twice per search), not at every step.

Still not exhaustive: only the single greedy-best candidate is explored at
each lookahead step, not every branch (a full branch-and-bound would cost
combinatorially more real solves per rejected step). This catches the
reported (1,2,3) > (1) > (1,2) failure mode but not every possible
non-monotonic path a true joint optimization would find -- see the
LIMITATIONS section below, which this doesn't retire, only narrows.

LIMITATIONS -- HONEST, NOT HIDDEN
------------------------------------
Still a GREEDY search over a SEEDED REGION shape, not the full
optimization-over-the-GIC-circuit the literature's exact methods use
(arXiv:1701.01469's MIQCP, or a DRO formulation over storm-severity
uncertainty like arXiv:2104.00204). Scoring against real storms fixes
the direction-blindness diagnosed in Phase 6c, and bounded lookahead
(Phase 6g) fixes the one-step-stopping blind spot, but the search itself
is still greedy region-growing (single seed, single best candidate per
step/lookahead-step), not a joint optimization over which candidates to
consider. Worst-case-over-three-storms is also not the
same as worst-case-over-all-possible-directions -- gannon/halloween/
st_patricks are the only three real, independently-measured directional
forcings this project has data for, not an exhaustive sample of every
direction a real storm could take. If this still doesn't produce a
consistently positive, direction-robust result once run against real
data, the next honest step is the bigger optimization rung, not another
heuristic tweak.

PHASE 6h: BETWEENNESS-WIDENED SEED POOL -- A DISCOVERY GAP, NOT A SCORING GAP
-------------------------------------------------------------------------------
Real-data motivation: Island G's Halloween/St. Patrick's backfill (Phase 6g)
confirmed it fails all three real storms (0/3) -- so its apparent strength
was a genuine direction-blindness artifact, not evidence the search was
missing a better candidate. But the person running this project reasonably
asked why the real-storm-scored search (which SHOULD generalize better than
G's topology-only proxy) hadn't already found something at least as
economically large as G, just correctly signed. Checked directly against
real data before writing any code: G's own seed substation (96396692) does
NOT appear anywhere in Island K's `seed_candidate_trace` -- the full list of
every seed this search actually evaluated. So the search was never blind to
G's location because real-storm scoring rejected it; it was blind because
G's seed was never even a CANDIDATE.

Why not: the seed pool has always been (going back to this file's very
first, pre-real-storm revision that produced G itself -- see git history,
commit 32eebc8) the top `n_seed_candidates` substations by `risk_weight`
(0.5*GIC risk + 0.5*econ exposure density -- how valuable/exposed a
substation's OWN location is). That is a different axis entirely from what
made G interesting: G's huge redistribution effect comes from being an
electrically PIVOTAL node (current from all over the network routes through
it), not from G's own location being especially GIC-risky or GDP-dense.
current_flow_betweenness_centrality (Newman 2005; Brandes & Fleischer 2005)
is exactly the measure of that pivotalness -- it's what the deprecated
proxy that originally selected G was built on (see PHASE 6d section 1
above) -- but this file never used it to pick CANDIDATES, only (in that
now-fully-retired revision) to isotropically SCORE them.

Fix: `compute_betweenness_seed_candidates()` computes current-flow
betweenness ONCE on the full baseline network (a single one-time cost, not
per-candidate -- contrast with the retired revision's much more expensive
per-cut recomputation) and takes the top `n_betweenness_candidates`
eligible substations by that measure, UNION'd with the existing top-
`n_seed_candidates`-by-`risk_weight` pool (deduplicated). Every seed in
the combined pool -- however it was nominated -- is then evaluated by
the exact same real-storm/fragility-weighted `evaluate_cut()` every other
seed already goes through. Betweenness only WIDENS which substations get a
chance to be judged by real physics; it never touches net_score or the
accept/reject decision itself, deliberately -- putting an isotropic,
direction-blind signal into the actual scoring is the specific mistake
Phase 6d retired G's method for making, and this fix is careful not to
reintroduce it through a side door.

Cost: one current_flow_betweenness_centrality solve on ~2600 substations
(observed cost unknown against this session's real data -- the retired
per-candidate version of this computation was already flagged as
expensive; this is cheaper since it happens once, not once per cut, but
still likely on the order of minutes, not seconds), plus up to
`n_betweenness_candidates` additional real admittance/GIC solves (~20-40s
each) for whichever newly-nominated substations weren't already in the
risk_weight pool.

Honest expectation, and what actually happened: Island L (this fix's real
run) DID nominate substations current_flow_betweenness ranked highly that
weren't already in the risk_weight seed pool, but still came back 0/3 on
real storms (gannon -0.005, halloween -0.037, st_patricks -0.001 net_musd)
-- confirming G's failure mode (and apparently its whole neighborhood's)
is structural, not a Phase 6h-sized discovery gap. That result is what
motivated Phase 6i below.

PHASE 6i: HARD REAL-STORM FILTER -- FIXING A SECOND, DEEPER PROXY GAP
------------------------------------------------------------------------
Real-data motivation: Island L still failing all three real storms despite
a widened seed pool raised a sharper question -- why does a search that
explicitly worst-cases redistribution_cost over real storms keep landing
on real-storm failures at all? Answer, found by re-reading evaluate_cut()
rather than running more searches: `benefit` was NEVER storm-aware.

   benefit = sum(risk_weights.get(n, 0.0) for n in members)   # static, isotropic
   redistribution_cost = worst_case_redistribution_cost(per_storm)  # real, worst-cased
   net_score = benefit - redistribution_cost_weight * redistribution_cost - switching_cost_weight * switching_cost

`fragility_weighted_redistribution()` already computes a real, per-storm
`own_effect` (the members' own fragility-weighted improvement from being
islanded) inside `per_storm` -- but evaluate_cut() only ever pulled
`redistribution_cost_real` out of it for scoring, discarding `own_effect`
in favor of the static `risk_weight` proxy. So net_score was worst-casing
HALF of the real per-storm picture (the cost to the rest of the network)
while the other half (the benefit to the island's own members) stayed a
non-storm-specific topology guess. A candidate whose static risk_weight
overstated its real, direction-specific own-effect could look net-positive
at search time while genuinely failing in the real pipeline -- exactly
what G and L both did.

Fix, two parts:
1. `benefit` is now `-worst_case_total_effect`, where `total_effect[storm]
   = own_effect[storm] + redistribution_cost_real[storm]` (both already
   the same fragility-weighted-expected-GDP-affected units from the same
   call, so summing them needs no cross-unit calibration, unlike mixing
   in risk_weight would) and `worst_case_total_effect = max` over gannon/
   halloween/st_patricks. `net_score = redistribution_cost_weight *
   benefit - switching_cost_weight * switching_cost` -- 100% real,
   100% worst-cased, no static proxy left in it anywhere.
2. A HARD FILTER, not just a reweighted score: `passes_real_storm_filter
   = worst_case_total_effect <= 0.0`. Every selection point -- seed
   choice, growth-step choice, and every step inside bounded lookahead --
   now refuses to add ANY candidate that fails this, regardless of its
   net_score relative to other failing candidates. A failing candidate is
   simply never a member, not merely disfavored. (Exception: if literally
   every evaluated SEED candidate fails, the search still returns the
   least-bad one rather than crashing -- same "select best available, warn
   loudly" convention this file already used for net_score <= 0 -- but
   flags `seed_passed_real_storm_filter: false` in the output so this is
   never silently mistaken for a real pass.)

`risk_weight` keeps a role, just not in scoring: it's still one of the two
candidate-DISCOVERY heuristics (alongside Phase 6h's betweenness) that
decide who gets a real solve spent on them in the first place.

Also raised `fragility_draws` from real_storm_gic.py's CHEAP_FRAGILITY_
DRAWS default (300) to SEARCH_FRAGILITY_DRAWS (2000, `--fragility-draws`)
for this file's own evaluate_cut() calls specifically -- a hard filter is
only trustworthy if the quantity it gates on isn't dominated by Monte
Carlo noise, and 300 draws is noisier than 2000 for the same reason a
small island's effect was the one Phase 6f's precision bug hit first.
Still far cheaper than the real pipeline's 20,000+ draws, and still one
batch (see real_storm_gic.py: must stay under 5000 to keep the early-stop
convergence check inert).

Honest limit: this raises the odds a selected island survives the real
20,000-draw/IO-model pipeline, it cannot guarantee it -- the filter still
runs on an approximation, just a much less noisy one than before. Treat a
candidate that passes Phase 6i as "likely direction-robust, not yet
proven" until the real pipeline confirms it, same posture already taken
for every other in-loop score in this file.

Real result, Island M: 10/12 scenarios net-positive, gannon solidly
positive (the biggest real-storm margin of any candidate so far), but
halloween (-0.025 diff_per_day_musd) and st_patricks (-0.0003) both still
came back slightly negative in the real, full-precision pipeline despite
M passing the Phase 6i filter at search time -- exactly the honest limit
above, observed for real. st_patricks' miss is small enough (three orders
of magnitude below the other real storms' effects) that it may not even
be a stable result -- the fragility Monte Carlo is unseeded (no
torch.manual_seed() anywhere in econ/preprocess/p_gic_files.py), so it
could flip sign on a rerun from pure randomness alone. halloween's miss
is far larger in relative terms and more likely a real, structural
residual. This motivated Phase 6j below.

PHASE 6j: LOCAL-SEARCH POLISH -- GROWTH NEVER RECONSIDERS OR REMOVES
------------------------------------------------------------------------
The one significant piece of algorithm SHAPE still missing after 6g/6h/6i:
growth only ever ADDS a member, one at a time, forward. It never
reconsiders a member already added, and never removes one, no matter how
much better the island might be without it. Bounded lookahead (Phase 6g)
only searches forward through hypothetical future adds; it can't undo a
past one. Island M coming this close (positive on 10/12, including the
largest real-storm margins yet, with only two small negative real-storm
residuals left) is exactly the situation where a small structural
adjustment -- not a bigger search or a stricter filter -- is the more
likely fix.

Fix: `polish_island()`, run once after the growth loop (and any Phase 6g
lookahead recoveries within it) settles. Greedy local search, same
one-move-per-iteration shape as growth itself, up to `--polish-iterations`
(default POLISH_ITERATIONS) rounds:
1. REMOVE moves: try dropping each current member. Only valid if the
   remaining set stays a single connected component (`nx.is_connected`
   on G's induced subgraph) -- an island has to stay one contiguous
   region to mean anything physically; a remove that would split it into
   disconnected fragments is skipped, not evaluated.
2. SWAP moves: for each current member, try removing it and adding a
   different substation adjacent to the reduced set's new boundary (this
   automatically keeps the result connected, since it's exactly the same
   "add a boundary-adjacent node" operation growth already uses, just
   applied after a removal).
3. Every candidate move -- remove or swap -- goes through the identical
   `evaluate_cut()` and Phase 6i hard filter as every seed/growth
   candidate. A move is only accepted if it both passes the filter (when
   `require_real_storm_filter` is True) AND improves `net_score` over the
   current best. Greedily take the single best improving move each
   iteration; stop early if no move improves anything, otherwise stop at
   the iteration cap.

Cost, bluntly: this is the most expensive phase yet. One iteration costs
roughly (island_size remove-attempts + island_size * boundary-fanout
swap-attempts) real admittance/GIC solves -- for a 12-member island with
~15-candidate boundaries, that's on the order of 12 + 12*15 ~ 190 real
solves PER iteration, each ~20-40s. `--polish-iterations` defaults low
(3) deliberately; raise it only with the runtime cost in mind.

Honest expectation: polish can only IMPROVE on (or match) what growth
already found, by construction -- every accepted move strictly improves
net_score and never violates the hard filter, so it can never make the
final island worse. But it's still a local search around one specific
grown island, not a search over all possible islands -- if the fix
Island M needs isn't reachable by a single remove or swap from where
growth left it, polish won't find it either, and the honest next step
would be the bigger optimization rung already named in LIMITATIONS.

Usage:
    python scripts/select_island_by_redistribution_sensitivity.py --island j
    python scripts/select_island_by_redistribution_sensitivity.py --island j --exclude a b c d e f g
    python scripts/select_island_by_redistribution_sensitivity.py --island l --n-betweenness-candidates 10
    python scripts/select_island_by_redistribution_sensitivity.py --island m
    python scripts/select_island_by_redistribution_sensitivity.py --island n --polish-iterations 5
"""

import argparse
import json

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, get_search_scratch_dirs
from scripts.island_graph_utils import (
    load_graph,
    boundary_edges,
    total_boundary_cost,
    sanitize_for_json,
    excluded_nodes_for,
    attach_conductance,
    largest_component_subgraph,
)
from scripts.real_storm_gic import (
    solve_single_iteration,
    ensure_baseline_single_iteration,
    load_fragility_exposure,
    fragility_weighted_redistribution,
    worst_case_redistribution_cost,
)
from scripts.run_virginia_gic import load_gic_context
from scripts.virginia_risk_data import (
    load_topology_metrics,
    load_gic_risk,
    load_economic_exposure,
    normalize_min_max,
    restrict_to_virginia,
)

logger = setup_logger(log_file="logs/select_island_by_redistribution_sensitivity.log")

MAX_ISLAND_SIZE = 15
N_SEED_CANDIDATES = 10
SWITCHING_COST_WEIGHT = 0.015
REDISTRIBUTION_COST_WEIGHT = 1.0
LOOKAHEAD_STEPS = 2
N_BETWEENNESS_CANDIDATES = 10
CFB_SOLVER = "lu"
SEARCH_FRAGILITY_DRAWS = 2000
POLISH_ITERATIONS = 3


def load_risk_weights() -> dict:
    """sub_id -> risk_weight, 0.5*normalized GIC risk + 0.5*normalized
    economic exposure density, restricted to Virginia/main-component
    substations -- see module docstring's RISK WEIGHTS."""
    topology = restrict_to_virginia(load_topology_metrics())
    eligible = topology[topology["in_main_component"]].copy()

    gic = load_gic_risk()
    econ = load_economic_exposure()
    merged = eligible.merge(gic, on="sub_id", how="left").merge(econ, on="sub_id", how="left")
    merged["gic_risk"] = merged["gic_risk"].fillna(0)
    merged["gdp_total"] = merged["gdp_total"].fillna(0)
    merged["population"] = merged["population"].fillna(0)
    valid_area = merged["area_km2"] > 0

    merged["gdp_density"] = 0.0
    merged["population_density"] = 0.0
    merged.loc[valid_area, "gdp_density"] = merged.loc[valid_area, "gdp_total"] / merged.loc[valid_area, "area_km2"]
    merged.loc[valid_area, "population_density"] = merged.loc[valid_area, "population"] / merged.loc[valid_area, "area_km2"]

    merged["gic_risk_norm"] = normalize_min_max(merged["gic_risk"])
    merged["gdp_norm"] = normalize_min_max(merged["gdp_density"])
    merged["population_norm"] = normalize_min_max(merged["population_density"])
    merged["econ_exposure_norm"] = (merged["gdp_norm"] + merged["population_norm"]) / 2
    merged["risk_weight"] = 0.5 * merged["gic_risk_norm"] + 0.5 * merged["econ_exposure_norm"]

    # Explicit float() cast: pandas/numpy float64 values aren't JSON-
    # serializable by the standard library's json module, and every
    # downstream sum over this dict needs to stay plain Python float,
    # not silently promote to numpy and fail json.dump later.
    return {str(k): float(v) for k, v in zip(merged["sub_id"], merged["risk_weight"])}


def node_line_count(G: nx.Graph, node) -> int:
    """Physical line count incident to `node` -- summed line_ids across
    all its graph edges (build_virginia_network_graph.py aggregates
    parallel circuits onto one graph edge, so this is NOT just the
    node's graph degree). A node with node_line_count == 1 is a true
    dead-end: cutting it can't redistribute current anywhere, since
    nothing was routing through it in the first place."""
    return sum(len(data.get("line_ids", [])) for _, _, data in G.edges(node, data=True))


def evaluate_cut(
    G: nx.Graph,
    members: set,
    risk_weights: dict,
    gic_ctx: dict,
    baseline_gnd_df: pd.DataFrame,
    exposure: pd.DataFrame,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    fragility_draws: int = None,
) -> dict:
    """Score cutting `members`'s induced boundary -- see module
    docstring's SCORING and PHASE 6i. Requires one real single-iteration
    admittance/GIC solve (the expensive part -- see scripts/real_storm_
    gic.py's module docstring for the cost tradeoff this implies) plus
    two cheap fragility-model calls per real storm (before/after).

    PHASE 6i: `risk_weights` is no longer used here at all -- kept as a
    parameter only because callers (seed/growth loops) still pass it for
    other purposes (candidate discovery). `benefit` used to be the static
    sum of risk_weight (GIC risk + econ exposure density) over `members`
    -- an isotropic, non-storm-specific proxy. It is now the real,
    worst-case-across-the-three-real-storms whole-Virginia effect,
    sign-flipped so positive still means good: `own_effect` (the real
    fragility-weighted benefit TO members) and `redistribution_cost_real`
    (the real fragility-weighted cost to everyone else) are the same
    units from the same fragility_weighted_redistribution() call, so
    summing them per storm is legitimate (no cross-unit calibration
    needed, unlike the old risk_weight-vs-redistribution_cost split).
    `passes_real_storm_filter` is the Phase 6i hard gate: True only if
    this candidate's worst real storm doesn't make whole-Virginia worse
    off, at this search's own cheap-draw approximation -- callers use it
    to refuse ever adding a candidate that fails it, seed or growth."""
    boundary = boundary_edges(G, members)
    boundary_line_ids = set()
    for u, v in boundary:
        boundary_line_ids.update(str(x) for x in G.edges[u, v].get("line_ids", []))

    scratch_dirs = get_search_scratch_dirs()
    candidate_df = solve_single_iteration(gic_ctx, boundary_line_ids, scratch_dirs["gic_dir"], scratch_dirs["gnd_gic_dir"])
    fragility_kwargs = {} if fragility_draws is None else {"draws": fragility_draws}
    per_storm = fragility_weighted_redistribution(baseline_gnd_df, candidate_df, members, exposure, **fragility_kwargs)
    redistribution_cost = worst_case_redistribution_cost(per_storm)

    total_effect_by_storm = {
        storm: v["own_effect"] + v["redistribution_cost_real"] for storm, v in per_storm.items()
    }
    worst_case_total_effect = max(total_effect_by_storm.values()) if total_effect_by_storm else 0.0
    benefit = -worst_case_total_effect
    switching_cost = total_boundary_cost(G, boundary)
    net_score = redistribution_cost_weight * benefit - switching_cost_weight * switching_cost

    return {
        "boundary": boundary,
        "benefit": benefit,
        "worst_case_total_effect": worst_case_total_effect,
        "total_effect_by_storm": total_effect_by_storm,
        "passes_real_storm_filter": worst_case_total_effect <= 0.0,
        "redistribution_cost": redistribution_cost,
        "per_storm": per_storm,
        "switching_cost": switching_cost,
        "net_score": net_score,
    }


def compute_betweenness_seed_candidates(
    G: nx.Graph, seed_pool: set, n_betweenness_candidates: int, solver: str = CFB_SOLVER,
) -> list:
    """See module docstring's PHASE 6h. Computes current-flow betweenness
    ONCE on the full baseline network (largest connected component -- this
    measure requires a connected graph, see island_graph_utils.
    largest_component_subgraph) and returns the top `n_betweenness_
    candidates` substations within `seed_pool` (already min_lines-filtered
    by the caller) by that measure. This is a discovery aid only -- every
    substation returned still gets evaluated by the exact same real-storm/
    fragility-weighted evaluate_cut() every risk_weight-nominated seed
    goes through, so a high-betweenness-but-direction-wrong candidate
    (like Island G) is still correctly rejectable; this just gives it a
    chance to be judged at all."""
    if n_betweenness_candidates <= 0:
        return []

    logger.info(
        f"Computing current-flow betweenness centrality on the baseline network "
        f"({G.number_of_nodes()} nodes, solver={solver}) to widen the seed candidate pool -- "
        "a one-time cost, not per-candidate (see module docstring's PHASE 6h)..."
    )
    G_baseline = largest_component_subgraph(G)
    cfb = nx.current_flow_betweenness_centrality(G_baseline, weight="conductance", solver=solver)
    eligible_cfb = {n: v for n, v in cfb.items() if n in seed_pool}
    return sorted(eligible_cfb, key=lambda n: eligible_cfb[n], reverse=True)[:n_betweenness_candidates]


def _try_bounded_lookahead(
    G: nx.Graph,
    island: set,
    current_eval: dict,
    first_candidate,
    first_eval: dict,
    risk_weights: dict,
    gic_ctx: dict,
    baseline_gnd_df: pd.DataFrame,
    exposure: pd.DataFrame,
    eligible_nodes: set,
    switching_cost_weight: float,
    redistribution_cost_weight: float,
    max_size: int,
    lookahead_steps: int,
    fragility_draws: int = None,
    require_real_storm_filter: bool = True,
) -> list | None:
    """See module docstring's PHASE 6g. `first_candidate`/`first_eval` is
    the single best growth candidate at this step, already known to score
    WORSE than `current_eval` (the old stopping rule's trigger condition)
    -- and, since the caller only calls this with a candidate that already
    passed the Phase 6i hard real-storm filter, already known to be
    real-storm-eligible itself. Tentatively commits to it anyway and keeps
    greedily growing (single best candidate per step, not branching) for
    up to `lookahead_steps` more steps, checking after every step --
    including this first one -- whether net_score has climbed back past
    `current_eval`'s score (the score BEFORE the tentative step, not just
    past first_eval). Every lookahead step is ALSO restricted to
    real-storm-filter-passing candidates when require_real_storm_filter is
    True -- lookahead is only allowed to explore a path where every single
    member ever added is independently real-storm-eligible, never a
    temporarily-failing state.

    Returns the list of (candidate, eval) pairs to commit, ending at the
    first point recovery happened, or None if the lookahead budget ran
    out without recovering -- in which case the caller discards the whole
    tentative extension and stops exactly where the old one-step rule
    would have."""
    path = [(first_candidate, first_eval)]
    trial_island = set(island) | {first_candidate}
    if first_eval["net_score"] > current_eval["net_score"]:
        return path

    for _ in range(lookahead_steps):
        if len(trial_island) >= max_size:
            break
        boundary = path[-1][1]["boundary"]
        candidates = {v for (u, v) in boundary if u in trial_island} | {u for (u, v) in boundary if v in trial_island}
        candidates = {c for c in candidates if c not in trial_island and c in eligible_nodes}
        if not candidates:
            break

        best_c, best_c_eval = None, None
        for c in candidates:
            ev = evaluate_cut(
                G, trial_island | {c}, risk_weights, gic_ctx, baseline_gnd_df, exposure,
                switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
                fragility_draws=fragility_draws,
            )
            if require_real_storm_filter and not ev["passes_real_storm_filter"]:
                continue
            if best_c_eval is None or ev["net_score"] > best_c_eval["net_score"]:
                best_c, best_c_eval = c, ev

        if best_c is None:
            break

        trial_island.add(best_c)
        path.append((best_c, best_c_eval))
        if best_c_eval["net_score"] > current_eval["net_score"]:
            return path

    return None


def _boundary_candidates_for(G: nx.Graph, members: set, eligible_nodes: set) -> set:
    """Substations adjacent to `members`'s boundary, not already in
    `members` -- the same "who could grow into this set next" computation
    the main growth loop uses, factored out so polish_island() can reuse
    it against a hypothetically-reduced member set."""
    boundary = boundary_edges(G, members)
    candidates = {v for (u, v) in boundary if u in members} | {u for (u, v) in boundary if v in members}
    return {c for c in candidates if c not in members and c in eligible_nodes}


def polish_island(
    G: nx.Graph,
    island: set,
    current_eval: dict,
    risk_weights: dict,
    gic_ctx: dict,
    baseline_gnd_df: pd.DataFrame,
    exposure: pd.DataFrame,
    eligible_nodes: set,
    switching_cost_weight: float,
    redistribution_cost_weight: float,
    fragility_draws: int,
    require_real_storm_filter: bool,
    max_iterations: int,
) -> tuple:
    """PHASE 6j local-search polish -- see module docstring. Growth only
    ever ADDS members, one at a time, forward; this runs AFTER growth (and
    any Phase 6g lookahead) settles, trying single-member REMOVE and SWAP
    moves against the final island, greedily accepting whichever available
    move most improves net_score, same one-move-per-iteration shape as
    growth itself. Every candidate move goes through the identical
    evaluate_cut() and Phase 6i hard filter as every seed/growth candidate
    -- a move is only ever accepted if it passes the filter (when
    require_real_storm_filter is True) AND strictly improves net_score, so
    polish can only match or improve on what growth already found, never
    make it worse.

    Returns (island: set, current_eval: dict, polish_trace: list)."""
    polish_trace = []
    for iteration in range(max_iterations):
        best_move, best_eval, best_desc = None, None, None

        if len(island) > 1:
            for m in island:
                trial_island = island - {m}
                if not nx.is_connected(G.subgraph(trial_island)):
                    continue
                ev = evaluate_cut(
                    G, trial_island, risk_weights, gic_ctx, baseline_gnd_df, exposure,
                    switching_cost_weight=switching_cost_weight,
                    redistribution_cost_weight=redistribution_cost_weight, fragility_draws=fragility_draws,
                )
                if require_real_storm_filter and not ev["passes_real_storm_filter"]:
                    continue
                if ev["net_score"] > current_eval["net_score"] and (best_eval is None or ev["net_score"] > best_eval["net_score"]):
                    best_move, best_eval, best_desc = ("remove", trial_island, m, None), ev, f"remove {m}"

        for m in island:
            reduced = island - {m}
            if not reduced or not nx.is_connected(G.subgraph(reduced)):
                continue
            for c in _boundary_candidates_for(G, reduced, eligible_nodes):
                if c == m:
                    continue
                trial_island = reduced | {c}
                ev = evaluate_cut(
                    G, trial_island, risk_weights, gic_ctx, baseline_gnd_df, exposure,
                    switching_cost_weight=switching_cost_weight,
                    redistribution_cost_weight=redistribution_cost_weight, fragility_draws=fragility_draws,
                )
                if require_real_storm_filter and not ev["passes_real_storm_filter"]:
                    continue
                if ev["net_score"] > current_eval["net_score"] and (best_eval is None or ev["net_score"] > best_eval["net_score"]):
                    best_move, best_eval, best_desc = ("swap", trial_island, m, c), ev, f"swap {m} -> {c}"

        if best_move is None:
            logger.info(
                f"PHASE 6j polish converged after {iteration} accepted move(s): no remaining "
                "remove/swap move improves net_score while passing the real-storm filter."
            )
            break

        _, trial_island, removed, added = best_move
        prev_score = current_eval["net_score"]
        island = trial_island
        current_eval = best_eval
        polish_trace.append({
            "iteration": iteration, "move": best_move[0], "removed": removed, "added": added,
            "reason": f"{best_desc} ({prev_score:.4f} -> {current_eval['net_score']:.4f})",
            "island_size": len(island), **{k: v for k, v in current_eval.items() if k != "boundary"},
        })
        logger.info(
            f"PHASE 6j polish move {iteration}: {best_desc} -- net_score {prev_score:.4f} -> "
            f"{current_eval['net_score']:.4f}, worst_case_total_effect="
            f"{current_eval['worst_case_total_effect']:.6f}."
        )
    else:
        logger.info(f"PHASE 6j polish hit its {max_iterations}-iteration cap without converging naturally.")

    return island, current_eval, polish_trace


def select_island_by_sensitivity(
    G: nx.Graph,
    risk_weights: dict,
    gic_ctx: dict,
    baseline_gnd_df: pd.DataFrame,
    exposure: pd.DataFrame,
    excluded_nodes: set | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    n_seed_candidates: int = N_SEED_CANDIDATES,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    min_lines: int = 1,
    lookahead_steps: int = LOOKAHEAD_STEPS,
    n_betweenness_candidates: int = N_BETWEENNESS_CANDIDATES,
    cfb_solver: str = CFB_SOLVER,
    fragility_draws: int = SEARCH_FRAGILITY_DRAWS,
    require_real_storm_filter: bool = True,
    polish_iterations: int = POLISH_ITERATIONS,
) -> dict:
    """Run the real-storm-scored greedy search once. Returns a dict with
    the same members/boundary_line_ids/seed/growth_trace shape scripts/
    deprecated_sci_islanding/select_virginia_island.py produced, plus the
    net_score/benefit/redistribution_cost/switching_cost/per_storm
    breakdown at each step.

    min_lines restricts the SEED pool only -- see module docstring's
    SEED CANDIDATE POOL. lookahead_steps controls the Phase 6g bounded
    lookahead -- see module docstring's PHASE 6g section. 0 reproduces the
    old strict one-step-stopping behavior exactly. n_betweenness_candidates
    controls the Phase 6h betweenness-widened seed pool -- see module
    docstring's PHASE 6h section. 0 reproduces the pre-Phase-6h behavior
    (risk_weight-only seed pool) exactly. fragility_draws overrides
    evaluate_cut()'s Monte Carlo draw count (see real_storm_gic.py's
    CHEAP_FRAGILITY_DRAWS) -- None uses that module's default.
    require_real_storm_filter is the Phase 6i hard gate -- see module
    docstring's PHASE 6i section. True (default) means a candidate that
    fails on any of the three real storms is NEVER added to the island,
    seed or growth step, even if it's the "best available" by net_score;
    False reproduces pre-Phase-6i behavior (net_score alone decides).
    polish_iterations controls the Phase 6j local-search polish (single-
    member remove/swap moves after growth settles) -- see module
    docstring's PHASE 6j section. 0 disables polish and returns growth's
    result unchanged."""
    excluded_nodes = excluded_nodes or set()
    eligible_nodes = set(risk_weights) - excluded_nodes
    if not eligible_nodes:
        raise ValueError("No eligible substations left to seed from (excluded_nodes covers everything).")

    seed_pool = eligible_nodes
    if min_lines > 1:
        seed_pool = {n for n in eligible_nodes if node_line_count(G, n) >= min_lines}
        if not seed_pool:
            raise ValueError(
                f"min_lines={min_lines}: no eligible substation has that many physical lines. "
                "Lower --min-lines."
            )
        logger.info(
            f"min_lines={min_lines}: {len(seed_pool)}/{len(eligible_nodes)} eligible substations "
            "qualify as seed candidates."
        )

    seed_candidates = sorted(seed_pool, key=lambda n: risk_weights[n], reverse=True)[:n_seed_candidates]
    betweenness_candidates = compute_betweenness_seed_candidates(
        G, seed_pool, n_betweenness_candidates, solver=cfb_solver,
    )
    added_by_betweenness = [c for c in betweenness_candidates if c not in seed_candidates]
    if added_by_betweenness:
        logger.info(
            f"Widened seed pool with {len(added_by_betweenness)} substation(s) from current-flow "
            f"betweenness centrality not already in the top-{n_seed_candidates}-by-risk_weight pool "
            f"(Phase 6h -- see module docstring): {added_by_betweenness}"
        )
        seed_candidates = seed_candidates + added_by_betweenness
    elif n_betweenness_candidates > 0:
        logger.info(
            f"Top-{n_betweenness_candidates} substations by current-flow betweenness were already all "
            f"in the top-{n_seed_candidates}-by-risk_weight pool -- no new seed candidates added."
        )

    logger.info(
        f"Evaluating {len(seed_candidates)} seed candidates (top {n_seed_candidates} by raw risk weight"
        f"{f' + {len(added_by_betweenness)} by current-flow betweenness' if added_by_betweenness else ''}"
        f"{f', min_lines={min_lines}' if min_lines > 1 else ''}) by their real, worst-case-across-"
        "real-storms redistribution effect (one single-iteration admittance/GIC solve each)..."
    )

    seed_candidate_trace = []
    all_seed_evals = []
    for s in seed_candidates:
        ev = evaluate_cut(
            G, {s}, risk_weights, gic_ctx, baseline_gnd_df, exposure,
            switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
            fragility_draws=fragility_draws,
        )
        seed_candidate_trace.append({"candidate": s, **{k: v for k, v in ev.items() if k != "boundary"}})
        all_seed_evals.append((s, ev))

    passing_seed_evals = [(s, ev) for s, ev in all_seed_evals if ev["passes_real_storm_filter"]]
    seed_pool_for_selection = all_seed_evals
    seed_passed_filter = True
    if require_real_storm_filter:
        if passing_seed_evals:
            seed_pool_for_selection = passing_seed_evals
        else:
            seed_passed_filter = False
            logger.warning(
                f"PHASE 6i: NONE of the {len(seed_candidates)} evaluated seed candidates pass the hard "
                "real-storm-positivity filter (worst-case total effect over gannon/halloween/st_patricks "
                "<= 0). Falling back to the best available by net_score anyway, but the resulting island "
                "will NOT be positive across all three real storms even at this search's own cheap-draw "
                "approximation -- a real signal about this candidate pool, not a search bug."
            )

    best_seed, best_eval = max(seed_pool_for_selection, key=lambda pair: pair[1]["net_score"])

    if best_eval["net_score"] <= 0:
        logger.warning(
            f"Best seed candidate ({best_seed})'s net_score is {best_eval['net_score']:.4f} <= 0 -- "
            "even isolating the single best candidate doesn't beat its own worst-case redistribution + "
            "switching cost. Selecting it anyway (still the best available among the evaluated "
            "candidates), but this is a real signal that no single-substation cut in this candidate "
            "pool is worth making."
        )

    island = {best_seed}
    current_eval = best_eval
    trace = [{
        "step": 0, "added": best_seed,
        "reason": "seed (best net_score among top-risk candidates, measured by real worst-case-"
                  "across-real-storms redistribution effect)",
        "island_size": 1, **{k: v for k, v in current_eval.items() if k != "boundary"},
    }]

    hit_cap = False
    stop_reason = None
    rejected_candidate = None
    while len(island) < max_size:
        boundary = current_eval["boundary"]
        candidates = {v for (u, v) in boundary if u in island} | {u for (u, v) in boundary if v in island}
        candidates = {c for c in candidates if c not in island and c in eligible_nodes}
        if not candidates:
            stop_reason = "no_eligible_neighbors"
            logger.info(
                f"Growth stopped at {len(island)} member(s): no eligible neighbors left on the "
                "boundary to consider (not that a candidate was evaluated and rejected)."
            )
            break

        best_candidate, best_candidate_eval = None, None
        n_failed_filter = 0
        for c in candidates:
            trial_eval = evaluate_cut(
                G, island | {c}, risk_weights, gic_ctx, baseline_gnd_df, exposure,
                switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
                fragility_draws=fragility_draws,
            )
            if require_real_storm_filter and not trial_eval["passes_real_storm_filter"]:
                n_failed_filter += 1
                continue
            if best_candidate_eval is None or trial_eval["net_score"] > best_candidate_eval["net_score"]:
                best_candidate, best_candidate_eval = c, trial_eval

        if best_candidate is None:
            stop_reason = "no_candidate_passes_real_storm_filter"
            logger.info(
                f"Growth stopped at {len(island)} member(s): all {len(candidates)} evaluated boundary "
                "candidate(s) failed the Phase 6i hard real-storm-positivity filter (worst-case total "
                "effect over gannon/halloween/st_patricks > 0) -- none are eligible to grow into, "
                "regardless of net_score. The current island is kept as-is (it already passes)."
            )
            break

        if n_failed_filter:
            logger.info(
                f"{n_failed_filter}/{len(candidates)} boundary candidate(s) failed the Phase 6i hard "
                f"real-storm-positivity filter and were excluded from consideration this step "
                f"(best remaining, passing candidate: {best_candidate})."
            )

        if best_candidate_eval["net_score"] <= current_eval["net_score"]:
            lookahead_path = None
            if lookahead_steps > 0:
                lookahead_path = _try_bounded_lookahead(
                    G, island, current_eval, best_candidate, best_candidate_eval,
                    risk_weights, gic_ctx, baseline_gnd_df, exposure, eligible_nodes,
                    switching_cost_weight, redistribution_cost_weight, max_size, lookahead_steps,
                    fragility_draws=fragility_draws, require_real_storm_filter=require_real_storm_filter,
                )

            if lookahead_path is None:
                stop_reason = "no_improving_candidate"
                rejected_candidate = {
                    "candidate": best_candidate,
                    **{k: v for k, v in best_candidate_eval.items() if k != "boundary"},
                }
                logger.info(
                    f"Growth stopped at {len(island)} member(s): the best of {len(candidates)} evaluated "
                    f"candidate(s), {best_candidate}, WAS considered and rejected -- its net_score "
                    f"({best_candidate_eval['net_score']:.4f}) would be worse than the current island's "
                    f"({current_eval['net_score']:.4f}), i.e. adding it makes things worse, not just "
                    "'no better'."
                    + (f" Bounded lookahead ({lookahead_steps} step(s)) did not recover a better state either."
                       if lookahead_steps > 0 else "")
                )
                break

            logger.info(
                f"Growth would have stopped at {len(island)} member(s) ({best_candidate}'s net_score "
                f"{best_candidate_eval['net_score']:.4f} <= current {current_eval['net_score']:.4f}), but "
                f"Phase 6g bounded lookahead recovered a better state {len(lookahead_path)} step(s) ahead: "
                f"committing path {[c for c, _ in lookahead_path]} (final net_score "
                f"{lookahead_path[-1][1]['net_score']:.4f} > {current_eval['net_score']:.4f})."
            )
            for step_candidate, step_eval in lookahead_path:
                island.add(step_candidate)
                prev_score = current_eval["net_score"]
                current_eval = step_eval
                trace.append({
                    "step": len(trace), "added": step_candidate,
                    "reason": f"bounded-lookahead recovery step ({prev_score:.4f} -> {current_eval['net_score']:.4f})",
                    "island_size": len(island), **{k: v for k, v in current_eval.items() if k != "boundary"},
                })
            continue

        island.add(best_candidate)
        prev_score = current_eval["net_score"]
        current_eval = best_candidate_eval
        trace.append({
            "step": len(trace), "added": best_candidate,
            "reason": f"best net_score improvement ({prev_score:.4f} -> {current_eval['net_score']:.4f})",
            "island_size": len(island), **{k: v for k, v in current_eval.items() if k != "boundary"},
        })
    else:
        hit_cap = True
        stop_reason = "max_size_reached"
        logger.warning(
            f"Island growth hit the {max_size}-substation safety cap without net_score naturally "
            "turning over -- worth reviewing risk weights or switching_cost_weight."
        )

    pre_polish_net_score = current_eval["net_score"]
    polish_trace = []
    if polish_iterations > 0:
        logger.info(
            f"Growth complete at {len(island)} member(s), net_score={pre_polish_net_score:.4f}. Starting "
            f"PHASE 6j local-search polish (up to {polish_iterations} iteration(s))..."
        )
        island, current_eval, polish_trace = polish_island(
            G, island, current_eval, risk_weights, gic_ctx, baseline_gnd_df, exposure, eligible_nodes,
            switching_cost_weight, redistribution_cost_weight, fragility_draws, require_real_storm_filter,
            polish_iterations,
        )

    boundary = current_eval["boundary"]
    boundary_line_ids = []
    for u, v in boundary:
        boundary_line_ids.extend(G.edges[u, v].get("line_ids", []))

    return {
        "seed": best_seed,
        "members": sorted(island),
        "member_count": len(island),
        "boundary_line_ids": boundary_line_ids,
        "boundary_line_count": len(boundary_line_ids),
        "boundary_edge_count": len(boundary),
        "voltage_weighted_cost": current_eval["switching_cost"],
        "redistribution_cost": current_eval["redistribution_cost"],
        "per_storm": current_eval["per_storm"],
        "worst_case_total_effect": current_eval["worst_case_total_effect"],
        "worst_case_storm_benefit": current_eval["benefit"],
        "net_score": current_eval["net_score"],
        "pre_polish_net_score": pre_polish_net_score,
        "seed_passed_real_storm_filter": seed_passed_filter,
        "final_island_passes_real_storm_filter": current_eval["passes_real_storm_filter"],
        "hit_size_cap": hit_cap,
        "stop_reason": stop_reason,
        "rejected_candidate": rejected_candidate,
        "seed_candidate_trace": seed_candidate_trace,
        "growth_trace": trace,
        "polish_trace": polish_trace,
    }


def main(
    island_id: str = "j",
    exclude_island_ids: list | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    n_seed_candidates: int = N_SEED_CANDIDATES,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    min_lines: int = 1,
    lookahead_steps: int = LOOKAHEAD_STEPS,
    n_betweenness_candidates: int = N_BETWEENNESS_CANDIDATES,
    cfb_solver: str = CFB_SOLVER,
    fragility_draws: int = SEARCH_FRAGILITY_DRAWS,
    require_real_storm_filter: bool = True,
    polish_iterations: int = POLISH_ITERATIONS,
):
    G = load_graph()
    attach_conductance(G)
    risk_weights = load_risk_weights()
    logger.info(f"{len(risk_weights)} eligible substations with computed risk weights.")

    logger.info("Loading real GIC context (network topology, transformer samples, storm E-field data)...")
    gic_ctx = load_gic_context()
    logger.info("Solving real single-iteration baseline (no cut)...")
    baseline_gnd_df = ensure_baseline_single_iteration(gic_ctx)
    logger.info("Loading Virginia economic exposure data (for fragility-weighted scoring)...")
    exposure = load_fragility_exposure()

    excluded_nodes = None
    if exclude_island_ids:
        excluded_nodes = excluded_nodes_for(G, exclude_island_ids)
        logger.info(
            f"Excluding {len(excluded_nodes)} substations (members + boundary neighbors) "
            f"from prior island(s) {exclude_island_ids}."
        )

    island = select_island_by_sensitivity(
        G, risk_weights, gic_ctx, baseline_gnd_df, exposure, excluded_nodes=excluded_nodes, max_size=max_size,
        n_seed_candidates=n_seed_candidates, switching_cost_weight=switching_cost_weight,
        redistribution_cost_weight=redistribution_cost_weight, min_lines=min_lines,
        lookahead_steps=lookahead_steps, n_betweenness_candidates=n_betweenness_candidates,
        cfb_solver=cfb_solver, fragility_draws=fragility_draws,
        require_real_storm_filter=require_real_storm_filter, polish_iterations=polish_iterations,
    )

    out_path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(island), f, indent=2, allow_nan=False)
    logger.info(f"Saved {out_path}")

    logger.info(
        f"Island {island_id}: seed={island['seed']}, {island['member_count']} members, "
        f"{island['boundary_line_count']} boundary lines cut (voltage-weighted cost "
        f"{island['voltage_weighted_cost']:.2f}). net_score={island['net_score']:.4f} "
        f"(worst_case_storm_benefit={island['worst_case_storm_benefit']:.4f}, "
        f"worst_case_total_effect={island['worst_case_total_effect']:.4f}, "
        f"worst-case redistribution_cost={island['redistribution_cost']:.4f}). "
        f"PHASE 6i: seed passed real-storm filter={island['seed_passed_real_storm_filter']}, "
        f"final island passes real-storm filter={island['final_island_passes_real_storm_filter']}. "
        f"Growth stopped: {island['stop_reason']}"
        + (
            f" (best rejected candidate: {island['rejected_candidate']['candidate']}, would-be "
            f"net_score {island['rejected_candidate']['net_score']:.4f})"
            if island["rejected_candidate"] else ""
        )
    )
    logger.info(f"Members: {island['members']}")
    logger.info(
        "Per-storm fragility-weighted redistribution_cost_real at final island "
        "(expected GDP-affected units, positive = worse): "
        + ", ".join(f"{p}={v['redistribution_cost_real']:+.6f}" for p, v in island["per_storm"].items())
    )
    logger.info(
        "PHASE 6i per-storm TOTAL effect (own_effect + redistribution_cost_real -- the quantity the "
        "hard filter and net_score are actually based on now, positive = worse for whole Virginia): "
        + ", ".join(
            f"{p}={v['own_effect'] + v['redistribution_cost_real']:+.6f}"
            for p, v in island["per_storm"].items()
        )
    )
    logger.info("Growth trace:\n" + "\n".join(
        f"  step {t['step']}: added {t['added']} -- {t['reason']} "
        f"(size={t['island_size']}, benefit={t['benefit']:.4f}, "
        f"redistribution_cost={t['redistribution_cost']:.4f}, net_score={t['net_score']:.4f})"
        for t in island["growth_trace"]
    ))
    if island["polish_trace"]:
        logger.info(
            f"PHASE 6j polish trace ({len(island['polish_trace'])} accepted move(s), "
            f"pre-polish net_score={island['pre_polish_net_score']:.4f} -> "
            f"final net_score={island['net_score']:.4f}):\n" + "\n".join(
                f"  iteration {p['iteration']}: {p['move']} removed={p['removed']} added={p['added']} -- "
                f"{p['reason']} (size={p['island_size']}, net_score={p['net_score']:.4f})"
                for p in island["polish_trace"]
            )
        )
    else:
        logger.info("PHASE 6j polish: no improving move found (or --polish-iterations 0) -- island unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-storm-scored islanding: greedy region growth scored by each candidate's "
                     "actual worst-case-across-real-storms redistribution effect (one single-iteration "
                     "admittance/GIC solve per candidate), not a topology-only proxy. Phase 6d "
                     "replacement for this file's own earlier current_flow_betweenness-based scoring."
    )
    parser.add_argument("--island", type=str, default="j", help="Island id to select and save, e.g. 'j'.")
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=None,
        help="Prior island id(s) whose members + boundary neighbors are ineligible for this island.",
    )
    parser.add_argument("--max-size", type=int, default=MAX_ISLAND_SIZE)
    parser.add_argument(
        "--n-seed-candidates", type=int, default=N_SEED_CANDIDATES,
        help="How many top-risk substations to evaluate as seed candidates (bounds the number of "
             "real single-iteration admittance/GIC solves spent on seed selection -- each one now "
             "costs ~20-40s, not a fast graph computation; see scripts/real_storm_gic.py).",
    )
    parser.add_argument("--switching-cost-weight", type=float, default=SWITCHING_COST_WEIGHT)
    parser.add_argument(
        "--redistribution-cost-weight", type=float, default=REDISTRIBUTION_COST_WEIGHT,
        help="Multiplier on the measured worst-case-across-real-storms redistribution effect in "
             "net_score. Needs real-data recalibration now that redistribution_cost is in raw "
             "risk-weighted |GIC| amps, not current-flow-betweenness units -- see module docstring's "
             "SCORING section.",
    )
    parser.add_argument(
        "--min-lines", type=int, default=1,
        help="Restrict SEED choice (not growth) to substations with at least this many physical "
             "lines incident to them. Default 1 (no restriction). Use --min-lines 2 to rule out "
             "degree-1 dead-end substations as seeds -- they tend to win on cost-effectiveness "
             "(minimal switching cost, ~zero redistribution risk) but can't grow past themselves.",
    )
    parser.add_argument(
        "--lookahead-steps", type=int, default=LOOKAHEAD_STEPS,
        help="Phase 6g bounded lookahead: when the best growth candidate at a step would make "
             "net_score worse, keep tentatively growing up to this many MORE steps before giving up, "
             "committing the whole tentative path if net_score ever recovers past its pre-step value "
             "(catches a temporarily-worse-but-eventually-better growth path the old strict one-step "
             "stopping rule couldn't see). Set to 0 to reproduce the old strict behavior exactly.",
    )
    parser.add_argument(
        "--n-betweenness-candidates", type=int, default=N_BETWEENNESS_CANDIDATES,
        help="Phase 6h: also evaluate this many top substations by current-flow betweenness "
             "centrality as seed candidates, in addition to the top --n-seed-candidates by "
             "risk_weight (deduplicated) -- widens seed DISCOVERY toward electrically pivotal "
             "substations (like Island G's) that risk_weight alone can miss, while still scoring "
             "every one of them with the same real-storm/fragility-weighted evaluate_cut(). Costs "
             "one one-time current-flow-betweenness solve on the full network (minutes, not "
             "seconds) plus one real admittance/GIC solve per newly-nominated substation. Set to 0 "
             "to reproduce the pre-Phase-6h behavior (risk_weight-only seed pool) exactly.",
    )
    parser.add_argument(
        "--cfb-solver", type=str, default=CFB_SOLVER, choices=["full", "lu", "cg"],
        help="networkx current_flow_betweenness_centrality solver (Phase 6h seed widening only). "
             "'lu' (default) is the same choice this file's original, now-retired current-flow-"
             "betweenness-based scoring used.",
    )
    parser.add_argument(
        "--fragility-draws", type=int, default=SEARCH_FRAGILITY_DRAWS,
        help="Phase 6i: Monte Carlo draw count for evaluate_cut()'s per-candidate fragility calls "
             "(real_storm_gic.py's CHEAP_FRAGILITY_DRAWS, default 300, is tuned for cheap in-loop "
             "scoring; the hard real-storm filter below needs less noise to be trustworthy, so this "
             "file's own default is higher). Must stay under 5000 to keep _simulate_scenario's "
             "early-stop convergence check inert (see real_storm_gic.py). Higher = less noise, more "
             "per-candidate wall-clock time.",
    )
    parser.add_argument(
        "--no-real-storm-filter", action="store_true",
        help="Disable the Phase 6i hard filter (reproduces pre-Phase-6i behavior: net_score alone "
             "decides, a candidate that fails a real storm can still be selected if its net_score "
             "is merely the least-bad available). Default: filter ON -- a candidate that makes any "
             "of gannon/halloween/st_patricks worse for whole Virginia is never added, seed or growth.",
    )
    parser.add_argument(
        "--polish-iterations", type=int, default=POLISH_ITERATIONS,
        help="Phase 6j: after growth (and any Phase 6g lookahead) settles, run up to this many "
             "rounds of single-member remove/swap local search, greedily accepting whichever move "
             "most improves net_score while still passing the Phase 6i filter. Growth only ever "
             "adds members forward and never reconsiders one already added -- this is the one "
             "significant algorithm-shape gap that leaves. Expensive: roughly (island_size + "
             "island_size * boundary_fanout) real admittance/GIC solves PER iteration. Set to 0 to "
             "disable and return growth's result unchanged.",
    )
    args = parser.parse_args()
    main(
        island_id=args.island, exclude_island_ids=args.exclude, max_size=args.max_size,
        n_seed_candidates=args.n_seed_candidates, switching_cost_weight=args.switching_cost_weight,
        redistribution_cost_weight=args.redistribution_cost_weight, min_lines=args.min_lines,
        lookahead_steps=args.lookahead_steps, n_betweenness_candidates=args.n_betweenness_candidates,
        cfb_solver=args.cfb_solver, fragility_draws=args.fragility_draws,
        require_real_storm_filter=not args.no_real_storm_filter, polish_iterations=args.polish_iterations,
    )
