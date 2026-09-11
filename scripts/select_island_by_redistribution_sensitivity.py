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
4. benefit = sum of risk_weight over `members` -- these substations'
   own GIC-risk/economic-exposure no longer needs to route through the
   part of the grid being protected. Still the static risk_weight (GIC
   risk + economic exposure density), NOT fragility-weighted -- only
   redistribution_cost changed; benefit is cheap to keep as a simple
   proxy since it's summed over a handful of members, not the whole rest
   of the network.
5. switching_cost = voltage-weighted boundary cost (unchanged --
   see scripts/island_graph_utils.py's edge_cost()).

   SWITCHING_COST_WEIGHT (0.015) and REDISTRIBUTION_COST_WEIGHT (1.0)
   were calibrated for the raw-|GIC| version's scale (per-storm
   magnitudes in the 1-30 range). Fragility-weighted redistribution_cost
   lives in expected-GDP-affected units instead -- a probability
   (typically << 1) times a GDP exposure value, likely a very different
   (probably much smaller) magnitude. This is a FOURTH real-data
   calibration pass in this file's history (after SUM-vs-MEAN, the
   0.1-vs-0.015 switching-cost-weight fix, and the raw-|GIC| rescaling
   itself) -- expect it, don't treat it as a sign of an error: it is the
   same "check real output magnitudes, recalibrate" step this file has
   needed every time its scoring function's underlying units changed.
6. net_score = benefit - redistribution_cost_weight * redistribution_cost
   - switching_cost_weight * switching_cost. Greedily maximized, same
   stopping rule as before (stop the moment no candidate improves it).

RISK WEIGHTS
------------
Two-term only: 0.5*normalized GIC risk + 0.5*normalized economic
exposure density (scripts/virginia_risk_data.py's loaders). No
generation term anywhere -- see scripts/deprecated_sci_islanding/
README.md's "Generation-anchoring is retired too" for why (Island G,
selected with zero generation requirement, beat generation-anchored
Island F's 6/10 with 9/10).

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

Honest expectation: this may still come back empty-handed. It's entirely
possible G's specific location is simply a bad place to cut under real
storm directions no matter how it's approached -- in which case this run
confirms G's failure is structural (already correctly avoidable, not
uncovered by a search gap) rather than finding a new winner. Both outcomes
are informative.

Usage:
    python scripts/select_island_by_redistribution_sensitivity.py --island j
    python scripts/select_island_by_redistribution_sensitivity.py --island j --exclude a b c d e f g
    python scripts/select_island_by_redistribution_sensitivity.py --island l --n-betweenness-candidates 10
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
) -> dict:
    """Score cutting `members`'s induced boundary -- see module
    docstring's SCORING. Requires one real single-iteration admittance/
    GIC solve (the expensive part -- see scripts/real_storm_gic.py's
    module docstring for the cost tradeoff this implies) plus two cheap
    fragility-model calls per real storm (before/after)."""
    boundary = boundary_edges(G, members)
    boundary_line_ids = set()
    for u, v in boundary:
        boundary_line_ids.update(str(x) for x in G.edges[u, v].get("line_ids", []))

    scratch_dirs = get_search_scratch_dirs()
    candidate_df = solve_single_iteration(gic_ctx, boundary_line_ids, scratch_dirs["gic_dir"], scratch_dirs["gnd_gic_dir"])
    per_storm = fragility_weighted_redistribution(baseline_gnd_df, candidate_df, members, exposure)
    redistribution_cost = worst_case_redistribution_cost(per_storm)

    benefit = sum(risk_weights.get(n, 0.0) for n in members)
    switching_cost = total_boundary_cost(G, boundary)
    net_score = benefit - redistribution_cost_weight * redistribution_cost - switching_cost_weight * switching_cost

    return {
        "boundary": boundary,
        "benefit": benefit,
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
) -> list | None:
    """See module docstring's PHASE 6g. `first_candidate`/`first_eval` is
    the single best growth candidate at this step, already known to score
    WORSE than `current_eval` (the old stopping rule's trigger condition).
    Tentatively commits to it anyway and keeps greedily growing (single
    best candidate per step, not branching) for up to `lookahead_steps`
    more steps, checking after every step -- including this first one --
    whether net_score has climbed back past `current_eval`'s score (the
    score BEFORE the tentative step, not just past first_eval).

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
            )
            if best_c_eval is None or ev["net_score"] > best_c_eval["net_score"]:
                best_c, best_c_eval = c, ev

        trial_island.add(best_c)
        path.append((best_c, best_c_eval))
        if best_c_eval["net_score"] > current_eval["net_score"]:
            return path

    return None


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
    (risk_weight-only seed pool) exactly."""
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
    best_seed, best_eval = None, None
    for s in seed_candidates:
        ev = evaluate_cut(
            G, {s}, risk_weights, gic_ctx, baseline_gnd_df, exposure,
            switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
        )
        seed_candidate_trace.append({"candidate": s, **{k: v for k, v in ev.items() if k != "boundary"}})
        if best_eval is None or ev["net_score"] > best_eval["net_score"]:
            best_seed, best_eval = s, ev

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
        for c in candidates:
            trial_eval = evaluate_cut(
                G, island | {c}, risk_weights, gic_ctx, baseline_gnd_df, exposure,
                switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
            )
            if best_candidate_eval is None or trial_eval["net_score"] > best_candidate_eval["net_score"]:
                best_candidate, best_candidate_eval = c, trial_eval

        if best_candidate_eval["net_score"] <= current_eval["net_score"]:
            lookahead_path = None
            if lookahead_steps > 0:
                lookahead_path = _try_bounded_lookahead(
                    G, island, current_eval, best_candidate, best_candidate_eval,
                    risk_weights, gic_ctx, baseline_gnd_df, exposure, eligible_nodes,
                    switching_cost_weight, redistribution_cost_weight, max_size, lookahead_steps,
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
        "total_protected_risk_weight": current_eval["benefit"],
        "net_score": current_eval["net_score"],
        "hit_size_cap": hit_cap,
        "stop_reason": stop_reason,
        "rejected_candidate": rejected_candidate,
        "seed_candidate_trace": seed_candidate_trace,
        "growth_trace": trace,
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
        cfb_solver=cfb_solver,
    )

    out_path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(island), f, indent=2, allow_nan=False)
    logger.info(f"Saved {out_path}")

    logger.info(
        f"Island {island_id}: seed={island['seed']}, {island['member_count']} members, "
        f"{island['boundary_line_count']} boundary lines cut (voltage-weighted cost "
        f"{island['voltage_weighted_cost']:.2f}). net_score={island['net_score']:.4f} "
        f"(benefit={island['total_protected_risk_weight']:.4f}, "
        f"worst-case redistribution_cost={island['redistribution_cost']:.4f}). "
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
    logger.info("Growth trace:\n" + "\n".join(
        f"  step {t['step']}: added {t['added']} -- {t['reason']} "
        f"(size={t['island_size']}, benefit={t['benefit']:.4f}, "
        f"redistribution_cost={t['redistribution_cost']:.4f}, net_score={t['net_score']:.4f})"
        for t in island["growth_trace"]
    ))


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
    args = parser.parse_args()
    main(
        island_id=args.island, exclude_island_ids=args.exclude, max_size=args.max_size,
        n_seed_candidates=args.n_seed_candidates, switching_cost_weight=args.switching_cost_weight,
        redistribution_cost_weight=args.redistribution_cost_weight, min_lines=args.min_lines,
        lookahead_steps=args.lookahead_steps, n_betweenness_candidates=args.n_betweenness_candidates,
        cfb_solver=args.cfb_solver,
    )
