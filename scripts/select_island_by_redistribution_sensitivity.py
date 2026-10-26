"""
Replacement for scripts/deprecated_sci_islanding/select_virginia_island.py
(the old SCI-driven greedy method). Same seeded-region-growing shape --
pick a seed, grow it by adding whichever adjacent substation most improves
a score, stop when nothing improves -- producing the same va_island_{id}.json
schema every downstream Phase 4-6 script consumes.

Scoring history (why it looks the way it does): an earlier revision scored
candidates by current_flow_betweenness_centrality, a cheap but
direction-blind topology proxy -- it has no storm in it, just an isotropic
average over all current flow directions. That picked Island G, which
looked great on 9/10 scenarios but failed the real Gannon storm. The fix
was to score every candidate with an actual single-iteration admittance/GIC
solve against the three real historical storms this project has data for
(gannon, halloween, st_patricks), taking the worst case, not the average.
That still wasn't quite enough on its own: raw |GIC| magnitude isn't what
drives $/day impact (the fragility model's failure curve is nonlinear and
threshold-sensitive), so scoring was changed again to use fragility-weighted
expected-GDP-affected via real_storm_gic.fragility_weighted_redistribution(),
which calls the real pipeline's own fragility function at a lower (cheaper)
draw count.

SCORING
-------
For a candidate node set `members` (a seed, or the island plus one growth
candidate): compute the induced boundary cut, run one real single-iteration
admittance/GIC solve with those lines removed (vs. a one-time real
baseline), then score fragility-weighted expected-GDP-affected before/after
for every substation NOT in members, taking the worst case across the three
real storms as `redistribution_cost`. `benefit` is the real worst-case
whole-Virginia effect (own improvement + cost to the rest), sign-flipped so
positive is good. `switching_cost` is the voltage-weighted boundary cost
(island_graph_utils.edge_cost()). `net_score` combines benefit and
switching_cost; SWITCHING_COST_WEIGHT may need recalibrating if the scoring
function's units change again.

RISK WEIGHTS
------------
0.5*normalized GIC risk + 0.5*normalized economic exposure density
(virginia_risk_data.py). No generation term -- it was tried and dropped
(a generation-anchored island did worse than one without it). risk_weight
no longer has any vote in scoring; it's purely a candidate-discovery
heuristic for ranking the seed pool, alongside current-flow betweenness.

SEED CANDIDATE POOL
--------------------
n_seed_candidates caps seed evaluation to the top-N substations by raw
risk_weight, since each seed candidate now costs one real admittance/GIC
solve (~20-40s), not a cheap graph computation. Growth-step candidates are
bounded by boundary adjacency. Lower --n-seed-candidates/--max-size if a
run is too slow.

min_lines filters the seed pool (not growth) to substations with at least
that many physical lines -- added because degree-1 dead-end substations
kept winning as seeds (minimal cost, ~zero redistribution risk) despite
being unable to grow into anything.

BOUNDED LOOKAHEAD (`_try_bounded_lookahead`)
---------------------------------------------
One-step greedy stopping can't tell a dead end from a path that's
temporarily worse but leads somewhere better (e.g. {1} > {1,2} but
{1,2,3} > both). When the best growth candidate would make net_score
worse, the search tentatively commits anyway and keeps growing up to
`lookahead_steps` more steps, watching whether net_score recovers past
its pre-step value. If it does, the tentative path is committed for real;
if the budget runs out first, it's discarded and the search stops where
the old rule would have. Strictly a relaxation -- never worse than the
old behavior. Only explores the single greedy-best candidate at each
lookahead step, not every branch, so it won't catch every possible
non-monotonic path.

BETWEENNESS-WIDENED SEED POOL (`compute_betweenness_seed_candidates`)
------------------------------------------------------------------------
risk_weight ranks a substation's own location; it says nothing about
whether a substation is electrically pivotal (lots of current routes
through it), which is what made Island G look so strong under the old
proxy. current_flow_betweenness_centrality measures that pivotalness.
This computes it once on the baseline network (not per-candidate) and
unions the top `n_betweenness_candidates` into the seed pool so
electrically-pivotal substations get a chance to be judged by the same
real-storm scoring -- it only widens discovery, never touches net_score
or the accept/reject decision itself.

HARD REAL-STORM FILTER
------------------------
`benefit` used to mix a static risk_weight proxy for the island's own
gain with a real worst-cased redistribution cost for everyone else --
half the picture was real, half was a topology guess. Now `benefit` is
fully real: `-worst_case_total_effect`, where total_effect combines the
real own_effect and real redistribution_cost per storm, maxed over the
three storms. `passes_real_storm_filter` is a hard gate (not just a
score term): any candidate with worst_case_total_effect > 0 (minus
`filter_margin`, see below) is never added at any selection point --
seed, growth, lookahead, or polish. If literally every seed candidate
fails, the search still returns the least-bad one rather than crashing,
but flags `seed_passed_real_storm_filter: false`.

fragility_draws defaults higher here (SEARCH_FRAGILITY_DRAWS=2000) than
real_storm_gic.py's cheap default (300), since a hard filter needs less
Monte Carlo noise to be trustworthy. Still far cheaper than the real
pipeline's 20,000+ draws, so passing this filter means "likely
direction-robust," not "proven" -- the real pipeline can still disagree
on a close call.

SAFETY MARGIN (`filter_margin`)
----------------------------------
A candidate at e.g. -0.0005 has no real margin over the search's own
sampling noise, so which side of zero the real pipeline lands on is
close to a coin flip. `filter_margin` requires the worst storm's effect
to clear this far below zero, not just be non-positive, before a
candidate is accepted anywhere. 0.0 reproduces the original behavior.

HARD FRAGMENTATION GATE
--------------------------
A candidate whose cut would strand other substations outside it (it's a
graph cut vertex) can look artificially cheap -- tiny own-outage cost,
tiny switching cost, large benefit from sitting on lots of current paths
-- even though no real utility would disconnect a substation knowing it
cuts off 148 others. `evaluate_cut()` checks this first, cheaply, before
running any solve: candidates that would fragment the rest of the
network (outside `members`' own baseline-connected component) hard-fail
the real-storm filter immediately.

LOCAL-SEARCH POLISH (`polish_island`)
----------------------------------------
Growth only ever adds members, forward, and never reconsiders one already
added. After growth (and any lookahead) settles, this runs up to
`--polish-iterations` rounds of greedy single-member remove/swap moves,
accepting whichever available move most improves net_score while still
passing the hard filter. Can only match or improve on what growth found,
never make it worse. Expensive: roughly (island_size + island_size *
boundary_fanout) real solves per iteration, so the default is low (3).

MIN-MEMBERS RETRY AND SHORTLIST
-----------------------------------
A 1-substation island can be numerically valid but too small to be an
interesting result. When `min_members > 1`, the top
`min_members_seed_attempts` seed candidates are each grown and polished
in net_score order, stopping at the first that reaches `min_members`
(never by relaxing what growth accepts -- only by trying a different
seed). If none reach it, the largest grown island is returned anyway,
flagged `meets_min_members: false`.

The proxy filter doesn't reliably predict a >1-member island's real
economic outcome the way it did for single-substation islands (the proxy
and the real IO model's $/day multiplier effects are different metrics
in uncalibrated units, and they can diverge more as more substations and
sectors are involved). `shortlist_size > 1` keeps the min-members retry
loop exploring until it collects that many qualifying candidates instead
of stopping at the first, so several real candidates can be validated
through the real pipeline instead of just one. The best is saved as
usual; the rest are saved as va_island_{id}_alt{n}.json for independent
validation. This doesn't fix the underlying proxy-vs-real-model gap, it
just gives more chances to land on a candidate where the gap doesn't flip
a sign.

LIMITATIONS -- HONEST, NOT HIDDEN
------------------------------------
Still a greedy search over a seeded-region shape, not a full joint
optimization over the GIC circuit (the literature's exact methods use
e.g. an MIQCP or a DRO formulation over storm-severity uncertainty).
Worst-case-over-three-storms is also not the same as worst-case-over-all
-possible-directions -- gannon/halloween/st_patricks are the only three
real, independently-measured directional forcings this project has data
for. If this still doesn't produce a consistently positive,
direction-robust result against real data, the honest next step is a
bigger optimization approach, not another heuristic tweak.

Usage:
    python scripts/select_island_by_redistribution_sensitivity.py --island j
    python scripts/select_island_by_redistribution_sensitivity.py --island j --exclude a b c d e f g
    python scripts/select_island_by_redistribution_sensitivity.py --island l --n-betweenness-candidates 10
    python scripts/select_island_by_redistribution_sensitivity.py --island m
    python scripts/select_island_by_redistribution_sensitivity.py --island n --polish-iterations 5
    python scripts/select_island_by_redistribution_sensitivity.py --island o --n-seed-candidates 500 --fragility-draws 8000 --filter-margin 0.05
    python scripts/select_island_by_redistribution_sensitivity.py --island p --n-seed-candidates 500 --fragility-draws 8000 --filter-margin 0.05 --min-members 5 --min-members-seed-attempts 15
    python scripts/select_island_by_redistribution_sensitivity.py --island q --n-seed-candidates 500 --fragility-draws 8000 --filter-margin 0.1 --min-members 5 --min-members-seed-attempts 20 --shortlist-size 5
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
MIN_MEMBERS = 1
MIN_MEMBERS_SEED_ATTEMPTS = 10
SHORTLIST_SIZE = 1


def load_risk_weights() -> dict:
    """sub_id -> risk_weight, 0.5*normalized GIC risk + 0.5*normalized
    economic exposure density, restricted to Virginia/main-component
    substations -- see module docstring's RISK WEIGHTS section."""
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

    # float() cast: numpy float64 isn't JSON-serializable by stdlib json
    return {str(k): float(v) for k, v in zip(merged["sub_id"], merged["risk_weight"])}


def node_line_count(G: nx.Graph, node) -> int:
    """Physical line count incident to `node` -- not the same as graph
    degree, since parallel circuits get aggregated onto one graph edge.
    node_line_count == 1 means a true dead-end substation."""
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
    filter_margin: float = 0.0,
) -> dict:
    """Score cutting `members`'s induced boundary -- see module
    docstring's SCORING / hard filter / safety margin sections. Requires
    one real single-iteration admittance/GIC solve (the expensive part)
    plus two cheap fragility-model calls per real storm (before/after).

    `risk_weights` is no longer used for scoring, only kept as a param
    since callers still pass it for candidate discovery.

    Fragmentation is checked first, cheap and disqualifying, before any
    solve: the check is restricted to `members`' own baseline-connected
    component, not the whole graph, since the graph has ~48 pre-existing
    orphan components at baseline unrelated to any candidate -- checking
    against the whole graph would flag every candidate as fragmenting.
    check_island_connectivity() in run_virginia_islanded_gic.py has this
    same restriction and needs it for the same reason."""
    boundary = boundary_edges(G, members)
    boundary_line_ids = set()
    for u, v in boundary:
        boundary_line_ids.update(str(x) for x in G.edges[u, v].get("line_ids", []))

    home_component = nx.node_connected_component(G, next(iter(members)))
    outside_within_home = home_component - members
    outside_components = nx.number_connected_components(G.subgraph(outside_within_home)) if outside_within_home else 1
    if outside_components > 1:
        return {
            "boundary": boundary,
            "benefit": float("-inf"),
            "worst_case_total_effect": float("inf"),
            "total_effect_by_storm": {},
            "passes_real_storm_filter": False,
            "redistribution_cost": float("inf"),
            "per_storm": {},
            "switching_cost": total_boundary_cost(G, boundary),
            "net_score": float("-inf"),
            "fragments_outside_network": True,
            "outside_component_count": outside_components,
        }

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
        "passes_real_storm_filter": worst_case_total_effect <= -filter_margin,
        "redistribution_cost": redistribution_cost,
        "per_storm": per_storm,
        "switching_cost": switching_cost,
        "net_score": net_score,
        "fragments_outside_network": False,
        "outside_component_count": outside_components,
    }


def compute_betweenness_seed_candidates(
    G: nx.Graph, seed_pool: set, n_betweenness_candidates: int, solver: str = CFB_SOLVER,
) -> list:
    """See module docstring's betweenness-widened seed pool section.
    Computes current-flow betweenness once on the baseline network
    (largest connected component -- this measure requires a connected
    graph) and returns the top `n_betweenness_candidates` substations
    within `seed_pool` by that measure. Discovery aid only -- every
    substation returned still gets evaluated by the same real-storm
    evaluate_cut() as any other seed, so a high-betweenness-but-wrong-
    direction candidate is still rejectable; this just gives it a
    chance to be judged."""
    if n_betweenness_candidates <= 0:
        return []

    logger.info(
        f"Computing current-flow betweenness centrality on the baseline network "
        f"({G.number_of_nodes()} nodes, solver={solver}) to widen the seed candidate pool "
        "(one-time cost, not per-candidate)..."
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
    filter_margin: float = 0.0,
) -> list | None:
    """See module docstring's bounded lookahead section. `first_candidate`/
    `first_eval` is the single best growth candidate at this step, already
    known to score worse than `current_eval`. Tentatively commits to it
    and keeps greedily growing for up to `lookahead_steps` more steps,
    checking each step whether net_score climbs back past `current_eval`'s
    score. Every lookahead step is also restricted to real-storm-filter-
    passing candidates when require_real_storm_filter is True.

    Returns the list of (candidate, eval) pairs to commit, up to the
    recovery point, or None if the budget ran out without recovering."""
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
                fragility_draws=fragility_draws, filter_margin=filter_margin,
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
    `members` -- same computation the growth loop uses, factored out so
    polish_island() can reuse it against a reduced member set."""
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
    filter_margin: float = 0.0,
) -> tuple:
    """Local-search polish -- see module docstring. Runs after growth (and
    any lookahead) settles, trying single-member remove/swap moves against
    the final island, greedily accepting whichever move most improves
    net_score. Every move goes through the same evaluate_cut() and hard
    filter as seed/growth candidates, so polish can only match or improve
    on what growth found, never make it worse.

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
                    filter_margin=filter_margin,
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
                    filter_margin=filter_margin,
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


def _grow_and_polish_from_seed(
    G: nx.Graph,
    seed,
    seed_eval: dict,
    risk_weights: dict,
    gic_ctx: dict,
    baseline_gnd_df: pd.DataFrame,
    exposure: pd.DataFrame,
    eligible_nodes: set,
    max_size: int,
    switching_cost_weight: float,
    redistribution_cost_weight: float,
    lookahead_steps: int,
    fragility_draws: int,
    require_real_storm_filter: bool,
    polish_iterations: int,
    filter_margin: float,
) -> dict:
    """Grow one island from a single fixed seed through the growth loop,
    lookahead, and polish. Factored out so the min_members retry loop can
    call this against more than one seed candidate in net_score order,
    stopping at the first whose grown island reaches min_members.

    Returns everything about this one attempt except the search-level
    seed_candidate_trace/seed_passed_real_storm_filter/min_members
    bookkeeping, which the caller adds once it's picked the winner."""
    if seed_eval["net_score"] <= 0:
        logger.warning(
            f"Seed candidate ({seed})'s net_score is {seed_eval['net_score']:.4f} <= 0 -- even "
            "isolating this single candidate doesn't beat its own worst-case redistribution + "
            "switching cost. Growing from it anyway (still among the best available), but this is "
            "a real signal that no single-substation cut in this candidate pool is worth making."
        )

    island = {seed}
    current_eval = seed_eval
    trace = [{
        "step": 0, "added": seed,
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
                fragility_draws=fragility_draws, filter_margin=filter_margin,
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
                    filter_margin=filter_margin,
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
            polish_iterations, filter_margin=filter_margin,
        )

    boundary = current_eval["boundary"]
    boundary_line_ids = []
    for u, v in boundary:
        boundary_line_ids.extend(G.edges[u, v].get("line_ids", []))

    return {
        "seed": seed,
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
        "final_island_passes_real_storm_filter": current_eval["passes_real_storm_filter"],
        "hit_size_cap": hit_cap,
        "stop_reason": stop_reason,
        "rejected_candidate": rejected_candidate,
        "growth_trace": trace,
        "polish_trace": polish_trace,
    }


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
    filter_margin: float = 0.0,
    min_members: int = MIN_MEMBERS,
    min_members_seed_attempts: int = MIN_MEMBERS_SEED_ATTEMPTS,
    shortlist_size: int = SHORTLIST_SIZE,
) -> dict:
    """Run the real-storm-scored greedy search once. Returns a dict with
    the same members/boundary_line_ids/seed/growth_trace shape the old
    select_virginia_island.py produced, plus the net_score/benefit/
    redistribution_cost/switching_cost/per_storm breakdown at each step,
    plus a `candidate_shortlist` of runner-up candidates.

    See the module docstring for what each parameter controls in detail;
    briefly: min_lines restricts the seed pool only. lookahead_steps is
    the bounded-lookahead budget (0 = old strict one-step stopping).
    n_betweenness_candidates widens the seed pool by betweenness (0 =
    risk_weight-only pool). fragility_draws overrides the Monte Carlo
    draw count used for scoring. require_real_storm_filter toggles the
    hard real-storm gate. polish_iterations controls local-search polish
    (0 disables it). filter_margin is the safety margin below zero the
    hard filter requires (0.0 = bare <= 0). min_members/
    min_members_seed_attempts control the min-size retry (min_members=1
    = only the best seed is ever grown). shortlist_size controls how many
    qualifying candidates to collect before stopping (1 = stop at first)."""
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
            fragility_draws=fragility_draws, filter_margin=filter_margin,
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
                f"PHASE 6i/6k: NONE of the {len(seed_candidates)} evaluated seed candidates pass the hard "
                f"real-storm-positivity filter (worst-case total effect over gannon/halloween/st_patricks "
                f"<= {-filter_margin:.6f}). Falling back to the best available by net_score anyway, but "
                "the resulting island will NOT be positive across all three real storms (with the "
                "required margin) even at this search's own cheap-draw approximation -- a real signal "
                "about this candidate pool, not a search bug."
            )

    sorted_by_net_score = sorted(seed_pool_for_selection, key=lambda pair: pair[1]["net_score"], reverse=True)
    if min_members > 1:
        seeds_to_try = sorted_by_net_score[:max(min_members_seed_attempts, 1)]
        logger.info(
            f"PHASE 6m: min_members={min_members} requested -- will grow up to {len(seeds_to_try)} seed "
            f"candidate(s) in net_score order, {'stopping at the first' if shortlist_size <= 1 else f'collecting up to {shortlist_size}'} "
            f"whose final island reaches {min_members} member(s)."
        )
    else:
        seeds_to_try = sorted_by_net_score[:1]

    # All attempts are kept, not just the winner, so shortlist_size > 1
    # can hand back several real candidates to validate instead of one
    # (the proxy score doesn't reliably predict a >1-member island's real
    # outcome -- see module docstring). shortlist_size=1 stops at first.
    all_attempts = []
    qualifying_count = 0
    for attempt_idx, (seed, seed_eval) in enumerate(seeds_to_try):
        result = _grow_and_polish_from_seed(
            G, seed, seed_eval, risk_weights, gic_ctx, baseline_gnd_df, exposure, eligible_nodes,
            max_size, switching_cost_weight, redistribution_cost_weight, lookahead_steps,
            fragility_draws, require_real_storm_filter, polish_iterations, filter_margin,
        )
        all_attempts.append(result)
        if min_members > 1:
            logger.info(
                f"PHASE 6m attempt {attempt_idx + 1}/{len(seeds_to_try)}: seed {seed} grew to "
                f"{result['member_count']} member(s) (min_members={min_members}, "
                f"net_score={result['net_score']:.4f})."
            )
        if result["member_count"] >= min_members:
            qualifying_count += 1
            if qualifying_count >= max(shortlist_size, 1):
                break

    qualifying = [r for r in all_attempts if r["member_count"] >= min_members]
    meets_min_members = len(qualifying) > 0
    ranked = sorted(
        qualifying if qualifying else all_attempts,
        key=lambda r: (r["member_count"], r["net_score"]), reverse=True,
    )
    best_result = ranked[0]

    if min_members > 1 and not meets_min_members:
        logger.warning(
            f"PHASE 6m: none of the {len(all_attempts)} attempted seed candidates grew to at least "
            f"min_members={min_members} (best reached {best_result['member_count']}). Returning the "
            "largest grown island found among those attempted -- try raising "
            "--min-members-seed-attempts to search further, or accept this smaller result. A "
            "min_members requirement can never force growth to add a member that fails the "
            "real-storm filter or hurts net_score, only choose a different seed to grow from."
        )
    elif min_members > 1 and shortlist_size > 1 and len(qualifying) < shortlist_size:
        logger.warning(
            f"PHASE 6n: only {len(qualifying)}/{shortlist_size} requested shortlist candidates were "
            f"found meeting min_members={min_members} after exhausting all {len(all_attempts)} "
            "attempted seeds -- raise --min-members-seed-attempts for a fuller shortlist."
        )

    best_result["seed_passed_real_storm_filter"] = seed_passed_filter
    best_result["seed_candidate_trace"] = seed_candidate_trace
    best_result["min_members_requested"] = min_members
    best_result["meets_min_members"] = best_result["member_count"] >= min_members

    # Lean summaries only (no growth_trace/polish_trace) -- enough for a
    # caller to decide which runner-up to validate through the real pipeline.
    best_result["candidate_shortlist"] = [
        {
            "seed": r["seed"], "members": r["members"], "member_count": r["member_count"],
            "boundary_line_ids": r["boundary_line_ids"], "boundary_line_count": r["boundary_line_count"],
            "net_score": r["net_score"], "worst_case_total_effect": r["worst_case_total_effect"],
            "per_storm": r["per_storm"], "final_island_passes_real_storm_filter": r["final_island_passes_real_storm_filter"],
        }
        for r in ranked[1:shortlist_size]
    ]
    return best_result


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
    filter_margin: float = 0.0,
    min_members: int = MIN_MEMBERS,
    min_members_seed_attempts: int = MIN_MEMBERS_SEED_ATTEMPTS,
    shortlist_size: int = SHORTLIST_SIZE,
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
        filter_margin=filter_margin, min_members=min_members,
        min_members_seed_attempts=min_members_seed_attempts, shortlist_size=shortlist_size,
    )

    candidate_shortlist = island.pop("candidate_shortlist", [])
    out_path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(island), f, indent=2, allow_nan=False)
    logger.info(f"Saved {out_path}")

    if candidate_shortlist:
        for n, alt in enumerate(candidate_shortlist, start=2):
            alt_id = f"{island_id}_alt{n}"
            alt_path = VA_NETWORK_DIR / f"va_island_{alt_id}.json"
            with open(alt_path, "w") as f:
                json.dump(sanitize_for_json(alt), f, indent=2, allow_nan=False)
            logger.info(
                f"PHASE 6n: saved shortlist candidate {n}/{len(candidate_shortlist) + 1} to {alt_path} "
                f"({alt['member_count']} members, net_score={alt['net_score']:.4f}) -- validate it "
                f"through the real pipeline the same way as the primary candidate, e.g. "
                f"`python scripts/run_virginia_islanded_gic.py --island {alt_id}`."
            )

    logger.info(
        f"Island {island_id}: seed={island['seed']}, {island['member_count']} members, "
        f"{island['boundary_line_count']} boundary lines cut (voltage-weighted cost "
        f"{island['voltage_weighted_cost']:.2f}). net_score={island['net_score']:.4f} "
        f"(worst_case_storm_benefit={island['worst_case_storm_benefit']:.4f}, "
        f"worst_case_total_effect={island['worst_case_total_effect']:.4f}, "
        f"worst-case redistribution_cost={island['redistribution_cost']:.4f}). "
        f"PHASE 6i: seed passed real-storm filter={island['seed_passed_real_storm_filter']}, "
        f"final island passes real-storm filter={island['final_island_passes_real_storm_filter']}. "
        f"PHASE 6m: min_members={island['min_members_requested']}, met={island['meets_min_members']}. "
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
    parser.add_argument(
        "--filter-margin", type=float, default=0.0,
        help="Phase 6k: require the worst real storm's total effect to clear this much below zero, "
             "not just be non-positive, before a candidate is ever accepted (seed, growth, lookahead, "
             "or polish). Islands H/J/K/L/M/N all passed the plain <= 0 filter at search time yet "
             "still landed with a real storm slightly negative in the full validating pipeline -- a "
             "candidate at e.g. -0.0005 has no real margin over the search's own Monte Carlo sampling "
             "noise. Try something on the order of the smallest real per-storm effect you've seen "
             "survive to the full pipeline as a starting point (recalibrate from real data, same as "
             "every other weight in this file). 0.0 (default) reproduces pre-Phase-6k behavior exactly.",
    )
    parser.add_argument(
        "--min-members", type=int, default=MIN_MEMBERS,
        help="Phase 6m: require the final island to have at least this many members. When > 1, the "
             "top --min-members-seed-attempts seed candidates (by net_score) are each grown and "
             "polished in full, in order, stopping at the first that reaches this size -- never by "
             "relaxing what growth is allowed to accept, only by trying a different seed. If none of "
             "the attempted seeds reach it, the largest one actually grown is returned anyway (see "
             "the saved JSON's meets_min_members field). 1 (default) reproduces pre-Phase-6m behavior "
             "exactly (only the single best-net_score seed is ever grown).",
    )
    parser.add_argument(
        "--min-members-seed-attempts", type=int, default=MIN_MEMBERS_SEED_ATTEMPTS,
        help="Phase 6m: how many top-net_score seed candidates to try growing (in order) when "
             "--min-members > 1. Each attempt is a FULL growth run, not a cheap check -- this "
             "multiplies search cost roughly linearly. Ignored when --min-members <= 1.",
    )
    parser.add_argument(
        "--shortlist-size", type=int, default=SHORTLIST_SIZE,
        help="Phase 6n: when --min-members > 1, keep exploring (up to --min-members-seed-attempts) "
             "until this many candidates meeting --min-members are found, instead of stopping at the "
             "first. The best is saved as usual (va_island_<id>.json); the rest are saved as "
             "va_island_<id>_alt2.json, _alt3.json, etc. for independent real-pipeline validation -- "
             "useful because the search's own filter doesn't reliably predict a >1-member island's "
             "real economic outcome (see module docstring's PHASE 6n section). 1 (default) reproduces "
             "pre-Phase-6n behavior exactly.",
    )
    args = parser.parse_args()
    main(
        island_id=args.island, exclude_island_ids=args.exclude, max_size=args.max_size,
        n_seed_candidates=args.n_seed_candidates, switching_cost_weight=args.switching_cost_weight,
        redistribution_cost_weight=args.redistribution_cost_weight, min_lines=args.min_lines,
        lookahead_steps=args.lookahead_steps, n_betweenness_candidates=args.n_betweenness_candidates,
        cfb_solver=args.cfb_solver, fragility_draws=args.fragility_draws,
        require_real_storm_filter=not args.no_real_storm_filter, polish_iterations=args.polish_iterations,
        filter_margin=args.filter_margin, min_members=args.min_members,
        min_members_seed_attempts=args.min_members_seed_attempts, shortlist_size=args.shortlist_size,
    )
