"""
Replacement for scripts/deprecated_sci_islanding/select_virginia_island.py
(see that folder's README.md for why the SCI-driven greedy method was
retired). Same seeded-region-growing SHAPE -- pick a seed, grow it by
adding whichever adjacent substation most improves a score, stop when
nothing improves -- so this stays a drop-in producer of the same
va_island_{id}.json schema (members, boundary_line_ids, seed,
growth_trace) that every downstream Phase 4-6 script already consumes.
What changed is the SCORE: instead of a static, purely local
SCI-weighted value/cost ratio, every candidate is scored by actually
recomputing current-flow betweenness (Newman 2005; Brandes & Fleischer
2005 -- the electrically-faithful analog of betweenness for a resistor
network, same math scripts/deprecated_sci_islanding/check_redistribution_
proxy.py used to diagnose the old method's blind spot) on the REST of
the network after the candidate cut, and comparing it to a one-time
baseline. That's the direct fix for what six real island selections (A-F)
and a literature review both converged on: local greedy criteria can't
see that a cut's benefit at one substation gets paid for -- or
overpaid for -- by increased current importance somewhere else in the
network. See Nagarajan et al., "Heuristic Algorithms for Placing
Geomagnetically Induced Current Blocking Devices" (arXiv:2310.09409) and
"A Review of the GIC Blocker Placement Problem" (arXiv:2402.07302):
"installing a blocking device at one location can exacerbate GIC levels
at others across the power grid," which is exactly why they find
"simplistic strategies like greedy methods... likely inadequate" for
this problem class in general -- this module's whole point is to stop
being blind to that effect, not to abandon a greedy search entirely
(a full MIQCP/DRO formulation over the actual GIC circuit, as in
arXiv:1701.01469 and arXiv:2006.02461, would be the next rung up if this
still isn't enough; see this module's own limitations section below).

SCORING
-------
For a candidate node set `members` (a single seed, or the current
island plus one growth candidate):
1. boundary = induced cut of members (scripts/island_graph_utils.py).
2. Remove boundary edges, restrict to the largest remaining component
   outside `members`, recompute current-flow betweenness there
   (cfb_after).
3. redistribution_cost = mean risk-weighted cfb per remaining substation,
   AFTER the cut, SUMMED (not averaged) over remaining substations,
   minus the same quantity from a one-time BASELINE current-flow
   betweenness computed on the untouched full network (restricted to
   the same remaining node set each time, so it's an apples-to-apples
   comparison, not confounded by which nodes happen to still be
   reachable). Positive means the cut made the rest of the network's
   risk-weighted current importance worse in total; negative means it
   improved.

   Originally used the MEAN, out of a concern that summing over ~2600
   substations would produce a number two to three orders of magnitude
   larger than `benefit` (summed over at most a few dozen protected
   substations), making benefit numerically irrelevant. Real data
   proved that concern backwards: current_flow_betweenness_centrality's
   normalized per-node values are already tiny at real network scale
   (~1e-4 to 1e-7), so the MEAN crushed redistribution_cost down to
   ~1e-5 to 1e-8 -- numerically negligible next to switching_cost and
   benefit (both order 0.1-10), meaning net_score was silently ignoring
   redistribution entirely and degenerating into "pick highest benefit,
   penalize switching cost" -- exactly the local-proxy blindness this
   algorithm exists to fix. Switching to the SUM (first caught against
   a real ~2600-node Virginia run, not a synthetic fixture) restored a
   comparable, meaningful magnitude. redistribution_cost_weight is
   still exposed as a tunable multiplier -- this scale calibration is
   itself a modeling assumption worth further sensitivity-testing
   against real results, not a permanently settled choice.
4. benefit = sum of risk_weight over `members` -- these substations'
   own GIC-risk/economic-exposure no longer needs to route through the
   part of the grid being protected. Same role as the old algorithm's
   SCI-weighted "value", just without a betweenness term baked in
   (betweenness is now measured dynamically via cfb instead).
5. switching_cost = voltage-weighted boundary cost (unchanged from the
   old algorithm -- see scripts/island_graph_utils.py's edge_cost()).
6. net_score = benefit - redistribution_cost - switching_cost_weight *
   switching_cost. Greedily maximized, same stopping rule as before
   (stop the moment no candidate improves it).

RISK WEIGHTS
------------
Two-term only: 0.5*normalized GIC risk + 0.5*normalized economic
exposure density (scripts/virginia_risk_data.py's loaders). Deliberately
NOT the old three/four-term SCI -- no static betweenness term is baked
in here at all, since this algorithm measures redistribution effects
directly via current-flow betweenness at evaluation time instead of
approximating "importance" with a fixed per-node score computed once
upfront.

SEED CANDIDATE POOL
--------------------
Evaluating cfb for every substation in Virginia as a seed candidate
would mean one current-flow-betweenness solve per candidate before
growth even starts -- for a ~2600-node graph, one solve per the existing
check_redistribution_proxy.py's own (untested-at-real-scale) expectation
should be fast, but a few hundred solves just for seed selection is
worth bounding rather than assuming away. n_seed_candidates restricts
seed evaluation to the top-N substations by raw risk_weight (highest GIC
risk + economic exposure) -- the plausible high-value seeds -- rather
than every eligible substation. Growth-step candidates are NOT
similarly restricted beyond the existing boundary-adjacency constraint
(same as the old algorithm), since that pool is naturally small
(a boundary's fan-out, typically single digits to tens).

min_lines filters the SEED pool (not growth) to substations with at
least that many physical lines incident to them (summed line_ids across
all their graph edges) -- added after a real run against Virginia data
kept selecting a genuine degree-1 dead-end substation as seed every
time: real GIC risk, real economic exposure, minimal switching cost,
and (correctly, not a bug) essentially zero redistribution cost, since
nothing routes through a true leaf. That's a legitimate, defensible
result on its own (see ISLANDING_MITIGATION_PLAN.md's Phase 6b), but it
never gives growth a chance to run, since a degree-1 node has at most
one neighbor to grow into. min_lines >= 2 forces seed selection toward
substations that could plausibly anchor a multi-substation region,
without touching growth itself (a growth step CAN still add a
low-degree substation once the island already has real size -- this
only constrains where the search starts).

PERFORMANCE, UNTESTED AT REAL SCALE
-------------------------------------
Same caveat as check_redistribution_proxy.py: current_flow_betweenness_
centrality's cost grows with graph size; this has not been run against
the real ~2600-node VA+PJM graph in this sandbox (no real data
available here). Expect roughly (n_seed_candidates + sum of boundary
fan-out across growth steps) total current-flow-betweenness solves for
one island selection -- report back how long this actually takes on
real data, and lower n_seed_candidates / max_size if it's too slow.

LIMITATIONS -- HONEST, NOT HIDDEN
------------------------------------
This is still a GREEDY search over a SEEDED REGION shape, not the full
optimization-over-the-GIC-circuit the literature's exact methods use
(arXiv:1701.01469's MIQCP, or a DRO formulation over storm-severity
uncertainty like arXiv:2104.00204). It fixes the specific blindness
diagnosed in this project (candidates scored by a local proxy instead
of their real network-wide effect) without rebuilding the whole
selection problem as a formal optimization. If this still doesn't
produce a consistently positive result once run against real data and
pushed through Phase 4/5's econ pipeline, the next honest step is that
bigger rung, not another heuristic tweak.

Usage:
    python scripts/select_island_by_redistribution_sensitivity.py --island g
    python scripts/select_island_by_redistribution_sensitivity.py --island g --exclude a b c d e f
"""

import argparse
import json

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR
from scripts.island_graph_utils import (
    load_graph,
    boundary_edges,
    total_boundary_cost,
    sanitize_for_json,
    excluded_nodes_for,
    attach_conductance,
    largest_component_subgraph,
)
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
SWITCHING_COST_WEIGHT = 0.1
REDISTRIBUTION_COST_WEIGHT = 1.0


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
    cfb_baseline_full: dict,
    solver: str = "lu",
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
) -> dict:
    """Score cutting `members`'s induced boundary -- see module
    docstring's SCORING. Requires one current_flow_betweenness_centrality
    solve (the expensive part)."""
    boundary = boundary_edges(G, members)
    G_cut = G.copy()
    G_cut.remove_edges_from(boundary)
    rest = largest_component_subgraph(G_cut, exclude=members)
    cfb_after = nx.current_flow_betweenness_centrality(rest, weight="conductance", solver=solver)

    rest_nodes = list(rest.nodes())
    n_rest = len(rest_nodes)
    baseline_sum = sum(risk_weights.get(n, 0.0) * cfb_baseline_full.get(n, 0.0) for n in rest_nodes)
    after_sum = sum(risk_weights.get(n, 0.0) * cfb_after.get(n, 0.0) for n in rest_nodes)
    redistribution_cost = after_sum - baseline_sum

    benefit = sum(risk_weights.get(n, 0.0) for n in members)
    switching_cost = total_boundary_cost(G, boundary)
    net_score = benefit - redistribution_cost_weight * redistribution_cost - switching_cost_weight * switching_cost

    return {
        "boundary": boundary,
        "benefit": benefit,
        "redistribution_cost": redistribution_cost,
        "switching_cost": switching_cost,
        "net_score": net_score,
        "rest_node_count": n_rest,
    }


def select_island_by_sensitivity(
    G: nx.Graph,
    risk_weights: dict,
    excluded_nodes: set | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    n_seed_candidates: int = N_SEED_CANDIDATES,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    min_lines: int = 1,
    solver: str = "lu",
) -> dict:
    """Run the redistribution-sensitivity-guided greedy search once.
    Returns a dict with the same members/boundary_line_ids/seed/
    growth_trace shape scripts/deprecated_sci_islanding/select_virginia_
    island.py produced, plus the new net_score/benefit/
    redistribution_cost/switching_cost breakdown at each step.

    min_lines restricts the SEED pool only -- see module docstring's
    SEED CANDIDATE POOL."""
    excluded_nodes = excluded_nodes or set()
    eligible_nodes = set(risk_weights) - excluded_nodes
    if not eligible_nodes:
        raise ValueError("No eligible substations left to seed from (excluded_nodes covers everything).")

    logger.info(f"Computing baseline current-flow betweenness on the full network ({G.number_of_nodes()} nodes)...")
    G_baseline = largest_component_subgraph(G)
    cfb_baseline_full = nx.current_flow_betweenness_centrality(G_baseline, weight="conductance", solver=solver)

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
    logger.info(
        f"Evaluating {len(seed_candidates)} seed candidates (top {n_seed_candidates} by raw risk weight"
        f"{f', min_lines={min_lines}' if min_lines > 1 else ''}) by their actual whole-network "
        "redistribution effect..."
    )

    seed_candidate_trace = []
    best_seed, best_eval = None, None
    for s in seed_candidates:
        ev = evaluate_cut(
            G, {s}, risk_weights, cfb_baseline_full, solver=solver,
            switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
        )
        seed_candidate_trace.append({"candidate": s, **{k: v for k, v in ev.items() if k != "boundary"}})
        if best_eval is None or ev["net_score"] > best_eval["net_score"]:
            best_seed, best_eval = s, ev

    if best_eval["net_score"] <= 0:
        logger.warning(
            f"Best seed candidate ({best_seed})'s net_score is {best_eval['net_score']:.4f} <= 0 -- "
            "even isolating the single best candidate doesn't beat its own redistribution + switching "
            "cost. Selecting it anyway (still the best available among the evaluated candidates), but "
            "this is a real signal that no single-substation cut in this candidate pool is worth making."
        )

    island = {best_seed}
    current_eval = best_eval
    trace = [{
        "step": 0, "added": best_seed,
        "reason": "seed (best net_score among top-risk candidates, measured by actual whole-network redistribution effect)",
        "island_size": 1, **{k: v for k, v in current_eval.items() if k != "boundary"},
    }]

    hit_cap = False
    while len(island) < max_size:
        boundary = current_eval["boundary"]
        candidates = {v for (u, v) in boundary if u in island} | {u for (u, v) in boundary if v in island}
        candidates = {c for c in candidates if c not in island and c in eligible_nodes}
        if not candidates:
            break

        best_candidate, best_candidate_eval = None, None
        for c in candidates:
            trial_eval = evaluate_cut(
                G, island | {c}, risk_weights, cfb_baseline_full, solver=solver,
                switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
            )
            if best_candidate_eval is None or trial_eval["net_score"] > best_candidate_eval["net_score"]:
                best_candidate, best_candidate_eval = c, trial_eval

        if best_candidate_eval["net_score"] <= current_eval["net_score"]:
            break  # no candidate improves net_score -- growth is done

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
        "total_protected_risk_weight": current_eval["benefit"],
        "net_score": current_eval["net_score"],
        "hit_size_cap": hit_cap,
        "seed_candidate_trace": seed_candidate_trace,
        "growth_trace": trace,
    }


def main(
    island_id: str = "g",
    exclude_island_ids: list | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    n_seed_candidates: int = N_SEED_CANDIDATES,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    min_lines: int = 1,
    solver: str = "lu",
):
    G = load_graph()
    attach_conductance(G)
    risk_weights = load_risk_weights()
    logger.info(f"{len(risk_weights)} eligible substations with computed risk weights.")

    excluded_nodes = None
    if exclude_island_ids:
        excluded_nodes = excluded_nodes_for(G, exclude_island_ids)
        logger.info(
            f"Excluding {len(excluded_nodes)} substations (members + boundary neighbors) "
            f"from prior island(s) {exclude_island_ids}."
        )

    island = select_island_by_sensitivity(
        G, risk_weights, excluded_nodes=excluded_nodes, max_size=max_size,
        n_seed_candidates=n_seed_candidates, switching_cost_weight=switching_cost_weight,
        redistribution_cost_weight=redistribution_cost_weight, min_lines=min_lines, solver=solver,
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
        f"redistribution_cost={island['redistribution_cost']:.4f})."
    )
    logger.info(f"Members: {island['members']}")
    logger.info("Growth trace:\n" + "\n".join(
        f"  step {t['step']}: added {t['added']} -- {t['reason']} "
        f"(size={t['island_size']}, benefit={t['benefit']:.4f}, "
        f"redistribution_cost={t['redistribution_cost']:.4f}, net_score={t['net_score']:.4f})"
        for t in island["growth_trace"]
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redistribution-sensitivity-guided islanding: greedy region growth scored by each "
                     "candidate's actual measured whole-network current-flow-betweenness effect, not a "
                     "local proxy. Replacement for scripts/deprecated_sci_islanding/select_virginia_island.py."
    )
    parser.add_argument("--island", type=str, default="g", help="Island id to select and save, e.g. 'g'.")
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=None,
        help="Prior island id(s) whose members + boundary neighbors are ineligible for this island.",
    )
    parser.add_argument("--max-size", type=int, default=MAX_ISLAND_SIZE)
    parser.add_argument(
        "--n-seed-candidates", type=int, default=N_SEED_CANDIDATES,
        help="How many top-risk substations to evaluate as seed candidates (bounds the number of "
             "current-flow-betweenness solves spent on seed selection).",
    )
    parser.add_argument("--switching-cost-weight", type=float, default=SWITCHING_COST_WEIGHT)
    parser.add_argument(
        "--redistribution-cost-weight", type=float, default=REDISTRIBUTION_COST_WEIGHT,
        help="Multiplier on the measured whole-network redistribution effect in net_score. "
             "Default 1.0 (unscaled) -- tune if results seem to over- or under-weight redistribution "
             "relative to benefit/switching_cost.",
    )
    parser.add_argument(
        "--min-lines", type=int, default=1,
        help="Restrict SEED choice (not growth) to substations with at least this many physical "
             "lines incident to them. Default 1 (no restriction). Use --min-lines 2 to rule out "
             "degree-1 dead-end substations as seeds -- they tend to win on cost-effectiveness "
             "(minimal switching cost, ~zero redistribution risk) but can't grow past themselves.",
    )
    parser.add_argument("--solver", type=str, default="lu", choices=["full", "lu", "cg"])
    args = parser.parse_args()
    main(
        island_id=args.island, exclude_island_ids=args.exclude, max_size=args.max_size,
        n_seed_candidates=args.n_seed_candidates, switching_cost_weight=args.switching_cost_weight,
        redistribution_cost_weight=args.redistribution_cost_weight, min_lines=args.min_lines, solver=args.solver,
    )
