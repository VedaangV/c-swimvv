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
2. Current fix: `evaluate_cut()` now scores every candidate by an actual
   single-iteration real admittance/GIC solve (scripts/real_storm_gic.py,
   reusing the exact machinery Phase 6c already proved out) against the
   three genuinely independent real historical storms this pipeline has
   data for -- gannon, halloween, st_patricks -- and takes the WORST
   case across them, not the average. A candidate only scores well if it
   doesn't hurt badly under ANY of the three real directions. This
   directly targets what was asked of this method: "a method that is
   positive for all cases," not just on average across scenarios that
   mostly share one storm's directional signature (see
   VIRGINIA_ADAPTATION.md Section 5.5-5.6 for the full diagnosis).

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
   risk-weighted change in |GIC| summed over every substation NOT in
   `members` (scripts.real_storm_gic.worst_case_redistribution_cost).
   |GIC|, not signed GIC, because magnitude is what drives transformer
   fragility in this project's econ model, not direction. Positive means
   the cut made the rest of the network's risk-weighted current
   magnitude worse, under the worst of the three real directions;
   negative means every one of the three real storms improved it.
4. benefit = sum of risk_weight over `members` -- these substations'
   own GIC-risk/economic-exposure no longer needs to route through the
   part of the grid being protected.
5. switching_cost = voltage-weighted boundary cost (unchanged --
   see scripts/island_graph_utils.py's edge_cost()).

   SWITCHING_COST_WEIGHT (0.015) and REDISTRIBUTION_COST_WEIGHT (1.0)
   were calibrated for the OLD current_flow_betweenness-scale
   redistribution_cost (~0.001-0.009 typical magnitude). real_storm_gic's
   redistribution_cost lives in raw risk-weighted |GIC| amps instead --
   a completely different scale (Phase 6c's real Island G check showed
   per-storm magnitudes in the 1-30 range). Both weights need a THIRD
   real-data calibration pass, same pattern as the two calibration fixes
   already made to this file (SUM-vs-MEAN, then the 0.1-vs-0.015
   switching-cost-weight fix) -- expect this, don't be alarmed by it: it
   is not a modeling error, it is the same "check real output magnitudes,
   recalibrate" step this file has needed every time its scoring
   function's underlying units changed.
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

LIMITATIONS -- HONEST, NOT HIDDEN
------------------------------------
Still a GREEDY search over a SEEDED REGION shape, not the full
optimization-over-the-GIC-circuit the literature's exact methods use
(arXiv:1701.01469's MIQCP, or a DRO formulation over storm-severity
uncertainty like arXiv:2104.00204). Scoring against real storms fixes
the direction-blindness diagnosed in Phase 6c, but the search itself is
still greedy region-growing, not a joint optimization over which
candidates to consider. Worst-case-over-three-storms is also not the
same as worst-case-over-all-possible-directions -- gannon/halloween/
st_patricks are the only three real, independently-measured directional
forcings this project has data for, not an exhaustive sample of every
direction a real storm could take. If this still doesn't produce a
consistently positive, direction-robust result once run against real
data, the next honest step is the bigger optimization rung, not another
heuristic tweak.

Usage:
    python scripts/select_island_by_redistribution_sensitivity.py --island h
    python scripts/select_island_by_redistribution_sensitivity.py --island h --exclude a b c d e f g
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
)
from scripts.real_storm_gic import (
    solve_single_iteration,
    ensure_baseline_single_iteration,
    real_storm_redistribution,
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
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
) -> dict:
    """Score cutting `members`'s induced boundary -- see module
    docstring's SCORING. Requires one real single-iteration admittance/
    GIC solve (the expensive part -- see scripts/real_storm_gic.py's
    module docstring for the cost tradeoff this implies)."""
    boundary = boundary_edges(G, members)
    boundary_line_ids = set()
    for u, v in boundary:
        boundary_line_ids.update(str(x) for x in G.edges[u, v].get("line_ids", []))

    scratch_dirs = get_search_scratch_dirs()
    candidate_df = solve_single_iteration(gic_ctx, boundary_line_ids, scratch_dirs["gic_dir"], scratch_dirs["gnd_gic_dir"])
    per_storm = real_storm_redistribution(baseline_gnd_df, candidate_df, members, risk_weights)
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


def select_island_by_sensitivity(
    G: nx.Graph,
    risk_weights: dict,
    gic_ctx: dict,
    baseline_gnd_df: pd.DataFrame,
    excluded_nodes: set | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    n_seed_candidates: int = N_SEED_CANDIDATES,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    min_lines: int = 1,
) -> dict:
    """Run the real-storm-scored greedy search once. Returns a dict with
    the same members/boundary_line_ids/seed/growth_trace shape scripts/
    deprecated_sci_islanding/select_virginia_island.py produced, plus the
    net_score/benefit/redistribution_cost/switching_cost/per_storm
    breakdown at each step.

    min_lines restricts the SEED pool only -- see module docstring's
    SEED CANDIDATE POOL."""
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
    logger.info(
        f"Evaluating {len(seed_candidates)} seed candidates (top {n_seed_candidates} by raw risk weight"
        f"{f', min_lines={min_lines}' if min_lines > 1 else ''}) by their real, worst-case-across-"
        "real-storms redistribution effect (one single-iteration admittance/GIC solve each)..."
    )

    seed_candidate_trace = []
    best_seed, best_eval = None, None
    for s in seed_candidates:
        ev = evaluate_cut(
            G, {s}, risk_weights, gic_ctx, baseline_gnd_df,
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
                G, island | {c}, risk_weights, gic_ctx, baseline_gnd_df,
                switching_cost_weight=switching_cost_weight, redistribution_cost_weight=redistribution_cost_weight,
            )
            if best_candidate_eval is None or trial_eval["net_score"] > best_candidate_eval["net_score"]:
                best_candidate, best_candidate_eval = c, trial_eval

        if best_candidate_eval["net_score"] <= current_eval["net_score"]:
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
            )
            break

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
    island_id: str = "h",
    exclude_island_ids: list | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    n_seed_candidates: int = N_SEED_CANDIDATES,
    switching_cost_weight: float = SWITCHING_COST_WEIGHT,
    redistribution_cost_weight: float = REDISTRIBUTION_COST_WEIGHT,
    min_lines: int = 1,
):
    G = load_graph()
    attach_conductance(G)
    risk_weights = load_risk_weights()
    logger.info(f"{len(risk_weights)} eligible substations with computed risk weights.")

    logger.info("Loading real GIC context (network topology, transformer samples, storm E-field data)...")
    gic_ctx = load_gic_context()
    logger.info("Solving real single-iteration baseline (no cut)...")
    baseline_gnd_df = ensure_baseline_single_iteration(gic_ctx)

    excluded_nodes = None
    if exclude_island_ids:
        excluded_nodes = excluded_nodes_for(G, exclude_island_ids)
        logger.info(
            f"Excluding {len(excluded_nodes)} substations (members + boundary neighbors) "
            f"from prior island(s) {exclude_island_ids}."
        )

    island = select_island_by_sensitivity(
        G, risk_weights, gic_ctx, baseline_gnd_df, excluded_nodes=excluded_nodes, max_size=max_size,
        n_seed_candidates=n_seed_candidates, switching_cost_weight=switching_cost_weight,
        redistribution_cost_weight=redistribution_cost_weight, min_lines=min_lines,
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
        "Per-storm redistribution_cost_real at final island (positive = worse): "
        + ", ".join(f"{p}={v['redistribution_cost_real']:+.4f}" for p, v in island["per_storm"].items())
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
    parser.add_argument("--island", type=str, default="h", help="Island id to select and save, e.g. 'h'.")
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
    args = parser.parse_args()
    main(
        island_id=args.island, exclude_island_ids=args.exclude, max_size=args.max_size,
        n_seed_candidates=args.n_seed_candidates, switching_cost_weight=args.switching_cost_weight,
        redistribution_cost_weight=args.redistribution_cost_weight, min_lines=args.min_lines,
    )
