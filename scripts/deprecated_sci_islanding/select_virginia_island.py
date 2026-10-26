"""
Seeded greedy region-growing to select a minimal-cut island around the
highest-SCI region of Virginia's grid. Pure graph algorithm, no dependency
on the electrical solve.

ALGORITHM
---------
1. Seed: argmax(SCI) among eligible substations (Virginia, main connected
   component, with a computed SCI).
2. Initialize: island = {seed}; boundary = every line connecting it to a
   neighbor.
3. Score: value = sum(SCI over members), cost = voltage-weighted sum over
   boundary lines, ratio = value/cost.
4. Grow: for every substation adjacent to the boundary, compute the ratio
   the island would have if added, and add whichever candidate gives the
   best resulting ratio.
5. Stop when no candidate improves the ratio, or none remain.
6. MAX_ISLAND_SIZE is a hard safety cap; hitting it logs a warning since
   it means the growth never naturally turned over.

Boundary edges are recomputed from scratch on each candidate rather than
incrementally patched -- simpler, and cheap at this scale.

LITERATURE GROUNDING
---------------------
Based on Clauset (2005), "Finding local community structure in networks,"
Phys. Rev. E 72, 026132 -- same greedy seed-and-grow structure, but with
his unweighted "local modularity" ratio replaced by a domain-weighted one
(SCI value over real line-cut cost).

This is not mainstream power-systems "controlled islanding" (spectral/
slow-coherency methods for real-time cascading-contingency response) --
it's a proactive, risk-index-driven selection ahead of a predicted
geomagnetic storm, a different problem.

COST WEIGHTING
--------------
Boundary cost is voltage-weighted (kV / 100 per physical line) rather than
a flat per-line count, since cutting a 500kV line is a much bigger deal
than cutting a 138kV one. Summed per physical line, not per graph edge
(build_virginia_network_graph.py already aggregates parallel circuits
onto one edge), so this reflects the real number of lines to switch.

ELIGIBILITY
-----------
Both the seed and every growth candidate must have a computed SCI (i.e.
be in Virginia and in the main connected component) -- a substation the
graph connects to but that's outside that scope is never added, no matter
how good its ratio would look. `select_island()` also takes an
`excluded_nodes` set so a second island can avoid overlapping a first
one's members and boundary.

GENERATION-ANCHORED SEEDING (require_local_generation)
--------------------------------------------------------
Inspired by NZ's Scenario 5 (arXiv:2507.12495), which anchors its island
on Manapouri hydro generation so the protected load keeps being served
locally regardless of the wider grid. `generation_lookup`/
`min_generation_mw` only constrain the SEED choice (to substations with
enough nearby EIA-860 capacity) -- growth afterward is unconstrained,
same as the base algorithm. Requiring every member to have its own
generation would be a much stronger constraint than Scenario 5 itself.

Usage:
    python scripts/select_virginia_island.py
    python scripts/select_virginia_island.py --island f --require-local-generation --min-generation-mw 50
"""

import argparse
import json

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR
import scripts.deprecated_sci_islanding.check_island_generation as gen_check
from scripts.island_graph_utils import (
    load_graph,
    boundary_edges as _boundary_edges,
    total_boundary_cost as _total_cost,
    sanitize_for_json as _sanitize_for_json,
    excluded_nodes_for as _excluded_nodes_for,
)

logger = setup_logger(log_file="logs/select_virginia_island.log")

MAX_ISLAND_SIZE = 15


def load_sci_lookup() -> dict:
    path = VA_ECON_DIR / "va_substation_criticality.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/deprecated_sci_islanding/compute_virginia_criticality.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return dict(zip(df["sub_id"], df["SCI"]))


def select_island(
    G: nx.Graph,
    sci_lookup: dict,
    excluded_nodes: set | None = None,
    max_size: int = MAX_ISLAND_SIZE,
    generation_lookup: dict | None = None,
    min_generation_mw: float = 0.0,
) -> dict:
    """Run the seeded greedy region-growing algorithm once. Returns a dict
    with seed, members, boundary edges/lines, final value/cost/ratio, and
    a step-by-step growth trace (which substation was added when and why).

    generation_lookup/min_generation_mw (see module docstring) restrict
    only the SEED choice to substations with nearby EIA-860 generation."""
    excluded_nodes = excluded_nodes or set()
    eligible_nodes = set(sci_lookup) - excluded_nodes
    if not eligible_nodes:
        raise ValueError("No eligible substations left to seed from (excluded_nodes covers everything).")

    seed_candidates = eligible_nodes
    if generation_lookup is not None:
        # Check `n in generation_lookup` explicitly -- unmatched substations
        # are omitted from the dict entirely, not zero-valued, so `.get(n,
        # 0.0) >= min_generation_mw` alone would let them all pass when
        # min_generation_mw is 0.0 (the default).
        seed_candidates = {
            n for n in eligible_nodes
            if n in generation_lookup and generation_lookup[n] >= min_generation_mw
        }
        if not seed_candidates:
            raise ValueError(
                f"require_local_generation: no eligible substation has >= {min_generation_mw} MW "
                "of nearby EIA-860 generation. Lower --min-generation-mw or check that EIA-860 "
                "was downloaded into data/virginia/network/raw_eia860/ (see scripts/"
                "check_island_generation.py's module docstring)."
            )

    seed = max(seed_candidates, key=lambda n: sci_lookup[n])
    island = {seed}
    trace = []

    boundary = _boundary_edges(G, island)
    cost = _total_cost(G, boundary)
    value = sci_lookup[seed]
    ratio = value / cost if cost > 0 else float("inf")
    seed_reason = "seed (argmax SCI)" if generation_lookup is None else (
        f"seed (argmax SCI among substations with >= {min_generation_mw} MW local generation, "
        f"{generation_lookup.get(seed, 0.0):.1f} MW nearby)"
    )
    trace.append({
        "step": 0, "added": seed, "reason": seed_reason,
        "island_size": 1, "value": value, "cost": cost, "ratio": ratio,
    })

    hit_cap = False
    while len(island) < max_size:
        candidates = {v for (u, v) in boundary if u in island} | {u for (u, v) in boundary if v in island}
        candidates = {c for c in candidates if c not in island and c in eligible_nodes}
        if not candidates:
            break  # nothing left this island is allowed to grow into

        best_candidate, best_ratio, best_value, best_cost, best_boundary = None, ratio, None, None, None
        for c in candidates:
            new_island = island | {c}
            new_boundary = _boundary_edges(G, new_island)
            new_value = value + sci_lookup[c]
            new_cost = _total_cost(G, new_boundary)
            new_ratio = new_value / new_cost if new_cost > 0 else float("inf")
            if new_ratio > best_ratio:
                best_candidate, best_ratio = c, new_ratio
                best_value, best_cost, best_boundary = new_value, new_cost, new_boundary

        if best_candidate is None:
            break  # no candidate improves the ratio -- growth is done

        island.add(best_candidate)
        boundary, value, cost, ratio = best_boundary, best_value, best_cost, best_ratio
        trace.append({
            "step": len(trace), "added": best_candidate,
            "reason": f"best ratio improvement ({trace[-1]['ratio']:.4f} -> {ratio:.4f})",
            "island_size": len(island), "value": value, "cost": cost, "ratio": ratio,
        })
    else:
        hit_cap = True
        logger.warning(
            f"Island growth hit the {max_size}-substation safety cap without the "
            "ratio naturally turning over -- worth reviewing SCI/cost weights."
        )

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
        "voltage_weighted_cost": cost,
        "total_value_sci": value,
        "final_ratio": None if cost <= 0 else ratio,
        "hit_size_cap": hit_cap,
        "growth_trace": trace,
    }


def main(
    island_id: str = "a",
    exclude_island_ids: list | None = None,
    require_local_generation: bool = False,
    min_generation_mw: float = 0.0,
):
    G = load_graph()
    sci_lookup = load_sci_lookup()
    logger.info(f"{len(sci_lookup)} eligible substations to seed/grow from.")

    excluded_nodes = None
    if exclude_island_ids:
        excluded_nodes = _excluded_nodes_for(G, exclude_island_ids)
        logger.info(
            f"Excluding {len(excluded_nodes)} substations (members + boundary neighbors) "
            f"from prior island(s) {exclude_island_ids}."
        )

    generation_lookup = None
    if require_local_generation:
        topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
        if not topo_path.exists():
            raise FileNotFoundError(f"Missing {topo_path}. Run scripts/build_virginia_network_graph.py first.")
        topo = pd.read_csv(topo_path)
        topo["sub_id"] = topo["sub_id"].astype(str)
        sub_coords = topo[topo["sub_id"].isin(sci_lookup)][["sub_id", "latitude", "longitude"]]
        generation_lookup = gen_check.compute_generation_by_substation(sub_coords)
        if not generation_lookup:
            raise ValueError(
                "require_local_generation was set but no EIA-860 generation data is available. "
                "Download EIA-860 into data/virginia/network/raw_eia860/ first -- see "
                "scripts/check_island_generation.py's module docstring."
            )
        logger.info(
            f"require_local_generation: {len(generation_lookup)}/{len(sci_lookup)} eligible "
            f"substations have matched EIA-860 generation nearby; seed will be restricted to "
            f"those with >= {min_generation_mw} MW."
        )

    island = select_island(
        G, sci_lookup, excluded_nodes=excluded_nodes,
        generation_lookup=generation_lookup, min_generation_mw=min_generation_mw,
    )

    out_path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    with open(out_path, "w") as f:
        # allow_nan=False as a backstop in case _sanitize_for_json missed an inf
        json.dump(_sanitize_for_json(island), f, indent=2, allow_nan=False)
    logger.info(f"Saved {out_path}")

    logger.info(
        f"Island {island_id}: seed={island['seed']}, {island['member_count']} members, "
        f"{island['boundary_line_count']} boundary lines cut "
        f"(voltage-weighted cost {island['voltage_weighted_cost']:.2f}), "
        f"final ratio {island['final_ratio']}."
    )
    logger.info(f"Members: {island['members']}")
    logger.info("Growth trace:\n" + "\n".join(
        f"  step {t['step']}: added {t['added']} -- {t['reason']} "
        f"(size={t['island_size']}, value={t['value']:.4f}, cost={t['cost']:.4f})"
        for t in island["growth_trace"]
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seeded region-growing islanding algorithm.")
    parser.add_argument("--island", type=str, default="a", help="Island id to select and save, e.g. 'a' or 'b'.")
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=None,
        help="Prior island id(s) whose members + boundary neighbors are ineligible for this "
             "island (Phase 6: e.g. '--island b --exclude a').",
    )
    parser.add_argument(
        "--require-local-generation", action="store_true",
        help="Restrict SEED choice (not growth) to substations with real nearby EIA-860 "
             "generation capacity, anchoring the island the way Manapouri anchors NZ's "
             "Scenario 5 (arXiv:2507.12495). Requires EIA-860 downloaded into "
             "data/virginia/network/raw_eia860/.",
    )
    parser.add_argument(
        "--min-generation-mw", type=float, default=0.0,
        help="Minimum nearby EIA-860 generation capacity (MW) for a substation to be eligible "
             "as SEED when --require-local-generation is set. Default 0.0 (any nonzero match).",
    )
    args = parser.parse_args()
    main(
        island_id=args.island, exclude_island_ids=args.exclude,
        require_local_generation=args.require_local_generation,
        min_generation_mw=args.min_generation_mw,
    )
