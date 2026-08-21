"""
Phase 3 of ISLANDING_MITIGATION_PLAN.md: seeded greedy region-growing to
select a specific, minimal-cut island around the highest-SCI region of
Virginia's grid. Pure graph algorithm -- no dependency on the electrical
solve, so it's testable/debuggable in isolation before Phase 4 touches the
admittance pipeline.

ALGORITHM
---------
1. Seed: argmax(SCI) among eligible substations (in va_substation_
   criticality.csv -- already restricted to Virginia, main-connected-
   component substations by Phase 2).
2. Initialize: island = {seed}; boundary = every line connecting the seed
   to a neighbor.
3. Score: value = sum(SCI over island members), cost = voltage-weighted
   sum over boundary lines (see COST WEIGHTING below), ratio = value/cost.
4. Grow: for every substation adjacent to the current boundary (and
   itself eligible -- see ELIGIBILITY below), compute the ratio the
   island would have if it were added. Add whichever candidate produces
   the best resulting ratio.
5. Stop the moment no candidate improves the ratio, or no eligible
   candidates remain.
6. Safety valve: MAX_ISLAND_SIZE hard-caps growth; hitting it logs a
   warning rather than failing silently, since it's a sign the graph or
   weights need a look, not an expected outcome.

Boundary edges are recomputed from scratch (induced cut of the current
node set) rather than incrementally patched on each candidate evaluation
-- simpler to get right than incremental bookkeeping, and cheap at this
scale (max 15 members, boundary sizes in the single digits to low tens).

LITERATURE GROUNDING
---------------------
Direct algorithmic ancestor: Clauset (2005), "Finding local community
structure in networks," Phys. Rev. E 72, 026132. Clauset's algorithm is
structurally this one -- start from a single seed node, greedily add
whichever neighbor most improves a local ratio metric (his "local
modularity" R = edges-fully-inside / edges-touching-the-boundary), stop
when no neighbor improves it -- specifically so it needs no global network
knowledge, only the current boundary. The deliberate departure here: R is
replaced with a domain-weighted ratio (SCI-weighted node value over real
physical line-cut cost) instead of raw unweighted edge counts. That's the
actual methodological move worth naming explicitly -- adapting a general
local-community-detection template to a risk-weighted, engineering-cost-
aware setting, not inventing greedy region-growing from scratch.

Explicitly NOT mainstream power-systems "controlled islanding" (spectral
clustering / slow-coherency methods optimizing generator dynamic-stability
coherency and minimum power-flow disruption, for real-time response to
cascading contingencies across the WHOLE network). This is a proactive,
single-region, risk-index-driven selection ahead of a slower-onset hazard
(a predicted geomagnetic storm) -- a different problem with different
constraints, not a lighter version of that literature.

GIC-specific precedent for topology reconfiguration as a mitigation
strategy generally: "Optimal Transmission Line Switching under Geomagnetic
Disturbances" (arXiv:1701.01469). Islanding specifically -- as opposed to
general topology reconfiguration -- doesn't have a direct precedent paper
combining it with GIC the way this project does; that's an honest gap,
not a hidden one, and the actual contribution here is a defensible
combination of two separately-established ideas applied to an under-
explored specific use case.

COST WEIGHTING
--------------
Boundary cost is voltage-weighted (kV / 100 per physical line), not a
flat per-line count -- cutting a 500kV line is a materially bigger deal
than cutting a 138kV line, and treating them identically would let the
algorithm cut high-capacity lines as casually as low-capacity ones. This
is itself a modeling assumption (linear in voltage, not capacity/current-
rating-based) worth a sensitivity check later, same spirit as the SCI
weight-sensitivity note in compute_virginia_criticality.py.

Summed per PHYSICAL line, not per graph edge -- build_virginia_network_
graph.py aggregates parallel circuits onto a single edge specifically so
this cost accounting reflects the real number of lines that would need to
be switched, not the number of substation-pairs.

ELIGIBILITY
-----------
Both the seed and every growth candidate must be a key in sci_lookup
(built from va_substation_criticality.csv) -- i.e., inside Virginia, in
the main connected component, with a computed SCI. A substation the graph
connects to but that Phase 2 excluded (non-Virginia, or a different
component) is never added, regardless of how good its ratio would look --
growing outside Virginia doesn't make sense for a Virginia-scoped
mitigation study, and Phase 4/5's downstream econ re-solve is VA-scoped
too. `select_island()` also accepts an `excluded_nodes` set so Phase 6's
second island can exclude Island A's members and boundary, keeping the
two structurally independent (per the plan's second-island design).

GENERATION-ANCHORED SEEDING (require_local_generation)
--------------------------------------------------------
Added after reading the NZ precedent paper (arXiv:2507.12495) in full:
its strongest scenario (Scenario 5) isn't a generic protective cut, it's
an island deliberately anchored on Manapouri, a real hydro generation
site, so the protected load stays served by local generation regardless
of what the wider grid does. Every island this project selected before
this point (A, B, C, D, E) had zero local generation -- islanding them
just isolates load from the grid's supply, it doesn't substitute a local
one, which is a materially weaker form of protection than Scenario 5's.

`select_island()`'s `generation_lookup`/`min_generation_mw` params only
constrain the SEED, not the whole island: eligible_nodes for seeding is
narrowed to substations with >= min_generation_mw of nearby EIA-860
capacity (via scripts/check_island_generation.py's
compute_generation_by_substation()), but growth candidates afterward are
unconstrained, same as before. This mirrors Manapouri's actual role --
it's the anchor the island is built around, not a requirement every
member independently satisfy; requiring every member to have its own
generation would be a much stronger (and for a transmission substation
graph, largely unsatisfiable) constraint that Scenario 5 itself doesn't
impose either.

Usage:
    python scripts/select_virginia_island.py
    python scripts/select_virginia_island.py --island f --require-local-generation --min-generation-mw 50
"""

import argparse
import json
import pickle

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR
import scripts.check_island_generation as gen_check

logger = setup_logger(log_file="logs/select_virginia_island.log")

MAX_ISLAND_SIZE = 15


def _sanitize_for_json(obj):
    """Recursively replace float('inf')/-inf with None. json.dump would
    otherwise emit the bare (non-standard) token Infinity for a ratio of
    inf -- valid to Python's own json.load, but not spec-compliant JSON,
    so a different tool reading va_island_a.json could choke on it. A
    ratio of inf only occurs when cost hits exactly 0 (the island has
    absorbed its entire reachable component, or an edge case) -- None
    communicates "undefined/no boundary cost" without smuggling a
    non-standard numeric token into the file."""
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def load_graph() -> nx.Graph:
    path = VA_NETWORK_DIR / "va_network_graph.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sci_lookup() -> dict:
    path = VA_ECON_DIR / "va_substation_criticality.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/compute_virginia_criticality.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return dict(zip(df["sub_id"], df["SCI"]))


def _boundary_edges(G: nx.Graph, island: set) -> set:
    """Induced cut of island against the rest of the graph -- every edge
    with exactly one endpoint inside island. Recomputed fresh rather than
    incrementally patched; see module docstring for why."""
    boundary = set()
    for u in island:
        for v in G.neighbors(u):
            if v not in island:
                boundary.add((u, v) if u < v else (v, u))
    return boundary


def _edge_cost(G: nx.Graph, u: str, v: str) -> float:
    """Voltage-weighted cost (kV / 100), summed per physical parallel line
    this graph edge represents -- see module docstring's COST WEIGHTING."""
    voltages = G.edges[u, v].get("voltages", [])
    return sum(v_kv / 100 for v_kv in voltages if v_kv is not None)


def _total_cost(G: nx.Graph, boundary: set) -> float:
    return sum(_edge_cost(G, u, v) for u, v in boundary)


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
    a step-by-step growth trace (for the methods section: which substation
    was added at each step and why).

    generation_lookup/min_generation_mw (see module docstring's GENERATION-
    ANCHORED SEEDING) restrict only the SEED choice to substations with
    real nearby EIA-860 generation capacity -- growth afterward is
    unconstrained, same as the base algorithm."""
    excluded_nodes = excluded_nodes or set()
    eligible_nodes = set(sci_lookup) - excluded_nodes
    if not eligible_nodes:
        raise ValueError("No eligible substations left to seed from (excluded_nodes covers everything).")

    seed_candidates = eligible_nodes
    if generation_lookup is not None:
        seed_candidates = {n for n in eligible_nodes if generation_lookup.get(n, 0.0) >= min_generation_mw}
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


def _excluded_nodes_for(G: nx.Graph, island_ids: list) -> set:
    """Phase 6: members + boundary neighbors of each prior island, so a
    second (third, ...) island can't overlap or sit immediately adjacent
    to one already selected -- "excluded_nodes... Island A's members and
    boundary, keeping the two structurally independent" per this module's
    own ELIGIBILITY section above. Boundary neighbors (not just members)
    are excluded because seeding right next to Island A's cut wouldn't be
    a structurally independent second island, just an extension of the
    first one by another name."""
    excluded = set()
    for island_id in island_ids:
        path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- can't exclude Island {island_id}'s footprint.")
        with open(path) as f:
            prior = json.load(f)
        members = set(prior["members"])
        boundary = _boundary_edges(G, members)
        boundary_neighbors = {v if u in members else u for u, v in boundary}
        excluded |= members | boundary_neighbors
    return excluded


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
        # allow_nan=False as a backstop: _sanitize_for_json should already
        # have removed every inf, so this raises loudly instead of writing
        # non-standard JSON if something was missed.
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
