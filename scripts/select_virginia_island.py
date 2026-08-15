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

Usage:
    python scripts/select_virginia_island.py
"""

import json
import pickle

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR

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
) -> dict:
    """Run the seeded greedy region-growing algorithm once. Returns a dict
    with seed, members, boundary edges/lines, final value/cost/ratio, and
    a step-by-step growth trace (for the methods section: which substation
    was added at each step and why)."""
    excluded_nodes = excluded_nodes or set()
    eligible_nodes = set(sci_lookup) - excluded_nodes
    if not eligible_nodes:
        raise ValueError("No eligible substations left to seed from (excluded_nodes covers everything).")

    seed = max(eligible_nodes, key=lambda n: sci_lookup[n])
    island = {seed}
    trace = []

    boundary = _boundary_edges(G, island)
    cost = _total_cost(G, boundary)
    value = sci_lookup[seed]
    ratio = value / cost if cost > 0 else float("inf")
    trace.append({
        "step": 0, "added": seed, "reason": "seed (argmax SCI)",
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


def main():
    G = load_graph()
    sci_lookup = load_sci_lookup()
    logger.info(f"{len(sci_lookup)} eligible substations to seed/grow from.")

    island = select_island(G, sci_lookup)

    out_path = VA_NETWORK_DIR / "va_island_a.json"
    with open(out_path, "w") as f:
        # allow_nan=False as a backstop: _sanitize_for_json should already
        # have removed every inf, so this raises loudly instead of writing
        # non-standard JSON if something was missed.
        json.dump(_sanitize_for_json(island), f, indent=2, allow_nan=False)
    logger.info(f"Saved {out_path}")

    logger.info(
        f"Island A: seed={island['seed']}, {island['member_count']} members, "
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
    main()
