"""
Cheap, graph-only proxy for GIC redistribution risk -- no Monte Carlo, no
stochastic transformer/grounding sampling, just a deterministic linear
solve on the network's real electrical structure. Prompted by Phase 6's
finding that SCI's existing betweenness term (Freeman 1977's shortest-path
betweenness -- the "traffic takes the single cheapest route" model) can't
see this coming, because GIC doesn't route that way: it's Ohm's-law
current splitting across every parallel path simultaneously, in inverse
proportion to resistance. See ISLANDING_MITIGATION_PLAN.md's Phase 6
section for the full discussion.

Uses current-flow betweenness centrality instead (Newman, M.E.J. (2005),
"A measure of betweenness centrality based on random walks," Social
Networks 27(1), 39-54 -- derives it via random walks weighted by edge
conductance; Brandes, U. & Fleischer, D. (2005), "Centrality Measures
Based on Current Flow," STACS 2005 -- proves the random-walk formulation
is identical to literally modeling the graph as a resistor network and
computing current flow directly, and gives the efficient graph-Laplacian-
based computation networkx's current_flow_betweenness_centrality()
implements). This is the electrically-faithful analog of betweenness for
a network where things actually split across parallel paths rather than
picking one shortest route -- the same Laplacian/nodal-admittance math
scripts/build_admittance_matrix.py's real GIC solve uses, just without
the stochastic sampling layered on top for the real Monte Carlo.

METHOD
------
1. Attach real conductance (1/R, summed over each graph edge's parallel
   physical lines) to va_network_graph.pkl's edges, from the same
   transmission_lines.csv R values scripts/build_admittance_matrix.py's
   network_admittance()/get_injection_currents() actually use -- not the
   voltage-based cost proxy select_virginia_island.py uses for a different
   purpose (weighting cut cost, not modeling current flow).
2. Compute current-flow betweenness on the full baseline graph (all paths
   available, including through the candidate island).
3. Remove the candidate island's boundary edges, restrict to the largest
   remaining non-island component (current-flow betweenness requires a
   connected graph -- the island becomes its own separate component once
   every boundary edge is cut, by construction), and recompute.
4. Rank non-island substations by the biggest INCREASE -- substations that
   picked up current-carrying importance specifically because the cut
   removed edges that used to serve as one of several parallel routes
   (whether for island-adjacent traffic or for other pairs entirely
   within the rest of the network -- removing any edge changes the whole
   network's Laplacian, not just paths that used that exact edge).

VALIDATION -- NOT YET DONE, NEXT STEP FOR THE USER
---------------------------------------------------
The real, Monte-Carlo-measured hotspots are already known: 96396693 for
Island A, 81597861 for Island B (both consistently the #1 non-island GIC
riser across scenario types, per scripts/compare_virginia_gic.py's real
output). Run this for both islands and check whether either substation
shows up near the top of the predicted list. If it does, this becomes a
legitimate cheap pre-screening tool for future candidate islands --
letting select_virginia_island.py's growth or a candidate-comparison step
avoid boundaries this proxy flags as likely to concentrate redistribution,
without paying the ~3000-iteration Monte Carlo cost just to find out.

PERFORMANCE, UNTESTED AT REAL SCALE
------------------------------------
current_flow_betweenness_centrality's solve cost grows with graph size;
solver="lu" (sparse LU factorization, the default here) should handle a
~2600-node, sparse power-grid graph in well under the time the real Monte
Carlo takes, but this has not been run against the real VA+PJM graph in
this sandbox (no real data available here) -- report back how long it
actually takes.

Usage:
    python scripts/check_redistribution_proxy.py [--island a] [--top N]
"""

import argparse
import json

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ADMITTANCE_DIR, VA_NETWORK_DIR, VA_ECON_DIR
from scripts.select_virginia_island import load_graph, _boundary_edges

logger = setup_logger(log_file="logs/check_redistribution_proxy.log")


def attach_conductance(G: nx.Graph) -> nx.Graph:
    """Sum 1/R (parallel-resistor conductance) over each edge's line_ids,
    using the real per-line resistances from transmission_lines.csv."""
    lines = pd.read_csv(VA_ADMITTANCE_DIR / "transmission_lines.csv")
    lines["name"] = lines["name"].astype(str)
    r_lookup = dict(zip(lines["name"], lines["R"]))

    missing = 0
    for _, _, data in G.edges(data=True):
        line_ids = data.get("line_ids", [])
        conductance = 0.0
        for lid in line_ids:
            r = r_lookup.get(str(lid))
            if r is None or r <= 0:
                missing += 1
                continue
            conductance += 1.0 / r
        # A zero/near-zero conductance edge would make the Laplacian
        # singular in a way current_flow_betweenness_centrality can't
        # handle gracefully -- floor it instead of dropping the edge
        # entirely (dropping could disconnect something that's supposed
        # to still be connected).
        data["conductance"] = conductance if conductance > 0 else 1e-9
    if missing:
        logger.warning(
            f"{missing} line_ids had no matching/valid R in transmission_lines.csv "
            "(floored to near-zero conductance rather than dropped)."
        )
    return G


def largest_component_subgraph(G: nx.Graph, exclude: set | None = None) -> nx.Graph:
    """current_flow_betweenness_centrality requires a connected graph --
    restrict to the largest connected component after optionally removing
    a node set (an island's members, once its boundary is cut)."""
    exclude = exclude or set()
    H = G.copy()
    H.remove_nodes_from(exclude)
    components = list(nx.connected_components(H))
    if not components:
        raise ValueError("Graph has no nodes left after exclusion.")
    largest = max(components, key=len)
    if len(components) > 1:
        logger.info(
            f"{len(components)} components after exclusion; using the largest "
            f"({len(largest)} nodes, {sum(len(c) for c in components) - len(largest)} "
            "nodes in smaller components dropped from this proxy's comparison)."
        )
    return H.subgraph(largest).copy()


def main(island_id: str = "a", top_n: int = 25, solver: str = "lu"):
    G = load_graph()
    attach_conductance(G)

    island_path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not island_path.exists():
        raise FileNotFoundError(f"Missing {island_path}. Run scripts/select_virginia_island.py first.")
    with open(island_path) as f:
        island = json.load(f)
    members = set(island["members"])
    boundary = _boundary_edges(G, members)

    logger.info(f"Computing baseline current-flow betweenness ({G.number_of_nodes()} nodes)...")
    G_baseline = largest_component_subgraph(G)
    baseline_scores = nx.current_flow_betweenness_centrality(G_baseline, weight="conductance", solver=solver)

    logger.info(f"Computing post-cut current-flow betweenness (Island {island_id}'s {len(boundary)} boundary edges removed)...")
    G_cut = G.copy()
    G_cut.remove_edges_from(boundary)
    G_cut_rest = largest_component_subgraph(G_cut, exclude=members)
    cut_scores = nx.current_flow_betweenness_centrality(G_cut_rest, weight="conductance", solver=solver)

    rows = [
        {
            "sub_id": node,
            "baseline_cfb": baseline_scores[node],
            "cut_cfb": cut_scores[node],
            "diff": cut_scores[node] - baseline_scores[node],
        }
        for node in G_cut_rest.nodes()
        if node in baseline_scores
    ]
    result = pd.DataFrame(rows).sort_values("diff", ascending=False)
    out_path = VA_ECON_DIR / f"va_island_{island_id}_redistribution_proxy.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(result)} rows)")

    print(f"\n=== Top {top_n} substations by predicted current-flow-betweenness increase (Island {island_id} cut) ===")
    print(result.head(top_n).to_string(index=False))
    print(f"\nFull detail: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cheap current-flow-betweenness proxy for GIC redistribution risk.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'b'.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--solver", type=str, default="lu", choices=["full", "lu", "cg"])
    args = parser.parse_args()
    main(island_id=args.island, top_n=args.top, solver=args.solver)
