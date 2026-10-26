"""
Cheap, graph-only proxy for GIC redistribution risk -- no Monte Carlo, just
a deterministic linear solve on the network's electrical structure. SCI's
betweenness term uses shortest-path betweenness, which doesn't capture how
GIC actually redistributes: current splits across every parallel path at
once (Ohm's law), not along a single cheapest route.

Uses current-flow betweenness centrality instead (Newman 2005; Brandes &
Fleischer 2005), which is equivalent to modeling the graph as a resistor
network and solving for current flow directly -- same idea as the real
GIC solve in build_admittance_matrix.py, without the Monte Carlo sampling.

METHOD
------
1. Attach real conductance (1/R, from transmission_lines.csv) to the
   graph's edges -- not the voltage-based cost proxy select_virginia_
   island.py uses, which weights cut cost rather than modeling flow.
2. Compute current-flow betweenness on the full baseline graph.
3. Remove the candidate island's boundary edges, keep only the largest
   remaining non-island component (current-flow betweenness needs a
   connected graph), and recompute.
4. Rank non-island substations by the biggest increase -- these are the
   ones that picked up current-carrying importance because the cut
   removed one of their parallel routes.

VALIDATION -- NOT YET DONE
---------------------------------------------------
Known real hotspots from the Monte Carlo runs: 96396693 for Island A,
81597861 for Island B. Run this for both and check whether either shows
up near the top of the predicted list -- if so, this becomes a legitimate
cheap pre-screen for future candidate islands, without paying the full
Monte Carlo cost.

PERFORMANCE, UNTESTED AT REAL SCALE
------------------------------------
solver="lu" should handle a ~2600-node graph well under Monte Carlo time,
but hasn't been run against the real VA+PJM graph yet.

Usage:
    python scripts/check_redistribution_proxy.py [--island a] [--top N]
"""

import argparse
import json

import networkx as nx
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, VA_ECON_DIR
from scripts.island_graph_utils import (
    load_graph,
    boundary_edges as _boundary_edges,
    attach_conductance,
    largest_component_subgraph,
)

logger = setup_logger(log_file="logs/check_redistribution_proxy.log")


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
