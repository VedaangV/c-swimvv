"""
Shared graph utilities for working with va_network_graph.pkl and
island member/boundary sets -- loading the graph, computing the
induced-cut boundary of a node set, voltage-weighted and conductance-
weighted edge costs, JSON-safe serialization, and attaching real
per-line conductance for current-flow-based analysis.

Extracted out of scripts/deprecated_sci_islanding/select_virginia_
island.py and check_redistribution_proxy.py when the SCI-driven greedy
islanding strategy was deprecated (see that folder's README.md). These
functions are pure graph-structure utilities with no SCI dependency --
any island-selection strategy, old or new, needs to load the graph,
compute a boundary cut, and cost it, so they're kept active and shared
rather than moving with the deprecated selection algorithm.

Usage: import as a library, e.g.
    from scripts.island_graph_utils import load_graph, boundary_edges
"""

import networkx as nx
import pandas as pd

from configs.virginia_config import VA_ADMITTANCE_DIR, VA_NETWORK_DIR
from configs import setup_logger

logger = setup_logger(log_file="logs/island_graph_utils.log")


def sanitize_for_json(obj):
    """Recursively replace float('inf')/-inf with None. json.dump would
    otherwise emit the bare (non-standard) token Infinity for a ratio of
    inf -- valid to Python's own json.load, but not spec-compliant JSON,
    so a different tool reading an island JSON file could choke on it."""
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_graph() -> nx.Graph:
    path = VA_NETWORK_DIR / "va_network_graph.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def boundary_edges(G: nx.Graph, members: set) -> set:
    """Induced cut of members against the rest of the graph -- every edge
    with exactly one endpoint inside members. Recomputed fresh rather than
    incrementally patched -- simple to get right, cheap at power-grid
    scale (thousands of nodes, boundary sizes in the tens)."""
    boundary = set()
    for u in members:
        for v in G.neighbors(u):
            if v not in members:
                boundary.add((u, v) if u < v else (v, u))
    return boundary


def edge_cost(G: nx.Graph, u: str, v: str) -> float:
    """Voltage-weighted cost (kV / 100), summed per physical parallel line
    this graph edge represents -- cutting a 500kV line is a materially
    bigger deal than cutting a 138kV line. Summed per PHYSICAL line, not
    per graph edge -- build_virginia_network_graph.py aggregates parallel
    circuits onto a single edge specifically so this cost accounting
    reflects the real number of lines that would need to be switched."""
    voltages = G.edges[u, v].get("voltages", [])
    return sum(v_kv / 100 for v_kv in voltages if v_kv is not None)


def total_boundary_cost(G: nx.Graph, boundary: set) -> float:
    return sum(edge_cost(G, u, v) for u, v in boundary)


def excluded_nodes_for(G: nx.Graph, island_ids: list) -> set:
    """Members + boundary neighbors of each prior island (by id, reading
    va_island_{id}.json), so a new island can't overlap or sit immediately
    adjacent to one already selected. Boundary neighbors (not just
    members) are excluded because seeding right next to a prior island's
    cut wouldn't be a structurally independent new island, just an
    extension of the first one by another name."""
    excluded = set()
    for island_id in island_ids:
        import json
        path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- can't exclude Island {island_id}'s footprint.")
        with open(path) as f:
            prior = json.load(f)
        members = set(prior["members"])
        boundary = boundary_edges(G, members)
        boundary_neighbors = {v if u in members else u for u, v in boundary}
        excluded |= members | boundary_neighbors
    return excluded


def attach_conductance(G: nx.Graph) -> nx.Graph:
    """Sum 1/R (parallel-resistor conductance) over each edge's line_ids,
    using the real per-line resistances from transmission_lines.csv -- the
    same physical quantity the real GIC solve (admittance-matrix pipeline)
    uses, not the voltage-based cost proxy above (which weights cut cost,
    not current flow)."""
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
        # A zero/near-zero conductance edge would make a Laplacian-based
        # solve singular in a way it can't handle gracefully -- floor it
        # instead of dropping the edge entirely (dropping could disconnect
        # something that's supposed to still be connected).
        data["conductance"] = conductance if conductance > 0 else 1e-9
    if missing:
        logger.warning(
            f"{missing} line_ids had no matching/valid R in transmission_lines.csv "
            "(floored to near-zero conductance rather than dropped)."
        )
    return G


def largest_component_subgraph(G: nx.Graph, exclude: set | None = None) -> nx.Graph:
    """Some analyses (e.g. current_flow_betweenness_centrality) require a
    connected graph -- restrict to the largest connected component after
    optionally removing a node set (an island's members, once its
    boundary is cut, which makes them their own separate component)."""
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
            "nodes in smaller components dropped)."
        )
    return H.subgraph(largest).copy()
