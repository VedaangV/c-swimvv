"""
Shared graph utilities for va_network_graph.pkl and island member/boundary
sets: loading the graph, induced-cut boundary of a node set, voltage- and
conductance-weighted edge costs, JSON-safe serialization, and attaching
real per-line conductance for current-flow analysis.

Extracted from deprecated_sci_islanding/select_virginia_island.py and
check_redistribution_proxy.py when that greedy islanding strategy was
deprecated -- these are pure graph utilities any selection strategy needs,
so they stayed active rather than moving with the deprecated code.

Usage: import as a library, e.g.
    from scripts.island_graph_utils import load_graph, boundary_edges
"""

import networkx as nx
import pandas as pd

from configs.virginia_config import VA_ADMITTANCE_DIR, VA_NETWORK_DIR
from configs import setup_logger

logger = setup_logger(log_file="logs/island_graph_utils.log")


def sanitize_for_json(obj):
    """Recursively replace float('inf')/-inf with None -- json.dump would
    otherwise emit the non-spec-compliant Infinity token, which chokes
    other tools reading these files."""
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
    """Induced cut of members against the rest of the graph: every edge
    with exactly one endpoint in members. Recomputed fresh each time --
    cheap enough at this scale and simpler than patching incrementally."""
    boundary = set()
    for u in members:
        for v in G.neighbors(u):
            if v not in members:
                boundary.add((u, v) if u < v else (v, u))
    return boundary


def edge_cost(G: nx.Graph, u: str, v: str) -> float:
    """Voltage-weighted cost (kV / 100), summed per physical parallel line
    (cutting a 500kV line matters more than a 138kV one). Summed per
    physical line, not per graph edge, since build_virginia_network_graph.py
    aggregates parallel circuits onto one edge."""
    voltages = G.edges[u, v].get("voltages", [])
    return sum(v_kv / 100 for v_kv in voltages if v_kv is not None)


def total_boundary_cost(G: nx.Graph, boundary: set) -> float:
    return sum(edge_cost(G, u, v) for u, v in boundary)


def excluded_nodes_for(G: nx.Graph, island_ids: list) -> set:
    """Members + boundary neighbors of each prior island (by id, reading
    va_island_{id}.json), so a new island can't overlap or sit right next
    to one already selected -- otherwise it'd just be an extension of the
    first island, not a structurally independent one."""
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
    from transmission_lines.csv's real per-line resistances -- the actual
    physical quantity the GIC solve uses, not the voltage-based cost proxy
    above."""
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
        # Floor instead of dropping zero-conductance edges -- dropping
        # could disconnect something that should stay connected, and a
        # zero edge would make a Laplacian solve singular anyway.
        data["conductance"] = conductance if conductance > 0 else 1e-9
    if missing:
        logger.warning(
            f"{missing} line_ids had no matching/valid R in transmission_lines.csv "
            "(floored to near-zero conductance rather than dropped)."
        )
    return G


def largest_component_subgraph(G: nx.Graph, exclude: set | None = None) -> nx.Graph:
    """Some analyses (e.g. current_flow_betweenness_centrality) need a
    connected graph -- restrict to the largest component, optionally
    after removing a node set (e.g. an island once its boundary is cut)."""
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
