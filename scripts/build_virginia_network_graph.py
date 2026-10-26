"""
Phase 1 of ISLANDING_MITIGATION_PLAN.md: build a substation-level network
graph for Virginia+PJM and compute topology-criticality metrics (betweenness
centrality, degree, articulation points) feeding the Substation Criticality
Index in scripts/compute_virginia_criticality.py.

Built from run_virginia_admittance.py's substation_info.csv/
transmission_lines.csv (not the raw geojsons) so the graph topology matches
what the admittance/GIC solve actually sees -- buses are already remapped
through bus_ids_map by that point, so we just invert substation_info's
"buses" column to map bus_idx -> substation and look up each line's
from_bus/to_bus in it.

Two things handled on purpose:
1. A line whose endpoints resolve to the same substation (internal
   multi-bus wiring) isn't a real inter-substation edge -- dropped, with a
   logged count.
2. Parallel lines between the same two substations are aggregated onto one
   edge (line_ids list + n_parallel_lines) rather than collapsed silently,
   so Phase 3's min-cut cost accounting reflects the real switching count.

Usage:
    python scripts/build_virginia_network_graph.py
"""

import ast
import pickle
from collections import defaultdict

import pandas as pd
import networkx as nx

from shapely.geometry import Point

from configs import setup_logger
from configs.virginia_config import VA_ADMITTANCE_DIR, VA_NETWORK_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/build_virginia_network_graph.log")


def _load_bus_to_substation(substation_info: pd.DataFrame) -> dict:
    bus_to_sub = {}
    collisions = 0
    for _, row in substation_info.iterrows():
        sub_id = str(row["name"])
        buses = row["buses"]
        for bus in buses:
            if bus in bus_to_sub and bus_to_sub[bus] != sub_id:
                collisions += 1
                continue
            bus_to_sub[bus] = sub_id
    if collisions:
        logger.warning(
            f"{collisions} bus indices appeared under more than one substation's "
            "'buses' list; kept the first assignment. Investigate before trusting "
            "the graph if this number is large."
        )
    return bus_to_sub


def build_graph() -> nx.Graph:
    substation_info_path = VA_ADMITTANCE_DIR / "substation_info.csv"
    lines_path = VA_ADMITTANCE_DIR / "transmission_lines.csv"
    if not substation_info_path.exists() or not lines_path.exists():
        raise FileNotFoundError(
            f"Missing {substation_info_path} or {lines_path}. "
            "Run scripts/run_virginia_admittance.py first."
        )

    substation_info = pd.read_csv(substation_info_path)
    substation_info["buses"] = substation_info["buses"].apply(ast.literal_eval)
    lines = pd.read_csv(lines_path)

    bus_to_sub = _load_bus_to_substation(substation_info)

    G = nx.Graph()
    for _, row in substation_info.iterrows():
        G.add_node(
            str(row["name"]),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
        )

    internal_skipped = 0
    unresolved_skipped = 0
    edge_lines = defaultdict(list)
    edge_voltages = defaultdict(list)

    for _, row in lines.iterrows():
        sub_a = bus_to_sub.get(row["from_bus"])
        sub_b = bus_to_sub.get(row["to_bus"])
        if sub_a is None or sub_b is None:
            unresolved_skipped += 1
            continue
        if sub_a == sub_b:
            internal_skipped += 1
            continue
        key = tuple(sorted((sub_a, sub_b)))
        edge_lines[key].append(row["name"])
        edge_voltages[key].append(row.get("V"))

    for (sub_a, sub_b), line_ids in edge_lines.items():
        G.add_edge(
            sub_a,
            sub_b,
            line_ids=line_ids,
            n_parallel_lines=len(line_ids),
            voltages=edge_voltages[(sub_a, sub_b)],
        )

    logger.info(
        f"Graph built: {G.number_of_nodes()} substations, {G.number_of_edges()} "
        f"substation-pair edges ({sum(len(v) for v in edge_lines.values())} physical "
        f"lines). Skipped {internal_skipped} internal (same-substation) lines and "
        f"{unresolved_skipped} lines with an unresolved bus endpoint."
    )
    return G


def _log_orphan_va_membership(G: nx.Graph, orphans: set) -> None:
    """Of the substations outside the main connected component, how many are
    inside Virginia? An out-of-state orphan is just edge-of-region data
    coverage; one inside Virginia is a substation the SCI/islanding work
    would silently drop."""
    try:
        va_boundary = get_virginia_boundary()
    except Exception as e:
        logger.warning(f"Could not load Virginia boundary to check orphan substations: {e!r}")
        return

    va_orphans = []
    for node in orphans:
        lat = G.nodes[node].get("latitude")
        lon = G.nodes[node].get("longitude")
        if lat is None or lon is None:
            continue
        if va_boundary.contains(Point(lon, lat)):
            va_orphans.append(node)

    logger.info(
        f"{len(va_orphans)}/{len(orphans)} out-of-main-component substations are "
        f"inside Virginia: {va_orphans}"
    )


def compute_topology_metrics(G: nx.Graph) -> pd.DataFrame:
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    main_component = components[0] if components else set()
    if len(components) > 1:
        sizes = [len(c) for c in components]
        orphans = set().union(*components[1:]) if len(components) > 1 else set()
        logger.warning(
            f"Graph has {len(components)} connected components (sizes {sizes}). "
            "Substations outside the main component are flagged, not dropped -- "
            "they can't be meaningfully centrality-ranked or islanded against the "
            "rest of the network, so exclude them at the SCI/islanding step, not here."
        )
        _log_orphan_va_membership(G, orphans)
    else:
        logger.info("Graph is fully connected: 1 component.")

    component_id = {}
    for i, comp in enumerate(components):
        for node in comp:
            component_id[node] = i

    betweenness = nx.betweenness_centrality(G, normalized=True)
    degree = dict(G.degree())
    articulation = set(nx.articulation_points(G))

    rows = []
    for node in G.nodes():
        rows.append(
            {
                "sub_id": node,
                "latitude": G.nodes[node].get("latitude"),
                "longitude": G.nodes[node].get("longitude"),
                "degree": degree[node],
                "betweenness": betweenness[node],
                "is_articulation_point": node in articulation,
                "component_id": component_id[node],
                "in_main_component": node in main_component,
            }
        )
    return pd.DataFrame(rows).sort_values("betweenness", ascending=False).reset_index(drop=True)


def main():
    G = build_graph()
    metrics = compute_topology_metrics(G)

    VA_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = VA_NETWORK_DIR / "va_network_graph.pkl"
    metrics_path = VA_NETWORK_DIR / "va_topology_metrics.csv"

    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    metrics.to_csv(metrics_path, index=False)

    logger.info(f"Saved graph to {graph_path}")
    logger.info(f"Saved topology metrics to {metrics_path}")
    logger.info(
        f"Top 5 by betweenness centrality:\n"
        f"{metrics[['sub_id', 'betweenness', 'degree', 'is_articulation_point']].head(5).to_string(index=False)}"
    )


if __name__ == "__main__":
    main()
