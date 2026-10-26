"""
Re-solve GIC with an island's boundary lines removed, so we can compare the
resulting avoided-damage fragility/economic output against the unmodified
baseline topology.

Thin wrapper around run_virginia_gic.py's main() (exclude_line_ids/gic_dir/
gnd_gic_dir) -- this script just loads an island's cut set, runs a couple
safety checks, and delegates. Takes an island_id ("a", "b", ...); reads
va_island_{id}.json and writes to get_island_dirs(island_id)'s paths.

Safety checks:
1. Each sub-island's own members must form one connected component via
   their internal edges alone (boundary edges excluded -- they're the cut).
   Should always hold by construction; a failure means a bug upstream, so
   this hard-fails rather than solving GIC on a bogus topology.
2. Whether the rest of the network fragments further from the cut is
   logged, not gated -- the existing keep_nodes_with_ground() /
   non-zero-row filtering in run_virginia_gic.py already handles any
   isolated nodes numerically; this is just diagnostic visibility.

Non-island substations stranded by the cut (check 2) are written to
va_island_{id}_stranded.json so compute_island_benefit_cost.py can account
for their GDP loss, not just the formal island members'.

Run after scripts/select_virginia_island.py (and run_virginia_admittance.py).
Output: get_island_dirs(island_id)'s gic_dir/gnd_gic_dir (never touches the
baseline va_gic/va_gnd_gic directories).

Usage:
    python scripts/run_virginia_islanded_gic.py [--island b] [--start N] [--end N]
"""

import argparse
import json

import networkx as nx

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, get_island_dirs
from scripts.run_virginia_gic import main as run_gic_main
from scripts.island_graph_utils import load_graph, boundary_edges as _boundary_edges, sanitize_for_json

logger = setup_logger(log_file="logs/run_virginia_islanded_gic.log")


def load_island(island_id: str = "a") -> dict:
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run an island-selection script "
            f"(island_id='{island_id}') first."
        )
    with open(path) as f:
        return json.load(f)


def check_island_connectivity(
    G: nx.Graph, members: set, boundary: set, island_id: str = "a",
    component_groups: list | None = None,
) -> dict:
    """Returns a dict of diagnostics; raises if any sub-island's own
    members aren't internally connected (check 1). How the outside
    network responds to the cut is reported, not enforced (check 2).

    component_groups: for a combined multi-region island (combine_
    islands.py), each sub-island's own member set, checked individually
    instead of treating the full `members` union as one component (they're
    deliberately separate, non-adjacent regions). Defaults to [members]."""
    component_groups = component_groups or [members]

    for group in component_groups:
        subG = G.subgraph(group)
        if len(group) > 0 and not nx.is_connected(subG):
            raise ValueError(
                f"Island {island_id}: a sub-island's {len(group)} members are NOT internally "
                "connected via their own edges (independent of the boundary cut) -- this should "
                "be structurally impossible given how scripts/select_virginia_island.py grows "
                "islands (every added node is adjacent to the current boundary), so this "
                "indicates a real bug upstream, not just an unusual topology. Aborting rather "
                "than solving GIC on a bogus cut."
            )
    island_connected = True  # would have raised above otherwise

    G_cut = G.copy()
    G_cut.remove_edges_from(boundary)
    n_components_before = nx.number_connected_components(G)
    n_components_after = nx.number_connected_components(G_cut)

    cut_components = list(nx.connected_components(G_cut))
    groups_that_are_own_component = sum(1 for group in component_groups if group in cut_components)
    island_is_own_component = groups_that_are_own_component == len(component_groups)

    # G isn't one connected component at baseline (~48 pre-existing orphan
    # components, unrelated to any island). Restrict to components that
    # are subsets of the island's own baseline region (home_components) so
    # we don't count those pre-existing orphans as newly stranded.
    home_components = set()
    for group in component_groups:
        if group:
            home_components |= nx.node_connected_component(G, next(iter(group)))
    non_island_components = [c for c in cut_components if c.isdisjoint(members) and c <= home_components]
    outside_fragmented = len(non_island_components) > 1
    stranded_substations = []
    if outside_fragmented:
        # Largest non-island component is "the rest of PJM"; smaller ones
        # lost their only path through the island and are stranded.
        non_island_components_by_size = sorted(non_island_components, key=len, reverse=True)
        stranded_groups = non_island_components_by_size[1:]
        stranded_substations = sorted({sub for group in stranded_groups for sub in group})
        sizes = [len(g) for g in non_island_components_by_size]
        logger.warning(
            f"Cutting Island {island_id}'s boundary split the REST of the network into "
            f"{len(non_island_components)} components (sizes {sizes[:10]}"
            f"{'...' if len(sizes) > 10 else ''}) instead of staying as one. "
            f"{len(stranded_substations)} substation(s) outside Island {island_id} apparently "
            "reached the rest of PJM only via a path through it, and are now stranded -- "
            "written to va_island_{}_stranded.json for compute_island_benefit_cost.py to cost. "
            "Not a numerical error -- the Monte Carlo loop's existing non-zero-row filtering "
            "handles isolated nodes numerically -- but their own-outage cost was previously "
            "uncharged anywhere.".format(island_id)
        )

    stranded_path = VA_NETWORK_DIR / f"va_island_{island_id}_stranded.json"
    with open(stranded_path, "w") as f:
        json.dump(sanitize_for_json({
            "island_id": island_id,
            "outside_fragmented": outside_fragmented,
            "components_before": n_components_before,
            "components_after": n_components_after,
            "stranded_substations": stranded_substations,
        }), f, indent=2)
    logger.info(f"Saved {stranded_path} ({len(stranded_substations)} stranded substation(s)).")

    logger.info(
        f"Connectivity after cut: {n_components_before} -> {n_components_after} components. "
        f"{groups_that_are_own_component}/{len(component_groups)} sub-island(s) form their own "
        f"separate component. Rest of network fragmented further: {outside_fragmented}."
    )
    return {
        "island_internally_connected": island_connected,
        "components_before": n_components_before,
        "components_after": n_components_after,
        "island_is_own_component": island_is_own_component,
        "outside_fragmented": outside_fragmented,
        "stranded_substations": stranded_substations,
    }


def main(start_idx: int = 0, end_idx: int | None = None, island_id: str = "a"):
    island = load_island(island_id)
    members = set(island["members"])
    boundary_line_ids = {str(x) for x in island["boundary_line_ids"]}
    dirs = get_island_dirs(island_id)
    component_groups = (
        [set(v) for v in island["sub_islands"].values()] if "sub_islands" in island else None
    )

    G = load_graph()
    boundary_edges = _boundary_edges(G, members)
    check_island_connectivity(G, members, boundary_edges, island_id, component_groups)

    logger.info(
        f"Re-solving GIC with Island {island_id} ({len(members)} members) boundary removed: "
        f"{len(boundary_edges)} edges / {len(boundary_line_ids)} physical lines cut. "
        f"Output: {dirs['gic_dir']}, {dirs['gnd_gic_dir']}."
    )
    run_gic_main(
        start_idx=start_idx,
        end_idx=end_idx,
        exclude_line_ids=boundary_line_ids,
        gic_dir=dirs["gic_dir"],
        gnd_gic_dir=dirs["gnd_gic_dir"],
    )
    logger.info(f"Island {island_id} GIC Monte Carlo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-solve VA+PJM GIC with an island's boundary removed.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'b'.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    main(start_idx=args.start, end_idx=args.end, island_id=args.island)
