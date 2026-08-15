"""
Phase 4 of ISLANDING_MITIGATION_PLAN.md: re-solve GIC with Island A's
boundary lines removed, so Phase 5 can compare the resulting avoided-damage
fragility/economic output against the unmodified-topology baseline.

Thin wrapper around scripts/run_virginia_gic.py's main() (see that file's
module docstring for the exclude_line_ids/gic_dir/gnd_gic_dir parameters
added specifically for this) -- no separate Monte Carlo loop here, this
script only does what's specific to Phase 4: load Island A's cut set, run
the safety checks the plan calls for, then delegate.

SAFETY CHECKS (per ISLANDING_MITIGATION_PLAN.md Phase 4's stated risks):
1. Island A's members must form a single connected component in the
   topology graph via their *internal* edges alone (boundary edges
   excluded from this check by definition -- they're the cut). If not,
   that's a Phase 3 correctness bug, not just an interesting network fact
   -- hard-fails rather than silently solving GIC on a bogus topology. By
   construction this should always hold (region-growing only ever adds a
   node adjacent to the current boundary), so a failure here means
   something upstream changed in a way that broke that invariant.
2. How the *rest* of the network (everything outside Island A) responds
   to the cut is logged, not gated -- cutting boundary edges necessarily
   makes Island A its own connected component, but whether some outside
   substation loses its only path to the rest of PJM through a route that
   happened to run through Island A is a legitimate thing to know, not
   something to block on. The existing keep_nodes_with_ground() /
   non-zero-row filtering already in run_virginia_gic.py's Monte Carlo
   loop handles any resulting fully-isolated node numerically (same
   mechanism that already handles orphaned nodes in the baseline run) --
   nothing new needed there, this check is purely diagnostic visibility
   into whether that machinery is doing something unexpected on this run.

Run after scripts/select_virginia_island.py (and, as always,
scripts/run_virginia_admittance.py). Output: VA_ISLAND_A_GIC_DIR /
VA_ISLAND_A_GND_GIC_DIR (never touches the baseline va_gic/va_gnd_gic
directories).

Usage:
    python scripts/run_virginia_islanded_gic.py [--start N] [--end N]
"""

import argparse
import json

import networkx as nx

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, VA_ISLAND_A_GIC_DIR, VA_ISLAND_A_GND_GIC_DIR
from scripts.run_virginia_gic import main as run_gic_main
from scripts.select_virginia_island import load_graph, _boundary_edges

logger = setup_logger(log_file="logs/run_virginia_islanded_gic.log")


def load_island() -> dict:
    path = VA_NETWORK_DIR / "va_island_a.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/select_virginia_island.py first.")
    with open(path) as f:
        return json.load(f)


def check_island_connectivity(G: nx.Graph, members: set, boundary: set) -> dict:
    """Returns a dict of diagnostics; raises if Island A's own members
    aren't internally connected (see module docstring, check 1). The rest
    -- how the outside network responds to the cut -- is reported, not
    enforced (check 2)."""
    subG = G.subgraph(members)
    island_connected = nx.is_connected(subG) if len(members) > 0 else True
    if not island_connected:
        raise ValueError(
            f"Island A's {len(members)} members are NOT internally connected via their own "
            "edges (independent of the boundary cut) -- this should be structurally impossible "
            "given how scripts/select_virginia_island.py grows islands (every added node is "
            "adjacent to the current boundary), so this indicates a real bug upstream, not just "
            "an unusual topology. Aborting rather than solving GIC on a bogus cut."
        )

    G_cut = G.copy()
    G_cut.remove_edges_from(boundary)
    n_components_before = nx.number_connected_components(G)
    n_components_after = nx.number_connected_components(G_cut)

    island_is_own_component = False
    outside_fragmented = False
    for comp in nx.connected_components(G_cut):
        if comp == members:
            island_is_own_component = True
        elif not comp.isdisjoint(members):
            # partial overlap should be impossible if island_connected passed above
            island_is_own_component = False
            break

    non_island_components = [c for c in nx.connected_components(G_cut) if c.isdisjoint(members)]
    if len(non_island_components) > 1:
        outside_fragmented = True
        sizes = sorted((len(c) for c in non_island_components), reverse=True)
        logger.warning(
            f"Cutting Island A's boundary split the REST of the network into "
            f"{len(non_island_components)} components (sizes {sizes[:10]}"
            f"{'...' if len(sizes) > 10 else ''}) instead of staying as one. Some substation(s) "
            "outside Island A apparently reached the rest of PJM only via a path through Island "
            "A. Not an error -- the Monte Carlo loop's existing non-zero-row filtering handles "
            "isolated nodes numerically -- but worth knowing before trusting results for those "
            "specific substations."
        )

    logger.info(
        f"Connectivity after cut: {n_components_before} -> {n_components_after} components. "
        f"Island A forms its own separate component: {island_is_own_component}. "
        f"Rest of network fragmented further: {outside_fragmented}."
    )
    return {
        "island_internally_connected": island_connected,
        "components_before": n_components_before,
        "components_after": n_components_after,
        "island_is_own_component": island_is_own_component,
        "outside_fragmented": outside_fragmented,
    }


def main(start_idx: int = 0, end_idx: int | None = None):
    island = load_island()
    members = set(island["members"])
    boundary_line_ids = {str(x) for x in island["boundary_line_ids"]}

    G = load_graph()
    boundary_edges = _boundary_edges(G, members)
    check_island_connectivity(G, members, boundary_edges)

    logger.info(
        f"Re-solving GIC with Island A ({len(members)} members) boundary removed: "
        f"{len(boundary_edges)} edges / {len(boundary_line_ids)} physical lines cut. "
        f"Output: {VA_ISLAND_A_GIC_DIR}, {VA_ISLAND_A_GND_GIC_DIR}."
    )
    run_gic_main(
        start_idx=start_idx,
        end_idx=end_idx,
        exclude_line_ids=boundary_line_ids,
        gic_dir=VA_ISLAND_A_GIC_DIR,
        gnd_gic_dir=VA_ISLAND_A_GND_GIC_DIR,
    )
    logger.info("Islanded GIC Monte Carlo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-solve VA+PJM GIC with Island A's boundary removed.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    main(start_idx=args.start, end_idx=args.end)
