"""
Phase 4 of ISLANDING_MITIGATION_PLAN.md: re-solve GIC with an island's
boundary lines removed, so Phase 5 can compare the resulting avoided-damage
fragility/economic output against the unmodified-topology baseline.

Thin wrapper around scripts/run_virginia_gic.py's main() (see that file's
module docstring for the exclude_line_ids/gic_dir/gnd_gic_dir parameters
added specifically for this) -- no separate Monte Carlo loop here, this
script only does what's specific to Phase 4: load an island's cut set, run
the safety checks the plan calls for, then delegate.

Takes an island_id ("a", "b", ...; default "a") -- reads va_island_{id}.json
and writes to configs.virginia_config.get_island_dirs(island_id)'s paths.
island_id="a" behaves exactly as this script always has (same paths as the
standalone VA_ISLAND_A_GIC_DIR/VA_ISLAND_A_GND_GIC_DIR constants); "b" and
beyond were added once Phase 6 made a second real island (not a
hypothetical one) worth running through the identical pipeline, at which
point copy-pasting this file per island was clearly the wrong call.

SAFETY CHECKS (per ISLANDING_MITIGATION_PLAN.md Phase 4's stated risks):
1. Every constituent sub-island's own members must form a single connected
   component in the topology graph via their *internal* edges alone
   (boundary edges excluded from this check by definition -- they're the
   cut). If not, that's a Phase 3 correctness bug, not just an interesting
   network fact -- hard-fails rather than silently solving GIC on a bogus
   topology. By construction this should always hold (region-growing only
   ever adds a node adjacent to the current boundary), so a failure here
   means something upstream changed in a way that broke that invariant.
   For a single island (the normal case), this is just "members form one
   component." For a combined multi-region island (scripts/combine_
   islands.py's output, e.g. Island A and B run simultaneously), each
   sub-island is checked separately -- the *union* is never expected to be
   one component, since they're deliberately separate, non-adjacent
   regions cut at the same time, not one contiguous island.
2. How the *rest* of the network (everything outside every sub-island)
   responds to the cut is logged, not gated -- cutting boundary edges
   necessarily makes each sub-island its own connected component, but
   whether some outside substation loses its only path to the rest of PJM
   through a route that happened to run through one of them is a
   legitimate thing to know, not something to block on. The existing
   keep_nodes_with_ground() / non-zero-row filtering already in
   run_virginia_gic.py's Monte Carlo loop handles any resulting fully-
   isolated node numerically (same mechanism that already handles orphaned
   nodes in the baseline run) -- nothing new needed there, this check is
   purely diagnostic visibility into whether that machinery is doing
   something unexpected on this run.

Run after scripts/select_virginia_island.py (and, as always,
scripts/run_virginia_admittance.py). Output: get_island_dirs(island_id)'s
gic_dir/gnd_gic_dir (never touches the baseline va_gic/va_gnd_gic
directories).

Usage:
    python scripts/run_virginia_islanded_gic.py [--island b] [--start N] [--end N]
"""

import argparse
import json

import networkx as nx

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, get_island_dirs
from scripts.run_virginia_gic import main as run_gic_main
from scripts.select_virginia_island import load_graph, _boundary_edges

logger = setup_logger(log_file="logs/run_virginia_islanded_gic.log")


def load_island(island_id: str = "a") -> dict:
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/select_virginia_island.py "
            f"(island_id='{island_id}') first."
        )
    with open(path) as f:
        return json.load(f)


def check_island_connectivity(
    G: nx.Graph, members: set, boundary: set, island_id: str = "a",
    component_groups: list | None = None,
) -> dict:
    """Returns a dict of diagnostics; raises if any constituent sub-island's
    own members aren't internally connected (see module docstring, check
    1). The rest -- how the outside network responds to the cut -- is
    reported, not enforced (check 2).

    component_groups: for a combined multi-region island (scripts/combine_
    islands.py), the list of each sub-island's own member set -- checked
    individually for internal connectivity, instead of treating the full
    `members` union as one blob (which it structurally isn't -- separate,
    non-adjacent regions cut simultaneously, not one contiguous island).
    Defaults to [members] (the normal single-island case, and the exact
    original behavior)."""
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
    island_connected = True  # every group passed, or would have raised above

    G_cut = G.copy()
    G_cut.remove_edges_from(boundary)
    n_components_before = nx.number_connected_components(G)
    n_components_after = nx.number_connected_components(G_cut)

    cut_components = list(nx.connected_components(G_cut))
    groups_that_are_own_component = sum(1 for group in component_groups if group in cut_components)
    island_is_own_component = groups_that_are_own_component == len(component_groups)

    non_island_components = [c for c in cut_components if c.isdisjoint(members)]
    outside_fragmented = len(non_island_components) > 1
    if outside_fragmented:
        sizes = sorted((len(c) for c in non_island_components), reverse=True)
        logger.warning(
            f"Cutting Island {island_id}'s boundary split the REST of the network into "
            f"{len(non_island_components)} components (sizes {sizes[:10]}"
            f"{'...' if len(sizes) > 10 else ''}) instead of staying as one. Some substation(s) "
            f"outside Island {island_id} apparently reached the rest of PJM only via a path "
            "through it. Not an error -- the Monte Carlo loop's existing non-zero-row filtering "
            "handles isolated nodes numerically -- but worth knowing before trusting results for "
            "those specific substations."
        )

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
