"""
Builds a combined multi-region island from two (or more) already-selected
islands, so Phase 4/5's pipeline can test islanding Island A and Island B
*simultaneously* (open question from ISLANDING_MITIGATION_PLAN.md Phase 6:
does protecting multiple regions at once help or compound losses -- untested
until now since A and B were only ever run in isolation).

A and B are non-adjacent by construction (select_virginia_island.py's Phase
6 extension excludes A's members and boundary neighbors when selecting B),
so the combined boundary is just the union recomputed fresh from the graph
-- not a manual union of the saved boundary_line_ids lists, so it stays
correct even if that non-adjacency assumption ever breaks (the adjacency
check below catches that case explicitly).

Output's "sub_islands" field (each island's own member set) lets
run_virginia_islanded_gic.py's connectivity check validate each sub-island's
internal connectivity separately, since the combined member set isn't one
connected blob -- it's two separate regions.

Usage:
    python scripts/combine_islands.py --islands a b --output ab
"""

import argparse
import json

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR
from scripts.island_graph_utils import (
    load_graph,
    boundary_edges as _boundary_edges,
    sanitize_for_json as _sanitize_for_json,
)

logger = setup_logger(log_file="logs/combine_islands.log")


def main(island_ids: list, output_id: str):
    G = load_graph()

    sub_islands = {}
    all_members = set()
    for island_id in island_ids:
        path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/select_virginia_island.py (island_id='{island_id}') first.")
        with open(path) as f:
            island = json.load(f)
        members = set(island["members"])
        if not members.isdisjoint(all_members):
            raise ValueError(
                f"Island {island_id}'s members overlap with an already-included island: "
                f"{members & all_members}. Combining overlapping islands isn't meaningful."
            )
        sub_islands[island_id] = sorted(members)
        all_members |= members

    # If two sub-islands are directly connected, that edge has both endpoints
    # in all_members so _boundary_edges would miss it -- catch it explicitly
    # instead of silently mis-cutting.
    for i, id_i in enumerate(island_ids):
        for id_j in island_ids[i + 1:]:
            members_i, members_j = set(sub_islands[id_i]), set(sub_islands[id_j])
            direct_edges = [
                (u, v) for u in members_i for v in G.neighbors(u) if v in members_j
            ]
            if direct_edges:
                raise ValueError(
                    f"Island {id_i} and Island {id_j} are directly adjacent ({direct_edges}) -- "
                    "combining them isn't a simultaneous-but-separate cut, it's effectively one "
                    "larger contiguous island. This script assumes non-adjacent inputs (true for "
                    "A and B, which Phase 6's selection explicitly enforced) and doesn't handle "
                    "this case."
                )

    boundary = _boundary_edges(G, all_members)
    boundary_line_ids = []
    for u, v in boundary:
        boundary_line_ids.extend(G.edges[u, v].get("line_ids", []))

    combined = {
        "members": sorted(all_members),
        "member_count": len(all_members),
        "boundary_line_ids": boundary_line_ids,
        "boundary_line_count": len(boundary_line_ids),
        "boundary_edge_count": len(boundary),
        "sub_islands": sub_islands,
        "combined_from": island_ids,
    }

    out_path = VA_NETWORK_DIR / f"va_island_{output_id}.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize_for_json(combined), f, indent=2, allow_nan=False)
    logger.info(f"Saved {out_path}")
    logger.info(
        f"Combined Island {output_id} from {island_ids}: {len(all_members)} total members "
        f"({', '.join(f'{k}={len(v)}' for k, v in sub_islands.items())}), "
        f"{len(boundary)} boundary edges / {len(boundary_line_ids)} physical lines cut."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple already-selected islands into one simultaneous-cut candidate.")
    parser.add_argument("--islands", type=str, nargs="+", required=True, help="Island ids to combine, e.g. --islands a b")
    parser.add_argument("--output", type=str, required=True, help="Output island id, e.g. --output ab")
    args = parser.parse_args()
    main(island_ids=args.islands, output_id=args.output)
