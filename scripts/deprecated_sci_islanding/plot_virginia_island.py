"""
Sanity-check figure for scripts/select_virginia_island.py's output --
confirms Island A is a sane, contiguous cluster before anything else
builds on it (a past density bug was only caught by eyeballing a map, not
by numeric checks).

Two-panel figure: a map of all substations colored by SCI with Island A's
members highlighted/numbered in growth order and the cut boundary edges
drawn as dashed lines, plus a text table of the growth trace.

Boundary edges aren't stored directly in va_island_a.json (only the flat
boundary_line_ids list is), so they're recomputed here via
select_virginia_island._boundary_edges(G, members).

Usage:
    python viz/plot_virginia_island.py
"""

import json
import pickle

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import (
    VA_DATA_DIR,
    VA_ECON_DIR,
    VA_NETWORK_DIR,
    VA_FIGURES_DIR,
    get_virginia_boundary,
)
from scripts.island_graph_utils import boundary_edges as _boundary_edges, load_graph

logger = setup_logger(log_file="logs/plot_virginia_island.log")


def _load_substation_labels() -> dict:
    """Same tiered fallback as plot_virginia_criticality.py's identically
    named helper (external name -> OSM name -> operator/voltage/coords ->
    "Unnamed" + coords). Duplicated rather than shared, per project convention."""
    external_names = {}
    names_path = VA_NETWORK_DIR / "va_substation_names.csv"
    if names_path.exists():
        names_df = pd.read_csv(names_path)
        names_df["sub_id"] = names_df["sub_id"].astype(str)
        external_names = dict(zip(names_df["sub_id"], names_df["name"]))

    ss_df_path = VA_DATA_DIR / "grid_processed" / "ss_df.pkl"
    if not ss_df_path.exists():
        logger.warning(f"{ss_df_path} not found; falling back to external names (if any) or raw sub_id labels.")
        return {sid: str(name).strip() for sid, name in external_names.items() if pd.notna(name)}

    with open(ss_df_path, "rb") as f:
        ss_df = pickle.load(f)

    labels = {}
    for _, row in ss_df.iterrows():
        ss_id = str(row["SS_ID"])
        if ss_id in external_names and pd.notna(external_names[ss_id]):
            labels[ss_id] = str(external_names[ss_id]).strip()
            continue
        name = row.get("SS_NAME")
        if pd.notna(name) and str(name).strip() not in ("", "Unknown"):
            labels[ss_id] = str(name).strip()
            continue
        operator = row.get("SS_OPERATOR")
        voltage = row.get("SS_VOLTAGE")
        parts = []
        if pd.notna(operator) and str(operator).strip() not in ("", "Unknown"):
            parts.append(str(operator).strip())
        if pd.notna(voltage) and str(voltage).strip() not in ("", "Unknown"):
            parts.append(f"{voltage}")
        if parts:
            label = " ".join(parts) + f" substation ({row['lat']:.2f}, {row['lon']:.2f}) [SS {ss_id}]"
        else:
            label = f"Unnamed substation ({row['lat']:.2f}, {row['lon']:.2f}) [SS {ss_id}]"
        labels[ss_id] = label
    return labels


def main():
    setup_matplotlib()

    island_path = VA_NETWORK_DIR / "va_island_a.json"
    if not island_path.exists():
        raise FileNotFoundError(f"Missing {island_path}. Run scripts/select_virginia_island.py first.")
    with open(island_path) as f:
        island = json.load(f)

    sci_path = VA_ECON_DIR / "va_substation_criticality.csv"
    topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not sci_path.exists() or not topo_path.exists():
        raise FileNotFoundError(
            f"Missing {sci_path} or {topo_path}. Run scripts/compute_virginia_criticality.py "
            "and scripts/build_virginia_network_graph.py first."
        )
    sci = pd.read_csv(sci_path)
    sci["sub_id"] = sci["sub_id"].astype(str)
    topo = pd.read_csv(topo_path)
    topo["sub_id"] = topo["sub_id"].astype(str)
    summary = sci.merge(topo[["sub_id", "latitude", "longitude"]], on="sub_id", how="left")

    labels = _load_substation_labels()
    summary["display_name"] = summary["sub_id"].map(labels).fillna(summary["sub_id"])

    members = set(island["members"])
    G = load_graph()
    boundary = _boundary_edges(G, members)
    logger.info(f"Recomputed {len(boundary)} boundary edges for Island A ({len(members)} members).")

    growth_order = {t["added"]: t["step"] for t in island["growth_trace"]}
    member_rows = summary[summary["sub_id"].isin(members)].copy()
    member_rows["growth_step"] = member_rows["sub_id"].map(growth_order)
    member_rows = member_rows.sort_values("growth_step")

    fig, (ax_map, ax_table) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.4, 1]})

    va_boundary = get_virginia_boundary()
    gpd.GeoSeries([va_boundary], crs="EPSG:4326").plot(
        ax=ax_map, color="#f5f5f5", edgecolor="black", linewidth=1.2, zorder=1
    )

    lines_path = VA_NETWORK_DIR / "va_pjm_lines.geojson"
    if lines_path.exists():
        lines = gpd.read_file(lines_path).clip(va_boundary)
        if len(lines):
            lines.plot(ax=ax_map, color="0.85", linewidth=0.4, zorder=2)

    ax_map.scatter(
        summary["longitude"], summary["latitude"],
        c=summary["SCI"], cmap="YlOrRd", s=15,
        edgecolor="0.5", linewidth=0.2, zorder=3, alpha=0.6,
    )

    # Cut boundary: straight lines between island members and their
    # outside neighbors, at substation-pair resolution (not real line
    # geometry, since that's what the algorithm itself operates on).
    node_coords = {n: (G.nodes[n].get("longitude"), G.nodes[n].get("latitude")) for n in G.nodes()}
    for u, v in boundary:
        lon_u, lat_u = node_coords.get(u, (None, None))
        lon_v, lat_v = node_coords.get(v, (None, None))
        if None in (lon_u, lat_u, lon_v, lat_v):
            logger.warning(f"Boundary edge ({u}, {v}) missing coordinates; skipped in plot.")
            continue
        ax_map.plot([lon_u, lon_v], [lat_u, lat_v], color="#d62728", linewidth=2.2, linestyle="--", zorder=4)

    ax_map.scatter(
        member_rows["longitude"], member_rows["latitude"],
        facecolor="#1f77b4", edgecolor="black", linewidth=1.3, s=220, zorder=5,
    )
    for _, row in member_rows.iterrows():
        ax_map.annotate(
            f"{int(row['growth_step'])}: {row['display_name']}",
            (row["longitude"], row["latitude"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight="bold", zorder=6,
        )

    generation_note = ""
    if island.get("checked"):
        mw = island.get("generation_capacity_mw") or 0.0
        generation_note = f", existing local generation: {mw:.1f} MW"
    ax_map.set_title(
        f"Island A — {island['member_count']} substations, "
        f"{island['boundary_line_count']} boundary lines cut{generation_note}"
    )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")

    ax_table.axis("off")
    trace_lines = [f"seed = {island['seed']}", ""]
    for t in island["growth_trace"]:
        trace_lines.append(
            f"step {t['step']}: +{t['added']}\n"
            f"    {t['reason']}\n"
            f"    size={t['island_size']}  value={t['value']:.4f}  "
            f"cost={t['cost']:.4f}  ratio={t['ratio']:.4f}"
        )
    trace_lines.append("")
    trace_lines.append(f"final: {island['member_count']} members, cost={island['voltage_weighted_cost']:.2f}")
    if island.get("hit_size_cap"):
        trace_lines.append("(hit MAX_ISLAND_SIZE cap -- see plan doc)")
    ax_table.text(
        0.02, 0.98, "\n".join(trace_lines), transform=ax_table.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
    )
    ax_table.set_title("Growth trace")

    fig.tight_layout()
    out_path = VA_FIGURES_DIR / "va_island_a.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
