"""
Per-island map figure: a zoomed-in view of one island's own substations and
the transmission lines being cut to isolate it, plus a small inset locator
map showing where that zoomed area sits within Virginia as a whole.

Two panels per island:
- Main panel: island members (large, labeled with real names) and the
  specific boundary lines being switched out (highlighted, dashed),
  surrounded by enough of the local network (other nearby substations and
  transmission lines, shown small/unlabeled) to give real geographic
  context for what's being isolated.
- Inset panel: Virginia's own outline with a marker at the island's
  location -- lets a reader place the zoomed view without already knowing
  Virginia geography.

Reuses existing Virginia infrastructure rather than reimplementing it:
- viz/plot_virginia_gic_risk.py's _load_substation_labels() for real
  substation names (HIFLD/Wikidata/OSM-Overpass tiered lookup, same
  fallback labeling every other Virginia figure in this project uses).
- configs.virginia_config.get_virginia_boundary() for the Virginia outline.
- The same voltage-class color/width convention viz/plots.py's
  plot_vuln_trafos() uses, so this figure reads consistently against the
  substation-vulnerability figure in the same paper.

No cartopy here, deliberately -- viz/plot_virginia_gic_risk.py already
established the plain matplotlib/geopandas pattern for Virginia-local
(not CONUS-scale) maps in this project, and a zoomed-in local area doesn't
need a real map projection the way a state- or continent-spanning figure
does.

Usage:
    python viz/plot_island_maps.py --island a
    python viz/plot_island_maps.py --island a g o
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import (
    VA_ADMITTANCE_DIR,
    VA_NETWORK_DIR,
    VA_FIGURES_DIR,
    get_virginia_boundary,
)
from viz.plot_virginia_gic_risk import _load_substation_labels
from viz.plot_utils import savefig_with_retry

logger = setup_logger(log_file="logs/plot_island_maps.log")

VOLTAGE_LEVELS = [161, 230, 345, 500, 765]
VOLTAGE_COLORS = ["purple", "blue", "green", "orange", "maroon"]
VOLTAGE_WIDTHS = [0.5, 0.7, 0.9, 1.2, 1.5]
VOLTAGE_COLOR_MAP = dict(zip(VOLTAGE_LEVELS, VOLTAGE_COLORS))
VOLTAGE_WIDTH_MAP = dict(zip(VOLTAGE_LEVELS, VOLTAGE_WIDTHS))

# Padding around the island's own bounding box for the zoomed main panel --
# a percentage of the box's own size, floored by a minimum absolute degree
# pad so a single-substation island (a zero-size box) still gets a sensible
# amount of surrounding context instead of a degenerate zero-width plot.
ZOOM_PAD_FRACTION = 0.5
ZOOM_PAD_MIN_DEG = 0.06


def load_island(island_id: str) -> dict:
    import json

    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run an island-selection script first.")
    with open(path) as f:
        return json.load(f)


def load_substation_coords() -> pd.DataFrame:
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", "latitude", "longitude"]]


def load_transmission_lines() -> gpd.GeoDataFrame:
    path = VA_ADMITTANCE_DIR / "transmission_lines.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_admittance_matrix.py first.")
    df = pd.read_csv(path)
    df["name"] = df["name"].astype(str)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    if "V" not in df.columns:
        raise ValueError(f"Expected a 'V' voltage-class column in {path}; got {list(df.columns)}")
    return df


def plot_one_island(island_id: str, coords: pd.DataFrame, lines: gpd.GeoDataFrame, labels: dict, va_boundary):
    island = load_island(island_id)
    members = [str(m) for m in island["members"]]
    boundary_line_ids = {str(x) for x in island["boundary_line_ids"]}

    member_coords = coords[coords["sub_id"].isin(members)]
    if len(member_coords) != len(members):
        missing = set(members) - set(member_coords["sub_id"])
        logger.warning(f"Island {island_id}: {len(missing)} member(s) missing coordinates: {missing}")

    boundary_lines = lines[lines["name"].isin(boundary_line_ids)]
    if len(boundary_lines) != len(boundary_line_ids):
        missing = boundary_line_ids - set(boundary_lines["name"])
        logger.warning(f"Island {island_id}: {len(missing)} boundary line(s) not found in transmission_lines.csv: {missing}")

    # Zoom extent: bounding box of members' own coordinates UNION every
    # boundary line's full geometry (so the far/outside end of each cut
    # line -- what it used to connect to -- is visible too, not just the
    # island's own side of the cut).
    xs = list(member_coords["longitude"])
    ys = list(member_coords["latitude"])
    for geom in boundary_lines.geometry:
        gxs, gys = geom.xy
        xs.extend(gxs)
        ys.extend(gys)
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    pad_x = max((maxx - minx) * ZOOM_PAD_FRACTION, ZOOM_PAD_MIN_DEG)
    pad_y = max((maxy - miny) * ZOOM_PAD_FRACTION, ZOOM_PAD_MIN_DEG)
    ext_minx, ext_maxx = minx - pad_x, maxx + pad_x
    ext_miny, ext_maxy = miny - pad_y, maxy + pad_y

    ctx_subs = coords[
        coords["longitude"].between(ext_minx, ext_maxx) & coords["latitude"].between(ext_miny, ext_maxy)
    ]
    ctx_lines = lines.cx[ext_minx:ext_maxx, ext_miny:ext_maxy]
    # Excluded from the background layer below -- boundary lines get their
    # own red-dashed highlight instead, and plotting them twice (once in
    # their voltage color, once in red on top) let thin slivers of the
    # original color show through the dash gaps.
    ctx_lines_non_boundary = ctx_lines[~ctx_lines["name"].isin(boundary_line_ids)]

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    inset_ax = fig.add_subplot(gs[1])

    # Context: all nearby transmission lines, colored by voltage class, thin.
    for v in VOLTAGE_LEVELS:
        subset = ctx_lines_non_boundary[ctx_lines_non_boundary["V"] == v]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=VOLTAGE_COLOR_MAP[v], linewidth=VOLTAGE_WIDTH_MAP[v], alpha=0.5, zorder=2)

    # Context: all nearby substations, small gray dots, unlabeled.
    ax.scatter(
        ctx_subs["longitude"], ctx_subs["latitude"], s=12, facecolors="0.6", edgecolors="none", zorder=3,
    )

    # Highlight: the boundary lines being cut -- thick, dashed, red, on top
    # of everything else.
    if not boundary_lines.empty:
        boundary_lines.plot(ax=ax, color="red", linewidth=2.8, linestyle=(0, (5, 3)), zorder=5)

    # Highlight: island members -- large stars, labeled with real names.
    ax.scatter(
        member_coords["longitude"], member_coords["latitude"],
        s=260, marker="*", facecolors="#d73027", edgecolors="black", linewidths=0.8, zorder=6,
    )
    for _, row in member_coords.iterrows():
        name = labels.get(row["sub_id"], row["sub_id"])
        ax.annotate(
            name, (row["longitude"], row["latitude"]), xytext=(6, 6), textcoords="offset points",
            fontsize=9, fontweight="bold", zorder=7,
        )

    ax.set_xlim(ext_minx, ext_maxx)
    ax.set_ylim(ext_miny, ext_maxy)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Island {island_id.upper()}: isolated substation(s) and cut boundary lines")
    ax.set_facecolor("#F5F5F5")
    # Degrees of longitude cover less real ground distance than degrees of
    # latitude away from the equator -- plotting raw lon/lat with a plain
    # 1:1 aspect ratio visibly stretches everything east-west. Compensate
    # with a standard equirectangular correction (1/cos(latitude)) rather
    # than a real map projection, which isn't needed at this local, zoomed
    # scale.
    mean_lat_main = (ext_miny + ext_maxy) / 2
    ax.set_aspect(1.0 / max(np.cos(np.radians(mean_lat_main)), 1e-6))

    voltage_handles = [
        plt.Line2D([0], [0], color=VOLTAGE_COLOR_MAP[v], linewidth=2, label=f"{v} kV") for v in VOLTAGE_LEVELS
    ]
    voltage_handles.append(plt.Line2D([0], [0], color="red", linewidth=2.5, linestyle=(0, (5, 3)), label="Cut boundary line"))
    voltage_handles.append(
        plt.Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="#d73027", markeredgecolor="black", markersize=14, label="Island member")
    )
    ax.legend(handles=voltage_handles, loc="lower left", fontsize=8, framealpha=0.9)

    # Locator panel: Virginia's outline, with a star at the island's
    # location -- a genuinely separate subplot beside the main map, not
    # layered on top of it, so it never covers real map content.
    va_x, va_y = va_boundary.exterior.xy if va_boundary.geom_type == "Polygon" else (None, None)
    if va_x is not None:
        inset_ax.plot(va_x, va_y, color="black", linewidth=0.8)
        inset_ax.fill(va_x, va_y, color="#EEEEEE", zorder=1)
    else:
        for poly in va_boundary.geoms:
            px, py = poly.exterior.xy
            inset_ax.plot(px, py, color="black", linewidth=0.8)
            inset_ax.fill(px, py, color="#EEEEEE", zorder=1)
    inset_ax.scatter([(minx + maxx) / 2], [(miny + maxy) / 2], s=90, marker="*", facecolors="#d73027", edgecolors="black", linewidths=0.6, zorder=3)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_visible(False)
    inset_ax.set_title("Location in Virginia", fontsize=9)
    va_bounds = va_boundary.bounds
    mean_lat_va = (va_bounds[1] + va_bounds[3]) / 2
    inset_ax.set_aspect(1.0 / max(np.cos(np.radians(mean_lat_va)), 1e-6))

    out_path = VA_FIGURES_DIR / f"va_island_{island_id}_map.png"
    savefig_with_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")

    member_names = [labels.get(m, m) for m in members]
    logger.info(f"Island {island_id.upper()} members ({len(members)}): {member_names}")
    return member_names


def main(island_ids):
    setup_matplotlib()
    VA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    coords = load_substation_coords()
    lines = load_transmission_lines()
    labels = _load_substation_labels()
    va_boundary = get_virginia_boundary()

    all_names = {}
    for island_id in island_ids:
        all_names[island_id] = plot_one_island(island_id, coords, lines, labels, va_boundary)

    logger.info("Summary of island member names:")
    for island_id, names in all_names.items():
        logger.info(f"  Island {island_id.upper()}: {names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zoomed island map + Virginia locator inset, one figure per island.")
    parser.add_argument("--island", type=str, nargs="+", default=["a"], help="One or more island ids, e.g. --island a g o")
    args = parser.parse_args()
    main(island_ids=args.island)
