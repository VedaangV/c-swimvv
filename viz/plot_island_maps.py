"""
Per-island map figure: a zoomed-in view of one island's substations and
the boundary lines being cut to isolate it, plus a small inset locator
map showing where it sits within Virginia.

Reuses viz/plot_virginia_gic_risk.py's substation label lookup and
get_virginia_boundary() for the state outline, and matches the voltage
color/width convention used by plot_vuln_trafos() in viz/plots.py.
No cartopy -- not needed for a zoomed, local-scale map.

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

# maps internal island_id (used everywhere else in the codebase) to the
# paper's A/B/C labels; data loading still uses island_id, only display uses this
PAPER_ISLAND_LABELS = {"a": "A", "g": "B", "o": "C"}

# padding for the zoomed box; floored by a min degree pad so a
# single-substation island (zero-size box) doesn't collapse to nothing
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


def plot_one_island(
    island_id: str, coords: pd.DataFrame, lines: gpd.GeoDataFrame, labels: dict, va_boundary,
    show_inset: bool = True,
):
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

    # zoom extent: members' bbox union each boundary line's full geometry,
    # so the far end of each cut line is visible too, not just the island side
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
    # excluded here since boundary lines get their own red-dashed highlight;
    # plotting them twice let slivers of the original color show through the dashes
    ctx_lines_non_boundary = ctx_lines[~ctx_lines["name"].isin(boundary_line_ids)]

    paper_label = PAPER_ISLAND_LABELS.get(island_id, island_id.upper())

    fig = plt.figure(figsize=(10, 8))
    # leave the right ~1/3 free for the locator inset when show_inset; an
    # aspect-locked axes can only shrink within this allocation, never grow
    # past it, so reserving the width up front guarantees the inset has room
    ax = fig.add_axes([0.07, 0.09, 0.60, 0.86]) if show_inset else fig.add_axes([0.09, 0.09, 0.86, 0.86])

    for v in VOLTAGE_LEVELS:
        subset = ctx_lines_non_boundary[ctx_lines_non_boundary["V"] == v]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=VOLTAGE_COLOR_MAP[v], linewidth=VOLTAGE_WIDTH_MAP[v], alpha=0.5, zorder=2)

    ax.scatter(
        ctx_subs["longitude"], ctx_subs["latitude"], s=12, facecolors="0.6", edgecolors="none", zorder=3,
    )

    # boundary lines being cut
    if not boundary_lines.empty:
        boundary_lines.plot(ax=ax, color="red", linewidth=2.8, linestyle=(0, (5, 3)), zorder=5)

    # island members
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
    ax.set_title(f"Island {paper_label}: isolated substation(s) and cut boundary lines")
    ax.set_facecolor("#F5F5F5")
    # equirectangular correction (1/cos(lat)) so lon/lat don't stretch
    # east-west; a real projection isn't needed at this local scale
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

    # locator panel: Virginia outline with a star at the island's location.
    # Skipped when show_inset=False (e.g. the combined-islands figure,
    # where all three cluster in central VA so repeating this is redundant).
    #
    # Positioned against ax's actual post-aspect box rather than a guessed
    # rect, since set_aspect() above shrinks it by an amount that depends
    # on the island's own extent and isn't known in advance; forcing a
    # draw first makes get_position() return the real box.
    if show_inset:
        fig.canvas.draw()
        main_pos = ax.get_position()
        inset_gap = 0.02
        inset_width = 0.16
        inset_height = 0.32
        inset_x0 = min(main_pos.x1 + inset_gap, 1.0 - inset_width - 0.01)
        inset_y0 = main_pos.y0 + (main_pos.height - inset_height) / 2
        inset_ax = fig.add_axes([inset_x0, inset_y0, inset_width, inset_height])

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

    out_path = VA_FIGURES_DIR / f"va_island_{paper_label.lower()}_map.png"
    savefig_with_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path} (internal island_id={island_id}, paper label={paper_label})")

    member_names = [labels.get(m, m) for m in members]
    logger.info(f"Island {paper_label} (internal id {island_id}) members ({len(members)}): {member_names}")
    return member_names


def main(island_ids, show_inset: bool = True):
    setup_matplotlib()
    VA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    coords = load_substation_coords()
    lines = load_transmission_lines()
    labels = _load_substation_labels()
    va_boundary = get_virginia_boundary()

    all_names = {}
    for island_id in island_ids:
        all_names[island_id] = plot_one_island(island_id, coords, lines, labels, va_boundary, show_inset=show_inset)

    logger.info("Summary of island member names:")
    for island_id, names in all_names.items():
        paper_label = PAPER_ISLAND_LABELS.get(island_id, island_id.upper())
        logger.info(f"  Island {paper_label} (internal id {island_id}): {names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zoomed island map + Virginia locator inset, one figure per island.")
    parser.add_argument("--island", type=str, nargs="+", default=["a"], help="One or more island ids, e.g. --island a g o")
    parser.add_argument(
        "--no-inset", action="store_true",
        help="Skip the 'Location in Virginia' locator panel (e.g. when compositing all islands into "
             "one combined figure, where repeating the same near-identical inset on every panel is redundant).",
    )
    args = parser.parse_args()
    main(island_ids=args.island, show_inset=not args.no_inset)
