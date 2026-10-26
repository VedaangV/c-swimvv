"""
Visualization of the VA+PJM EHV network; source for the paper's Figure 3.

Deliberately does NOT read build_virginia_network.py's own output
(va_pjm_lines/substations.geojson) -- that script overwrites those files
unconditionally on every run and they can drift out of sync with what
every other script (centrality, islanding, GIC) actually reads. Instead
this reads transmission_lines.csv and va_topology_metrics.csv, same as
viz/plot_island_maps.py, so it stays consistent with the real pipeline.

Usage:
    python viz/plot_virginia_network.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import (
    VA_NETWORK_DIR,
    VA_ADMITTANCE_DIR,
    VA_FIGURES_DIR,
    get_pjm_geodataframe,
    get_virginia_boundary,
    _load_states,
)
from viz.plot_utils import savefig_with_retry

logger = setup_logger(log_file="logs/plot_virginia_network.log")

VOLTAGE_COLORS = {765: "#d73027", 500: "#fc8d59", 345: "#fee090", 230: "#91bfdb", 161: "#4575b4"}


def main():
    setup_matplotlib()

    topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    lines_path = VA_ADMITTANCE_DIR / "transmission_lines.csv"
    if not topo_path.exists() or not lines_path.exists():
        raise FileNotFoundError(
            f"Missing {topo_path} or {lines_path}. Run "
            "scripts/build_virginia_network_graph.py and "
            "scripts/build_admittance_matrix.py first."
        )

    topo = pd.read_csv(topo_path)
    subs = gpd.GeoDataFrame(
        topo, geometry=gpd.points_from_xy(topo["longitude"], topo["latitude"]), crs="EPSG:4326"
    )

    lines_df = pd.read_csv(lines_path)
    lines_df["geometry"] = lines_df["geometry"].apply(wkt.loads)
    lines = gpd.GeoDataFrame(lines_df, geometry="geometry", crs="EPSG:4326")
    lines = lines.rename(columns={"V": "VOLTAGE"})

    fig = plt.figure(figsize=(12, 9))
    # inset is placed inside this box (top-right), so no separate column needed
    ax = fig.add_axes([0.08, 0.08, 0.88, 0.86])

    pjm_states = get_pjm_geodataframe()
    pjm_states.plot(ax=ax, color="#f5f5f0", edgecolor="0.6", linewidth=0.6, zorder=1)

    va_boundary = get_virginia_boundary()
    gpd.GeoSeries([va_boundary], crs="EPSG:4326").plot(
        ax=ax, color="#e8e4d8", edgecolor="black", linewidth=1.5, zorder=2
    )

    for voltage, color in VOLTAGE_COLORS.items():
        subset = lines[lines["VOLTAGE"] == voltage]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=1.0, label=f"{voltage} kV", zorder=3)

    subs.plot(ax=ax, color="0.25", markersize=1.5, zorder=4, label="Substations")

    ax.set_title(f"VA+PJM EHV network — {len(lines)} lines, {len(subs)} substations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # equirectangular correction, same as other Virginia-local maps in this project
    _, miny, _, maxy = pjm_states.total_bounds
    mean_lat = (miny + maxy) / 2
    ax.set_aspect(1.0 / max(np.cos(np.radians(mean_lat)), 1e-6))

    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(handles=line_handles, labels=line_labels, loc="lower left", fontsize=8, title="Line voltage")

    # Locator inset: PJM footprint with Virginia highlighted, positioned
    # against the main axes' actual post-draw box (same approach as
    # viz/plot_island_maps.py) so it sits snug in the top-right corner.
    fig.canvas.draw()
    main_pos = ax.get_position()
    inset_width, inset_height = 0.20, 0.24
    inset_pad = 0.012
    inset_title_h = 0.035  # headroom for inset's title
    inset_x0 = main_pos.x1 - inset_width - inset_pad
    inset_y0 = main_pos.y1 - inset_height - inset_pad - inset_title_h
    inset_ax = fig.add_axes([inset_x0, inset_y0, inset_width, inset_height], zorder=10)

    # white backing so the inset stays legible over the map content beneath it
    backing_pad = 0.008
    fig.patches.append(
        plt.Rectangle(
            (inset_x0 - backing_pad, inset_y0 - backing_pad),
            inset_width + 2 * backing_pad, inset_height + 2 * backing_pad + inset_title_h,
            transform=fig.transFigure, facecolor="white", edgecolor="0.4", linewidth=0.8, zorder=9,
        )
    )

    us_states = _load_states()
    us_states.plot(ax=inset_ax, color="#eeeeee", edgecolor="0.6", linewidth=0.4, zorder=1)
    pjm_states.boundary.plot(ax=inset_ax, color="0.3", linewidth=0.6, zorder=2)
    gpd.GeoSeries([va_boundary], crs="EPSG:4326").plot(
        ax=inset_ax, color="#d73027", edgecolor="black", linewidth=0.8, zorder=3
    )
    conus_bounds = us_states.cx[-125:-66, 24:50].total_bounds
    inset_ax.set_xlim(conus_bounds[0], conus_bounds[2])
    inset_ax.set_ylim(conus_bounds[1], conus_bounds[3])
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_visible(False)
    inset_ax.set_title("Location in the PJM footprint", fontsize=8, pad=4)
    inset_mean_lat = (conus_bounds[1] + conus_bounds[3]) / 2
    inset_ax.set_aspect(1.0 / max(np.cos(np.radians(inset_mean_lat)), 1e-6))

    out_path = VA_FIGURES_DIR / "va_pjm_network.png"
    savefig_with_retry(fig, out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    logger.info(f"Lines: {len(lines)}, Substations: {len(subs)}")
    for voltage in sorted(VOLTAGE_COLORS, reverse=True):
        n = (lines["VOLTAGE"] == voltage).sum()
        if n:
            logger.info(f"  {voltage} kV: {n} lines")


if __name__ == "__main__":
    main()
