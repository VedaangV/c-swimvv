"""
Visualization of the VA+PJM EHV network, and the source for the paper's
Figure 3.

DELIBERATELY DOES NOT READ scripts/build_virginia_network.py's own output
(va_pjm_lines.geojson / va_pjm_substations.geojson) ANY MORE. Those two
files are overwritten unconditionally on every run of that script (only
its raw-OSM-download cache is checked before re-fetching), so they can
silently drift out of sync with the network actually in use everywhere
else -- confirmed against real data: a re-run of build_virginia_network.py
produced 2,226 substations / 3,880 lines, while va_network_graph.pkl /
va_topology_metrics.csv (built earlier by build_virginia_network_graph.py,
and what every other script -- centrality, islanding, GIC -- actually
reads) reflect 2,624 substations / 4,586 lines, matching the paper's
already-published 2,728/4,587. Reading transmission_lines.csv and
va_topology_metrics.csv here instead -- the exact same two files
viz/plot_island_maps.py already reads for the same reason -- means this
figure is always consistent with the real pipeline output, regardless of
whether build_virginia_network.py's own snapshot ever drifts again.

Three fixes applied per paper review feedback (Ed's Oct 2026 review):
- Filled PJM-state polygons as an actual basemap layer (zorder=1, light
  fill), not just thin boundary outlines over blank white -- the previous
  version had state borders but no sense of land vs. not-land.
- Added a "Location in the PJM footprint" inset locator, same technique
  viz/plot_island_maps.py already uses: build the main axes first, force a
  draw, then position the inset against the main axes' ACTUAL post-aspect
  rendered box (ax.get_position()) rather than a pre-guessed rect, so it
  sits snug regardless of how tall/wide the PJM footprint's own aspect
  ratio turns out to be.
- Substations are now in the legend (previous version's subs.plot() call
  had no label=, so matplotlib silently excluded it from ax.legend()) and
  smaller/lighter so they read as context rather than competing visually
  with the colored transmission lines.

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
    # Main axes claims a fixed-width left allocation, leaving room on the
    # right for the inset -- same reasoning as plot_island_maps.py: an
    # aspect-locked axes (set_aspect below) can only ever shrink within
    # this allocation, never grow past it, so the inset is guaranteed
    # space regardless of the PJM footprint's own aspect ratio.
    ax = fig.add_axes([0.08, 0.08, 0.66, 0.86])

    pjm_states = get_pjm_geodataframe()
    # Basemap layer: filled state polygons, not just boundary outlines --
    # gives the figure an actual land/not-land reference instead of thin
    # lines over blank white.
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
    # Equirectangular correction (1/cos(latitude)) -- same convention as
    # every other Virginia-local map in this project (viz/plot_island_maps.py,
    # viz/plot_virginia_gic_risk.py) -- plain 1:1 aspect visibly stretches
    # the network east-west at this latitude.
    _, miny, _, maxy = pjm_states.total_bounds
    mean_lat = (miny + maxy) / 2
    ax.set_aspect(1.0 / max(np.cos(np.radians(mean_lat)), 1e-6))

    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(handles=line_handles, labels=line_labels, loc="lower left", fontsize=8, title="Line voltage")

    # Locator inset: PJM footprint outline with Virginia highlighted,
    # positioned against the main axes' ACTUAL rendered box (post-draw),
    # not a pre-guessed rect -- see module docstring and
    # viz/plot_island_maps.py's identical pattern for why.
    fig.canvas.draw()
    main_pos = ax.get_position()
    inset_width, inset_height = 0.22, 0.30
    inset_x0 = min(main_pos.x1 + 0.02, 1.0 - inset_width - 0.01)
    inset_y0 = main_pos.y0 + (main_pos.height - inset_height) / 2
    inset_ax = fig.add_axes([inset_x0, inset_y0, inset_width, inset_height])

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
    inset_ax.set_title("Location in the PJM footprint", fontsize=9)
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
