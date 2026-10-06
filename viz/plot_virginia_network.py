"""
Visualization of the VA+PJM EHV network built by
scripts/build_virginia_network.py — a fast, no-Monte-Carlo sanity check of
topology before running the expensive storm-recompute/GIC steps, and the
source for the paper's Figure 3.

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
import matplotlib.pyplot as plt
import geopandas as gpd

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import (
    VA_NETWORK_DIR,
    VA_FIGURES_DIR,
    get_pjm_geodataframe,
    get_virginia_boundary,
    _load_states,
)

logger = setup_logger(log_file="logs/plot_virginia_network.log")

VOLTAGE_COLORS = {765: "#d73027", 500: "#fc8d59", 345: "#fee090", 230: "#91bfdb", 161: "#4575b4"}


def main():
    setup_matplotlib()

    lines_path = VA_NETWORK_DIR / "va_pjm_lines.geojson"
    subs_path = VA_NETWORK_DIR / "va_pjm_substations.geojson"
    if not lines_path.exists() or not subs_path.exists():
        raise FileNotFoundError(
            f"Missing {lines_path} or {subs_path}. Run "
            "scripts/build_virginia_network.py first."
        )

    lines = gpd.read_file(lines_path)
    subs = gpd.read_file(subs_path)

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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    logger.info(f"Lines: {len(lines)}, Substations: {len(subs)}")
    for voltage in sorted(VOLTAGE_COLORS, reverse=True):
        n = (lines["VOLTAGE"] == voltage).sum()
        if n:
            logger.info(f"  {voltage} kV: {n} lines")


if __name__ == "__main__":
    main()
