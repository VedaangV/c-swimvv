"""
Quick visualization of the VA+PJM EHV network built by
scripts/build_virginia_network.py — a fast, no-Monte-Carlo sanity check of
topology before running the expensive storm-recompute/GIC steps.

Usage:
    python viz/plot_virginia_network.py
"""

import matplotlib.pyplot as plt
import geopandas as gpd

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import (
    VA_NETWORK_DIR,
    VA_FIGURES_DIR,
    get_pjm_geodataframe,
    get_virginia_boundary,
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

    fig, ax = plt.subplots(figsize=(12, 9))

    pjm_states = get_pjm_geodataframe()
    pjm_states.boundary.plot(ax=ax, color="0.6", linewidth=0.6)

    va_boundary = get_virginia_boundary()
    gpd.GeoSeries([va_boundary], crs="EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.5)

    for voltage, color in VOLTAGE_COLORS.items():
        subset = lines[lines["VOLTAGE"] == voltage]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=1.0, label=f"{voltage} kV", zorder=2)

    subs.plot(ax=ax, color="black", markersize=3, zorder=3)

    ax.set_title(f"VA+PJM EHV network — {len(lines)} lines, {len(subs)} substations")
    ax.legend(loc="lower left", fontsize=8, title="Line voltage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    out_path = VA_FIGURES_DIR / "va_pjm_network.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    logger.info(f"Lines: {len(lines)}, Substations: {len(subs)}")
    for voltage in sorted(VOLTAGE_COLORS, reverse=True):
        n = (lines["VOLTAGE"] == voltage).sum()
        if n:
            logger.info(f"  {voltage} kV: {n} lines")


if __name__ == "__main__":
    main()
