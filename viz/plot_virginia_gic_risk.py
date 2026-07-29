"""
Visualize Virginia's GIC vulnerability ranking from
scripts/extract_virginia_results.py's output.

Produces a two-panel figure:
  left  — map of all Virginia substations sized/colored by mean 100-year
          GIC, with the VA+PJM network lines as geographic context and the
          top-vulnerability substations labeled
  right — horizontal bar chart of the top-N substations ranked by
          100-year mean GIC, with 95th-percentile as an error bar

Usage:
    python viz/plot_virginia_gic_risk.py
"""

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import (
    VA_ECON_DIR,
    VA_NETWORK_DIR,
    VA_FIGURES_DIR,
    get_virginia_boundary,
)

logger = setup_logger(log_file="logs/plot_virginia_gic_risk.log")

RANK_COL = "100-year-hazard A/ph_mean"
RANK_COL_P95 = "100-year-hazard A/ph_p95"
TOP_N_LABELED = 10
TOP_N_BAR = 15


def main():
    setup_matplotlib()

    summary_path = VA_ECON_DIR / "va_gic_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run scripts/extract_virginia_results.py first.")

    summary = pd.read_csv(summary_path)
    if RANK_COL not in summary.columns:
        raise ValueError(f"Expected column '{RANK_COL}' in {summary_path}; got {list(summary.columns)}")

    summary = summary.sort_values(RANK_COL, ascending=False).reset_index(drop=True)
    top = summary.head(TOP_N_BAR)
    labeled = summary.head(TOP_N_LABELED)

    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.3, 1]})

    va_boundary = get_virginia_boundary()
    gpd.GeoSeries([va_boundary], crs="EPSG:4326").plot(
        ax=ax_map, color="#f5f5f5", edgecolor="black", linewidth=1.2, zorder=1
    )

    lines_path = VA_NETWORK_DIR / "va_pjm_lines.geojson"
    if lines_path.exists():
        lines = gpd.read_file(lines_path).clip(va_boundary)
        if len(lines):
            lines.plot(ax=ax_map, color="0.75", linewidth=0.5, zorder=2)

    sizes = 20 + 180 * (summary[RANK_COL] / summary[RANK_COL].max()).clip(lower=0)
    sc = ax_map.scatter(
        summary["longitude"], summary["latitude"],
        c=summary[RANK_COL], cmap="YlOrRd", s=sizes,
        edgecolor="0.3", linewidth=0.3, zorder=3,
    )
    ax_map.scatter(
        labeled["longitude"], labeled["latitude"],
        facecolor="none", edgecolor="black", linewidth=1.3, s=260, zorder=4,
    )
    for _, row in labeled.iterrows():
        ax_map.annotate(
            str(row["sub_id"]), (row["longitude"], row["latitude"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7, zorder=5,
        )

    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.04, pad=0.02)
    cbar.set_label("Mean 100-year GIC (A/ph)")
    ax_map.set_title(f"Virginia substation GIC risk — {len(summary)} substations")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")

    order = top.iloc[::-1]
    y = range(len(order))
    errs = (order[RANK_COL_P95] - order[RANK_COL]).clip(lower=0) if RANK_COL_P95 in order.columns else None
    ax_bar.barh(y, order[RANK_COL], xerr=errs, color="#d73027", ecolor="0.3", capsize=3)
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels(order["sub_id"].astype(str), fontsize=8)
    ax_bar.set_xlabel("Mean 100-year GIC (A/ph), with p95 whisker")
    ax_bar.set_title(f"Top {len(top)} most vulnerable VA substations")

    fig.tight_layout()
    out_path = VA_FIGURES_DIR / "va_gic_risk.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    logger.info(f"Top substation: {summary.iloc[0]['sub_id']} ({summary.iloc[0][RANK_COL]:.2f} A/ph mean)")


if __name__ == "__main__":
    main()
