"""
Visualize Virginia substations ranked by Substation Criticality Index (SCI)
from scripts/compute_virginia_criticality.py's output.

Produces a two-panel figure, styled consistently with
viz/plot_virginia_gic_risk.py:
  left  — map of all ranked Virginia substations sized/colored by SCI, with
          the VA+PJM network lines as geographic context and the top-
          criticality substations labeled
  right — horizontal bar chart of the top-N substations ranked by SCI

va_substation_criticality.csv doesn't carry latitude/longitude --
compute_sci()'s output is deliberately just sub_id + component scores, not
a geometry file. Coordinates are merged in here from Phase 1's
va_topology_metrics.csv by sub_id instead, rather than adding geometry
columns compute_virginia_criticality.py doesn't otherwise need, or asking
for a re-run of an already-validated computation just to get lat/lon along
for the ride.

Works whether va_substation_criticality.csv was computed with three terms
(GIC risk, economic exposure, betweenness) or four (+ critical_load_i from
Phase 2b) -- detected from whether critical_load_norm is present, and
reflected in the map's title so the figure is honest about what it's
actually showing.

Usage:
    python viz/plot_virginia_criticality.py
"""

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

logger = setup_logger(log_file="logs/plot_virginia_criticality.log")

RANK_COL = "SCI"
TOP_N_LABELED = 10
TOP_N_BAR = 15


def _load_substation_labels() -> dict:
    """sub_id (SS_ID) -> a human-readable label, built from ss_df.pkl.

    Duplicated from viz/plot_virginia_gic_risk.py rather than imported,
    matching this project's existing convention (see virginia_config.py's
    TRAFO_WINDING_R comment) of keeping Virginia-side scripts independent
    of each other rather than cross-importing small helpers.

    Fallback labels (operator/voltage, or fully "Unnamed") append the raw
    sub_id -- without it, two distinct substations that share a voltage
    class and both lack an OSM name tag produce visually near-identical
    labels (e.g. two different "115000 substation (lat, lon)" bars), which
    reads as a duplicate/bug on a chart even though the small-print
    coordinates do differ. A real OSM name is assumed unique enough to
    stand alone.
    """
    ss_df_path = VA_DATA_DIR / "grid_processed" / "ss_df.pkl"
    if not ss_df_path.exists():
        logger.warning(f"{ss_df_path} not found; falling back to raw sub_id labels.")
        return {}

    with open(ss_df_path, "rb") as f:
        ss_df = pickle.load(f)

    labels = {}
    for _, row in ss_df.iterrows():
        ss_id = str(row["SS_ID"])
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


def _load_coordinates() -> pd.DataFrame:
    """sub_id, latitude, longitude -- from Phase 1's va_topology_metrics.csv.
    va_substation_criticality.csv doesn't carry geometry itself."""
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", "latitude", "longitude"]]


def main():
    setup_matplotlib()

    summary_path = VA_ECON_DIR / "va_substation_criticality.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run scripts/compute_virginia_criticality.py first.")

    summary = pd.read_csv(summary_path)
    if RANK_COL not in summary.columns:
        raise ValueError(f"Expected column '{RANK_COL}' in {summary_path}; got {list(summary.columns)}")
    summary["sub_id"] = summary["sub_id"].astype(str)

    coords = _load_coordinates()
    summary = summary.merge(coords, on="sub_id", how="left")
    missing_coords = summary["latitude"].isna().sum()
    if missing_coords:
        logger.warning(
            f"{missing_coords} substations in {summary_path} have no matching "
            "coordinates in va_topology_metrics.csv -- dropped from the map "
            "(sub_id mismatch would be unexpected; investigate if this is nonzero)."
        )
        summary = summary.dropna(subset=["latitude", "longitude"])

    summary = summary.sort_values(RANK_COL, ascending=False).reset_index(drop=True)
    labels = _load_substation_labels()
    summary["display_name"] = summary["sub_id"].map(labels).fillna(summary["sub_id"])

    has_critical_load = "critical_load_norm" in summary.columns
    n_terms = "4-term" if has_critical_load else "3-term"

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
            row["display_name"], (row["longitude"], row["latitude"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7, zorder=5,
        )

    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.04, pad=0.02)
    cbar.set_label("Substation Criticality Index (SCI)")
    ax_map.set_title(f"Virginia substation criticality ({n_terms} SCI) — {len(summary)} substations")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")

    order = top.iloc[::-1]
    y = range(len(order))
    ax_bar.barh(y, order[RANK_COL], color="#4575b4")
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels(order["display_name"], fontsize=8)
    ax_bar.set_xlabel("Substation Criticality Index (SCI)")
    ax_bar.set_title(f"Top {len(top)} most critical VA substations ({n_terms})")

    fig.tight_layout()
    out_path = VA_FIGURES_DIR / "va_criticality_map.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    logger.info(f"Top substation: {summary.iloc[0]['display_name']} (SCI={summary.iloc[0][RANK_COL]:.4f})")


if __name__ == "__main__":
    main()
