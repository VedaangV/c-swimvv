"""
Visualizes Virginia's GIC vulnerability ranking from
scripts/extract_virginia_results.py's output: a map of substations
sized/colored by mean 100-year ground GIC (with top substations
labeled), plus a horizontal bar chart of the top-N ranked substations.

Ranks on *ground* GIC (va_ground_gic_summary.csv), not winding GIC --
ground GIC is what the fragility Monte Carlo uses for the headline
economic results, and is already a single per-substation scalar. Units
are "A" (combined across all three phases), not winding GIC's "A/ph" --
don't copy that label over if editing this file.

`sub_id` is the raw OSM-derived SS_ID, not a human-readable name (most
OSM substations here have no name tag). This script joins in a real
name where available (ss_df.pkl's SS_NAME), falling back to an
operator/voltage/location label instead of a bare ID.

Usage:
    python viz/plot_virginia_gic_risk.py
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
    get_virginia_counties,
)

logger = setup_logger(log_file="logs/plot_virginia_gic_risk.log")

RANK_COL = "GIC_100_mean"
RANK_COL_P95 = "GIC_100_p95"
TOP_N_LABELED = 10
TOP_N_BAR = 15

# Manual stand-ins for substations that fall back to bare coordinates
# after every automatic naming tier. Keyed on the exact "(lat, lon)"
# string the fallback produces. These are APPROXIMATE placements, not a
# verified geocode -- replace with real names when available. Shared by
# every figure calling _load_substation_labels(), so the stand-in name
# stays consistent everywhere a substation shows up.
COORD_NAME_OVERRIDES = {
    "(37.25, -77.61)": "Chesterfield County",
    "(37.45, -77.63)": "Chesterfield County",
    "(37.59, -77.39)": "New Kent County",
    "(37.78, -77.55)": "Hanover County",
    "(36.86, -81.42)": "Smyth County (AEP)",
    "(37.25, -77.18)": "Prince George County",
}


def _load_substation_labels(include_coords: bool = False) -> dict:
    """sub_id (SS_ID) -> a human-readable label.

    Priority: (1) external name (va_substation_names.csv, from HIFLD/
    Wikidata/OSM Overpass); (2) OSM's own name tag; (3) county/city
    reverse-geocode against TIGER polygons; (4) no name, if even that
    fails. Each tier is optional and falls through to the next.

    include_coords=True appends "(lat, lon)" to every label -- useful for
    a chart like plot_virginia_winding_gic_ranking.py where the label is
    the only place a reader sees location. Left False here since this
    module's own map already shows the point directly.
    """
    external_names = {}
    names_path = VA_NETWORK_DIR / "va_substation_names.csv"
    if names_path.exists():
        names_df = pd.read_csv(names_path)
        names_df["sub_id"] = names_df["sub_id"].astype(str)
        external_names = dict(zip(names_df["sub_id"], names_df["name"]))
        logger.info(f"Loaded {len(external_names)} external substation names from {names_path}.")

    ss_df_path = VA_DATA_DIR / "grid_processed" / "ss_df.pkl"
    if not ss_df_path.exists():
        logger.warning(f"{ss_df_path} not found; falling back to external names (if any) or raw sub_id labels.")
        return {sid: str(name).strip() for sid, name in external_names.items() if pd.notna(name)}

    with open(ss_df_path, "rb") as f:
        ss_df = pickle.load(f)

    labels = {}
    unnamed = []  # (ss_id, lat, lon); resolved in one batched sjoin below
    for _, row in ss_df.iterrows():
        ss_id = str(row["SS_ID"])
        coord_suffix = f" ({row['lat']:.2f}, {row['lon']:.2f})" if include_coords else ""

        if ss_id in external_names and pd.notna(external_names[ss_id]):
            labels[ss_id] = str(external_names[ss_id]).strip() + coord_suffix
            continue

        name = row.get("SS_NAME")
        if pd.notna(name) and str(name).strip() not in ("", "Unknown"):
            labels[ss_id] = str(name).strip() + coord_suffix
            continue

        unnamed.append((ss_id, row["lat"], row["lon"]))

    county_names = {}
    if unnamed:
        counties = get_virginia_counties()
        if counties is None:
            logger.warning(
                f"{len(unnamed)} unnamed substation(s) falling back to coordinates alone "
                "-- tl_2022_us_county.zip not found (run econ/preprocess/fetch_bea_census.py's "
                "fetch_tiger_county() to enable county/city reverse geocoding)."
            )
        else:
            points = gpd.GeoDataFrame(
                {"ss_id": [sid for sid, _, _ in unnamed]},
                geometry=gpd.points_from_xy([lon for _, _, lon in unnamed], [lat for _, lat, _ in unnamed]),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin(points, counties[["NAMELSAD", "geometry"]], how="left", predicate="within")
            county_names = dict(zip(joined["ss_id"], joined["NAMELSAD"]))
            n_matched = joined["NAMELSAD"].notna().sum()
            logger.info(f"Reverse-geocoded {n_matched}/{len(unnamed)} unnamed substations to a county/city.")

    for ss_id, lat, lon in unnamed:
        county = county_names.get(ss_id)
        # coord_str doubles as the override-dict key and the last-resort
        # label; only appended to named labels when include_coords=True.
        coord_str = f"({lat:.2f}, {lon:.2f})"
        coord_suffix = f" {coord_str}" if include_coords else ""
        if pd.notna(county):
            labels[ss_id] = f"{county}{coord_suffix}"
        elif coord_str in COORD_NAME_OVERRIDES:
            labels[ss_id] = f"{COORD_NAME_OVERRIDES[coord_str]}{coord_suffix}"
        else:
            # no name, no county, no override -- coords are all we've got
            labels[ss_id] = coord_str

    return labels


def main():
    setup_matplotlib()

    summary_path = VA_ECON_DIR / "va_ground_gic_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run scripts/extract_virginia_results.py first.")

    summary = pd.read_csv(summary_path)
    if RANK_COL not in summary.columns:
        raise ValueError(f"Expected column '{RANK_COL}' in {summary_path}; got {list(summary.columns)}")

    summary = summary.sort_values(RANK_COL, ascending=False).reset_index(drop=True)
    summary["sub_id"] = summary["sub_id"].astype(str)
    labels = _load_substation_labels()
    summary["display_name"] = summary["sub_id"].map(labels).fillna(summary["sub_id"])

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
    cbar.set_label("Mean 100-year ground GIC (A)")
    ax_map.set_title(f"Virginia substation GIC risk — {len(summary)} substations")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")

    order = top.iloc[::-1]
    y = range(len(order))
    errs = (order[RANK_COL_P95] - order[RANK_COL]).clip(lower=0) if RANK_COL_P95 in order.columns else None
    ax_bar.barh(y, order[RANK_COL], xerr=errs, color="#d73027", ecolor="0.3", capsize=3)
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels(order["display_name"], fontsize=8)
    ax_bar.set_xlabel("Mean 100-year ground GIC (A), with p95 whisker")
    ax_bar.set_title(f"Top {len(top)} most vulnerable VA substations")

    fig.tight_layout()
    out_path = VA_FIGURES_DIR / "va_gic_risk.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    logger.info(f"Top substation: {summary.iloc[0]['display_name']} ({summary.iloc[0][RANK_COL]:.2f} A mean)")


if __name__ == "__main__":
    main()
