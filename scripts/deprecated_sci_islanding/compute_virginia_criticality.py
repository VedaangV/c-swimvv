"""
Substation Criticality Index (SCI): combines GIC risk, economic exposure,
and betweenness centrality (and optionally critical-load count) into one
ranked score per substation.

    SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i) + w_t * norm(betweenness_i)

Restricted to Virginia substations in the main connected component only --
GIC/econ data upstream is already Virginia-only, so scoring the rest of
PJM would silently read as "confirmed zero risk" instead of "never modeled".

Economic exposure and critical-load count use density (per km2 of a
substation's Voronoi cell), not raw totals -- a border substation can get
an oversized, poorly-bounded Voronoi cell (areal interpolation's known
limitation) that absorbs way more raw population/facilities than it's
actually near. Dividing by cell area fixes that.

Usage:
    python scripts/compute_virginia_criticality.py
"""

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR
from scripts.virginia_risk_data import (
    load_topology_metrics,
    load_gic_risk,
    load_economic_exposure,
    load_critical_facility_counts,
    normalize_min_max,
    restrict_to_virginia,
)

logger = setup_logger(log_file="logs/compute_virginia_criticality.log")

# Equal weighting by default; compute_sci() picks 3-term or 4-term
# depending on whether critical-load data is available. Worth sensitivity-
# testing other weights once the pipeline runs end-to-end.
WEIGHTS_3 = {"gic_risk": 1 / 3, "econ_exposure": 1 / 3, "betweenness": 1 / 3}
WEIGHTS_4 = {"gic_risk": 1 / 4, "econ_exposure": 1 / 4, "betweenness": 1 / 4, "critical_load": 1 / 4}


def compute_sci(
    topology: pd.DataFrame,
    gic: pd.DataFrame,
    econ: pd.DataFrame,
    critical_load: pd.DataFrame | None = None,
) -> pd.DataFrame:
    topology = restrict_to_virginia(topology)

    eligible = topology[topology["in_main_component"]].copy()
    excluded = len(topology) - len(eligible)
    if excluded:
        logger.info(
            f"Excluding {excluded} Virginia substations outside the main connected "
            "component from SCI ranking."
        )

    merged = eligible.merge(gic, on="sub_id", how="left").merge(econ, on="sub_id", how="left")

    missing_gic = merged["gic_risk"].isna().sum()
    missing_econ = merged[["gdp_total", "population", "area_km2"]].isna().any(axis=1).sum()
    if missing_gic:
        logger.warning(
            f"{missing_gic} eligible Virginia substations have no GIC risk data -- "
            "unexpected given Virginia-only scoping, check sub_id matching between "
            "va_topology_metrics.csv and va_gic_summary.csv."
        )
    if missing_econ:
        logger.warning(
            f"{missing_econ} eligible Virginia substations have no economic exposure data -- "
            "unexpected given Virginia-only scoping, check sub_id matching with "
            "va_substation_exposure.gpkg."
        )
    merged["gic_risk"] = merged["gic_risk"].fillna(0)
    merged["gdp_total"] = merged["gdp_total"].fillna(0)
    merged["population"] = merged["population"].fillna(0)
    # area_km2 is NOT filled to 0 -- dividing by zero would produce inf/NaN
    # and corrupt normalize_min_max's min/max. Density stays 0 for rows
    # without a valid positive area, but area_km2 itself stays NaN in the
    # output so it's visible why a substation scored 0 here.
    valid_area = merged["area_km2"] > 0

    merged["gdp_density"] = 0.0
    merged["population_density"] = 0.0
    merged.loc[valid_area, "gdp_density"] = (
        merged.loc[valid_area, "gdp_total"] / merged.loc[valid_area, "area_km2"]
    )
    merged.loc[valid_area, "population_density"] = (
        merged.loc[valid_area, "population"] / merged.loc[valid_area, "area_km2"]
    )

    merged["gic_risk_norm"] = normalize_min_max(merged["gic_risk"])
    merged["gdp_norm"] = normalize_min_max(merged["gdp_density"])
    merged["population_norm"] = normalize_min_max(merged["population_density"])
    merged["econ_exposure_norm"] = (merged["gdp_norm"] + merged["population_norm"]) / 2
    merged["betweenness_norm"] = normalize_min_max(merged["betweenness"])

    if critical_load is not None:
        merged = merged.merge(critical_load, on="sub_id", how="left")
        merged["critical_load_count"] = merged["critical_load_count"].fillna(0)
        merged["critical_load_density"] = 0.0
        merged.loc[valid_area, "critical_load_density"] = (
            merged.loc[valid_area, "critical_load_count"] / merged.loc[valid_area, "area_km2"]
        )
        merged["critical_load_norm"] = normalize_min_max(merged["critical_load_density"])
        weights = WEIGHTS_4
        merged["SCI"] = (
            weights["gic_risk"] * merged["gic_risk_norm"]
            + weights["econ_exposure"] * merged["econ_exposure_norm"]
            + weights["betweenness"] * merged["betweenness_norm"]
            + weights["critical_load"] * merged["critical_load_norm"]
        )
        logger.info(f"SCI computed with 4 terms (weights: {weights}).")
        out_cols = [
            "sub_id", "gic_risk", "area_km2", "gdp_total", "population", "gdp_density", "population_density",
            "betweenness", "critical_load_count", "critical_load_density",
            "gic_risk_norm", "econ_exposure_norm", "betweenness_norm", "critical_load_norm", "SCI",
        ]
    else:
        weights = WEIGHTS_3
        merged["SCI"] = (
            weights["gic_risk"] * merged["gic_risk_norm"]
            + weights["econ_exposure"] * merged["econ_exposure_norm"]
            + weights["betweenness"] * merged["betweenness_norm"]
        )
        logger.info(f"SCI computed with 3 terms (weights: {weights}); critical_load_i not included.")
        out_cols = [
            "sub_id", "gic_risk", "area_km2", "gdp_total", "population", "gdp_density", "population_density",
            "betweenness", "gic_risk_norm", "econ_exposure_norm", "betweenness_norm", "SCI",
        ]

    return merged[out_cols]


def main():
    topology = load_topology_metrics()
    gic = load_gic_risk()
    econ = load_economic_exposure()
    critical_load = load_critical_facility_counts()

    sci = compute_sci(topology, gic, econ, critical_load)
    sci = sci.sort_values("SCI", ascending=False).reset_index(drop=True)

    out_path = VA_ECON_DIR / "va_substation_criticality.csv"
    sci.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")
    logger.info(f"Top 10 by SCI:\n{sci.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
