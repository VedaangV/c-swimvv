"""
Phase 2 of ISLANDING_MITIGATION_PLAN.md: Substation Criticality Index (SCI).

Combines three signals, each already computed by an earlier pipeline step,
into one ranked score per substation:

    SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i) + w_t * norm(betweenness_i)

- GIC_risk_i        <- va_ground_gic_summary.csv    (scripts/extract_virginia_results.py)
- econ_exposure_i   <- va_substation_exposure.gpkg  (scripts/run_virginia_areal_interpolation.py)
- betweenness_i     <- va_topology_metrics.csv      (scripts/build_virginia_network_graph.py)

GIC risk uses *ground* GIC (current to earth through the substation's
grounding connection), not winding GIC (current through one transformer
winding) -- ground GIC is the same physical quantity
scripts/run_virginia_fragility.py's fragility Monte Carlo already uses for
the headline economic-impact results, and it's already a single per-
substation scalar rather than something assembled by combining multiple
transformers'/windings' separately-signed values. See extract_virginia_
results.py's summarize_ground_gic() for the full reasoning.

Eligibility requires both in_main_component=True (see ISLANDING_MITIGATION_
PLAN.md's Phase 1 results -- 11 Virginia substations sit outside the main
connected component and can't be meaningfully ranked or islanded against a
network they aren't part of here) AND the substation being inside Virginia.

The Virginia restriction matters because va_gic_summary.csv and
va_substation_exposure.gpkg are each already filtered to Virginia-only
substations by the scripts that produce them (extract_virginia_results.py,
run_virginia_areal_interpolation.py) -- consistent with this project's
"model on the full PJM network, report on Virginia" convention throughout.
va_topology_metrics.csv, correctly, is NOT pre-filtered (centrality needs
the full PJM network for realistic connectivity). Left un-restricted here,
Phase 2 would try to rank ~2000 non-Virginia PJM substations that were
never in scope for GIC/econ data, and score them 0 on both -- which reads
as "modeled and confirmed zero risk" rather than the true "never modeled,"
silently corrupting the ranking. Restricting to Virginia substations first
using the same lat/lon + get_virginia_boundary() check already used
elsewhere in this pipeline avoids that.

Economic exposure is GDP + population (not establishment counts) -- decided
to keep the exposure term aligned with the $/day and population-affected
figures already reported elsewhere in the paper, rather than adding
establishment counts that are highly correlated with GDP within a sector
and would partly double-weight the same underlying signal.

GDP and population are normalized *separately* before being averaged into
one exposure score, not summed first -- GDP is in dollars (tens of
millions) and population is in raw headcounts (hundreds of thousands); a
pre-normalization sum would let population numerically dominate and the
"economic exposure" term would really just be measuring population.

Usage:
    python scripts/compute_virginia_criticality.py
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/compute_virginia_criticality.log")

# Default equal weighting. Worth testing alternative weight sets once the
# pipeline runs end-to-end -- report how sensitive the ranking is, don't
# just present one weight choice as ground truth.
WEIGHTS = {"gic_risk": 1 / 3, "econ_exposure": 1 / 3, "betweenness": 1 / 3}

RANK_COL = "GIC_100_mean"  # 100yr ground-GIC magnitude, from va_ground_gic_summary.csv

# Same sector taxonomy as run_virginia_areal_interpolation.py's GROUP_KEYS.
GDP_SECTOR_KEYS = (
    "AGR", "MINING", "UTIL_CONST", "MANUF", "TRADE_TRANSP",
    "INFO", "FIRE", "PROF_OTHER", "EDUC_ENT", "G",
)


def load_topology_metrics() -> pd.DataFrame:
    """sub_id, betweenness, in_main_component, latitude, longitude -- from
    Phase 1's output. Keeps lat/lon (unlike the version of this function in
    an earlier revision) so eligibility can be restricted to Virginia
    without re-reading substation_info.csv separately."""
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", "betweenness", "in_main_component", "latitude", "longitude"]]


def load_gic_risk() -> pd.DataFrame:
    """sub_id, gic_risk -- 100yr mean |ground GIC| from
    va_ground_gic_summary.csv (scripts/extract_virginia_results.py's
    summarize_ground_gic()), the same physical quantity the fragility
    Monte Carlo uses. Not va_gic_summary.csv's winding-GIC ranking, which
    is a different per-transformer quantity -- see module docstring."""
    path = VA_ECON_DIR / "va_ground_gic_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/extract_virginia_results.py first "
            "(needs the ground_gic*.csv files scripts/run_virginia_gic.py produces)."
        )
    df = pd.read_csv(path)
    if RANK_COL not in df.columns:
        raise ValueError(f"Expected column '{RANK_COL}' in {path}; got {list(df.columns)}")
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", RANK_COL]].rename(columns={RANK_COL: "gic_risk"})


def load_economic_exposure() -> pd.DataFrame:
    """sub_id, gdp_total, population -- from va_substation_exposure.gpkg.

    Returns gdp_total and population as separate columns, deliberately not
    summed -- see module docstring for why they need independent
    normalization before being combined.
    """
    path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_virginia_areal_interpolation.py first.")
    gdf = gpd.read_file(path)
    gdf["sub_id"] = gdf["sub_id"].astype(str)

    gdp_cols = [f"GDP_{k}" for k in GDP_SECTOR_KEYS if f"GDP_{k}" in gdf.columns]
    if not gdp_cols:
        raise ValueError(f"No GDP_* columns found in {path}; got {list(gdf.columns)}")
    if "POP20" not in gdf.columns:
        raise ValueError(f"Expected POP20 column in {path}; got {list(gdf.columns)}")

    gdf["gdp_total"] = gdf[gdp_cols].sum(axis=1)
    return gdf[["sub_id", "gdp_total", "POP20"]].rename(columns={"POP20": "population"})


def normalize_min_max(series: pd.Series) -> pd.Series:
    """Scale to [0, 1]. If every value is identical, return 0.5 for all of
    them -- "no variation" should read as "this signal has nothing to say
    here," not as everyone scoring lowest on that term."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def restrict_to_virginia(topology: pd.DataFrame) -> pd.DataFrame:
    """va_gic_summary.csv and va_substation_exposure.gpkg only cover
    Virginia substations -- restrict here too, using the same lat/lon +
    boundary check pattern already used in extract_virginia_results.py and
    build_virginia_network_graph.py, so eligibility for SCI ranking matches
    what the other two signals can actually speak to."""
    va_boundary = get_virginia_boundary()
    in_va = topology.apply(
        lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1
    )
    excluded = (~in_va).sum()
    if excluded:
        logger.info(f"Excluding {excluded} non-Virginia substations from SCI ranking.")
    return topology[in_va].copy()


def compute_sci(topology: pd.DataFrame, gic: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
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
    missing_econ = merged[["gdp_total", "population"]].isna().any(axis=1).sum()
    if missing_gic:
        logger.warning(
            f"{missing_gic} eligible Virginia substations have no GIC risk data -- this is "
            "now unexpected (Virginia-only scoping already applied) and worth investigating, "
            "not just noting. Check sub_id matching between va_topology_metrics.csv and "
            "va_gic_summary.csv directly."
        )
    if missing_econ:
        logger.warning(
            f"{missing_econ} eligible Virginia substations have no economic exposure data -- "
            "this is now unexpected (Virginia-only scoping already applied) and worth "
            "investigating, not just noting. Check sub_id matching with "
            "va_substation_exposure.gpkg directly."
        )
    merged[["gic_risk", "gdp_total", "population"]] = merged[
        ["gic_risk", "gdp_total", "population"]
    ].fillna(0)

    merged["gic_risk_norm"] = normalize_min_max(merged["gic_risk"])
    merged["gdp_norm"] = normalize_min_max(merged["gdp_total"])
    merged["population_norm"] = normalize_min_max(merged["population"])
    merged["econ_exposure_norm"] = (merged["gdp_norm"] + merged["population_norm"]) / 2
    merged["betweenness_norm"] = normalize_min_max(merged["betweenness"])

    merged["SCI"] = (
        WEIGHTS["gic_risk"] * merged["gic_risk_norm"]
        + WEIGHTS["econ_exposure"] * merged["econ_exposure_norm"]
        + WEIGHTS["betweenness"] * merged["betweenness_norm"]
    )

    return merged[
        [
            "sub_id", "gic_risk", "gdp_total", "population", "betweenness",
            "gic_risk_norm", "econ_exposure_norm", "betweenness_norm", "SCI",
        ]
    ]


def main():
    topology = load_topology_metrics()
    gic = load_gic_risk()
    econ = load_economic_exposure()

    sci = compute_sci(topology, gic, econ)
    sci = sci.sort_values("SCI", ascending=False).reset_index(drop=True)

    out_path = VA_ECON_DIR / "va_substation_criticality.csv"
    sci.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")
    logger.info(f"Top 10 by SCI:\n{sci.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
