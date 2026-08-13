"""
Phase 2 of ISLANDING_MITIGATION_PLAN.md: Substation Criticality Index (SCI).

Combines three signals, each already computed by an earlier pipeline step,
into one ranked score per substation:

    SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i) + w_t * norm(betweenness_i)

- GIC_risk_i        <- va_gic_summary.csv           (scripts/extract_virginia_results.py)
- econ_exposure_i   <- va_substation_exposure.gpkg  (scripts/run_virginia_areal_interpolation.py)
- betweenness_i     <- va_topology_metrics.csv      (scripts/build_virginia_network_graph.py)

Only substations with in_main_component=True in va_topology_metrics.csv are
eligible -- see ISLANDING_MITIGATION_PLAN.md's Phase 1 results for why (11
Virginia substations sit outside the main connected component and can't be
meaningfully ranked or islanded against a network they aren't part of here).

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

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR

logger = setup_logger(log_file="logs/compute_virginia_criticality.log")

# Default equal weighting. Worth testing alternative weight sets once the
# pipeline runs end-to-end -- report how sensitive the ranking is, don't
# just present one weight choice as ground truth.
WEIGHTS = {"gic_risk": 1 / 3, "econ_exposure": 1 / 3, "betweenness": 1 / 3}

RANK_COL = "100-year-hazard A/ph_mean"  # same column plot_virginia_gic_risk.py ranks on

# Same sector taxonomy as run_virginia_areal_interpolation.py's GROUP_KEYS.
GDP_SECTOR_KEYS = (
    "AGR", "MINING", "UTIL_CONST", "MANUF", "TRADE_TRANSP",
    "INFO", "FIRE", "PROF_OTHER", "EDUC_ENT", "G",
)


def load_topology_metrics() -> pd.DataFrame:
    """sub_id, betweenness, in_main_component -- from Phase 1's output."""
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", "betweenness", "in_main_component"]]


def load_gic_risk() -> pd.DataFrame:
    """sub_id, gic_risk -- 100yr mean GIC from va_gic_summary.csv."""
    path = VA_ECON_DIR / "va_gic_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/extract_virginia_results.py first.")
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


def compute_sci(topology: pd.DataFrame, gic: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    eligible = topology[topology["in_main_component"]].copy()
    excluded = len(topology) - len(eligible)
    if excluded:
        logger.info(f"Excluding {excluded} substations outside the main connected component from SCI ranking.")

    merged = eligible.merge(gic, on="sub_id", how="left").merge(econ, on="sub_id", how="left")

    missing_gic = merged["gic_risk"].isna().sum()
    missing_econ = merged[["gdp_total", "population"]].isna().any(axis=1).sum()
    if missing_gic:
        logger.warning(
            f"{missing_gic} eligible substations have no GIC risk data -- check sub_id "
            "matching between va_topology_metrics.csv and va_gic_summary.csv."
        )
    if missing_econ:
        logger.warning(
            f"{missing_econ} eligible substations have no economic exposure data -- check "
            "sub_id matching with va_substation_exposure.gpkg."
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
