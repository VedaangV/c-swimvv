"""
Filter Virginia+PJM GIC Monte Carlo outputs down to Virginia, and filter the
national ZCTA-level socioeconomic dataset down to Virginia ZCTAs.

The GIC simulation (scripts/run_virginia_gic.py) is correct only because it
ran on the full PJM network — filtering happens here, after the fact, purely
for reporting.

Scope note on the economic model itself: econ/preprocess/p_gic_files.py
converts ground_gic*.csv into the "effective GIC" files econ/scripts/
econ_analysis.py consumes, but that conversion (and econ_analysis.py itself)
also depends on econ/preprocess/p_areal_intp.py's Voronoi-polygon mapping of
ZCTAs to the nearest EHV substation — built from
data/admittance_matrix/substation_info.csv, i.e. the *original US* network,
not Virginia's. Hooking Virginia's GIC results into the full IO/CGE run
would mean rebuilding that Voronoi mapping against
data/virginia/admittance_matrix/substation_info.csv and deciding whether the
downstream SAM/technology tables should be applied at national or regional
scope — a methodology question for Dennies, not a mechanical path swap. This
script does not attempt it; it stops at producing Virginia-scoped GIC
summaries and a Virginia-scoped socioeconomic_data.pkl, which is as far as
the original plan's Step 6 specifies concretely.
"""

import pickle
from glob import glob

import pandas as pd
from shapely.geometry import Point

from configs import setup_logger, get_data_dir, ECON_DATA_DIR
from configs.virginia_config import (
    VA_ADMITTANCE_DIR,
    VA_GIC_DIR,
    VA_GND_GIC_DIR,
    VA_ECON_DIR,
    get_virginia_boundary,
)

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/extract_virginia_results.log")


def _load_va_substation_lookup(va_boundary) -> pd.DataFrame:
    """substation name -> (lat, lon, in_virginia) from the VA+PJM admittance build."""
    sub_info = pd.read_csv(VA_ADMITTANCE_DIR / "substation_info.csv")
    sub_info["name"] = sub_info["name"].astype(str)
    sub_info["in_virginia"] = sub_info.apply(
        lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1
    )
    return sub_info


def filter_winding_gic(va_boundary) -> pd.DataFrame:
    files = sorted(glob(str(VA_GIC_DIR / "winding_gic_rand_*.csv")))
    if not files:
        raise FileNotFoundError(f"No winding_gic_rand_*.csv under {VA_GIC_DIR}")

    frames = []
    for fp in files:
        it = int(fp.rsplit("_", 1)[-1].split(".")[0])
        df = pd.read_csv(fp)
        df["sub_id"] = df["sub_id"].astype(str)
        df["in_virginia"] = df.apply(
            lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1
        )
        df = df[df["in_virginia"]].drop(columns=["in_virginia"])
        df["iteration"] = it
        frames.append(df)

    va_gic = pd.concat(frames, ignore_index=True)
    va_gic.to_csv(VA_ECON_DIR / "va_gic_results.csv", index=False)
    logger.info(f"Virginia winding-GIC rows: {len(va_gic)} across {len(files)} iterations")
    return va_gic


def summarize_and_rank(va_gic: pd.DataFrame, top_n: int = 25) -> None:
    hazard_cols = [c for c in va_gic.columns if c.endswith("-year-hazard A/ph")]
    if not hazard_cols:
        logger.warning("No hazard columns found in Virginia GIC results.")
        return

    grouped = va_gic.groupby("sub_id")[hazard_cols]
    summary = grouped.mean().add_suffix("_mean")
    summary = summary.join(grouped.quantile(0.95).add_suffix("_p95"))
    meta = va_gic.groupby("sub_id")[["latitude", "longitude"]].first()
    summary = summary.join(meta).reset_index()
    summary.to_csv(VA_ECON_DIR / "va_gic_summary.csv", index=False)
    logger.info(f"Substation-level VA GIC summary: {len(summary)} substations")

    rank_col = "100-year-hazard A/ph_mean"
    if rank_col in summary.columns:
        top = summary.sort_values(rank_col, ascending=False).head(top_n)
        top.to_csv(VA_ECON_DIR / "va_top_substations.csv", index=False)
        logger.info(f"Top {len(top)} most vulnerable VA substations -> va_top_substations.csv")
    else:
        logger.warning(f"'{rank_col}' not present; skipping vulnerability ranking.")


def filter_ground_gic(sub_lookup: pd.DataFrame) -> None:
    files = sorted(glob(str(VA_GND_GIC_DIR / "ground_gic*.csv")))
    if not files:
        logger.warning(f"No ground_gic*.csv under {VA_GND_GIC_DIR}; skipping.")
        return

    va_names = set(sub_lookup.loc[sub_lookup["in_virginia"], "name"])
    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df["Substation"] = df["Substation"].astype(str)
        df = df[df["Substation"].isin(va_names)]
        frames.append(df)

    va_ground_gic = pd.concat(frames, ignore_index=True)
    va_ground_gic.to_csv(VA_ECON_DIR / "va_ground_gic_results.csv", index=False)
    logger.info(f"Virginia ground-GIC rows: {len(va_ground_gic)} across {len(files)} files")


def filter_socioeconomic_data(va_boundary) -> None:
    src = ECON_DATA_DIR / "processed_econ" / "socioeconomic_data.pkl"
    if not src.exists():
        logger.warning(f"{src} not found; skipping econ ZCTA filtering.")
        return

    with open(src, "rb") as f:
        naics_est_gdp, zcta_pop_20, regions_pop_df, zcta_business_gdf, states_gdf = pickle.load(f)

    if "STABBR" not in zcta_pop_20.columns:
        raise ValueError(
            "Expected a STABBR column on zcta_pop_20 (econ/preprocess/p_econ_data.py's "
            "create_zcta_population_csv output) to identify Virginia ZCTAs."
        )

    va_zctas = set(zcta_pop_20.loc[zcta_pop_20["STABBR"] == "VA", "ZCTA"])
    logger.info(f"Virginia ZCTAs: {len(va_zctas)}")

    va_naics_est_gdp = naics_est_gdp[naics_est_gdp["ZCTA"].isin(va_zctas)].copy()
    va_zcta_pop_20 = zcta_pop_20[zcta_pop_20["ZCTA"].isin(va_zctas)].copy()
    va_regions_pop_df = regions_pop_df[regions_pop_df["ZCTA"].isin(va_zctas)].copy()
    va_zcta_business_gdf = zcta_business_gdf[zcta_business_gdf["ZCTA"].isin(va_zctas)].copy()

    if "STUSPS" in states_gdf.columns:
        va_states_gdf = states_gdf[states_gdf["STUSPS"] == "VA"].copy()
    elif "STATEFP" in states_gdf.columns:
        va_states_gdf = states_gdf[states_gdf["STATEFP"] == "51"].copy()
    else:
        logger.warning("Could not identify VA row in states_gdf by STUSPS/STATEFP; keeping all.")
        va_states_gdf = states_gdf

    out = VA_ECON_DIR / "va_socioeconomic_data.pkl"
    with open(out, "wb") as f:
        pickle.dump(
            (va_naics_est_gdp, va_zcta_pop_20, va_regions_pop_df, va_zcta_business_gdf, va_states_gdf),
            f,
        )
    logger.info(
        f"Saved Virginia socioeconomic data: {len(va_zcta_pop_20)} ZCTAs, "
        f"{len(va_naics_est_gdp)} sector rows -> {out}"
    )


def main():
    va_boundary = get_virginia_boundary()

    va_gic = filter_winding_gic(va_boundary)
    summarize_and_rank(va_gic)

    sub_lookup = _load_va_substation_lookup(va_boundary)
    filter_ground_gic(sub_lookup)

    filter_socioeconomic_data(va_boundary)


if __name__ == "__main__":
    main()
