"""
Filters Virginia+PJM GIC Monte Carlo outputs down to Virginia, and the
national ZCTA-level socioeconomic dataset down to Virginia ZCTAs.

The GIC simulation itself (run_virginia_gic.py) runs on the full PJM
network -- filtering happens here, after the fact, for reporting only.

Note: hooking Virginia's GIC results into the full IO/CGE econ run would
still require rebuilding p_areal_intp.py's ZCTA-to-substation Voronoi
mapping against Virginia's own substation_info.csv (it's currently built
from the national one) and deciding national vs regional SAM/technology
table scope -- a methodology question, not done here. This script stops at
producing Virginia-scoped GIC summaries and socioeconomic_data.pkl.
"""

import pickle
from glob import glob

import pandas as pd
from shapely.geometry import Point

from configs import setup_logger, get_data_dir, ECON_DATA_DIR
from configs.virginia_config import (
    VA_ADMITTANCE_DIR,
    VA_FIPS,
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


def summarize_and_rank(va_gic: pd.DataFrame) -> None:
    hazard_cols = [c for c in va_gic.columns if c.endswith("-year-hazard A/ph")]
    if not hazard_cols:
        logger.warning("No hazard columns found in Virginia GIC results.")
        return

    # Winding GIC is signed (see est_gic.py's calc_trafo_gic()) and its sign
    # depends on storm field direction, not current magnitude. Averaging
    # signed values can cancel out real risk, so rank on |GIC| instead.
    va_gic_abs = va_gic.copy()
    va_gic_abs[hazard_cols] = va_gic_abs[hazard_cols].abs()

    grouped = va_gic_abs.groupby("sub_id")[hazard_cols]
    summary = grouped.mean().add_suffix("_mean")
    summary = summary.join(grouped.quantile(0.95).add_suffix("_p95"))
    meta = va_gic.groupby("sub_id")[["latitude", "longitude"]].first()
    summary = summary.join(meta).reset_index()
    summary.to_csv(VA_ECON_DIR / "va_gic_summary.csv", index=False)
    logger.info(f"Substation-level VA GIC summary: {len(summary)} substations")
    logger.info(
        "Per-transformer-winding detail only -- va_top_substations.csv (the paper's "
        "reported ranking) now comes from summarize_ground_gic(), not this function; "
        "see that function's docstring for why."
    )


def filter_ground_gic(sub_lookup: pd.DataFrame) -> pd.DataFrame:
    files = sorted(glob(str(VA_GND_GIC_DIR / "ground_gic*.csv")))
    if not files:
        logger.warning(f"No ground_gic*.csv under {VA_GND_GIC_DIR}; skipping.")
        return pd.DataFrame()

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
    return va_ground_gic


def summarize_ground_gic(
    va_ground_gic: pd.DataFrame, sub_lookup: pd.DataFrame, top_n: int = 25
) -> None:
    """sub_id + mean/p95 ground-GIC magnitude per storm scenario, from
    va_ground_gic_results.csv -- the same quantity (current to earth via
    the grounding connection, see est_gic.py's solve_total_nodal_gic())
    that run_virginia_fragility.py's Monte Carlo uses for the headline
    impact results. Using it here too keeps va_top_substations.csv,
    plot_virginia_gic_risk.py, and the SCI agreeing on what "GIC risk"
    means, rather than the separate per-winding quantity summarize_and_rank()
    reports.

    Signed for the same reason winding GIC is, so magnitude is taken here
    too.

    Units: solve_total_nodal_gic() multiplies by 3 for all three phases,
    so this is total current ("A"), not "A/ph" like winding GIC's columns
    -- label it accordingly.

    va_ground_gic_results.csv has no lat/lon of its own, so sub_lookup
    (loaded in main() from substation_info.csv) supplies it.
    """
    gic_cols = [c for c in va_ground_gic.columns if c.startswith("GIC_")]
    if not gic_cols:
        logger.warning("No GIC_* columns found in Virginia ground-GIC results.")
        return

    va_ground_gic_abs = va_ground_gic.copy()
    va_ground_gic_abs[gic_cols] = va_ground_gic_abs[gic_cols].abs()

    grouped = va_ground_gic_abs.groupby("Substation")[gic_cols]
    summary = grouped.mean().add_suffix("_mean")
    summary = summary.join(grouped.quantile(0.95).add_suffix("_p95"))
    summary = summary.reset_index().rename(columns={"Substation": "sub_id"})

    coords = sub_lookup[["name", "latitude", "longitude"]].rename(columns={"name": "sub_id"})
    summary = summary.merge(coords, on="sub_id", how="left")
    missing_coords = summary["latitude"].isna().sum()
    if missing_coords:
        logger.warning(
            f"{missing_coords} substations in the ground-GIC summary have no matching "
            "coordinates in substation_info.csv -- check sub_id/name matching."
        )

    summary.to_csv(VA_ECON_DIR / "va_ground_gic_summary.csv", index=False)
    logger.info(f"Substation-level VA ground-GIC summary: {len(summary)} substations")

    rank_col = "GIC_100_mean"
    if rank_col in summary.columns:
        top = summary.sort_values(rank_col, ascending=False).head(top_n)
        top.to_csv(VA_ECON_DIR / "va_top_substations.csv", index=False)
        logger.info(f"Top {len(top)} most vulnerable VA substations -> va_top_substations.csv")
    else:
        logger.warning(f"'{rank_col}' not present; skipping vulnerability ranking.")


def filter_socioeconomic_data(va_boundary) -> None:
    src = ECON_DATA_DIR / "processed_econ" / "socioeconomic_data.pkl"
    if not src.exists():
        logger.warning(f"{src} not found; skipping econ ZCTA filtering.")
        return

    with open(src, "rb") as f:
        naics_est_gdp, zcta_pop_20, regions_pop_df, zcta_business_gdf, states_gdf = pickle.load(f)

    # None of these tables carries a state abbreviation directly, but
    # naics_est_gdp/regions_pop_df have a county FIPS ("GeoFips") whose first
    # two digits are the state FIPS -- use that to get the Virginia ZCTA
    # set, then filter everything else by ZCTA membership.
    if "GeoFips" not in regions_pop_df.columns:
        raise ValueError(
            "Expected a GeoFips column on regions_pop_df to derive Virginia ZCTAs "
            f"from county FIPS; got columns {list(regions_pop_df.columns)}."
        )

    geo_fips = regions_pop_df["GeoFips"].astype(str).str.zfill(5)
    va_zctas = set(regions_pop_df.loc[geo_fips.str[:2] == VA_FIPS, "ZCTA"])
    logger.info(f"Virginia ZCTAs: {len(va_zctas)}")

    va_naics_est_gdp = naics_est_gdp[naics_est_gdp["ZCTA"].isin(va_zctas)].copy()
    va_zcta_pop_20 = zcta_pop_20[zcta_pop_20["ZCTA"].isin(va_zctas)].copy()
    va_regions_pop_df = regions_pop_df[regions_pop_df["ZCTA"].isin(va_zctas)].copy()
    va_zcta_business_gdf = zcta_business_gdf[zcta_business_gdf["ZCTA"].isin(va_zctas)].copy()

    if "STUSPS" in states_gdf.columns:
        va_states_gdf = states_gdf[states_gdf["STUSPS"] == "VA"].copy()
    elif "STATEFP" in states_gdf.columns:
        va_states_gdf = states_gdf[states_gdf["STATEFP"] == VA_FIPS].copy()
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
    va_ground_gic = filter_ground_gic(sub_lookup)
    if not va_ground_gic.empty:
        summarize_ground_gic(va_ground_gic, sub_lookup)

    filter_socioeconomic_data(va_boundary)


if __name__ == "__main__":
    main()
