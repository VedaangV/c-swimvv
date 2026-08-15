"""
Bonus check for Phase 3's Island A (ISLANDING_MITIGATION_PLAN.md): does the
island contain any existing generation capacity? Not required for the core
protective-islanding methodology -- Phase 3/4/5 model islanding as a
temporary, planned disconnection during a storm's peak-risk window (see the
plan doc's Phase 5 section and the NZ precedent it's grounded in), which
doesn't require the island to stay powered. This is purely a "does it turn
out Island A already has some capacity" factual data point, worth reporting
if true, not a design assumption anything else depends on.

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS to download EIA-860
first -- this script itself only reads local files, no live fetch, since
EIA's exact current-year download URL couldn't be verified from this
sandbox (unlike the ArcGIS services elsewhere in this project, EIA-860 is
a well-established, stable government survey, but the file naming is
versioned by year and best downloaded by hand rather than guessed blind).

SETUP (one-time):
1. Go to https://www.eia.gov/electricity/data/eia860/ and download the most
   recent year's full data ZIP (labeled something like the current year's
   "Form EIA-860 Detailed Data").
2. Extract it into data/virginia/network/raw_eia860/ -- this script globs
   for "3_1_Generator*.xlsx" (generator-level nameplate capacity/fuel/
   status) and a plant file matching "*Plant*.xlsx" (plant-level lat/lon)
   inside that folder, so exact year-versioned filenames don't matter.

EIA-860 files are known to have a title row above the real header (real
column headers usually start on the second row) and slightly inconsistent
column-name spacing/capitalization across years -- handled defensively
below (tries header=0 then header=1, matches key columns case/whitespace-
insensitively) rather than assuming one exact layout.

Matched to Island A's members by nearest plant location within
SNAP_DISTANCE_M, same tolerance and reasoning as scripts/fetch_virginia_
substation_names.py's HIFLD/Wikidata matching.

Usage:
    python scripts/check_island_generation.py
"""

import json

import geopandas as gpd
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, SNAP_DISTANCE_M

logger = setup_logger(log_file="logs/check_island_generation.log")

RAW_EIA860_DIR = VA_NETWORK_DIR / "raw_eia860"

PLANT_CODE_CANDIDATES = ("plant code", "plantcode")
LAT_CANDIDATES = ("latitude",)
LON_CANDIDATES = ("longitude",)
CAPACITY_CANDIDATES = ("nameplate capacity (mw)", "nameplate capacity", "nameplate capacity(mw)")
FUEL_CANDIDATES = ("energy source 1", "energy source", "fuel type")
STATUS_CANDIDATES = ("status",)
PLANT_NAME_CANDIDATES = ("plant name",)


def _find_col(columns, candidates) -> str | None:
    lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _read_with_header_detection(path, required_candidates) -> pd.DataFrame:
    """EIA-860 sheets commonly have a title row above the real header --
    try header=0 first, fall back to header=1 if none of the required
    columns are found, and raise a clear, columns-listing error rather
    than silently returning garbage if neither works."""
    for header_row in (0, 1):
        df = pd.read_excel(path, header=header_row)
        if any(_find_col(df.columns, cands) for cands in required_candidates):
            return df
    df0 = pd.read_excel(path, header=0)
    raise ValueError(
        f"{path}: couldn't find expected columns with header=0 or header=1. "
        f"Columns seen at header=0: {list(df0.columns)}. Tell me what you see "
        "and I'll fix the column matching."
    )


def load_eia860_plants() -> pd.DataFrame:
    matches = list(RAW_EIA860_DIR.glob("*Plant*.xlsx")) + list(RAW_EIA860_DIR.glob("*plant*.xlsx"))
    if not matches:
        raise FileNotFoundError(
            f"No plant file found in {RAW_EIA860_DIR} (expected something matching "
            "'*Plant*.xlsx'). See module docstring for where to download EIA-860 from."
        )
    path = matches[0]
    df = _read_with_header_detection(path, [PLANT_CODE_CANDIDATES, LAT_CANDIDATES])

    code_col = _find_col(df.columns, PLANT_CODE_CANDIDATES)
    lat_col = _find_col(df.columns, LAT_CANDIDATES)
    lon_col = _find_col(df.columns, LON_CANDIDATES)
    name_col = _find_col(df.columns, PLANT_NAME_CANDIDATES)
    if not (code_col and lat_col and lon_col):
        raise ValueError(
            f"{path}: found a plant file but couldn't identify Plant Code/Latitude/"
            f"Longitude columns. Columns: {list(df.columns)}."
        )

    out = df[[code_col, lat_col, lon_col] + ([name_col] if name_col else [])].copy()
    out.columns = ["plant_code", "latitude", "longitude"] + (["plant_name"] if name_col else [])
    return out.dropna(subset=["latitude", "longitude"])


def load_eia860_generators() -> pd.DataFrame:
    matches = list(RAW_EIA860_DIR.glob("3_1_Generator*.xlsx"))
    if not matches:
        raise FileNotFoundError(
            f"No generator file found in {RAW_EIA860_DIR} (expected something matching "
            "'3_1_Generator*.xlsx'). See module docstring for where to download EIA-860 from."
        )
    path = matches[0]
    df = _read_with_header_detection(path, [PLANT_CODE_CANDIDATES, CAPACITY_CANDIDATES])

    code_col = _find_col(df.columns, PLANT_CODE_CANDIDATES)
    cap_col = _find_col(df.columns, CAPACITY_CANDIDATES)
    fuel_col = _find_col(df.columns, FUEL_CANDIDATES)
    status_col = _find_col(df.columns, STATUS_CANDIDATES)
    if not (code_col and cap_col):
        raise ValueError(
            f"{path}: found a generator file but couldn't identify Plant Code/"
            f"Nameplate Capacity columns. Columns: {list(df.columns)}."
        )

    cols = [code_col, cap_col] + ([fuel_col] if fuel_col else []) + ([status_col] if status_col else [])
    out = df[cols].copy()
    out.columns = ["plant_code", "capacity_mw"] + (["fuel"] if fuel_col else []) + (["status"] if status_col else [])
    if "status" in out.columns:
        before = len(out)
        out = out[out["status"].astype(str).str.strip().str.upper() == "OP"]
        logger.info(f"{path}: {len(out)}/{before} generators are operating (status=='OP').")
    return out


def match_generation_to_island(island_members: list, member_coords: pd.DataFrame) -> dict:
    """member_coords: sub_id, latitude, longitude for Island A's members
    (from va_topology_metrics.csv). Returns total matched capacity_mw and
    the list of matched generators, or zero/empty if EIA-860 hasn't been
    downloaded -- this check is optional, its absence is never an error
    for anything else in this pipeline."""
    if not RAW_EIA860_DIR.exists() or not any(RAW_EIA860_DIR.iterdir()):
        logger.info(
            f"{RAW_EIA860_DIR} not found or empty -- skipping generation check. "
            "See module docstring for where to download EIA-860 from."
        )
        return {"generation_capacity_mw": None, "generators": [], "checked": False}

    plants = load_eia860_plants()
    generators = load_eia860_generators()
    plants_with_capacity = plants.merge(
        generators.groupby("plant_code")["capacity_mw"].sum().reset_index(), on="plant_code", how="inner"
    )
    logger.info(f"{len(plants_with_capacity)} EIA-860 plants with operating generation capacity and known location.")

    members_gdf = gpd.GeoDataFrame(
        member_coords[["sub_id"]],
        geometry=gpd.points_from_xy(member_coords["longitude"], member_coords["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    plants_gdf = gpd.GeoDataFrame(
        plants_with_capacity[["plant_code", "capacity_mw"] + (["plant_name"] if "plant_name" in plants_with_capacity.columns else [])],
        geometry=gpd.points_from_xy(plants_with_capacity["longitude"], plants_with_capacity["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    joined = gpd.sjoin_nearest(plants_gdf, members_gdf, max_distance=SNAP_DISTANCE_M, distance_col="match_distance_m")
    if joined.empty:
        logger.info(f"0 EIA-860 plants matched to Island A's {len(island_members)} members within {SNAP_DISTANCE_M}m.")
        return {"generation_capacity_mw": 0.0, "generators": [], "checked": True}

    total_mw = float(joined["capacity_mw"].sum())
    gen_list = joined[["plant_code", "capacity_mw", "sub_id", "match_distance_m"]].to_dict(orient="records")
    logger.info(
        f"{len(joined)} EIA-860 plants ({total_mw:.1f} MW total) matched to Island A "
        f"within {SNAP_DISTANCE_M}m."
    )
    return {"generation_capacity_mw": total_mw, "generators": gen_list, "checked": True}


def main():
    island_path = VA_NETWORK_DIR / "va_island_a.json"
    if not island_path.exists():
        raise FileNotFoundError(f"Missing {island_path}. Run scripts/select_virginia_island.py first.")
    with open(island_path) as f:
        island = json.load(f)

    topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not topo_path.exists():
        raise FileNotFoundError(f"Missing {topo_path}. Run scripts/build_virginia_network_graph.py first.")
    topo = pd.read_csv(topo_path)
    topo["sub_id"] = topo["sub_id"].astype(str)
    member_coords = topo[topo["sub_id"].isin(island["members"])][["sub_id", "latitude", "longitude"]]

    result = match_generation_to_island(island["members"], member_coords)
    island.update(result)

    with open(island_path, "w") as f:
        json.dump(island, f, indent=2)
    logger.info(f"Updated {island_path} with generation-capacity check.")

    if result["checked"]:
        logger.info(f"Island A existing generation capacity: {result['generation_capacity_mw']:.1f} MW")
    else:
        logger.info("Island A generation capacity: not checked (EIA-860 not downloaded).")


if __name__ == "__main__":
    main()
