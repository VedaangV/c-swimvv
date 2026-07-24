"""
Build the Virginia+PJM EHV transmission network from scratch.

Mirrors preprocess/06_dl_power_grid_update.py (OSM substation download) and
preprocess/08_p_power_grid.py (EHV line/substation processing + graph
topology), scoped to the PJM region instead of all of CONUS. The topology
functions in preprocess/08_p_power_grid.py (graph_node_edges, get_EHV_lines,
...) are geography-agnostic, so they are imported and reused unmodified via
importlib (the file's numeric-prefixed name isn't a valid `import` target).
The only piece that differs from the US script is *what* gets fed in:
- substations come from a PJM-bbox Overpass query instead of all of CONUS
- transmission lines are filtered to intersect the PJM boundary instead of
  being spatially joined against FERC order-1000 regions (nerc_gdf.geojson);
  the FERC-derived REGION_ID/REGIONS columns that get_EHV_lines() expects
  are stubbed with a constant "PJM" value since nothing downstream keys off
  their real content for the admittance-matrix build.

Output is written to configs.virginia_config.VA_DATA_DIR / "grid_processed",
i.e. laid out exactly like DATA_DIR / "grid_processed" in the US pipeline.
That lets scripts/build_admittance_matrix.py's load_and_process_data() work
against Virginia data completely unmodified, by passing VA_DATA_DIR in place
of DATA_DIR (see scripts/run_virginia_admittance.py).

Does not download HIFLD lines or run Overpass again if cached files already
exist under VA_NETWORK_DIR.
"""

import importlib.util
import pickle
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from configs import setup_logger, get_data_dir
from configs.virginia_config import (
    EHV_VOLTAGE_THRESHOLD,
    VA_NETWORK_DIR,
    get_pjm_boundary,
    get_pjm_geodataframe,
)

warnings.filterwarnings("ignore")

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/build_virginia_network.log")

ROOT_DIR = Path(__file__).resolve().parent.parent
GRID_OUT_DIR = VA_NETWORK_DIR.parent / "grid_processed"
GRID_OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Same voltage standardization used by preprocess/08_p_power_grid.py, copied
# verbatim so the two pipelines classify voltage classes identically.
LINE_VOLTAGE_RATINGS = {
    345: 345, 230: 230, 450: 500, 500: 500, 765: 765, 250: 230, 400: 345,
    232: 230, 1000: 765, 220: 230, 273: 230, 218: 230, 236: 230, 287: 345,
    238: 230, 200: 230, 167: 161, 169: 161, 161: 161,
}


def _load_module(name: str, relpath: str):
    """Import a numeric-prefixed preprocess script as a module by file path."""
    path = ROOT_DIR / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p06 = _load_module("_c_swim_preprocess_06", "preprocess/06_dl_power_grid_update.py")
p08 = _load_module("_c_swim_preprocess_08", "preprocess/08_p_power_grid.py")


# --------------------------------------------------------------------------
# Step A: PJM substations from OpenStreetMap (Overpass)
# --------------------------------------------------------------------------
def _query_overpass(query: str, timeout: int = 180, retries: int = 2) -> dict | None:
    """POST an Overpass query, trying each server in OVERPASS_SERVERS in turn.
    Logs the real reason for every failed attempt — a single whole-PJM query
    was silently swallowing 400/429 responses from overpass-api.de/kumi
    behind a fast failure, leaving only the last (unrelated) timeout from
    openstreetmap.ru visible in the logs. Chunking by state (see
    download_pjm_substations) also means a failed state doesn't need to
    retry with the same aggressiveness as a one-shot whole-region query, so
    the retry budget here is intentionally smaller than the original."""
    for url in OVERPASS_SERVERS:
        for attempt in range(retries):
            try:
                r = requests.post(url, data={"data": query}, timeout=timeout + 30)
                if r.status_code == 200:
                    return r.json()
                logger.warning(
                    f"Overpass {url} (attempt {attempt + 1}/{retries}) -> "
                    f"HTTP {r.status_code}: {r.text[:300]!r}"
                )
            except Exception as e:
                logger.warning(f"Overpass {url} (attempt {attempt + 1}/{retries}) -> {e!r}")
            time.sleep(3 * (attempt + 1))
    logger.error(f"All Overpass servers failed for this query (first 120 chars: {query.strip()[:120]!r})")
    return None


def _substations_from_nodeway(data: dict | None) -> gpd.GeoDataFrame:
    if not data or not data.get("elements"):
        return gpd.GeoDataFrame()
    return p06.process_substations(p06.convert_to_geodataframe(data))


def _substations_from_relations(data: dict | None) -> gpd.GeoDataFrame:
    if not data or not data.get("elements"):
        return gpd.GeoDataFrame()
    feats = []
    for el in data["elements"]:
        if el.get("type") != "relation" or "center" not in el:
            continue
        tags = el.get("tags", {})
        feats.append(
            {
                "osmid": el["id"],
                "element_type": "relation",
                "original_geom_type": "RelationCenter",
                **tags,
                "geometry": Point(el["center"]["lon"], el["center"]["lat"]),
            }
        )
    if not feats:
        return gpd.GeoDataFrame()
    return p06.process_substations(gpd.GeoDataFrame(feats, crs="EPSG:4326"))


def download_pjm_substations() -> gpd.GeoDataFrame:
    """Download PJM substations (nodes/ways + relations) from OSM, one state
    at a time.

    A single query over the whole 13-state PJM bounding box was getting
    rejected quickly by overpass-api.de/kumi (public-instance query-size/rate
    limits) while overpass.openstreetmap.ru genuinely timed out at the
    network level. Chunking by state — the same strategy
    preprocess/06_dl_power_grid_update.py already uses for the full-CONUS
    download — keeps each request small enough to succeed, and a failure on
    one state no longer loses the other twelve.
    """
    cache = VA_NETWORK_DIR / "va_pjm_substations_raw.geojson"
    if cache.exists():
        logger.info(f"Using cached PJM substations: {cache}")
        return gpd.read_file(cache)

    pjm_states = get_pjm_geodataframe()
    parts = []
    failed_states = []

    for _, row in pjm_states.iterrows():
        state = row["STUSPS"]
        west, south, east, north = row.geometry.bounds
        logger.info(f"Downloading OSM substations for {state} ({south:.2f},{west:.2f},{north:.2f},{east:.2f})...")

        nodeway_query = f"""
        [out:json][timeout:150][bbox:{south},{west},{north},{east}];
        (
          node["power"="substation"];
          way["power"="substation"];
        );
        out body geom;
        """
        relation_query = f"""
        [out:json][timeout:150][bbox:{south},{west},{north},{east}];
        (
          relation["power"="substation"];
        );
        out body center;
        """

        nodeway_gdf = _substations_from_nodeway(_query_overpass(nodeway_query))
        relation_gdf = _substations_from_relations(_query_overpass(relation_query))

        state_parts = [g for g in (nodeway_gdf, relation_gdf) if len(g) > 0]
        if state_parts:
            logger.info(f"  {state}: {sum(len(g) for g in state_parts)} substations")
            parts.extend(state_parts)
        else:
            failed_states.append(state)
            logger.warning(f"  {state}: no substations downloaded (all servers failed)")

        time.sleep(3)

    if not parts:
        raise RuntimeError("No PJM substations downloaded from any Overpass server for any state.")
    if failed_states:
        logger.warning(
            f"Substation download failed for: {failed_states}. The network "
            "will be missing substations in those states — re-run this "
            "script (cached states aren't re-downloaded once "
            f"{cache} exists, so delete it first to retry all states, or "
            "patch in a partial retry for just the failed ones)."
        )

    merged = pd.concat(parts, ignore_index=False)
    merged = merged[~merged.index.duplicated(keep="first")]
    merged = merged.dropna(subset=["geometry"])
    merged = gpd.GeoDataFrame(merged, crs="EPSG:4326").reset_index().rename(
        columns={"index": "osmid"}
    )

    merged.to_file(cache, driver="GeoJSON")
    logger.info(f"Downloaded {len(merged)} PJM substations -> {cache}")
    return merged


# --------------------------------------------------------------------------
# Step B: PJM-filtered HIFLD EHV transmission lines
# --------------------------------------------------------------------------
def load_and_process_pjm_transmission_lines(pjm_boundary) -> gpd.GeoDataFrame:
    """Like preprocess/08's load_and_process_transmission_lines(), but filtered
    to the PJM boundary geographically instead of a FERC-region sjoin."""
    filename = "Electric__Power_Transmission_Lines.shp"
    folder = DATA_LOC / "Electric__Power_Transmission_Lines"
    transmission_lines_path = folder / filename

    gdf = gpd.read_file(transmission_lines_path).to_crs("EPSG:4326")
    gdf.rename(columns={"ID": "line_id"}, inplace=True)
    gdf = gdf.reset_index(drop=True).explode(index_parts=True).reset_index(level=1)
    gdf["line_id"] = gdf.apply(
        lambda row: (
            f"{row['line_id']}_{row['level_1']}" if row["level_1"] > 0 else row["line_id"]
        ),
        axis=1,
    )
    gdf = gdf[gdf["VOLTAGE"] >= EHV_VOLTAGE_THRESHOLD].drop(columns=["level_1"])
    gdf["VOLTAGE"] = gdf["VOLTAGE"].map(LINE_VOLTAGE_RATINGS).fillna(gdf["VOLTAGE"])

    gdf = gdf[gdf.intersects(pjm_boundary)].copy()

    # get_EHV_lines()'s downstream FERC filter step only ever selects these
    # columns by name; PJM has no FERC-region concept, so stub a constant.
    gdf["REGION_ID"] = "PJM"
    gdf["REGIONS"] = "PJM"

    import bezpy

    gdf["length"] = gdf.apply(lambda row: bezpy.tl.TransmissionLine(row).length, axis=1)
    return gdf


def load_and_process_pjm_substations(substations_gdf, pjm_boundary) -> gpd.GeoDataFrame:
    """Like preprocess/08's load_and_process_substations(), but filtered to the
    PJM boundary geographically instead of a FERC-region sjoin."""
    substations_gdf = substations_gdf.to_crs("EPSG:4326")

    if "osmid" in substations_gdf.columns:
        substations_gdf["ss_id"] = substations_gdf["osmid"].astype(str)
    else:
        substations_gdf["ss_id"] = substations_gdf.index.astype(str)

    substations_gdf.dropna(subset=["geometry"], inplace=True)
    substations_gdf = substations_gdf[substations_gdf.intersects(pjm_boundary)].copy()
    substations_gdf["REGION_ID"] = "PJM"
    substations_gdf["REGIONS"] = "PJM"

    column_mapping = {
        "name": "SS_NAME",
        "operator": "SS_OPERATOR",
        "voltage": "SS_VOLTAGE",
        "substation": "SS_TYPE",
    }
    for osm_col, expected_col in column_mapping.items():
        if osm_col in substations_gdf.columns and expected_col not in substations_gdf.columns:
            substations_gdf[expected_col] = substations_gdf[osm_col]

    for col in ("SS_NAME", "SS_OPERATOR", "SS_VOLTAGE", "SS_TYPE"):
        if col not in substations_gdf.columns:
            substations_gdf[col] = "Unknown"

    return p08.prepare_substations_for_grid_analysis(substations_gdf)


def validate_network(ss_df: pd.DataFrame, df_lines_EHV: pd.DataFrame) -> None:
    """Log basic connectivity/coverage sanity checks."""
    n_lines = len(df_lines_EHV)
    n_subs = len(ss_df)
    connected = set(df_lines_EHV["substation_a"]) | set(df_lines_EHV["substation_b"])
    n_connected = len(connected & set(ss_df["SS_ID"]))
    missing_sub = df_lines_EHV[
        df_lines_EHV["substation_a"].isna() | df_lines_EHV["substation_b"].isna()
    ]
    logger.info(f"VA+PJM network: {n_lines} lines, {n_subs} substations")
    logger.info(f"Substations connected to >=1 EHV line: {n_connected}/{n_subs}")
    if len(missing_sub) > 0:
        logger.warning(f"{len(missing_sub)} lines missing a substation endpoint")


def main():
    pjm_boundary = get_pjm_boundary()

    logger.info("Downloading/loading PJM substations from OSM...")
    substations_raw = download_pjm_substations()

    logger.info("Loading and filtering HIFLD EHV lines to PJM...")
    translines_gdf = load_and_process_pjm_transmission_lines(pjm_boundary)
    logger.info(f"PJM EHV lines (pre-topology): {len(translines_gdf)}")

    logger.info("Preparing PJM substations for grid analysis...")
    substation_gdf = load_and_process_pjm_substations(substations_raw, pjm_boundary)

    logger.info("Building graph topology (substation <-> line intersections)...")
    substation_gdf, tl_gdf_subset, ss_gdf_subset = p08.graph_node_edges(
        substation_gdf, translines_gdf, buffer_distance=150
    )

    logger.info("Extracting EHV lines and bus topology...")
    (
        unique_sub_voltage_pairs,
        substation_to_line_voltages,
        ss_df,
        df_lines_EHV,
        trans_lines_within_FERC_filtered_,
        tl_gdf_subset,
    ) = p08.get_EHV_lines(substation_gdf, tl_gdf_subset, translines_gdf)

    validate_network(ss_df, df_lines_EHV)

    logger.info(f"Saving Virginia+PJM network to {GRID_OUT_DIR}")
    with open(GRID_OUT_DIR / "unique_sub_voltage_pairs.pkl", "wb") as f:
        pickle.dump(unique_sub_voltage_pairs, f)
    with open(GRID_OUT_DIR / "substation_to_line_voltages.pkl", "wb") as f:
        pickle.dump(substation_to_line_voltages, f)
    with open(GRID_OUT_DIR / "ss_df.pkl", "wb") as f:
        pickle.dump(ss_df, f)
    with open(GRID_OUT_DIR / "df_lines_EHV.pkl", "wb") as f:
        pickle.dump(df_lines_EHV, f)
    with open(GRID_OUT_DIR / "trans_lines_within_FERC_filtered.pkl", "wb") as f:
        pickle.dump(trans_lines_within_FERC_filtered_, f)

    # Visualization copies (not consumed by any downstream script).
    df_lines_EHV.to_file(VA_NETWORK_DIR / "va_pjm_lines.geojson", driver="GeoJSON")
    ss_gdf = gpd.GeoDataFrame(
        ss_df,
        geometry=gpd.points_from_xy(ss_df["lon"], ss_df["lat"]),
        crs="EPSG:4326",
    )
    ss_gdf.to_file(VA_NETWORK_DIR / "va_pjm_substations.geojson", driver="GeoJSON")

    logger.info("Virginia+PJM network build complete.")
    logger.info(
        "NOTE: scripts/build_admittance_matrix.py's load_and_process_data() also "
        "expects grid_processed/trans_lines_pickle.pkl and a top-level "
        "grid_mapping.csv next to this directory — both are produced by "
        "scripts/recompute_virginia_line_voltages.py and "
        "scripts/run_virginia_admittance.py respectively, not by this script."
    )


if __name__ == "__main__":
    main()
