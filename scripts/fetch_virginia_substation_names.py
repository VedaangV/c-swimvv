"""
Optional refinement to substation labels used in viz/plot_virginia_criticality.py
and viz/plot_virginia_gic_risk.py: HIFLD's Electric Substations dataset carries
real utility-assigned NAME/OWNER fields that OSM's power=substation tagging
often lacks in this region (most substation features here carry no OSM name
tag at all, which is why those two scripts fall back to operator/voltage/
coordinates for so many of the top-ranked substations).

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS -- see scripts/fetch_
virginia_critical_loads.py's module docstring for why (this sandbox's
egress policy blocks every ArcGIS-hosting domain outright). HIFLD_
SUBSTATIONS_URL below was found via search, not verified by fetching it --
may be wrong or stale; check the logged record count on your first run. If
it's 0 or errors out, this dataset may have been restricted the same way
HIFLD Open's Hospitals dataset was (see fetch_virginia_critical_loads.py's
docstring) -- report back what you see and this becomes fully droppable
(both viz scripts already fall back gracefully when this file doesn't
exist).

Nearest-neighbor matched to this project's own sub_id coordinates (from
va_topology_metrics.csv) within SNAP_DISTANCE_M -- reusing virginia_
config.py's existing line-endpoint snap tolerance rather than inventing a
new distance threshold, since it's the same kind of "same real-world
substation, two different datasets' idea of its exact point" problem
build_admittance_matrix.py already solves for line endpoints. HIFLD's
substation points and this project's OSM-derived ones won't share IDs and
won't be pixel-identical locations, so nearest-within-tolerance is the
match rule, not exact coordinate equality.

Usage:
    python scripts/fetch_virginia_substation_names.py
"""

import time

import geopandas as gpd
import pandas as pd
import requests

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, SNAP_DISTANCE_M, get_virginia_boundary

logger = setup_logger(log_file="logs/fetch_virginia_substation_names.log")

FETCH_USER_AGENT = "c-swim-virginia-adaptation/1.0 (+https://github.com/VedaangV/c-swimvv)"

HIFLD_SUBSTATIONS_URL = (
    "https://services.arcgis.com/G4S1dGvn7PIgYd6Y/ArcGIS/rest/services/"
    "HIFLD_electric_power_substations/FeatureServer/0/query"
)

# Matched case-insensitively-by-exact-name against whatever field names the
# service actually uses -- couldn't verify the real schema from this
# sandbox, so this tries the common HIFLD conventions in order.
NAME_FIELD_CANDIDATES = ("NAME", "SUBSTATION", "SUBST_NAME", "SS_NAME")
OWNER_FIELD_CANDIDATES = ("OWNER", "UTILITY_OWNER", "OPERATOR")

PAGE_SIZE = 2000


def _fetch_hifld_substations(bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Same paginate-until-a-genuinely-empty-page pattern as fetch_
    virginia_critical_loads.py's _fetch_category() -- see that function's
    docstring for why "got fewer than requested" isn't a safe stop signal
    (ArcGIS services commonly cap a page below what's asked for)."""
    headers = {"User-Agent": FETCH_USER_AGENT}
    minx, miny, maxx, maxy = bbox
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": PAGE_SIZE,
        "resultOffset": 0,
    }

    rows = []
    max_pages = 500
    for _ in range(max_pages):
        r = None
        for attempt in range(retries):
            try:
                r = requests.get(HIFLD_SUBSTATIONS_URL, params=params, headers=headers, timeout=60)
                if r.status_code == 200:
                    break
                logger.warning(
                    f"HIFLD substations (attempt {attempt + 1}/{retries}) -> HTTP "
                    f"{r.status_code}: {r.text[:300]!r}"
                )
            except Exception as e:
                logger.warning(f"HIFLD substations (attempt {attempt + 1}/{retries}) -> {e!r}")
                r = None
            time.sleep(2 * (attempt + 1))
        else:
            logger.error(
                f"HIFLD substations: all {retries} attempts failed at offset "
                f"{params['resultOffset']}; keeping {len(rows)} rows fetched so far."
            )
            break

        try:
            data = r.json()
        except Exception as e:
            logger.error(f"HIFLD substations: response wasn't valid JSON ({e!r}); first 300 chars: {r.text[:300]!r}")
            break
        if "error" in data:
            logger.error(f"HIFLD substations: service returned an error: {data['error']}")
            break

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            lon, lat = geom["coordinates"][:2]
            props = feat.get("properties", {}) or {}
            name = next((props[c] for c in NAME_FIELD_CANDIDATES if props.get(c)), None)
            owner = next((props[c] for c in OWNER_FIELD_CANDIDATES if props.get(c)), None)
            if not name:
                continue  # nothing to add over the existing OSM-based fallback
            rows.append({"hifld_name": name, "hifld_owner": owner, "latitude": lat, "longitude": lon})

        params["resultOffset"] += len(features)
    else:
        logger.warning(
            f"HIFLD substations: stopped after {max_pages} consecutive non-empty "
            "pages without ever seeing an empty one -- results may be truncated."
        )

    logger.info(f"HIFLD substations: fetched {len(rows)} named point features within the query bbox.")
    return pd.DataFrame(rows)


def match_to_virginia_substations(hifld: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor join HIFLD's named points onto this project's own
    sub_id coordinates, within SNAP_DISTANCE_M. Returns sub_id, hifld_name,
    hifld_owner, match_distance_m -- only for substations that found a
    match. A sub_id absent from the output file means "no HIFLD name
    available," not an error; callers (the two viz scripts) must keep
    falling back to their existing OSM-based label in that case.
    """
    topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not topo_path.exists():
        raise FileNotFoundError(f"Missing {topo_path}. Run scripts/build_virginia_network_graph.py first.")
    topo = pd.read_csv(topo_path)
    topo["sub_id"] = topo["sub_id"].astype(str)

    ours = gpd.GeoDataFrame(
        topo[["sub_id"]],
        geometry=gpd.points_from_xy(topo["longitude"], topo["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")  # meters, so max_distance below is a real distance

    theirs = gpd.GeoDataFrame(
        hifld[["hifld_name", "hifld_owner"]],
        geometry=gpd.points_from_xy(hifld["longitude"], hifld["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    joined = gpd.sjoin_nearest(ours, theirs, max_distance=SNAP_DISTANCE_M, distance_col="match_distance_m")
    # sjoin_nearest can return more than one row per left geometry on exact
    # distance ties -- keep only the single closest match per sub_id.
    joined = joined.sort_values("match_distance_m").drop_duplicates(subset="sub_id", keep="first")

    unmatched = len(ours) - len(joined)
    logger.info(
        f"{len(joined)}/{len(ours)} Virginia substations matched to a named HIFLD "
        f"substation within {SNAP_DISTANCE_M}m ({unmatched} unmatched, will keep "
        "their existing OSM-based fallback label)."
    )
    return joined[["sub_id", "hifld_name", "hifld_owner", "match_distance_m"]]


def main():
    va_boundary = get_virginia_boundary()
    minx, miny, maxx, maxy = va_boundary.buffer(0.3).bounds

    hifld = _fetch_hifld_substations((minx, miny, maxx, maxy))
    if hifld.empty:
        raise RuntimeError(
            "0 named substations fetched from HIFLD -- check HIFLD_SUBSTATIONS_URL "
            "is still valid (see module docstring); this may mean the dataset has "
            "been restricted, same as HIFLD Open's Hospitals dataset was."
        )

    matches = match_to_virginia_substations(hifld)
    out_path = VA_NETWORK_DIR / "va_hifld_substation_names.csv"
    matches.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
