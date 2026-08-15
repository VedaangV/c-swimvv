"""
Optional refinement to substation labels used in viz/plot_virginia_criticality.py
and viz/plot_virginia_gic_risk.py: pulls real substation names from external
sources that OSM's power=substation tagging often lacks in this region (most
substation features here carry no OSM name tag at all, which is why those
two scripts fall back to operator/voltage/coordinates for so many of the
top-ranked substations).

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS -- see scripts/fetch_
virginia_critical_loads.py's module docstring for why (this sandbox's
egress policy blocks essentially every external domain outright -- not just
ArcGIS; en.wikipedia.org and query.wikidata.org were both confirmed
egress-blocked too while building this).

Two independent sources, tried separately and merged (HIFLD takes priority
where both matched the same substation, as the more authoritative,
utility-sourced name over a crowd-sourced one):

1. HIFLD's Electric Substations dataset (HIFLD_SUBSTATIONS_URL, ORNL-
   hosted, found via search). On the first real run this returned 0 raw
   features for the whole Virginia bounding box -- confirmed via a direct
   diagnostic query (not just a schema/field-name mismatch) -- so this
   dataset may have been restricted the same way HIFLD Open's Hospitals
   dataset was (see fetch_virginia_critical_loads.py's docstring). Left in
   place since it's harmless if empty and might work in the future or in
   a different network environment; do not be surprised if it returns 0.
2. Wikidata, via its public SPARQL query service: items that are
   (transitively, via wdt:P31/wdt:P279*) an instance of "electrical
   substation" (Q174814 -- verified against Wikidata directly via search,
   not guessed; this transitive path also catches subclasses like
   "distribution substation") with a coordinate inside Virginia's bounding
   box. A genuinely different kind of source than HIFLD -- community-
   maintained, not a federal/DHS-administered dataset -- so it fails (or
   doesn't) for different reasons. Expect sparse, notability-skewed
   coverage: Wikidata is not an exhaustive infrastructure inventory, and
   will likely only have larger/more notable substations, not most of the
   402 in this project's network.

Both sources nearest-neighbor matched to this project's own sub_id
coordinates (from va_topology_metrics.csv) within SNAP_DISTANCE_M --
reusing virginia_config.py's existing line-endpoint snap tolerance rather
than inventing a new distance threshold, since it's the same kind of "same
real-world substation, two different datasets' idea of its exact point"
problem build_admittance_matrix.py already solves for line endpoints.

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

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SUBSTATION_QUERY = """
SELECT ?item ?itemLabel ?coord WHERE {{
  SERVICE wikibase:box {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerWest "Point({minx} {miny})"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerEast "Point({maxx} {maxy})"^^geo:wktLiteral .
  }}
  ?item wdt:P31/wdt:P279* wd:Q174814 .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

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
            rows.append({"name": name, "owner": owner, "latitude": lat, "longitude": lon})

        params["resultOffset"] += len(features)
    else:
        logger.warning(
            f"HIFLD substations: stopped after {max_pages} consecutive non-empty "
            "pages without ever seeing an empty one -- results may be truncated."
        )

    logger.info(f"HIFLD substations: fetched {len(rows)} named point features within the query bbox.")
    return pd.DataFrame(rows)


def _fetch_wikidata_substations(bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Query Wikidata Query Service for electrical-substation-like items
    (see module docstring for the Q174814 class and why the transitive
    P31/P279* path is used) with a coordinate inside bbox."""
    minx, miny, maxx, maxy = bbox
    query = WIKIDATA_SUBSTATION_QUERY.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    headers = {"User-Agent": FETCH_USER_AGENT, "Accept": "application/sparql-results+json"}

    r = None
    for attempt in range(retries):
        try:
            r = requests.get(
                WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"}, headers=headers, timeout=90
            )
            if r.status_code == 200:
                break
            logger.warning(f"Wikidata SPARQL (attempt {attempt + 1}/{retries}) -> HTTP {r.status_code}: {r.text[:300]!r}")
        except Exception as e:
            logger.warning(f"Wikidata SPARQL (attempt {attempt + 1}/{retries}) -> {e!r}")
            r = None
        time.sleep(3 * (attempt + 1))
    else:
        logger.error(f"Wikidata SPARQL: all {retries} attempts failed.")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])

    try:
        data = r.json()
    except Exception as e:
        logger.error(f"Wikidata SPARQL: response wasn't valid JSON ({e!r}); first 300 chars: {r.text[:300]!r}")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])

    bindings = data.get("results", {}).get("bindings", [])
    rows = []
    for b in bindings:
        label = b.get("itemLabel", {}).get("value")
        coord = b.get("coord", {}).get("value")  # "Point(lon lat)"
        if not label or not coord:
            continue
        try:
            lon_str, lat_str = coord.replace("Point(", "").replace(")", "").split(" ")
            lon, lat = float(lon_str), float(lat_str)
        except Exception:
            continue
        rows.append({"name": label, "latitude": lat, "longitude": lon})

    logger.info(f"Wikidata: fetched {len(rows)} named substation-like points within the query bbox.")
    return pd.DataFrame(rows)


def _nearest_match(ours_3857: gpd.GeoDataFrame, named_points: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor join named_points (must have name/latitude/
    longitude in EPSG:4326) onto ours_3857 (this project's own sub_id
    points, already projected to EPSG:3857) within SNAP_DISTANCE_M. Returns
    sub_id, name, match_distance_m -- one row per matched sub_id (closest
    kept on ties). Empty input -> empty, correctly-shaped output, not an
    error."""
    if named_points.empty:
        return pd.DataFrame(columns=["sub_id", "name", "match_distance_m"])

    theirs = gpd.GeoDataFrame(
        named_points[["name"]],
        geometry=gpd.points_from_xy(named_points["longitude"], named_points["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    joined = gpd.sjoin_nearest(ours_3857, theirs, max_distance=SNAP_DISTANCE_M, distance_col="match_distance_m")
    joined = joined.sort_values("match_distance_m").drop_duplicates(subset="sub_id", keep="first")
    return joined[["sub_id", "name", "match_distance_m"]]


def main():
    va_boundary = get_virginia_boundary()
    minx, miny, maxx, maxy = va_boundary.buffer(0.3).bounds

    topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not topo_path.exists():
        raise FileNotFoundError(f"Missing {topo_path}. Run scripts/build_virginia_network_graph.py first.")
    topo = pd.read_csv(topo_path)
    topo["sub_id"] = topo["sub_id"].astype(str)
    ours_3857 = gpd.GeoDataFrame(
        topo[["sub_id"]],
        geometry=gpd.points_from_xy(topo["longitude"], topo["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    hifld = _fetch_hifld_substations((minx, miny, maxx, maxy))
    hifld_matches = _nearest_match(ours_3857, hifld)
    hifld_matches = hifld_matches.assign(source="hifld")
    logger.info(f"HIFLD: {len(hifld_matches)}/{len(ours_3857)} substations matched within {SNAP_DISTANCE_M}m.")

    wikidata = _fetch_wikidata_substations((minx, miny, maxx, maxy))
    wikidata_matches = _nearest_match(ours_3857, wikidata)
    wikidata_matches = wikidata_matches.assign(source="wikidata")
    logger.info(f"Wikidata: {len(wikidata_matches)}/{len(ours_3857)} substations matched within {SNAP_DISTANCE_M}m.")

    # HIFLD listed first -> keep="first" on a duplicate sub_id prefers the
    # HIFLD (utility-sourced) name over Wikidata's (crowd-sourced) one.
    combined = pd.concat([hifld_matches, wikidata_matches], ignore_index=True)
    combined = combined.drop_duplicates(subset="sub_id", keep="first")

    if combined.empty:
        logger.warning(
            "No named substations matched from either source -- writing an "
            "empty result. Both viz scripts fall back cleanly to OSM-based "
            "labels when this file is empty/missing."
        )

    out_path = VA_NETWORK_DIR / "va_substation_names.csv"
    combined.to_csv(out_path, index=False)
    logger.info(
        f"Saved {out_path}: {len(combined)} substations with a real name "
        f"({(combined['source'] == 'hifld').sum()} from HIFLD, "
        f"{(combined['source'] == 'wikidata').sum()} from Wikidata)."
    )


if __name__ == "__main__":
    main()
